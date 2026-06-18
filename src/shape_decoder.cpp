#include "shape_decoder.h"
#include "sparse.h"
#include "trellis_model.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <string>
#include <stdexcept>

namespace trellis {
using T = ggml_tensor;

// run a graph producing `out`, with the given input tensors set from host data
static std::vector<float> run1(const Model& m, ggml_context* c, T* out,
                               std::vector<std::pair<T*, const void*>> ins) {
    ggml_set_output(out);
    ggml_cgraph* g = ggml_new_graph_custom(c, 65536, false);
    ggml_build_forward_expand(g, out);
    ggml_gallocr_t a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    if (!ggml_gallocr_alloc_graph(a, g)) throw std::runtime_error("shape_dec alloc");
    for (auto& [t, d] : ins) ggml_backend_tensor_set(t, d, 0, ggml_nbytes(t));
    if (ggml_backend_graph_compute(m.backend, g) != GGML_STATUS_SUCCESS) throw std::runtime_error("shape_dec compute");
    std::vector<float> r = tensor_to_f32(out);
    ggml_gallocr_free(a);
    return r;
}

static ggml_context* mkctx() {
    size_t meta = ggml_tensor_overhead() * 65536 + ggml_graph_overhead_custom(65536, false) + (1 << 20);
    return ggml_init({ meta, nullptr, true });
}

// Generic SparseUnetVaeDecoder: from_latent -> 4×(ConvNeXt stage + C2S) -> final LN -> output_layer.
// guide_subs!=null -> tex (C2S driven by provided masks); subs_out!=null -> shape (collect masks).
static std::vector<float> decode_unet(const Model& m, const std::vector<float>& latent, int out_ch,
                                      std::vector<std::array<int,3>>& coords,
                                      const std::vector<std::vector<uint8_t>>* guide_subs,
                                      std::vector<std::vector<uint8_t>>* subs_out,
                                      bool coords_only = false) {
    int N = (int)coords.size();
    std::vector<float> h;
    {   // from_latent [32,N] -> [1024,N]
        ggml_context* c = mkctx();
        T* gf = ggml_new_tensor_2d(c, GGML_TYPE_F32, 32, N); ggml_set_input(gf);
        T* o = ggml_add(c, ggml_mul_mat(c, m.get("from_latent.weight"), gf), m.get("from_latent.bias"));
        h = run1(m, c, o, { {gf, latent.data()} });
        ggml_free(c);
    }
    struct Stage { int C, nblk, Cout, c2si; const char* s; };
    const Stage stages[4] = { {1024,4,512,4,"0"}, {512,16,256,16,"1"}, {256,8,128,8,"2"}, {128,4,64,4,"3"} };
    for (int si = 0; si < 4; ++si) {
        const Stage& st = stages[si];
        std::vector<int32_t> nbr = build_neighbor_table(coords);
        {   // ConvNeXt stage
            ggml_context* c = mkctx();
            T* gh = ggml_new_tensor_2d(c, GGML_TYPE_F32, st.C, N); ggml_set_input(gh);
            T* gn = ggml_new_tensor_2d(c, GGML_TYPE_I32, N, 27);   ggml_set_input(gn);
            T* x = gh;
            for (int j = 0; j < st.nblk; ++j)
                x = sparse_convnext(c, m, std::string("blocks.") + st.s + "." + std::to_string(j), x, gn, N);
            h = run1(m, c, x, { {gh, h.data()}, {gn, nbr.data()} });
            ggml_free(c);
        }
        const std::vector<uint8_t>* ext = guide_subs ? &(*guide_subs)[si] : nullptr;
        C2SResult r = sparse_c2s(m, std::string("blocks.") + st.s + "." + std::to_string(st.c2si), h, st.C, coords, st.Cout, ext);
        if (subs_out) subs_out->push_back(r.subdiv);
        h = std::move(r.feats); coords = std::move(r.coords); N = (int)coords.size();
    }
    if (coords_only) return {};   // cascade upsample: just need the grown coords
    std::vector<float> out;
    {   // final LN(no affine) + output_layer -> [out_ch, M]
        ggml_context* c = mkctx();
        T* gh = ggml_new_tensor_2d(c, GGML_TYPE_F32, 64, N); ggml_set_input(gh);
        T* x = ggml_norm(c, gh, 1e-5f);
        x = ggml_add(c, ggml_mul_mat(c, m.get("output_layer.weight"), x), m.get("output_layer.bias"));
        out = run1(m, c, x, { {gh, h.data()} });
        ggml_free(c);
    }
    return out;
}

ShapeOut shape_decode(const Model& m, const std::vector<float>& latent,
                      const std::vector<std::array<int,3>>& coords0, int resolution) {
    ShapeOut so; so.coords = coords0; so.res = resolution;
    so.feats7 = decode_unet(m, latent, 7, so.coords, nullptr, &so.subs);
    return so;
}

std::vector<std::array<int,3>> shape_upsample(const Model& m, const std::vector<float>& latent,
                                              const std::vector<std::array<int,3>>& coords0) {
    std::vector<std::array<int,3>> coords = coords0;
    decode_unet(m, latent, 7, coords, nullptr, nullptr, /*coords_only=*/true);
    return coords;
}

std::vector<float> tex_decode(const Model& m, const std::vector<float>& tex_latent,
                              const std::vector<std::array<int,3>>& coords0,
                              const std::vector<std::vector<uint8_t>>& subs) {
    std::vector<std::array<int,3>> coords = coords0;
    return decode_unet(m, tex_latent, 6, coords, &subs, nullptr);   // 6-ch PBR at the same final coords
}

} // namespace trellis
