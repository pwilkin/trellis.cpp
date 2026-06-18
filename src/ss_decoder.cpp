#include "ss_decoder.h"
#include "trellis_model.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <array>
#include <functional>
#include <stdexcept>

namespace trellis {
using T = ggml_tensor;

// ---- graph ops (features held channels-last [s0,s1,s2,C], ne0=s0 fastest) ----

// ChannelLayerNorm over C (=ne3): permute C->ne0, ggml_norm, affine, permute back.
static T* chln(ggml_context* c, const Model& m, const std::string& p, T* x) {
    T* h = ggml_cont(c, ggml_permute(c, x, 1, 2, 3, 0));        // [C,s0,s1,s2]
    h = ggml_norm(c, h, 1e-5f);
    h = ggml_mul(c, h, m.get(p + ".weight"));
    h = ggml_add(c, h, m.get(p + ".bias"));
    return ggml_cont(c, ggml_permute(c, h, 3, 0, 1, 2));        // back to [s0,s1,s2,C]
}

// Conv3d (channels-last in/out). w: [k,k,k,IC*OC]; bias [OC].
static T* conv3d(ggml_context* c, const Model& m, const std::string& p, T* x, int IC) {
    T* w = m.get(p + ".weight");
    const int k = (int)w->ne[0];
    const int pad = (k - 1) / 2;
    T* y = ggml_conv_3d(c, w, x, IC, 1, 1, 1, pad, pad, pad, 1, 1, 1);  // [s0,s1,s2,OC]
    const int64_t OC = w->ne[3] / IC;
    T* b = ggml_reshape_4d(c, m.get(p + ".bias"), 1, 1, 1, OC);
    return ggml_add(c, y, b);
}

static T* resblock(ggml_context* c, const Model& m, const std::string& p, T* x, int ch) {
    T* h = chln(c, m, p + ".norm1", x);
    h = ggml_silu(c, h);
    h = conv3d(c, m, p + ".conv1", h, ch);
    h = chln(c, m, p + ".norm2", h);
    h = ggml_silu(c, h);
    h = conv3d(c, m, p + ".conv2", h, ch);
    return ggml_add(c, h, x);          // ResBlock3d(ch,ch): skip is Identity
}

// run a single-graph segment: in_host laid channels-last [n0,n1,n2,C]
struct Seg { std::vector<float> data; };
static Seg run_seg(const Model& m, const std::vector<float>& in_host,
                   std::array<int64_t,4> ne,
                   const std::function<T*(ggml_context*, T*)>& build) {
    size_t meta = ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(8192, false) + (1 << 20);
    ggml_context* c = ggml_init({ meta, nullptr, true });
    T* in = ggml_new_tensor_4d(c, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]); ggml_set_input(in);
    T* out = build(c, in); ggml_set_output(out);
    ggml_cgraph* g = ggml_new_graph_custom(c, 8192, false);
    ggml_build_forward_expand(g, out);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    if (!ggml_gallocr_alloc_graph(alloc, g)) throw std::runtime_error("ss_dec: alloc failed");
    ggml_backend_tensor_set(in, in_host.data(), 0, in_host.size() * 4);
    if (ggml_backend_graph_compute(m.backend, g) != GGML_STATUS_SUCCESS) throw std::runtime_error("ss_dec: compute failed");
    Seg s; s.data = tensor_to_f32(out);
    ggml_gallocr_free(alloc); ggml_free(c);
    return s;
}

// pixel_shuffle_3d(scale 2): in torch [C=Co*8,H,W,D] -> out torch [Co,2H,2W,2D]
// (both stored == ggml channels-last). c = co*8 + s0*4+s1*2+s2 ; s0->H,s1->W,s2->D.
static std::vector<float> pixel_shuffle(const std::vector<float>& in, int H, int W, int D, int Co) {
    const int H2 = 2*H, W2 = 2*W, D2 = 2*D;
    std::vector<float> out((size_t)Co * H2 * W2 * D2);
    for (int co = 0; co < Co; ++co)
      for (int h = 0; h < H; ++h) for (int w = 0; w < W; ++w) for (int d = 0; d < D; ++d)
        for (int s = 0; s < 8; ++s) {
            int s0 = (s >> 2) & 1, s1 = (s >> 1) & 1, s2 = s & 1;
            int cc = co * 8 + s;
            size_t isrc = (((size_t)cc * H + h) * W + w) * D + d;
            size_t odst = (((size_t)co * H2 + (2*h+s0)) * W2 + (2*w+s1)) * D2 + (2*d+s2);
            out[odst] = in[isrc];
        }
    return out;
}

std::vector<float> ss_decode(const Model& m, const std::vector<float>& z) {
    // Segment 1: input_layer + middle(2) + blocks.0,1 (res512) + blocks.2.conv  -> [16,16,16,128*8]
    Seg s1 = run_seg(m, z, {16,16,16,8}, [&](ggml_context* c, T* x) {
        T* h = conv3d(c, m, "input_layer", x, 8);
        h = resblock(c, m, "middle_block.0", h, 512);
        h = resblock(c, m, "middle_block.1", h, 512);
        h = resblock(c, m, "blocks.0", h, 512);
        h = resblock(c, m, "blocks.1", h, 512);
        return conv3d(c, m, "blocks.2.conv", h, 512);     // [16,16,16,1024]
    });
    std::vector<float> up1 = pixel_shuffle(s1.data, 16, 16, 16, 128);  // [32,32,32,128]

    // Segment 2: blocks.3,4 (res128) + blocks.5.conv -> [32,32,32,32*8]
    Seg s2 = run_seg(m, up1, {32,32,32,128}, [&](ggml_context* c, T* x) {
        T* h = resblock(c, m, "blocks.3", x, 128);
        h = resblock(c, m, "blocks.4", h, 128);
        return conv3d(c, m, "blocks.5.conv", h, 128);     // [32,32,32,256]
    });
    std::vector<float> up2 = pixel_shuffle(s2.data, 32, 32, 32, 32);   // [64,64,64,32]

    // Segment 3: blocks.6,7 (res32) + out_layer -> [64,64,64,1]
    Seg s3 = run_seg(m, up2, {64,64,64,32}, [&](ggml_context* c, T* x) {
        T* h = resblock(c, m, "blocks.6", x, 32);
        h = resblock(c, m, "blocks.7", h, 32);
        h = chln(c, m, "out_layer.0", h);
        h = ggml_silu(c, h);
        return conv3d(c, m, "out_layer.2", h, 32);        // [64,64,64,1]
    });
    return s3.data;   // torch [1,1,64,64,64] memory
}

std::vector<std::array<int,3>> ss_coords(const std::vector<float>& logits, int src_res, int out_res) {
    const int S = src_res, r = src_res / out_res;
    auto occ = [&](int x, int y, int z) { return logits[((size_t)x * S + y) * S + z] > 0.0f; };
    std::vector<std::array<int,3>> out;
    for (int x = 0; x < out_res; ++x)
      for (int y = 0; y < out_res; ++y)
        for (int z = 0; z < out_res; ++z) {
            bool active = false;
            for (int i = 0; i < r && !active; ++i)
              for (int j = 0; j < r && !active; ++j)
                for (int k = 0; k < r; ++k)
                    if (occ(x*r+i, y*r+j, z*r+k)) { active = true; break; }
            if (active) out.push_back({x, y, z});
        }
    return out;
}

} // namespace trellis
