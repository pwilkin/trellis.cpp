#include "sparse.h"
#include "trellis_model.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <unordered_map>
#include <stdexcept>

namespace trellis {
using T = ggml_tensor;

bool g_sparse_cast_f32 = false;   // cast f16 conv weights to f32 (precision check)
static T* w32(ggml_context* c, T* w) {
    return (g_sparse_cast_f32 && w->type == GGML_TYPE_F16) ? ggml_cast(c, w, GGML_TYPE_F32) : w;
}

std::vector<int32_t> build_neighbor_table(const std::vector<std::array<int,3>>& coords) {
    const int N = (int)coords.size();
    std::unordered_map<uint64_t, int> cmap;
    cmap.reserve(N * 2);
    auto key = [](int x, int y, int z) { return ((uint64_t)(uint32_t)x << 40) | ((uint64_t)(uint32_t)y << 20) | (uint32_t)z; };
    for (int i = 0; i < N; ++i) cmap[key(coords[i][0], coords[i][1], coords[i][2])] = i;
    std::vector<int32_t> nbr((size_t)27 * N, N);   // tap-major; default = sentinel zero row (N)
    for (int i = 0; i < N; ++i) {
        int x = coords[i][0], y = coords[i][1], z = coords[i][2];
        for (int kd = 0; kd < 3; ++kd)
          for (int kh = 0; kh < 3; ++kh)
            for (int kw = 0; kw < 3; ++kw) {
                int t = kd*9 + kh*3 + kw;
                auto it = cmap.find(key(x+kd-1, y+kh-1, z+kw-1));
                nbr[(size_t)t * N + i] = (it == cmap.end()) ? N : it->second;
            }
    }
    return nbr;
}

ggml_tensor* sparse_submconv(ggml_context* c, const Model& m, const std::string& prefix,
                             T* feats, T* nbr, int N) {
    T* W = w32(c, m.get(prefix + ".weight"));  // ggml ne [Ci,27,Co]
    const int64_t Ci = W->ne[0], Co = W->ne[2];
    T* zero = ggml_scale(c, ggml_view_2d(c, feats, Ci, 1, feats->nb[1], 0), 0.0f); // [Ci,1] zeros
    T* fz = ggml_concat(c, feats, ggml_cont(c, zero), 1);          // [Ci, N+1]
    T* acc = nullptr;
    for (int t = 0; t < 27; ++t) {
        T* idx = ggml_view_1d(c, nbr, N, (size_t)t * N * ggml_element_size(nbr));   // [N] i32
        T* g = ggml_get_rows(c, fz, idx);                          // [Ci, N]
        T* Wt = ggml_cont(c, ggml_view_2d(c, W, Ci, Co, W->nb[2], (size_t)t * W->nb[1])); // [Ci,Co]
        T* o = ggml_mul_mat(c, Wt, g);                             // [Co, N]
        acc = acc ? ggml_add(c, acc, o) : o;
    }
    return ggml_add(c, acc, ggml_reshape_2d(c, m.get(prefix + ".bias"), Co, 1));
}

ggml_tensor* sparse_convnext(ggml_context* c, const Model& m, const std::string& prefix,
                             T* feats, T* nbr, int N) {
    T* h = sparse_submconv(c, m, prefix + ".conv", feats, nbr, N);   // [C,N]
    h = ggml_norm(c, h, 1e-6f);                                       // rowLN over ne0=C
    h = ggml_add(c, ggml_mul(c, h, m.get(prefix + ".norm.weight")), m.get(prefix + ".norm.bias"));
    h = ggml_add(c, ggml_mul_mat(c, m.get(prefix + ".mlp.0.weight"), h), m.get(prefix + ".mlp.0.bias"));
    h = ggml_silu(c, h);
    h = ggml_add(c, ggml_mul_mat(c, m.get(prefix + ".mlp.2.weight"), h), m.get(prefix + ".mlp.2.bias"));
    return ggml_add(c, h, feats);                                     // residual on block input
}

// Run a graph with given inputs (name->host data), return one named output to host.
namespace {
struct GraphRun {
    const Model& m; ggml_context* c; ggml_gallocr_t alloc = nullptr;
    GraphRun(const Model& mm) : m(mm) {
        size_t meta = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(16384, false) + (1 << 20);
        c = ggml_init({ meta, nullptr, true });
    }
    ~GraphRun() { if (alloc) ggml_gallocr_free(alloc); ggml_free(c); }
    std::vector<float> run(ggml_tensor* out, const std::vector<std::pair<ggml_tensor*, const void*>>& inputs) {
        ggml_set_output(out);
        ggml_cgraph* g = ggml_new_graph_custom(c, 16384, false);
        ggml_build_forward_expand(g, out);
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
        if (!ggml_gallocr_alloc_graph(alloc, g)) throw std::runtime_error("c2s: alloc failed");
        for (auto& [t, data] : inputs) ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t));
        if (ggml_backend_graph_compute(m.backend, g) != GGML_STATUS_SUCCESS) throw std::runtime_error("c2s: compute failed");
        return tensor_to_f32(out);
    }
};
} // anon

C2SResult sparse_c2s(const Model& m, const std::string& prefix,
                     const std::vector<float>& feats_in, int Cin,
                     const std::vector<std::array<int,3>>& coords, int Cout,
                     const std::vector<uint8_t>* ext_subdiv) {
    const int N = (int)coords.size();
    std::vector<int32_t> nbr = build_neighbor_table(coords);

    // ---- graph 1: conv1_pre [Cout*8,N], and (shape decoder only) subdiv [8,N] ----
    std::vector<float> subdiv, conv1;
    {
        GraphRun gr1(m);
        ggml_context* c = gr1.c;
        ggml_tensor* gf = ggml_new_tensor_2d(c, GGML_TYPE_F32, Cin, N); ggml_set_input(gf);
        ggml_tensor* gn = ggml_new_tensor_2d(c, GGML_TYPE_I32, N, 27);  ggml_set_input(gn);
        ggml_tensor* sd = nullptr;
        if (!ext_subdiv)   // pred_subdiv: to_subdiv(feats)
            sd = ggml_add(c, ggml_mul_mat(c, m.get(prefix + ".to_subdiv.weight"), gf), m.get(prefix + ".to_subdiv.bias"));
        ggml_tensor* h = ggml_norm(c, gf, 1e-6f);
        h = ggml_add(c, ggml_mul(c, h, m.get(prefix + ".norm1.weight")), m.get(prefix + ".norm1.bias"));
        h = ggml_silu(c, h);
        ggml_tensor* cv = sparse_submconv(c, m, prefix + ".conv1", h, gn, N);  // [Cout*8, N]
        if (sd) ggml_set_output(sd);
        ggml_set_output(cv);
        ggml_cgraph* g = ggml_new_graph_custom(c, 16384, false);
        if (sd) ggml_build_forward_expand(g, sd);
        ggml_build_forward_expand(g, cv);
        gr1.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
        if (!ggml_gallocr_alloc_graph(gr1.alloc, g)) throw std::runtime_error("c2s g1 alloc");
        ggml_backend_tensor_set(gf, feats_in.data(), 0, (size_t)Cin * N * 4);
        ggml_backend_tensor_set(gn, nbr.data(), 0, nbr.size() * 4);
        if (ggml_backend_graph_compute(m.backend, g) != GGML_STATUS_SUCCESS) throw std::runtime_error("c2s g1 compute");
        if (sd) subdiv = tensor_to_f32(sd);
        conv1  = tensor_to_f32(cv);
        if (sd && getenv("TRELLIS_DBG_SUBDIV")) {
            size_t pos = 0; double smin = 0, smax = 0; bool first = true;
            for (float x : subdiv) { if (x > 0.0f) pos++; if (first) { smin = smax = x; first = false; }
                else { if (x < smin) smin = x; if (x > smax) smax = x; } }
            fprintf(stderr, "      [c2s] %-26s N=%d  subdiv>0: %zu/%zu (%.2f%%)  logit[min=%.3f max=%.3f]\n",
                    prefix.c_str(), N, pos, subdiv.size(),
                    subdiv.empty() ? 0.0 : 100.0 * pos / subdiv.size(), smin, smax);
        }
    }

    // ---- host: octant subdivision (predicted >0, or external mask), gather, new coords/nbr, skip ----
    const int K = Cin / 8, R = Cout / K;
    std::vector<uint8_t> mask_used((size_t)8 * N);
    std::vector<std::array<int,3>> nc;
    std::vector<float> hnew, xs;
    hnew.reserve((size_t)Cout * N); xs.reserve((size_t)K * N);
    for (int i = 0; i < N; ++i) {
        int x = coords[i][0], y = coords[i][1], z = coords[i][2];
        for (int o = 0; o < 8; ++o) {
            bool on = ext_subdiv ? (*ext_subdiv)[(size_t)o + 8*i] != 0 : (subdiv[(size_t)o + 8 * i] > 0.0f);
            mask_used[(size_t)o + 8*i] = on ? 1 : 0;
            if (!on) continue;
            nc.push_back({ 2*x + (o & 1), 2*y + ((o>>1) & 1), 2*z + ((o>>2) & 1) });
            for (int k = 0; k < Cout; ++k) hnew.push_back(conv1[(size_t)(o*Cout + k) + (size_t)(Cout*8) * i]);
            for (int k = 0; k < K;    ++k) xs.push_back(feats_in[(size_t)(o*K + k) + (size_t)Cin * i]);
        }
    }
    const int M = (int)nc.size();
    std::vector<int32_t> nnbr = build_neighbor_table(nc);
    // skip = repeat_interleave(xs[K,M], R) -> [Cout, M]
    std::vector<float> skip((size_t)Cout * M);
    for (int cidx = 0; cidx < M; ++cidx)
        for (int k = 0; k < K; ++k) { float v = xs[(size_t)k + (size_t)K*cidx]; for (int r = 0; r < R; ++r) skip[(size_t)(k*R + r) + (size_t)Cout*cidx] = v; }

    // ---- graph 2: out = conv2(silu(norm2_noaffine(h_new))) + skip ----
    GraphRun gr2(m);
    ggml_context* c = gr2.c;
    ggml_tensor* gh = ggml_new_tensor_2d(c, GGML_TYPE_F32, Cout, M); ggml_set_input(gh);
    ggml_tensor* gn2 = ggml_new_tensor_2d(c, GGML_TYPE_I32, M, 27);  ggml_set_input(gn2);
    ggml_tensor* gsk = ggml_new_tensor_2d(c, GGML_TYPE_F32, Cout, M); ggml_set_input(gsk);
    ggml_tensor* h = ggml_silu(c, ggml_norm(c, gh, 1e-6f));          // norm2 no affine
    h = sparse_submconv(c, m, prefix + ".conv2", h, gn2, M);
    ggml_tensor* out = ggml_add(c, h, gsk);
    std::vector<float> outv = gr2.run(out, { {gh, hnew.data()}, {gn2, nnbr.data()}, {gsk, skip.data()} });

    return { std::move(outv), std::move(nc), Cout, std::move(mask_used) };
}

} // namespace trellis
