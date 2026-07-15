#include "sparse.h"
#include "trellis_model.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cstdlib>
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

// mul_mat with src1 row-chunked inside the graph: Vulkan packs matmul rows into
// dispatch dim 1 (65535 workgroups; n-tile as small as 32 for few-output
// pipelines -> ~2.1M-row ceiling), and dense res-1024 models exceed it.
static T* mul_mat_rows(ggml_context* c, T* W, T* x) {
    constexpr int64_t max_rows = 1000000;
    const int64_t n = x->ne[1];
    if (n <= max_rows) return ggml_mul_mat(c, W, x);
    T* out = nullptr;
    for (int64_t r0 = 0; r0 < n; r0 += max_rows) {
        const int64_t nr = std::min(max_rows, n - r0);
        T* xv = ggml_cont(c, ggml_view_2d(c, x, x->ne[0], nr, x->nb[1], (size_t)r0 * x->nb[1]));
        T* o = ggml_mul_mat(c, W, xv);
        out = out ? ggml_concat(c, out, o, 1) : o;
    }
    return out;
}

// Voxel range [r0, r0+nr) of the submanifold conv. The gather reads arbitrary rows of the
// FULL input (a neighbour can be any voxel), so `feats` must stay whole -- but the OUTPUT
// is per-voxel, so restricting it to a range needs no halo.
static T* submconv_range(ggml_context* c, const Model& m, const std::string& prefix,
                         T* fz, T* W, T* nbr, int N, int r0, int nr) {
    const int64_t Ci = W->ne[0], Co = W->ne[2];
    T* acc = nullptr;
    for (int t = 0; t < 27; ++t) {
        // cont() so idx is contiguous at buffer offset 0: the Vulkan get_rows
        // kernel asserts a zero index offset (CUDA tolerates the view offset).
        // nbr is tap-major [27, N], so tap t's slice for this range starts at t*N + r0.
        T* idx = ggml_cont(c, ggml_view_1d(c, nbr, nr, ((size_t)t * N + r0) * ggml_element_size(nbr)));
        T* g = ggml_get_rows(c, fz, idx);                          // [Ci, nr]
        T* Wt = ggml_cont(c, ggml_view_2d(c, W, Ci, Co, W->nb[2], (size_t)t * W->nb[1])); // [Ci,Co]
        T* o = mul_mat_rows(c, Wt, g);                             // [Co, nr]
        acc = acc ? ggml_add(c, acc, o) : o;
    }
    return ggml_add(c, acc, ggml_reshape_2d(c, m.get(prefix + ".bias"), Co, 1));
}

// Zero-padded input [Ci, N+1]: row N is the sentinel that absent neighbours index into.
static T* submconv_pad(ggml_context* c, T* feats, int64_t Ci) {
    T* zero = ggml_scale(c, ggml_view_2d(c, feats, Ci, 1, feats->nb[1], 0), 0.0f); // [Ci,1] zeros
    return ggml_concat(c, feats, ggml_cont(c, zero), 1);                            // [Ci, N+1]
}

ggml_tensor* sparse_submconv(ggml_context* c, const Model& m, const std::string& prefix,
                             T* feats, T* nbr, int N) {
    T* W = w32(c, m.get(prefix + ".weight"));  // ggml ne [Ci,27,Co]
    T* fz = submconv_pad(c, feats, W->ne[0]);
    return submconv_range(c, m, prefix, fz, W, nbr, N, 0, N);
}

// Per-block peak is set by the ConvNeXt MLP, which widens to 4C: at res-1024 a stage-0
// [4096, 400K] intermediate is 6.6 GB, and a couple are live at once (~15 GB measured) --
// ~3x the reference's whole decode. Everything after the conv is per-voxel, so the block
// is chunked over output voxels: only the [C,N] input/output stay whole, while the wide
// intermediates are bounded to [4C, nr]. gallocr frees each chunk's temporaries at its
// residual add, so the peak is one chunk, not the sum.
static constexpr int64_t kBlockChunkBytes = 1500u * 1024 * 1024;   // budget per [4C, nr]

ggml_tensor* sparse_convnext(ggml_context* c, const Model& m, const std::string& prefix,
                             T* feats, T* nbr, int N) {
    T* Wc = w32(c, m.get(prefix + ".conv.weight"));
    const int64_t C = Wc->ne[2];
    T* fz = submconv_pad(c, feats, Wc->ne[0]);          // built once, shared by every chunk

    const int64_t per_vox = 4 * C * 4;                  // widest intermediate, bytes/voxel
    int64_t budget = kBlockChunkBytes;
    if (const char* e = getenv("TRELLIS_BLOCK_CHUNK_MB")) budget = atoll(e) * 1024 * 1024;  // A/B override
    int64_t chunk = std::max<int64_t>(1, budget / std::max<int64_t>(per_vox, 1));
    // Floor on the chunk size: every chunk adds ~145 graph nodes and decode_unet builds a
    // whole stage (up to 16 blocks) as ONE graph, so an unbounded chunk count exhausts the
    // ggml context -> GGML_ASSERT(obj_new). 24 chunks x 16 blocks x ~145 stays under the
    // 65536-node budget. Hitting this cap just means bigger chunks (more memory), not a crash.
    constexpr int64_t kMaxChunksPerBlock = 24;
    if (chunk * kMaxChunksPerBlock < N) chunk = (N + kMaxChunksPerBlock - 1) / kMaxChunksPerBlock;
    if (chunk >= N) chunk = N;                          // small stages: single chunk, no concat

    T* out = nullptr;
    for (int64_t r0 = 0; r0 < N; r0 += chunk) {
        const int nr = (int)std::min<int64_t>(chunk, N - r0);
        T* h = submconv_range(c, m, prefix + ".conv", fz, Wc, nbr, N, (int)r0, nr);   // [C,nr]
        h = ggml_norm(c, h, 1e-6f);                                                   // rowLN over ne0=C
        h = ggml_add(c, ggml_mul(c, h, m.get(prefix + ".norm.weight")), m.get(prefix + ".norm.bias"));
        h = ggml_add(c, mul_mat_rows(c, m.get(prefix + ".mlp.0.weight"), h), m.get(prefix + ".mlp.0.bias"));
        h = ggml_silu(c, h);
        h = ggml_add(c, mul_mat_rows(c, m.get(prefix + ".mlp.2.weight"), h), m.get(prefix + ".mlp.2.bias"));
        // residual on this range of the block input
        T* fr = (r0 == 0 && nr == N)
                    ? feats
                    : ggml_cont(c, ggml_view_2d(c, feats, C, nr, feats->nb[1], (size_t)r0 * feats->nb[1]));
        h = ggml_add(c, h, fr);
        out = out ? ggml_concat(c, out, h, 1) : h;
    }
    return out;
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
            sd = ggml_add(c, mul_mat_rows(c, m.get(prefix + ".to_subdiv.weight"), gf), m.get(prefix + ".to_subdiv.bias"));
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
