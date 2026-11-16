#include "sparse.h"
#include "trellis_model.h"
#include "trellis_debug.h"
#include "trellis_sched.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <functional>
#include <cmath>
#include <cstdlib>
#include <unordered_map>
#include <stdexcept>

namespace trellis {
using T = ggml_tensor;
void mem_probe(const char* tag);   // shape_decoder.cpp

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
// Callers that already chunk their rows can stay under this and skip the concat chain.
static constexpr int64_t kMulMatRowChunk = 1000000;

static T* mul_mat_rows(ggml_context* c, T* W, T* x) {
    constexpr int64_t max_rows = kMulMatRowChunk;
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
// C2S streaming uses many short-lived GraphRun instances. Keep the metadata ceiling generous:
// node/tensor metadata is host RAM only (~370 B each), while GPU workspace is owned and freed
// independently by each GraphRun's gallocr.
static constexpr size_t kGraphNodes = 65536;
struct GraphRun {
    const Model& m; ggml_context* c; trellis::GraphExec ex;
    GraphRun(const Model& mm) : m(mm), ex(mm) {
        size_t meta = ggml_tensor_overhead() * kGraphNodes + ggml_graph_overhead_custom(kGraphNodes, false) + (1 << 20);
        c = ggml_init({ meta, nullptr, true });
    }
    ~GraphRun() { ggml_free(c); }
    // `roots` are extra nodes to expand: writes into a preallocated `out` via ggml_cpy are
    // not reachable from `out` itself (nothing produces it), so they must be rooted explicitly.
    std::vector<float> run(ggml_tensor* out, const std::vector<std::pair<ggml_tensor*, const void*>>& inputs,
                           const std::vector<ggml_tensor*>& roots = {}) {
        ggml_set_output(out);
        ggml_cgraph* g = ggml_new_graph_custom(c, kGraphNodes, false);
        for (ggml_tensor* r : roots) ggml_build_forward_expand(g, r);
        ggml_build_forward_expand(g, out);
        if (!ex.alloc(g)) throw std::runtime_error("c2s: alloc failed");
        if (getenv("TRELLIS_DBG_ALLOC"))
            fprintf(stderr, "      [c2s-alloc] nodes=%d  gallocr buffer = %.2f GB\n",
                    ggml_graph_n_nodes(g), ex.buffer_size() / 1e9);
        for (auto& [t, data] : inputs) ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t));
        if (ex.compute(g, "sparse.c2s") != GGML_STATUS_SUCCESS) throw std::runtime_error("c2s: compute failed");
        return tensor_to_f32(out);
    }
};
} // anon

// Host-streamed SparseConvNeXt block. This is deliberately separate from the graph-builder
// sparse_convnext() above: the latter is excellent for ordinary sparse levels, but for very
// large stage-3 levels its chunk outputs are joined by a growing ggml_concat chain. The
// allocator then has to reserve every prefix of that chain and a single block can become a
// multi-GB monolith. Here every output chunk is an independent GraphRun, stitched on the host.
// As with compact C2S, only feature columns actually referenced by this chunk's 27-neighbour
// table are uploaded, so the full [C,N] sparse feature table never has to reside on the GPU.
std::vector<float> sparse_convnext_streamed(const Model& m, const std::string& prefix,
                                             const std::vector<float>& feats_in, int C,
                                             const std::vector<int32_t>& nbr, int N) {
    if (N <= 0 || C <= 0 || feats_in.size() != (size_t)C * N)
        throw std::runtime_error("shape_dec: invalid streamed ConvNeXt input in " + prefix);
    if (nbr.size() != (size_t)27 * N)
        throw std::runtime_error("shape_dec: invalid streamed ConvNeXt neighbour table in " + prefix);

    struct CompactNeighbors {
        std::vector<int32_t> table;       // tap-major local indices, sentinel = globals.size()
        std::vector<int32_t> globals;     // local column -> source column in feats_in
    };
    auto compact_neighbors = [](const std::vector<int32_t>& full, int full_n,
                                int64_t r0, int nr, int sentinel) {
        CompactNeighbors out;
        out.table.resize((size_t)27 * nr);
        std::unordered_map<int32_t, int32_t> remap;
        remap.reserve((size_t)nr * 2 + 1);
        for (int t = 0; t < 27; ++t) {
            for (int j = 0; j < nr; ++j) {
                const int32_t g = full[(size_t)t * full_n + (size_t)r0 + j];
                if (g == sentinel) {
                    out.table[(size_t)t * nr + j] = -1;
                    continue;
                }
                auto it = remap.find(g);
                if (it == remap.end()) {
                    const int32_t li = (int32_t)out.globals.size();
                    out.globals.push_back(g);
                    remap.emplace(g, li);
                    out.table[(size_t)t * nr + j] = li;
                } else {
                    out.table[(size_t)t * nr + j] = it->second;
                }
            }
        }
        const int32_t local_sentinel = (int32_t)out.globals.size();
        for (int32_t& x : out.table) if (x < 0) x = local_sentinel;
        return out;
    };
    auto gather_columns = [](const std::vector<float>& full, int ch,
                             const std::vector<int32_t>& globals) {
        std::vector<float> local((size_t)ch * (globals.size() + 1), 0.0f);
        for (size_t li = 0; li < globals.size(); ++li) {
            const size_t src = (size_t)ch * (size_t)globals[li];
            std::copy_n(full.data() + src, ch, local.data() + (size_t)ch * li);
        }
        return local;
    };

    static constexpr int64_t kDefaultConvNextChunkBytes = 256ll * 1024 * 1024;
    static constexpr int64_t kMaxConvNextStreamVoxels = 65536;
    int64_t budget = kDefaultConvNextChunkBytes;
    if (const char* e = getenv("TRELLIS_CONVNEXT_CHUNK_MB")) {
        const int64_t mb = atoll(e);
        if (mb > 0) budget = mb * 1024 * 1024;
    }
    // The MLP widens C -> 4C and is the widest voxel-local activation. Keep that wide
    // activation near the requested budget, then cap output rows just like compact C2S.
    const int64_t per_vox = std::max<int64_t>(1, 4ll * C * (int64_t)sizeof(float));
    int64_t chunk = std::max<int64_t>(1, budget / per_vox);
    if (chunk > kMaxConvNextStreamVoxels) chunk = kMaxConvNextStreamVoxels;
    if (chunk > N) chunk = N;
    const int64_t nchunks = (N + chunk - 1) / chunk;

    if (getenv("TRELLIS_DBG_ALLOC"))
        fprintf(stderr, "      [convnext-stream] %s C=%d N=%d chunks=%lldx%lld budget=%lld MB\n",
                prefix.c_str(), C, N, (long long)nchunks, (long long)chunk,
                (long long)(budget / (1024 * 1024)));

    std::vector<float> out((size_t)C * N);
    for (int64_t r0 = 0; r0 < N; r0 += chunk) {
        const int nr = (int)std::min<int64_t>(chunk, N - r0);
        CompactNeighbors cn = compact_neighbors(nbr, N, r0, nr, N);
        std::vector<float> local = gather_columns(feats_in, C, cn.globals);

        GraphRun gr(m);
        ggml_context* c = gr.c;
        const int nsrc = (int)cn.globals.size();
        T* gh = ggml_new_tensor_2d(c, GGML_TYPE_F32, C, nsrc + 1); ggml_set_input(gh);
        T* gn = ggml_new_tensor_2d(c, GGML_TYPE_I32, nr, 27);      ggml_set_input(gn);
        T* gres = ggml_new_tensor_2d(c, GGML_TYPE_F32, C, nr);     ggml_set_input(gres);

        T* Wc = w32(c, m.get(prefix + ".conv.weight"));
        T* h = submconv_range(c, m, prefix + ".conv", gh, Wc, gn, nr, 0, nr);
        h = ggml_norm(c, h, 1e-6f);
        h = ggml_add(c, ggml_mul(c, h, m.get(prefix + ".norm.weight")),
                     m.get(prefix + ".norm.bias"));
        h = ggml_add(c, mul_mat_rows(c, m.get(prefix + ".mlp.0.weight"), h),
                     m.get(prefix + ".mlp.0.bias"));
        h = ggml_silu(c, h);
        h = ggml_add(c, mul_mat_rows(c, m.get(prefix + ".mlp.2.weight"), h),
                     m.get(prefix + ".mlp.2.bias"));
        h = ggml_add(c, h, gres);

        std::vector<float> hv;
        try {
            hv = gr.run(h, { {gh, local.data()}, {gn, cn.table.data()},
                             {gres, feats_in.data() + (size_t)C * r0} });
        } catch (const std::runtime_error& e) {
            // GraphRun's generic label is c2s because it was originally introduced for C2S.
            // Reclassify here so the caller/queue reports the subsystem that actually failed.
            const std::string what = e.what();
            if (what == "c2s: alloc failed") throw std::runtime_error("shape_dec alloc");
            if (what == "c2s: compute failed") throw std::runtime_error("shape_dec compute");
            throw;
        }
        std::copy(hv.begin(), hv.end(), out.begin() + (size_t)C * r0);
    }
    return out;
}

// ---------------------------------------------------------------------------------------
// Encoder side (shape SLAT encoder, docs/spec/31-mesh-refine.md).
namespace {
struct CompactNbr {
    std::vector<int32_t> table;       // tap-major local indices, sentinel = globals.size()
    std::vector<int32_t> globals;     // local column -> source column in the full host table
};
CompactNbr compact_nbr(const std::vector<int32_t>& full, int full_n, int64_t r0, int nr, int sentinel) {
    CompactNbr out;
    out.table.resize((size_t)27 * nr);
    std::unordered_map<int32_t, int32_t> remap;
    remap.reserve((size_t)nr * 2 + 1);
    for (int t = 0; t < 27; ++t)
        for (int j = 0; j < nr; ++j) {
            const int32_t g = full[(size_t)t * full_n + (size_t)r0 + j];
            if (g == sentinel) { out.table[(size_t)t * nr + j] = -1; continue; }
            auto it = remap.find(g);
            if (it == remap.end()) {
                const int32_t li = (int32_t)out.globals.size();
                out.globals.push_back(g); remap.emplace(g, li);
                out.table[(size_t)t * nr + j] = li;
            } else out.table[(size_t)t * nr + j] = it->second;
        }
    const int32_t ls = (int32_t)out.globals.size();
    for (int32_t& x : out.table) if (x < 0) x = ls;
    return out;
}
std::vector<float> gather_cols(const float* full, int C, const std::vector<int32_t>& globals) {
    std::vector<float> local((size_t)C * (globals.size() + 1), 0.0f);
    for (size_t li = 0; li < globals.size(); ++li)
        std::copy_n(full + (size_t)C * (size_t)globals[li], C, local.data() + (size_t)C * li);
    return local;
}
constexpr int64_t kEncStreamVoxels = 65536;
int64_t enc_chunk(int64_t per_vox_bytes, int64_t N) {
    int64_t budget = 256ll * 1024 * 1024;
    if (const char* e = getenv("TRELLIS_ENC_CHUNK_MB")) { const int64_t mb = atoll(e); if (mb > 0) budget = mb * 1024 * 1024; }
    int64_t ch = std::max<int64_t>(1, budget / std::max<int64_t>(1, per_vox_bytes));
    return std::min<int64_t>(std::min<int64_t>(ch, kEncStreamVoxels), std::max<int64_t>(1, N));
}
// Streamed submanifold conv over a host table `in` [Ci*N] -> [Co*N]. `post` (optional) is applied
// to each output chunk inside the same graph.
std::vector<float> submconv_streamed(const Model& m, const std::string& prefix, const float* in, int Ci,
                                     const std::vector<int32_t>& nbr, int N, int Co,
                                     const std::function<T*(ggml_context*, T*)>& post = nullptr) {
    std::vector<float> out((size_t)Co * N);
    const int64_t chunk = enc_chunk(4ll * std::max(Ci, Co) * 4, N);
    for (int64_t r0 = 0; r0 < N; r0 += chunk) {
        const int nr = (int)std::min<int64_t>(chunk, N - r0);
        CompactNbr cn = compact_nbr(nbr, N, r0, nr, N);
        std::vector<float> local = gather_cols(in, Ci, cn.globals);
        GraphRun gr(m);
        ggml_context* c = gr.c;
        T* gh = ggml_new_tensor_2d(c, GGML_TYPE_F32, Ci, (int64_t)cn.globals.size() + 1); ggml_set_input(gh);
        T* gn = ggml_new_tensor_2d(c, GGML_TYPE_I32, nr, 27); ggml_set_input(gn);
        T* W = w32(c, m.get(prefix + ".weight"));
        T* o = submconv_range(c, m, prefix, gh, W, gn, nr, 0, nr);
        if (post) o = post(c, o);
        std::vector<float> ov = gr.run(o, { {gh, local.data()}, {gn, cn.table.data()} });
        std::copy(ov.begin(), ov.end(), out.begin() + (size_t)Co * r0);
    }
    return out;
}
}  // namespace

std::vector<float> sparse_linear_streamed(const Model& m, const std::string& prefix,
                                          const std::vector<float>& feats_in, int Cin, int N) {
    T* Wt = m.get(prefix + ".weight");     // ggml ne [Cin, Cout]
    const int Cout = (int)Wt->ne[1];
    if ((int)Wt->ne[0] != Cin || feats_in.size() != (size_t)Cin * N)
        throw std::runtime_error("sparse_linear_streamed: shape mismatch in " + prefix);
    std::vector<float> out((size_t)Cout * N);
    const int64_t chunk = enc_chunk(4ll * (Cin + Cout), N);
    for (int64_t r0 = 0; r0 < N; r0 += chunk) {
        const int nr = (int)std::min<int64_t>(chunk, N - r0);
        GraphRun gr(m);
        ggml_context* c = gr.c;
        T* gf = ggml_new_tensor_2d(c, GGML_TYPE_F32, Cin, nr); ggml_set_input(gf);
        T* o = ggml_add(c, ggml_mul_mat(c, m.get(prefix + ".weight"), gf), m.get(prefix + ".bias"));
        std::vector<float> ov = gr.run(o, { {gf, feats_in.data() + (size_t)Cin * r0} });
        std::copy(ov.begin(), ov.end(), out.begin() + (size_t)Cout * r0);
    }
    return out;
}

std::vector<float> sparse_norm_streamed(const Model& m, const std::vector<float>& feats_in, int C, int N, float eps) {
    std::vector<float> out((size_t)C * N);
    const int64_t chunk = enc_chunk(4ll * C * 2, N);
    for (int64_t r0 = 0; r0 < N; r0 += chunk) {
        const int nr = (int)std::min<int64_t>(chunk, N - r0);
        GraphRun gr(m);
        T* gf = ggml_new_tensor_2d(gr.c, GGML_TYPE_F32, C, nr); ggml_set_input(gf);
        std::vector<float> ov = gr.run(ggml_norm(gr.c, gf, eps), { {gf, feats_in.data() + (size_t)C * r0} });
        std::copy(ov.begin(), ov.end(), out.begin() + (size_t)C * r0);
    }
    return out;
}

S2CResult sparse_s2c(const Model& m, const std::string& prefix,
                     const std::vector<float>& feats_in, int Cin,
                     const std::vector<std::array<int,3>>& coords, int Cout) {
    const int N = (int)coords.size();
    if (N == 0) throw std::runtime_error("s2c: empty input in " + prefix);
    if (Cin <= 0 || Cout % 8 != 0 || (8 * Cin) % Cout != 0 || feats_in.size() != (size_t)Cin * N)
        throw std::runtime_error("s2c: invalid shapes in " + prefix);
    const int C1 = Cout / 8;               // conv1 width at the fine level
    const int G = 8 * Cin / Cout;          // skip: mean over groups of G packed channels

    // ---- parents: coord/2, lexicographic (x,y,z); child slot lx + 2ly + 4lz ----
    std::vector<uint64_t> keyed((size_t)N);
    auto pkey = [](int x, int y, int z) { return ((uint64_t)(uint32_t)x << 42) | ((uint64_t)(uint32_t)y << 21) | (uint64_t)(uint32_t)z; };
    for (int i = 0; i < N; ++i) keyed[(size_t)i] = pkey(coords[i][0] >> 1, coords[i][1] >> 1, coords[i][2] >> 1);
    std::vector<uint64_t> uniq = keyed;
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    const int P = (int)uniq.size();
    std::vector<std::array<int,3>> pc((size_t)P);
    for (int p = 0; p < P; ++p) {
        const uint64_t k = uniq[(size_t)p];
        pc[(size_t)p] = { (int)(k >> 42), (int)((k >> 21) & 0x1fffff), (int)(k & 0x1fffff) };
    }
    std::vector<int32_t> parent((size_t)N);
    std::vector<uint8_t> slot((size_t)N);
    for (int i = 0; i < N; ++i) {
        parent[(size_t)i] = (int32_t)(std::lower_bound(uniq.begin(), uniq.end(), keyed[(size_t)i]) - uniq.begin());
        slot[(size_t)i] = (uint8_t)((coords[i][0] & 1) | ((coords[i][1] & 1) << 1) | ((coords[i][2] & 1) << 2));
    }
    std::vector<uint64_t>().swap(keyed); std::vector<uint64_t>().swap(uniq);

    // ---- norm1 (affine) + SiLU, voxel-local ----
    std::vector<float> hnorm((size_t)Cin * N);
    {
        const int64_t chunk = enc_chunk(4ll * Cin * 2, N);
        for (int64_t r0 = 0; r0 < N; r0 += chunk) {
            const int nr = (int)std::min<int64_t>(chunk, N - r0);
            GraphRun gr(m);
            ggml_context* c = gr.c;
            T* gf = ggml_new_tensor_2d(c, GGML_TYPE_F32, Cin, nr); ggml_set_input(gf);
            T* h = ggml_norm(c, gf, 1e-6f);
            h = ggml_add(c, ggml_mul(c, h, m.get(prefix + ".norm1.weight")), m.get(prefix + ".norm1.bias"));
            h = ggml_silu(c, h);
            std::vector<float> hv = gr.run(h, { {gf, feats_in.data() + (size_t)Cin * r0} });
            std::copy(hv.begin(), hv.end(), hnorm.begin() + (size_t)Cin * r0);
        }
    }
    // ---- conv1 at the fine level ----
    std::vector<float> h1;
    {
        std::vector<int32_t> nbr = build_neighbor_table(coords);
        h1 = submconv_streamed(m, prefix + ".conv1", hnorm.data(), Cin, nbr, N, C1);
    }
    std::vector<float>().swap(hnorm);
    // ---- S2C of h (and the skip of x) on the host ----
    std::vector<float> h2((size_t)Cout * P, 0.0f), skip((size_t)Cout * P, 0.0f);
    for (int i = 0; i < N; ++i) {
        const size_t p = (size_t)parent[(size_t)i];
        const int s = slot[(size_t)i];
        std::copy_n(h1.data() + (size_t)C1 * i, C1, h2.data() + p * Cout + (size_t)s * C1);
        // packed x: channel s*Cin + c; skip output o averages packed channels [o*G, (o+1)*G)
        const float* xi = feats_in.data() + (size_t)Cin * i;
        for (int c = 0; c < Cin; ++c) skip[p * Cout + (size_t)((s * Cin + c) / G)] += xi[c];
    }
    for (float& v : skip) v /= (float)G;
    std::vector<float>().swap(h1);
    // ---- norm2 (no affine) + SiLU, then conv2 at the coarse level, + skip ----
    std::vector<float> h2n((size_t)Cout * P);
    {
        const int64_t chunk = enc_chunk(4ll * Cout * 2, P);
        for (int64_t r0 = 0; r0 < P; r0 += chunk) {
            const int nr = (int)std::min<int64_t>(chunk, P - r0);
            GraphRun gr(m);
            T* gf = ggml_new_tensor_2d(gr.c, GGML_TYPE_F32, Cout, nr); ggml_set_input(gf);
            std::vector<float> hv = gr.run(ggml_silu(gr.c, ggml_norm(gr.c, gf, 1e-6f)), { {gf, h2.data() + (size_t)Cout * r0} });
            std::copy(hv.begin(), hv.end(), h2n.begin() + (size_t)Cout * r0);
        }
    }
    std::vector<float>().swap(h2);
    std::vector<int32_t> pnbr = build_neighbor_table(pc);
    std::vector<float> out = submconv_streamed(m, prefix + ".conv2", h2n.data(), Cout, pnbr, P, Cout);
    for (size_t k = 0; k < out.size(); ++k) out[k] += skip[k];
    return { std::move(out), std::move(pc), Cout };
}

C2SResult sparse_c2s(const Model& m, const std::string& prefix,
                     const std::vector<float>& feats_in, int Cin,
                     const std::vector<std::array<int,3>>& coords, int Cout,
                     const std::vector<uint8_t>* ext_subdiv) {
    const int N = (int)coords.size();
    if (N == 0) throw std::runtime_error("c2s: empty input coordinate set in " + prefix);
    if (Cin <= 0 || Cout <= 0 || feats_in.size() != (size_t)Cin * N)
        throw std::runtime_error("c2s: invalid input feature shape in " + prefix);
    if (ext_subdiv && ext_subdiv->size() != (size_t)8 * N)
        throw std::runtime_error("c2s: invalid external subdivision mask in " + prefix);
    std::vector<int32_t> nbr = build_neighbor_table(coords);

    // ---- graph 1: subdiv [8,N] only. to_subdiv reads the RAW feats (the reference takes it
    // before norm1), so it does not depend on conv1 -- which lets the whole conv1 ->
    // subdivide -> conv2 chain live in one on-device graph below. The tex decoder supplies
    // the mask externally, so it skips this graph entirely.
    std::vector<float> subdiv;
    if (!ext_subdiv) {
        auto predict_subdiv = [&]() {
            // to_subdiv is a voxel-local linear projection [Cin,N] -> [8,N].  Keeping the
            // whole F32 input tensor on-device defeats C2S streaming for very dense shape
            // decoder stages (e.g. Cin=128,N~4.5M is >2 GiB before any workspace).  Stream
            // independent voxel ranges and stitch the tiny 8-channel logits on the host.
            static constexpr int64_t kSubdivStreamVoxels = 65536;
            std::vector<float> out((size_t)8 * N);
            if (N > kSubdivStreamVoxels)
                fprintf(stderr,
                        "      [c2s-subdiv-stream] %s N=%d chunks=%lldx%lld\n",
                        prefix.c_str(), N,
                        (long long)((N + kSubdivStreamVoxels - 1) / kSubdivStreamVoxels),
                        (long long)kSubdivStreamVoxels);

            for (int64_t r0 = 0; r0 < N; r0 += kSubdivStreamVoxels) {
                const int nr = (int)std::min<int64_t>(kSubdivStreamVoxels, (int64_t)N - r0);
                GraphRun gr1(m);
                ggml_context* c = gr1.c;
                T* gf = ggml_new_tensor_2d(c, GGML_TYPE_F32, Cin, nr); ggml_set_input(gf);
                T* sd = ggml_add(c, ggml_mul_mat(c, m.get(prefix + ".to_subdiv.weight"), gf),
                                 m.get(prefix + ".to_subdiv.bias"));
                std::vector<float> sv = gr1.run(sd, { {gf, feats_in.data() + (size_t)Cin * r0} });
                std::copy(sv.begin(), sv.end(), out.begin() + (size_t)8 * r0);
            }
            return out;
        };
        constexpr int kAttempts = 3;
        for (int attempt = 0; attempt < kAttempts; ++attempt) {
            subdiv = predict_subdiv();
            const size_t bad = std::count_if(subdiv.begin(), subdiv.end(),
                                             [](float x) { return !std::isfinite(x); });
            if (bad == 0) break;
            const size_t bad_input = std::count_if(feats_in.begin(), feats_in.end(),
                                                   [](float x) { return !std::isfinite(x); });
            if (bad_input != 0)
                throw std::runtime_error("c2s: non-finite input features in " + prefix);
            if (attempt + 1 == kAttempts)
                throw std::runtime_error("c2s: non-finite subdivision logits from finite input in " + prefix);
            fprintf(stderr,
                    "      [c2s] %s produced %zu/%zu non-finite subdivision logits; retrying (%d/%d)\n",
                    prefix.c_str(), bad, subdiv.size(), attempt + 1, kAttempts - 1);
        }
        if (getenv("TRELLIS_DBG_SUBDIV")) {
            size_t pos = 0; double smin = 0, smax = 0; bool first = true;
            for (float x : subdiv) { if (x > 0.0f) pos++; if (first) { smin = smax = x; first = false; }
                else { if (x < smin) smin = x; if (x > smax) smax = x; } }
            fprintf(stderr, "      [c2s] %-26s N=%d  subdiv>0: %zu/%zu (%.2f%%)  logit[min=%.3f max=%.3f]\n",
                    prefix.c_str(), N, pos, subdiv.size(),
                    subdiv.empty() ? 0.0 : 100.0 * pos / subdiv.size(), smin, smax);
        }
    }

    // ---- host: octant mask -> new coords + the channel2spatial gather index ----
    // gidx is built i-major, so it is ascending: every input-voxel range [r0, r0+nr) maps to
    // a CONTIGUOUS output range [mstart[r0], mstart[r0+nr]). That is what lets conv1 be
    // chunked below without disturbing the output order.
    const int K = Cin / 8, R = Cout / K;
    std::vector<uint8_t> mask_used((size_t)8 * N);
    std::vector<std::array<int,3>> nc;
    std::vector<int32_t> gidx;                 // gidx[m] = o + 8*i: source column of the [C, 8N] reshape
    std::vector<int32_t> mstart(N + 1);
    nc.reserve(N); gidx.reserve(N);
    for (int i = 0; i < N; ++i) {
        mstart[i] = (int32_t)gidx.size();
        const int x = coords[i][0], y = coords[i][1], z = coords[i][2];
        for (int o = 0; o < 8; ++o) {
            const bool on = ext_subdiv ? (*ext_subdiv)[(size_t)o + 8*i] != 0 : (subdiv[(size_t)o + 8*i] > 0.0f);
            mask_used[(size_t)o + 8*i] = on ? 1 : 0;
            if (!on) continue;
            nc.push_back({ 2*x + (o & 1), 2*y + ((o>>1) & 1), 2*z + ((o>>2) & 1) });
            gidx.push_back(o + 8*i);
        }
    }
    mstart[N] = (int32_t)gidx.size();
    const int M = (int)nc.size();
    if (M == 0) throw std::runtime_error("c2s: subdivision produced no active voxels in " + prefix);
    std::vector<int32_t> nnbr = build_neighbor_table(nc);
    if (getenv("TRELLIS_DBG_MEM"))
        fprintf(stderr, "      [c2s] %-22s N=%d -> M=%d | host nnbr=%.2f GB feats_in=%.2f GB out=%.2f GB\n",
                prefix.c_str(), N, M, (double)nnbr.size() * 4 / 1e9,
                (double)feats_in.size() * 4 / 1e9, (double)Cout * M * 4 / 1e9);

    // ---- streamed C2S ---------------------------------------------------------
    // The older implementation chunked conv1/conv2 logically but still built every chunk
    // into ONE ggml graph. gallocr therefore reserved the aggregate lifetime of the whole
    // graph, which defeats the memory cap and can request 10-20+ GB at res-1024.
    //
    // This path makes the chunk boundary a real allocation boundary: each normalization,
    // conv1 and conv2 chunk gets its own GraphRun. The GraphRun is destroyed before the next
    // chunk starts, so its gallocr buffer is released. Intermediate C2S features are stitched
    // on the host. This trades some PCIe traffic for a bounded GPU working set -- exactly what
    // the 16 GB Windows/Vulkan path needs.
    static constexpr int64_t kDefaultC2SChunkBytes = 256ll * 1024 * 1024;
    int64_t budget = kDefaultC2SChunkBytes;
    if (const char* e = getenv("TRELLIS_C2S_CHUNK_MB")) {
        const int64_t mb = atoll(e);
        if (mb > 0) budget = mb * 1024 * 1024;
    }

    auto chunk_count = [](int64_t total, int64_t chunk_sz) -> int64_t {
        return (total + chunk_sz - 1) / chunk_sz;
    };
    struct CompactNeighbors {
        std::vector<int32_t> table;       // tap-major local indices, sentinel = globals.size()
        std::vector<int32_t> globals;     // local column -> source column in the full host table
    };
    auto compact_neighbors = [](const std::vector<int32_t>& full, int full_n,
                                int64_t r0, int nr, int sentinel) {
        CompactNeighbors out;
        out.table.resize((size_t)27 * nr);
        // A streamed graph must not upload the full [C,N] feature table just because its
        // neighbours are global indices. Compact the referenced columns for this output chunk
        // and remap the neighbour table to that compact table. With a 65k output chunk this
        // stays small even when the full post-subdivision stage has 10M+ voxels.
        std::unordered_map<int32_t, int32_t> remap;
        remap.reserve((size_t)nr * 2 + 1);
        for (int t = 0; t < 27; ++t) {
            for (int j = 0; j < nr; ++j) {
                const int32_t g = full[(size_t)t * full_n + (size_t)r0 + j];
                if (g == sentinel) {
                    out.table[(size_t)t * nr + j] = -1;
                    continue;
                }
                auto it = remap.find(g);
                if (it == remap.end()) {
                    const int32_t li = (int32_t)out.globals.size();
                    out.globals.push_back(g);
                    remap.emplace(g, li);
                    out.table[(size_t)t * nr + j] = li;
                } else {
                    out.table[(size_t)t * nr + j] = it->second;
                }
            }
        }
        const int32_t local_sentinel = (int32_t)out.globals.size();
        for (int32_t& x : out.table) if (x < 0) x = local_sentinel;
        return out;
    };
    auto gather_columns = [](const std::vector<float>& full, int C,
                             const std::vector<int32_t>& globals) {
        std::vector<float> local((size_t)C * (globals.size() + 1), 0.0f);
        for (size_t li = 0; li < globals.size(); ++li) {
            const size_t src = (size_t)C * (size_t)globals[li];
            std::copy_n(full.data() + src, C, local.data() + (size_t)C * li);
        }
        return local;
    };

    // norm1 + affine + SiLU are voxel-local. Do them in independent row chunks and keep the
    // result on the host with one extra zero sentinel row. This also avoids submconv_pad's
    // full-size device concat in every conv1 chunk.
    const int64_t norm_bytes_per_vox = std::max<int64_t>(1, 4ll * Cin * (int64_t)sizeof(float));
    int64_t norm_chunk = std::max<int64_t>(1, budget / norm_bytes_per_vox);
    if (norm_chunk > N) norm_chunk = N;

    std::vector<float> hnorm((size_t)Cin * (N + 1), 0.0f);  // column N is the sentinel
    for (int64_t r0 = 0; r0 < N; r0 += norm_chunk) {
        const int nr = (int)std::min<int64_t>(norm_chunk, N - r0);
        GraphRun gr(m);
        ggml_context* c = gr.c;
        T* gf = ggml_new_tensor_2d(c, GGML_TYPE_F32, Cin, nr); ggml_set_input(gf);
        T* h = ggml_norm(c, gf, 1e-6f);
        h = ggml_add(c, ggml_mul(c, h, m.get(prefix + ".norm1.weight")),
                     m.get(prefix + ".norm1.bias"));
        h = ggml_silu(c, h);
        std::vector<float> hv = gr.run(h, { {gf, feats_in.data() + (size_t)Cin * r0} });
        std::copy(hv.begin(), hv.end(), hnorm.begin() + (size_t)Cin * r0);
    }

    // conv1 widens to Cout*8. Each input-voxel range maps to a contiguous output range
    // [mstart[r0], mstart[r1]), so we can gather the surviving octants immediately, apply
    // norm2+SiLU immediately, read that chunk back, and discard the graph before moving on.
    const int64_t per_vox = std::max<int64_t>(1, (int64_t)Cout * 8 * (int64_t)sizeof(float));
    int64_t chunk = std::max<int64_t>(1, budget / per_vox);
    // submconv has 27 gathers/matmuls; the output tensor alone is not a useful estimate of
    // its real Vulkan working set. Keep each streamed conv graph comfortably below the
    // multi-GB allocations seen on 16 GB cards.
    static constexpr int64_t kMaxC2SStreamVoxels = 65536;
    if (chunk > kMaxC2SStreamVoxels) chunk = kMaxC2SStreamVoxels;
    if (chunk > N) chunk = N;

    std::vector<float> hn((size_t)Cout * (M + 1), 0.0f);   // host table; column M is sentinel
    for (int64_t r0 = 0; r0 < N; r0 += chunk) {
        const int64_t r1 = std::min<int64_t>(r0 + chunk, N);
        const int nr = (int)(r1 - r0);
        const int32_t m0 = mstart[(size_t)r0], m1 = mstart[(size_t)r1];
        if (m1 == m0) continue;
        const int mc = m1 - m0;

        CompactNeighbors cn = compact_neighbors(nbr, N, r0, nr, N);
        std::vector<float> hnorm_local = gather_columns(hnorm, Cin, cn.globals);
        std::vector<int32_t> idx_local((size_t)mc);
        for (int j = 0; j < mc; ++j)
            idx_local[(size_t)j] = gidx[(size_t)m0 + j] - (int32_t)(8 * r0);

        GraphRun gr(m);
        ggml_context* c = gr.c;
        const int nsrc = (int)cn.globals.size();
        T* gh = ggml_new_tensor_2d(c, GGML_TYPE_F32, Cin, nsrc + 1); ggml_set_input(gh);
        T* gn = ggml_new_tensor_2d(c, GGML_TYPE_I32, nr, 27);         ggml_set_input(gn);
        T* gi = ggml_new_tensor_1d(c, GGML_TYPE_I32, mc);             ggml_set_input(gi);

        T* W1 = w32(c, m.get(prefix + ".conv1.weight"));
        T* cvc = submconv_range(c, m, prefix + ".conv1", gh, W1, gn, nr, 0, nr);
        T* hc = ggml_get_rows(c, ggml_reshape_2d(c, cvc, Cout, (int64_t)8 * nr), gi);
        hc = ggml_silu(c, ggml_norm(c, hc, 1e-6f));

        std::vector<float> hv = gr.run(hc, { {gh, hnorm_local.data()}, {gn, cn.table.data()},
                                              {gi, idx_local.data()} });
        std::copy(hv.begin(), hv.end(), hn.begin() + (size_t)Cout * m0);
    }

    // conv2 is streamed over POST-subdivision voxels. Neighbours are global on the host, but
    // each chunk compacts just the referenced feature columns and remaps its neighbour table,
    // so the GPU never receives the full [Cout,M] table. There is deliberately no full
    // [Cout,M] device input or output: each chunk is read back and stitched into `outv`.
    const int64_t per_vox2 = std::max<int64_t>(1, 3ll * Cout * (int64_t)sizeof(float));
    int64_t chunk2 = std::max<int64_t>(1, budget / per_vox2);
    if (chunk2 > kMaxC2SStreamVoxels) chunk2 = kMaxC2SStreamVoxels;
    if (chunk2 > kMulMatRowChunk) chunk2 = kMulMatRowChunk;
    if (chunk2 > M) chunk2 = M;

    if (getenv("TRELLIS_DBG_ALLOC"))
        fprintf(stderr,
                "      [c2s-stream] %s budget=%lld MB norm=%lldx%lld conv1=%lldx%lld conv2=%lldx%lld\n",
                prefix.c_str(), (long long)(budget / (1024 * 1024)),
                (long long)chunk_count(N, norm_chunk), (long long)norm_chunk,
                (long long)chunk_count(N, chunk), (long long)chunk,
                (long long)chunk_count(M, chunk2), (long long)chunk2);

    std::vector<float> outv((size_t)Cout * M);
    for (int64_t m0 = 0; m0 < M; m0 += chunk2) {
        const int nr2 = (int)std::min<int64_t>(chunk2, M - m0);
        CompactNeighbors cn = compact_neighbors(nnbr, M, m0, nr2, M);
        std::vector<float> hn_local = gather_columns(hn, Cout, cn.globals);

        GraphRun gr(m);
        ggml_context* c = gr.c;
        const int nsrc = (int)cn.globals.size();
        T* gh = ggml_new_tensor_2d(c, GGML_TYPE_F32, Cout, nsrc + 1); ggml_set_input(gh);
        T* gn = ggml_new_tensor_2d(c, GGML_TYPE_I32, nr2, 27);         ggml_set_input(gn);
        T* W2 = w32(c, m.get(prefix + ".conv2.weight"));
        T* o = submconv_range(c, m, prefix + ".conv2", gh, W2, gn, nr2, 0, nr2);

        std::vector<float> ov = gr.run(o, { {gh, hn_local.data()}, {gn, cn.table.data()} });

        // Skip path on host: channel2spatial(raw x) followed by repeat_interleave(R).
        // gidx[m] is the selected column in reshape(raw_x, [K, 8N]).
        for (int j = 0; j < nr2; ++j) {
            const int64_t mm = m0 + j;
            const int32_t src_col = gidx[(size_t)mm];
            for (int k = 0; k < K; ++k) {
                const float sv = feats_in[(size_t)K * src_col + k];
                const int c0 = k * R;
                for (int r = 0; r < R; ++r)
                    ov[(size_t)Cout * j + c0 + r] += sv;
            }
        }
        std::copy(ov.begin(), ov.end(), outv.begin() + (size_t)Cout * m0);
    }

    return { std::move(outv), std::move(nc), Cout, std::move(mask_used) };
}

} // namespace trellis
