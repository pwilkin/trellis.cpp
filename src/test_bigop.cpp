// Isolate which GPU op page-faults on a >2^31-element per-voxel decode tensor.
//   trellis-test-bigop <shape_dec.gguf> <op> [N] [gpu]
// op: silu | norm | gemm | concat | getrows | submconv | convnext
// Prints "<op> N=.. OK" on success; a GPU page fault aborts the process (=> that op is the culprit).
#include "trellis_model.h"
#include "sparse.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <cmath>

using std::vector;
using T = ggml_tensor;

static ggml_context* mkctx() {
    size_t meta = ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(8192, false) + (1 << 20);
    return ggml_init({ meta, nullptr, true });
}

// run graph producing `out`, set inputs from host data, return out as f32
static vector<float> run(const trellis::Model& m, ggml_context* c, T* out,
                         vector<std::pair<T*, const void*>> ins) {
    ggml_set_output(out);
    ggml_cgraph* g = ggml_new_graph_custom(c, 8192, false);
    ggml_build_forward_expand(g, out);
    ggml_gallocr_t a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    if (!ggml_gallocr_alloc_graph(a, g)) { fprintf(stderr, "alloc failed\n"); exit(2); }
    for (auto& [t, d] : ins) ggml_backend_tensor_set(t, d, 0, ggml_nbytes(t));
    if (ggml_backend_graph_compute(m.backend, g) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(3); }
    vector<float> r = trellis::tensor_to_f32(out);
    ggml_gallocr_free(a);
    return r;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <shape_dec.gguf> <op> [N] [gpu]\n", argv[0]); return 1; }
    const std::string op = argv[2];
    const int N   = argc > 3 ? atoi(argv[3]) : 1200000;
    const int gpu = argc > 4 ? atoi(argv[4]) : 0;
    trellis::Model m = trellis::Model::load(argv[1], gpu);
    printf("op=%s N=%d  (elements at 4096ch = %.2fG, at 1024ch = %.2fG)\n",
           op.c_str(), N, 4096.0*N/1e9, 1024.0*N/1e9);
    fflush(stdout);

    if (op == "silu" || op == "norm") {
        const int C = 4096;
        vector<float> x((size_t)C * N);
        for (size_t i = 0; i < x.size(); ++i) x[i] = ((i * 2654435761u) & 1023) / 512.0f - 1.0f;
        ggml_context* c = mkctx();
        T* gx = ggml_new_tensor_2d(c, GGML_TYPE_F32, C, N); ggml_set_input(gx);
        T* y = (op == "silu") ? ggml_silu(c, gx) : ggml_norm(c, gx, 1e-6f);
        vector<float> r = run(m, c, y, { {gx, x.data()} });
        printf("%s OK  out[0]=%.4f out[last]=%.4f\n", op.c_str(), r[0], r.back());
        ggml_free(c);
    } else if (op == "gemm") {
        // mlp.0.weight: [1024, 4096] -> mul_mat(W, x[1024,N]) = [4096, N]  (>2^31 output)
        T* W = m.get("blocks.0.0.mlp.0.weight");
        const int Cin = (int)W->ne[0];
        vector<float> x((size_t)Cin * N);
        for (size_t i = 0; i < x.size(); ++i) x[i] = ((i * 2654435761u) & 1023) / 512.0f - 1.0f;
        ggml_context* c = mkctx();
        T* gx = ggml_new_tensor_2d(c, GGML_TYPE_F32, Cin, N); ggml_set_input(gx);
        T* y = ggml_mul_mat(c, W, gx);   // [Wout, N]
        printf("gemm out shape = [%lld, %lld]\n", (long long)y->ne[0], (long long)y->ne[1]);
        vector<float> r = run(m, c, y, { {gx, x.data()} });
        printf("gemm OK  out[0]=%.4f out[last]=%.4f\n", r[0], r.back());
        ggml_free(c);
    } else if (op == "concat") {
        const int C = 4096, N2 = N / 2;
        vector<float> x((size_t)C * N2, 0.3f), y2((size_t)C * (N - N2), 0.7f);
        ggml_context* c = mkctx();
        T* a = ggml_new_tensor_2d(c, GGML_TYPE_F32, C, N2);     ggml_set_input(a);
        T* b = ggml_new_tensor_2d(c, GGML_TYPE_F32, C, N - N2); ggml_set_input(b);
        T* y = ggml_concat(c, a, b, 1);
        vector<float> r = run(m, c, y, { {a, x.data()}, {b, y2.data()} });
        printf("concat OK  out[0]=%.4f out[last]=%.4f\n", r[0], r.back());
        ggml_free(c);
    } else if (op == "getrows") {
        const int C = 1024;
        vector<float> x((size_t)C * (N + 1), 0.5f);
        vector<int32_t> idx(N); for (int i = 0; i < N; ++i) idx[i] = i % (N + 1);
        ggml_context* c = mkctx();
        T* gx = ggml_new_tensor_2d(c, GGML_TYPE_F32, C, N + 1); ggml_set_input(gx);
        T* gi = ggml_new_tensor_1d(c, GGML_TYPE_I32, N);        ggml_set_input(gi);
        T* y = ggml_get_rows(c, gx, gi);
        vector<float> r = run(m, c, y, { {gx, x.data()}, {gi, idx.data()} });
        printf("getrows OK  out[0]=%.4f out[last]=%.4f\n", r[0], r.back());
        ggml_free(c);
    } else if (op == "submconv" || op == "convnext") {
        // synthetic dense cube of ~N voxels -> real sparse path at blocks.0.0 (C=1024)
        int s = (int)std::cbrt((double)N) + 1;
        vector<std::array<int,3>> coords; coords.reserve(N);
        for (int x = 0; x < s && (int)coords.size() < N; ++x)
          for (int y = 0; y < s && (int)coords.size() < N; ++y)
            for (int z = 0; z < s && (int)coords.size() < N; ++z) coords.push_back({x,y,z});
        const int Nr = (int)coords.size();
        vector<int32_t> nbr = trellis::build_neighbor_table(coords);
        const int C = 1024;
        vector<float> feats((size_t)C * Nr);
        for (size_t i = 0; i < feats.size(); ++i) feats[i] = ((i * 2654435761u) & 1023) / 512.0f - 1.0f;
        ggml_context* c = mkctx();
        T* gf = ggml_new_tensor_2d(c, GGML_TYPE_F32, C, Nr); ggml_set_input(gf);
        T* gn = ggml_new_tensor_2d(c, GGML_TYPE_I32, Nr, 27); ggml_set_input(gn);
        T* y = (op == "submconv") ? trellis::sparse_submconv(c, m, "blocks.0.0.conv", gf, gn, Nr)
                                  : trellis::sparse_convnext(c, m, "blocks.0.0", gf, gn, Nr);
        printf("  out shape = [%lld, %lld]  (expect [%d, %d])\n",
               (long long)y->ne[0], (long long)y->ne[1], C, Nr);
        vector<float> r = run(m, c, y, { {gf, feats.data()}, {gn, nbr.data()} });
        printf("  out elems = %zu (expect %zu)\n", r.size(), (size_t)C*Nr);
        printf("%s OK  Nr=%d out[0]=%.4f out[last]=%.4f\n", op.c_str(), Nr, r[0], r.back());
        ggml_free(c);
    } else if (op == "add4096" || op == "addbias4096") {
        // add4096:     same-shape add  [4096,N] + [4096,N]   (c2s's acc = add(acc, o), 26x)
        // addbias4096: broadcast add   [4096,N] + [4096,1]   (convnext's add(x, bias))
        const int C = 4096;
        vector<float> a((size_t)C * N, 0.25f);
        ggml_context* c = mkctx();
        T* ga = ggml_new_tensor_2d(c, GGML_TYPE_F32, C, N); ggml_set_input(ga);
        T* y;
        vector<float> b;
        T* gb;
        if (op == "add4096") { b.assign((size_t)C * N, 0.5f); gb = ggml_new_tensor_2d(c, GGML_TYPE_F32, C, N); }
        else                 { b.assign((size_t)C, 0.5f);     gb = ggml_new_tensor_2d(c, GGML_TYPE_F32, C, 1); }
        ggml_set_input(gb);
        y = ggml_add(c, ga, gb);
        vector<float> r = run(m, c, y, { {ga, a.data()}, {gb, b.data()} });
        printf("%s OK  out[0]=%.4f out[last]=%.4f\n", op.c_str(), r[0], r.back());
        ggml_free(c);
    } else if (op == "submconv4096") {
        // the exact c2s conv1: sparse_submconv with Co = Cout*8 = 4096  -> [4096, N]
        int s = (int)std::cbrt((double)N) + 1;
        vector<std::array<int,3>> coords; coords.reserve(N);
        for (int x = 0; x < s && (int)coords.size() < N; ++x)
          for (int y = 0; y < s && (int)coords.size() < N; ++y)
            for (int z = 0; z < s && (int)coords.size() < N; ++z) coords.push_back({x,y,z});
        const int Nr = (int)coords.size();
        vector<int32_t> nbr = trellis::build_neighbor_table(coords);
        const int C = 1024;
        vector<float> feats((size_t)C * Nr, 0.1f);
        ggml_context* c = mkctx();
        T* gf = ggml_new_tensor_2d(c, GGML_TYPE_F32, C, Nr); ggml_set_input(gf);
        T* gn = ggml_new_tensor_2d(c, GGML_TYPE_I32, Nr, 27); ggml_set_input(gn);
        T* y = trellis::sparse_submconv(c, m, "blocks.0.4.conv1", gf, gn, Nr);
        printf("submconv4096 out shape = [%lld, %lld]\n", (long long)y->ne[0], (long long)y->ne[1]);
        vector<float> r = run(m, c, y, { {gf, feats.data()}, {gn, nbr.data()} });
        printf("submconv4096 OK  Nr=%d out[0]=%.4f out[last]=%.4f\n", Nr, r[0], r.back());
        ggml_free(c);
    } else if (op == "stage" || op == "c2s") {
        // stage: decode_unet's stage-0 ConvNeXt (4 chained blocks in ONE graph)
        // c2s:   the coords-to-subdiv at stage 0 (conv1 outputs [Cout*8=4096, N])
        int s = (int)std::cbrt((double)N) + 1;
        vector<std::array<int,3>> coords; coords.reserve(N);
        for (int x = 0; x < s && (int)coords.size() < N; ++x)
          for (int y = 0; y < s && (int)coords.size() < N; ++y)
            for (int z = 0; z < s && (int)coords.size() < N; ++z) coords.push_back({x,y,z});
        const int Nr = (int)coords.size();
        const int C = 1024;
        vector<float> feats((size_t)C * Nr);
        for (size_t i = 0; i < feats.size(); ++i) feats[i] = ((i * 2654435761u) & 1023) / 512.0f - 1.0f;
        if (op == "c2s") {
            // isolate graph1 (conv1 -> [Cout*8=4096, Nr] = the >2^31 tensor); minimal subdiv keeps graph2 trivial
            vector<uint8_t> ext((size_t)8 * Nr, 0);
            for (int o = 0; o < 8; ++o) ext[o] = 1;   // only voxel 0 subdivides -> M=8
            trellis::C2SResult r = trellis::sparse_c2s(m, "blocks.0.4", feats, C, coords, 512, &ext);
            printf("c2s OK  in Nr=%d -> out M=%zu Cout=512\n", Nr, r.coords.size());
        } else {
            vector<int32_t> nbr = trellis::build_neighbor_table(coords);
            ggml_context* c = mkctx();
            T* gf = ggml_new_tensor_2d(c, GGML_TYPE_F32, C, Nr); ggml_set_input(gf);
            T* gn = ggml_new_tensor_2d(c, GGML_TYPE_I32, Nr, 27); ggml_set_input(gn);
            T* x = gf;
            for (int j = 0; j < 4; ++j) x = trellis::sparse_convnext(c, m, "blocks.0." + std::to_string(j), x, gn, Nr);
            printf("  out shape = [%lld, %lld]  (expect [%d, %d])\n",
                   (long long)x->ne[0], (long long)x->ne[1], C, Nr);
            vector<float> r = run(m, c, x, { {gf, feats.data()}, {gn, nbr.data()} });
            printf("stage OK  Nr=%d out[0]=%.4f out[last]=%.4f\n", Nr, r[0], r.back());
            ggml_free(c);
        }
    } else { fprintf(stderr, "unknown op %s\n", op.c_str()); return 1; }

    m.free();
    return 0;
}
