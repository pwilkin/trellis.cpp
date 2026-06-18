// Validate submanifold sparse conv + ConvNeXt block vs pure-torch reference.
//   trellis-test-sparse-conv <shape_dec.gguf> <ref_dir> [gpu]
#include "trellis_model.h"
#include "sparse.h"
#include "npy.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cmath>
#include <cstdio>
#include <vector>
#include <array>

using std::vector;

static double rel(const vector<float>& mine_CN, const npy::Array& ref_NC, int C, int N) {
    double maxd = 0, gmax = 0;
    for (int n = 0; n < N; ++n) for (int cc = 0; cc < C; ++cc) {
        double d = std::fabs((double)mine_CN[cc + (size_t)C*n] - ref_NC.data[(size_t)n*C + cc]);
        maxd = std::max(maxd, d); gmax = std::max(gmax, std::fabs((double)ref_NC.data[(size_t)n*C + cc]));
    }
    return maxd / gmax;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <shape_dec.gguf> <ref_dir> [gpu]\n", argv[0]); return 1; }
    const std::string ref = argv[2];
    const int gpu = argc > 3 ? atoi(argv[3]) : 0;
    trellis::Model m = trellis::Model::load(argv[1], gpu);

    npy::Array co = npy::load(ref + "/coords.npy");    // [N,3]
    npy::Array fi = npy::load(ref + "/feats_in.npy");  // [N,Ci]
    npy::Array cref = npy::load(ref + "/conv_out.npy");// [N,Co]
    npy::Array bref = npy::load(ref + "/block_out.npy");
    const int N = (int)co.shape[0], C = (int)fi.shape[1];
    printf("N=%d C=%d\n", N, C);

    std::vector<std::array<int,3>> coords(N);
    for (int i = 0; i < N; ++i) coords[i] = { (int)co.data[i*3], (int)co.data[i*3+1], (int)co.data[i*3+2] };
    vector<int32_t> nbr = trellis::build_neighbor_table(coords);  // [27*N] tap-major

    // feats [N,Ci] -> [Ci,N]
    vector<float> feats((size_t)C * N);
    for (int n = 0; n < N; ++n) for (int cc = 0; cc < C; ++cc) feats[cc + (size_t)C*n] = fi.data[(size_t)n*C + cc];

    size_t meta = ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(8192, false) + (1<<20);
    ggml_context* c = ggml_init({ meta, nullptr, true });
    ggml_tensor* gf = ggml_new_tensor_2d(c, GGML_TYPE_F32, C, N);   ggml_set_input(gf);
    ggml_tensor* gn = ggml_new_tensor_2d(c, GGML_TYPE_I32, N, 27);  ggml_set_input(gn);
    ggml_tensor* convo = trellis::sparse_submconv(c, m, "blocks.3.0.conv", gf, gn, N);
    ggml_tensor* blko  = trellis::sparse_convnext(c, m, "blocks.3.0", gf, gn, N);
    ggml_set_output(convo); ggml_set_output(blko);
    ggml_cgraph* g = ggml_new_graph_custom(c, 8192, false);
    ggml_build_forward_expand(g, convo); ggml_build_forward_expand(g, blko);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    if (!ggml_gallocr_alloc_graph(alloc, g)) { fprintf(stderr, "alloc failed\n"); return 1; }
    ggml_backend_tensor_set(gf, feats.data(), 0, feats.size()*4);
    ggml_backend_tensor_set(gn, nbr.data(), 0, nbr.size()*4);
    if (ggml_backend_graph_compute(m.backend, g) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }

    double rc = rel(trellis::tensor_to_f32(convo), cref, C, N);
    double rb = rel(trellis::tensor_to_f32(blko),  bref, C, N);
    printf("conv_out  rel=%.4e  %s\n", rc, rc < 3e-2 ? "OK" : "**");
    printf("block_out rel=%.4e  %s\n", rb, rb < 3e-2 ? "OK" : "**");
    m.free();
    return 0;
}
