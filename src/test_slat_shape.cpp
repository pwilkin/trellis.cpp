// Validate the sparse SLAT shape flow (SLatFlowModel img2shape) vs PyTorch golden.
//   trellis-test-slat-shape <shape_flow.gguf> <ref_dir> [gpu] [GS] [GR]
#include "trellis_model.h"
#include "flow_runner.h"
#include "ss_decoder.h"   // (array include)
#include "npy.h"
#include <cmath>
#include <cstdio>
#include <vector>
#include <array>

using std::vector;

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <shape_flow.gguf> <ref_dir> [gpu] [GS] [GR]\n", argv[0]); return 1; }
    const std::string gguf = argv[1], ref = argv[2];
    const int gpu = argc > 3 ? atoi(argv[3]) : 0;
    const float GS = argc > 4 ? atof(argv[4]) : 7.5f;
    const float GR = argc > 5 ? atof(argv[5]) : 0.5f;

    trellis::Model m = trellis::Model::load(gguf, gpu);
    npy::Array co = npy::load(ref + "/coords.npy");   // [N,3] f32
    npy::Array no = npy::load(ref + "/noise.npy");     // [N,32]
    npy::Array cn = npy::load(ref + "/cond.npy");      // [N_img,1024]
    npy::Array ng = npy::load(ref + "/neg.npy");
    npy::Array gd = npy::load(ref + "/samples.npy");   // [N,32]
    const int64_t N = co.shape[0], Cin = no.shape[1], Nimg = cn.shape[0], Dc = cn.shape[1];
    printf("N=%lld Cin=%lld Nimg=%lld Dc=%lld GS=%.1f GR=%.1f\n",(long long)N,(long long)Cin,(long long)Nimg,(long long)Dc,GS,GR);

    std::vector<std::array<int,3>> coords(N);
    for (int64_t i = 0; i < N; ++i) coords[i] = { (int)co.data[i*3+0], (int)co.data[i*3+1], (int)co.data[i*3+2] };

    trellis::DiTParams p; p.in_ch = (int)Cin; p.out_ch = (int)Cin; p.d_cond = (int)Dc;
    if (getenv("TRELLIS_F32W")) { p.cast_f32 = true; printf("(f32 weight compute)\n"); }
    trellis::DitRunner* run = trellis::make_sparse_runner(m, p, coords, (int)Nimg);
    trellis::FlowFwd fwd = [&](const vector<float>& xt, float ts, const float* cd){ return run->forward(xt, ts, cd); };

    // noise [n*Cin+c] -> xt [c + Cin*n]
    vector<float> xt(N * Cin);
    for (int64_t n = 0; n < N; ++n) for (int64_t c = 0; c < Cin; ++c) xt[c + Cin*n] = no.data[n*Cin + c];

    trellis::SamplerParams sp;
    sp.steps = 12; sp.guidance_strength = GS; sp.guidance_rescale = GR;
    sp.gi0 = 0.6f; sp.gi1 = 1.0f; sp.rescale_t = 3.0f; sp.sigma_min = 1e-5f;

    vector<float> out = trellis::sample_flow(fwd, xt, cn.data.data(), ng.data.data(), sp);

    // out [c + Cin*n] -> compare to gd [n*Cin+c]
    double maxd = 0, gmax = 0, sumd = 0;
    for (int64_t n = 0; n < N; ++n) for (int64_t c = 0; c < Cin; ++c) {
        double d = std::fabs((double)out[c + Cin*n] - gd.data[n*Cin + c]);
        maxd = std::max(maxd, d); sumd += d; gmax = std::max(gmax, std::fabs((double)gd.data[n*Cin + c]));
    }
    printf("samples: max|d|=%.4e mean|d|=%.4e rel=%.4e  %s\n", maxd, sumd/(N*Cin), maxd/gmax, maxd/gmax < 3e-2 ? "OK" : "**");
    m.free();
    return 0;
}
