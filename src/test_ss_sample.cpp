// Validate the C++ FlowEuler sampler (driving the SS-flow DiT) vs PyTorch golden.
//   trellis-test-ss-sample <ss_flow.gguf> <ref_dir> [gpu]
#include "trellis_model.h"
#include "flow_runner.h"
#include "npy.h"
#include <cmath>
#include <cstdio>
#include <vector>

using std::vector;

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <ss_flow.gguf> <ref_dir> [gpu]\n", argv[0]); return 1; }
    const std::string gguf = argv[1], ref = argv[2];
    const int gpu = argc > 3 ? atoi(argv[3]) : 0;

    trellis::Model m = trellis::Model::load(gguf, gpu);

    npy::Array noise = npy::load(ref + "/noise.npy");      // [1,8,16,16,16]
    npy::Array cond  = npy::load(ref + "/cond.npy");       // [1,1029,1024]
    npy::Array neg   = npy::load(ref + "/neg_cond.npy");
    npy::Array gold  = npy::load(ref + "/samples.npy");
    const int64_t Cin = noise.shape[1], R = noise.shape[2], L = R * R * R;
    const int64_t Lc = cond.shape[1], Dc = cond.shape[2];
    printf("Cin=%lld R=%lld L=%lld Lc=%lld Dc=%lld\n",(long long)Cin,(long long)R,(long long)L,(long long)Lc,(long long)Dc);

    trellis::DiTParams p; p.in_ch = (int)Cin; p.out_ch = (int)Cin; p.d_cond = (int)Dc;
    if (getenv("TRELLIS_F32W")) { p.cast_f32 = true; printf("(f32 weight compute)\n"); }
    trellis::DitRunner* run = trellis::make_dense_runner(m, p, (int)R, (int)Lc);
    trellis::FlowFwd fwd = [&](const std::vector<float>& xt, float ts, const float* cd){ return run->forward(xt, ts, cd); };

    // noise [c*L+sp] -> sample [c + Cin*sp]
    vector<float> sample(Cin * L);
    for (int64_t c = 0; c < Cin; ++c) for (int64_t sp = 0; sp < L; ++sp) sample[c + Cin * sp] = noise.data[c * L + sp];

    trellis::SamplerParams sp;
    sp.steps = 12; sp.guidance_strength = 7.5f; sp.guidance_rescale = 0.7f;
    sp.gi0 = 0.6f; sp.gi1 = 1.0f; sp.rescale_t = 5.0f; sp.sigma_min = 1e-5f;
    if (getenv("GS")) sp.guidance_strength = atof(getenv("GS"));
    if (getenv("GR")) sp.guidance_rescale = atof(getenv("GR"));

    vector<vector<float>> trace;
    vector<float> out = trellis::sample_flow(fwd, sample, cond.data.data(), neg.data.data(), sp, &trace);

    // per-step element-wise diff vs reference step_{i}.npy (remap [c+Cin*sp]->[c*L+sp])
    for (size_t i = 0; i < trace.size(); ++i) {
        std::string sp_path = ref + "/step_" + std::to_string(i) + ".npy";
        FILE* f = fopen(sp_path.c_str(), "rb"); if (!f) continue; fclose(f);
        npy::Array rs = npy::load(sp_path);
        double maxd = 0, gmax = 0;
        for (int64_t c = 0; c < Cin; ++c) for (int64_t s2 = 0; s2 < L; ++s2) {
            double d = std::fabs((double)trace[i][c + Cin * s2] - rs.data[c * L + s2]);
            maxd = std::max(maxd, d); gmax = std::max(gmax, std::fabs((double)rs.data[c * L + s2]));
        }
        printf("  step %2zu rel=%.4e\n", i, maxd / gmax);
    }

    // remap out [c+Cin*sp] -> [c*L+sp] to match gold
    vector<float> remap(Cin * L);
    for (int64_t c = 0; c < Cin; ++c) for (int64_t sp2 = 0; sp2 < L; ++sp2) remap[c * L + sp2] = out[c + Cin * sp2];
    double maxd = 0, sumd = 0, gmax = 0;
    for (int64_t i = 0; i < Cin * L; ++i) { double d = std::fabs(remap[i] - gold.data[i]); maxd = std::max(maxd, d); sumd += d; gmax = std::max(gmax, std::fabs((double)gold.data[i])); }
    printf("samples: max|d|=%.4e mean|d|=%.4e rel=%.4e  %s\n", maxd, sumd / (Cin * L), maxd / gmax, (maxd / gmax) < 3e-2 ? "OK" : "**");
    m.free();
    return 0;
}
