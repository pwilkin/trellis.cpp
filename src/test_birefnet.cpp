// Validate the BiRefNet Swin-L backbone against tools/ref_birefnet.py dumps.
//   trellis-test-birefnet <birefnet.gguf> <dump_dir> [gpu]
#include "trellis_model.h"
#include "birefnet.h"
#include "npy.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>

static void cmp(const char* name, const std::vector<float>& a, const npy::Array& g) {
    if ((int64_t)a.size() != g.numel()) { printf("%-10s SIZE MISMATCH %zu vs %lld\n", name, a.size(), (long long)g.numel()); return; }
    double maxd = 0, gmax = 0, sad = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        maxd = std::max(maxd, (double)std::fabs(a[i] - g.data[i]));
        gmax = std::max(gmax, (double)std::fabs(g.data[i]));
        sad += std::fabs(a[i] - g.data[i]);
    }
    printf("%-10s max|d|=%.4e mean|d|=%.4e rel=%.4e  %s\n", name, maxd, sad / a.size(), maxd / gmax,
           maxd / gmax < 5e-2 ? "OK" : "**");
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <birefnet.gguf> <dump_dir> [gpu]\n", argv[0]); return 1; }
    int gpu = argc > 3 ? atoi(argv[3]) : 1;
    trellis::Model m = trellis::Model::load(argv[1], gpu);
    std::string d = argv[2];
    npy::Array in = npy::load(d + "/input.npy");        // [3,1024,1024]
    int S = (int)in.shape[2];
    printf("input %lldx%lldx%lld\n", (long long)in.shape[0], (long long)in.shape[1], (long long)in.shape[2]);
    trellis::BBOut bb = trellis::swin_backbone(m, in.data, S);
    for (int s = 0; s < 4; ++s) {
        npy::Array g = npy::load(d + "/bb_out" + std::to_string(s) + ".npy");
        printf("stage%d C=%d H=%d W=%d  ", s, bb.C[s], bb.H[s], bb.W[s]);
        cmp(("bb_out" + std::to_string(s)).c_str(), bb.f[s], g);
    }
    if (getenv("FULL")) {
        std::vector<float> logits = trellis::birefnet_matte(m, in.data, gpu);
        npy::Array g = npy::load(d + "/logits.npy");
        cmp("logits", logits, g);
        // write alpha for visual check
        npy::Array a; a.shape = {1024, 1024}; a.data.resize(logits.size());
        for (size_t i = 0; i < logits.size(); ++i) a.data[i] = 1.0f / (1.0f + std::exp(-logits[i]));
        npy::save("/tmp/birefnet_ref/alpha_cpp.npy", a.data.data(), a.shape);
        printf("wrote /tmp/birefnet_ref/alpha_cpp.npy\n");
    }
    m.free();
    return 0;
}
