// Validate the GGML DINOv3 ViT-L vs the timm golden.
//   trellis-test-dinov3 <dinov3.gguf> <ref_dir> [gpu]
#include "trellis_model.h"
#include "dinov3.h"
#include "npy.h"
#include <cmath>
#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <dinov3.gguf> <ref_dir> [gpu]\n", argv[0]); return 1; }
    const int gpu = argc > 3 ? atoi(argv[3]) : 0;
    trellis::Model m = trellis::Model::load(argv[1], gpu);
    npy::Array chw = npy::load(std::string(argv[2]) + "/input_chw.npy");  // [1,3,512,512]
    npy::Array gold = npy::load(std::string(argv[2]) + "/cond.npy");      // [1,1029,1024]
    const int S = (int)chw.shape[3];
    printf("S=%d gold tokens=%lld dim=%lld\n", S, (long long)gold.shape[1], (long long)gold.shape[2]);

    std::vector<float> cond = trellis::dinov3_encode(m, chw.data, S);     // [1024*Ntok], flat = d + 1024*tok
    if ((int64_t)cond.size() != gold.numel()) { printf("SIZE MISMATCH %zu vs %lld\n", cond.size(), (long long)gold.numel()); return 1; }
    double maxd = 0, sumd = 0, gmax = 0;
    for (size_t i = 0; i < cond.size(); ++i) {   // same flat layout (d+1024*tok == tok*1024+d)
        double d = std::fabs((double)cond[i] - gold.data[i]);
        maxd = std::max(maxd, d); sumd += d; gmax = std::max(gmax, std::fabs((double)gold.data[i]));
    }
    printf("cond: max|d|=%.4e mean|d|=%.4e rel=%.4e  %s\n", maxd, sumd / cond.size(), maxd / gmax, maxd / gmax < 5e-2 ? "OK" : "**");
    m.free();
    return 0;
}
