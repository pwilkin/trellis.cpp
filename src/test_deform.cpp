// Validate the custom deformable conv kernel against torchvision (tools/ref_deform.py).
//   trellis-test-deform <ref_dir> [gpu]
#include "deform_conv.h"
#include "npy.h"
#include <cstdio>
#include <cmath>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <ref_dir> [gpu]\n", argv[0]); return 1; }
    int gpu = argc > 2 ? atoi(argv[2]) : 1;
    std::string d = argv[1];
    npy::Array x = npy::load(d + "/x.npy"), off = npy::load(d + "/offset.npy"),
               mask = npy::load(d + "/mask.npy"), w = npy::load(d + "/weight.npy"),
               b = npy::load(d + "/bias.npy"), g = npy::load(d + "/out.npy");
    int Cin = (int)x.shape[0], H = (int)x.shape[1], W = (int)x.shape[2];
    int Cout = (int)w.shape[0], K = (int)w.shape[2];
    std::vector<float> out((size_t)Cout * H * W);
    trellis::deform_conv2d_run(x.data.data(), Cin, H, W, off.data.data(), mask.data.data(),
                               w.data.data(), b.data.data(), Cout, K, out.data(), gpu);
    double maxd = 0, gmax = 0, sad = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        maxd = std::max(maxd, (double)std::fabs(out[i] - g.data[i]));
        gmax = std::max(gmax, (double)std::fabs(g.data[i])); sad += std::fabs(out[i] - g.data[i]);
    }
    printf("deform K=%d Cin=%d Cout=%d %dx%d : max|d|=%.4e mean|d|=%.4e rel=%.4e  %s\n",
           K, Cin, Cout, H, W, maxd, sad / out.size(), maxd / gmax, maxd / gmax < 1e-4 ? "OK" : "**");
    return 0;
}
