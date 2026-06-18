// Validate the full FlexiDualGrid shape decoder vs pure-torch reference.
//   trellis-test-shape-dec <shape_dec.gguf> <ref_dir> [gpu]
#include "trellis_model.h"
#include "shape_decoder.h"
#include "npy.h"
#include <cmath>
#include <cstdio>
#include <vector>
#include <array>
#include <unordered_map>

using std::vector; using std::array;

namespace trellis { extern bool g_sparse_cast_f32; }
int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <shape_dec.gguf> <ref_dir> [gpu]\n", argv[0]); return 1; }
    const std::string ref = argv[2];
    const int gpu = argc > 3 ? atoi(argv[3]) : 0;
    if (getenv("TRELLIS_F32W")) trellis::g_sparse_cast_f32 = true;
    trellis::Model m = trellis::Model::load(argv[1], gpu);

    npy::Array c0 = npy::load(ref + "/coords0.npy");   // [N0,3]
    npy::Array la = npy::load(ref + "/latent.npy");     // [N0,32]
    npy::Array cf = npy::load(ref + "/coords.npy");     // [M,3] final
    npy::Array o7 = npy::load(ref + "/out7.npy");       // [M,7]
    const int N0 = (int)c0.shape[0], M = (int)cf.shape[0];
    printf("N0=%d ref M=%d\n", N0, M);

    vector<array<int,3>> coords0(N0);
    for (int i = 0; i < N0; ++i) coords0[i] = { (int)c0.data[i*3], (int)c0.data[i*3+1], (int)c0.data[i*3+2] };
    vector<float> latent((size_t)32 * N0);                 // [32,N0]
    for (int n = 0; n < N0; ++n) for (int cc = 0; cc < 32; ++cc) latent[cc + 32*n] = la.data[(size_t)n*32 + cc];

    trellis::ShapeOut so = trellis::shape_decode(m, latent, coords0);
    const int Mm = (int)so.coords.size();
    printf("mine M=%d\n", Mm);

    auto key = [](int x,int y,int z){ return ((uint64_t)(uint32_t)x<<40)|((uint64_t)(uint32_t)y<<20)|(uint32_t)z; };
    std::unordered_map<uint64_t,int> rm;
    for (int i = 0; i < M; ++i) rm[key((int)cf.data[i*3],(int)cf.data[i*3+1],(int)cf.data[i*3+2])] = i;
    int matched = 0; double maxd = 0, gmax = 0, sumd = 0; long cnt = 0;
    for (int i = 0; i < Mm; ++i) {
        auto it = rm.find(key(so.coords[i][0], so.coords[i][1], so.coords[i][2]));
        if (it == rm.end()) continue;
        matched++; int j = it->second;
        for (int ch = 0; ch < 7; ++ch) {
            double d = std::fabs((double)so.feats7[ch + 7*i] - o7.data[(size_t)j*7 + ch]);
            maxd = std::max(maxd, d); gmax = std::max(gmax, std::fabs((double)o7.data[(size_t)j*7+ch])); sumd += d; cnt++;
        }
    }
    double overlap = 100.0 * matched / std::max(M, Mm);
    printf("coord overlap=%.2f%% (matched %d)  feats7 rel=%.4e mean|d|=%.4e  %s\n",
           overlap, matched, maxd/gmax, sumd/cnt, (overlap > 90 && maxd/gmax < 8e-2) ? "OK" : "**");
    m.free();
    return 0;
}
