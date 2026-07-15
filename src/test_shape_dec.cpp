// Validate the full FlexiDualGrid shape decoder vs pure-torch reference.
//   trellis-test-shape-dec <shape_dec.gguf> <ref_dir> [gpu]
#include "trellis_model.h"
#include "shape_decoder.h"
#include "npy.h"
#include <algorithm>
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

    // rel is a MAX-based metric: one bad voxel out of 4.8M spikes it. What actually
    // matters downstream is (a) how the error is distributed, and (b) whether it flips
    // a DERIVED decision: channels 0..2 feed 2*sigmoid(x)-0.5 (saturated => harmless),
    // channels 3..5 are tested >0 (a sign flip changes topology), channel 6 is a
    // softplus split weight. Report those instead of a single outlier.
    {
        std::vector<double> ad; ad.reserve((size_t)matched * 7);
        double chan_sum[7] = {0}, chan_max[7] = {0};
        int64_t chan_n[7] = {0};
        int64_t flips = 0, vert_big = 0;
        for (int i = 0; i < Mm; ++i) {
            auto it = rm.find(key(so.coords[i][0], so.coords[i][1], so.coords[i][2]));
            if (it == rm.end()) continue;
            const int j = it->second;
            for (int ch = 0; ch < 7; ++ch) {
                const double a = (double)so.feats7[ch + 7*i];
                const double b = (double)o7.data[(size_t)j*7 + ch];
                const double d = std::fabs(a - b);
                ad.push_back(d);
                chan_sum[ch] += d; chan_n[ch]++;
                chan_max[ch] = std::max(chan_max[ch], d);
                if (ch >= 3 && ch <= 5 && ((a > 0) != (b > 0))) ++flips;   // topology flip
                if (ch <= 2) {                                             // vertex offset shift
                    const double sa = 2.0 / (1.0 + std::exp(-a)) - 0.5;
                    const double sb = 2.0 / (1.0 + std::exp(-b)) - 0.5;
                    if (std::fabs(sa - sb) > 0.05) ++vert_big;             // >5% of a voxel
                }
            }
        }
        std::sort(ad.begin(), ad.end());
        auto pct = [&](double p) { return ad.empty() ? 0.0 : ad[(size_t)(p * (ad.size() - 1))]; };
        printf("  |d| percentiles: p50=%.2e p90=%.2e p99=%.2e p99.9=%.2e p100=%.2e\n",
               pct(0.50), pct(0.90), pct(0.99), pct(0.999), pct(1.0));
        static const char* CH[7] = {"off.x","off.y","off.z","isect.x","isect.y","isect.z","split_w"};
        for (int ch = 0; ch < 7; ++ch)
            printf("  ch%d %-8s mean|d|=%.3e  max|d|=%.3e\n", ch, CH[ch],
                   chan_n[ch] ? chan_sum[ch]/chan_n[ch] : 0.0, chan_max[ch]);
        printf("  TOPOLOGY flips (ch3-5 sign differs): %lld / %lld  (%.4f%%)\n",
               (long long)flips, (long long)matched*3, 100.0*flips/std::max<int64_t>(matched*3,1));
        printf("  VERTEX offsets shifted >0.05 voxel: %lld / %lld  (%.4f%%)\n",
               (long long)vert_big, (long long)matched*3, 100.0*vert_big/std::max<int64_t>(matched*3,1));
    }
    m.free();
    return 0;
}
