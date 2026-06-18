// Validate SparseResBlockC2S3d up-block vs pure-torch reference.
//   trellis-test-c2s <shape_dec.gguf> <ref_dir> [gpu]
#include "trellis_model.h"
#include "sparse.h"
#include "npy.h"
#include <cmath>
#include <cstdio>
#include <vector>
#include <array>
#include <unordered_map>

using std::vector; using std::array;

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <shape_dec.gguf> <ref_dir> [gpu]\n", argv[0]); return 1; }
    const std::string ref = argv[2];
    const int gpu = argc > 3 ? atoi(argv[3]) : 0;
    trellis::Model m = trellis::Model::load(argv[1], gpu);

    npy::Array co = npy::load(ref + "/coords.npy");      // [N,3]
    npy::Array fi = npy::load(ref + "/feats_in.npy");    // [N,Cin]
    npy::Array nco = npy::load(ref + "/new_coords.npy"); // [M,3]
    npy::Array out = npy::load(ref + "/out.npy");        // [M,Cout]
    const int N = (int)co.shape[0], Cin = (int)fi.shape[1], M = (int)nco.shape[0], Cout = (int)out.shape[1];
    printf("N=%d Cin=%d ref M=%d Cout=%d\n", N, Cin, M, Cout);

    vector<array<int,3>> coords(N);
    for (int i = 0; i < N; ++i) coords[i] = { (int)co.data[i*3], (int)co.data[i*3+1], (int)co.data[i*3+2] };
    vector<float> feats((size_t)Cin * N);   // [Cin,N]
    for (int n = 0; n < N; ++n) for (int cc = 0; cc < Cin; ++cc) feats[cc + (size_t)Cin*n] = fi.data[(size_t)n*Cin + cc];

    trellis::C2SResult r = trellis::sparse_c2s(m, "blocks.3.4", feats, Cin, coords, Cout);
    printf("mine M=%d\n", (int)r.coords.size());

    // map ref coords -> row
    auto key = [](int x,int y,int z){ return ((uint64_t)(uint32_t)x<<40)|((uint64_t)(uint32_t)y<<20)|(uint32_t)z; };
    std::unordered_map<uint64_t,int> refmap;
    for (int i = 0; i < M; ++i) refmap[key((int)nco.data[i*3],(int)nco.data[i*3+1],(int)nco.data[i*3+2])] = i;
    int matched = 0; double maxd = 0, gmax = 0, sumd = 0; long cnt = 0;
    for (int i = 0; i < (int)r.coords.size(); ++i) {
        auto it = refmap.find(key(r.coords[i][0], r.coords[i][1], r.coords[i][2]));
        if (it == refmap.end()) continue;
        matched++;
        int j = it->second;
        for (int cc = 0; cc < Cout; ++cc) {
            double d = std::fabs((double)r.feats[cc + (size_t)Cout*i] - out.data[(size_t)j*Cout + cc]);
            maxd = std::max(maxd, d); gmax = std::max(gmax, std::fabs((double)out.data[(size_t)j*Cout+cc])); sumd += d; cnt++;
        }
    }
    double coord_agree = 100.0 * matched / std::max(M, (int)r.coords.size());
    printf("coords matched=%d (%.2f%%)  feats rel=%.4e mean|d|=%.4e  %s\n",
           matched, coord_agree, maxd / gmax, sumd / cnt, (coord_agree > 98 && maxd/gmax < 3e-2) ? "OK" : "**");
    m.free();
    return 0;
}
