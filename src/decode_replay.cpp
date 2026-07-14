// decode-replay — load a --dump-slat HR-slat dump and run just the FlexiDualGrid
// shape decode, so the ~18-minute flow is skipped when iterating on the decoder.
//   decode-replay <shape_dec.gguf> <hr_slat.bin> [gpu]
#include "trellis_model.h"
#include "shape_decoder.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <shape_dec.gguf> <hr_slat.bin> [gpu]\n", argv[0]); return 1; }
    const std::string gguf = argv[1], bin = argv[2];
    const int gpu = argc > 3 ? atoi(argv[3]) : 0;
    FILE* f = fopen(bin.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", bin.c_str()); return 1; }
    int N = 0, res = 0;
    if (fread(&N, 4, 1, f) != 1 || fread(&res, 4, 1, f) != 1) { fprintf(stderr, "bad header\n"); return 1; }
    printf("[replay] N=%d res=%d\n", N, res); fflush(stdout);
    std::vector<std::array<int,3>> coords(N);
    for (int i = 0; i < N; ++i) {
        int xyz[3];
        if (fread(xyz, 4, 3, f) != 3) { fprintf(stderr, "bad coords\n"); return 1; }
        coords[i] = { xyz[0], xyz[1], xyz[2] };
    }
    std::vector<float> feats((size_t)32 * N);
    if (fread(feats.data(), 4, feats.size(), f) != feats.size()) { fprintf(stderr, "bad feats\n"); return 1; }
    fclose(f);
    printf("[replay] loaded; decoding on gpu %d ...\n", gpu); fflush(stdout);
    trellis::Model m = trellis::Model::load(gguf, gpu);
    trellis::ShapeOut so = trellis::shape_decode(m, feats, coords, res);
    printf("[replay] OK decoded voxels = %zu  feats7 = %zu\n", so.coords.size(), so.feats7.size());
    m.free();
    return 0;
}
