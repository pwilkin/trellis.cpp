// Thin CLI entry point: parse args, then run the shared trellis_run() pipeline.
#include "trellis_run.h"
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <image.png> <out.glb> [gpu] [models_dir] [seed]\n", argv[0]);
        return 1;
    }
    const int gpu = argc > 3 ? atoi(argv[3]) : 1;
    const std::string M = argc > 4 ? argv[4] : "/media/ilintar/D_SSD/models/trellis2/gguf";
    const uint32_t seed = argc > 5 ? (uint32_t)atoi(argv[5]) : 42;
    return trellis_run(argv[1], argv[2], gpu, M, seed);
}
