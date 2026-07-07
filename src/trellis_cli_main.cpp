// Thin CLI entry point: parse args, then run the shared trellis_run() pipeline.
#include "trellis_args.h"
#include "trellis_run.h"
#include <cstdio>

int main(int argc, char** argv) {
    trellis::TrellisParams p;
    if (!trellis::parse_args(argc, argv, p)) {
        trellis::print_usage(argv[0], /*server=*/false);
        return p.help ? 0 : 1;
    }
    if (p.image.empty()) {
        fprintf(stderr, "[trellis] no input image (give <image.png> or --image)\n");
        trellis::print_usage(argv[0], /*server=*/false);
        return 1;
    }
    return trellis_run(p);
}
