// Thin CLI entry point: parse args, then run the shared trellis_run() pipeline.
#include "trellis_args.h"
#include "trellis_run.h"
#include <cstdio>
#include <exception>

#if defined(_WIN32)
#include <windows.h>
namespace {
// Long inference runs should survive the system idle timeout. The display may still blank.
struct KeepAwake {
    bool held = false;
    KeepAwake() {
        held = SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) != 0;
        if (!held) fprintf(stderr, "[trellis] could not hold a wake lock; sleep may interrupt this run\n");
    }
    ~KeepAwake() { if (held) SetThreadExecutionState(ES_CONTINUOUS); }
};
}  // namespace
#else
namespace { struct KeepAwake {}; }
#endif

int main(int argc, char** argv) {
    // Keep stage progress visible when stdout is redirected to a file or pipe.
    setvbuf(stdout, nullptr, _IONBF, 0);

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

    KeepAwake _awake;
    // trellis-server already reports these to the client; the CLI used to abort() through an
    // uncaught throw, which turns an "out of memory on Vulkan0 ..." message into a core dump.
    try {
        return trellis_run(p);
    } catch (const std::exception& e) {
        fprintf(stderr, "[trellis] failed: %s\n", e.what());
        return 1;
    }
}
