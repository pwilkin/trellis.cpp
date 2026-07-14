#pragma once
#include <cstdint>
#include <string>

namespace trellis {

// Cross-module runtime flags. Set from parsed params at the start of trellis_run;
// the modules that own them read them with an environment fallback so test
// binaries (which don't parse args) keep their historical TRELLIS_* behavior.
extern bool g_sparse_cast_f32;  // defined in sparse.cpp        (TRELLIS_F32)
extern bool g_no_fa;            // defined in dit.cpp           (TRELLIS_NOFA)
extern bool g_require_gpu;      // defined in trellis_model.cpp (TRELLIS_REQUIRE_GPU)

// Every knob for one TRELLIS.2 image->3D run. Resolved as default -> environment
// (the historical TRELLIS_* / GSS / GSH names) -> CLI flag, with the CLI winning.
// trellis-cli and trellis-server share the parser: the server runs it once for its
// launch defaults, then per request to apply overrides (resolution, bg removal, ...).
struct TrellisParams {
    std::string image;                                          // input image (image->3D)
    std::string output = "model.glb";                           // output .glb
    std::string models = "/media/ilintar/D_SSD/models/trellis2/gguf";
    std::string host   = "127.0.0.1";                           // trellis-server only
    int      port = 8080;                                       // trellis-server only
    int      gpu  = 0;                                          // >=0 GPU index, <0 CPU
    uint32_t seed = 0;

    bool cascade    = true;     // 1024 cascade (default); --res 512 selects the light path
    int  hr_res     = 1024;     // HR cascade target resolution (1024 / 1536)
    int  max_tokens = 49152;    // HR token budget (backoff floors at 1024)

    bool birefnet = false;      // BiRefNet bg removal (else white-background threshold)
    bool texture  = true;       // texture flow + UV bake (else geometry-only)
    bool xatlas   = true;       // xatlas UV unwrap (else voxel-native box projection)
    int  decim    = -1;         // decimation cluster grid   (-1 => per-cascade default)
    int  tex      = -1;         // UV atlas size in px        (-1 => per-cascade default)
    int  tex_res  = -1;         // texture PBR resolution: -1 => auto (drop dense res-1024 tex to
                                //   512, whose clean coarse PBR bakes onto the res-1024 mesh
                                //   without the partial-coverage "skin" speckle); else force 512/1024
    bool f32      = false;      // f32 sparse-conv compute
    bool no_fa    = false;      // disable FlashAttention (manual softmax)
    bool require_gpu = false;   // refuse CPU fallback if no GPU is usable
    float gss = 7.5f;           // sparse-structure guidance strength
    float gsh = 7.5f;           // shape-SLAT guidance strength
    bool voxply = false;        // dump out/myvox.ply              (debug)
    bool dump_slat = false;     // dump /tmp/hr_slat.bin           (debug)

    bool help = false;          // --help requested

    // 512 -> light single-res path; 1024/1536 -> cascade with that HR target.
    void set_res(int res) {
        if (res <= 512) { cascade = false; hr_res = 512; }
        else            { cascade = true;  hr_res = res; }
    }
};

void print_usage(const char* argv0, bool server);

// Apply environment fallbacks, then parse argv (CLI wins). The first two bare
// (non-flag) positionals fill `image` then `output`. Returns false on a parse
// error OR when --help was requested; check p.help to tell them apart.
bool parse_args(int argc, char** argv, TrellisParams& p);

}  // namespace trellis
