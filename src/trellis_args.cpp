#include "trellis_args.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace trellis {

void print_usage(const char* argv0, bool server) {
    if (server) {
        fprintf(stderr,
            "usage: %s [--host H] [--port P] [--models DIR] [--gpu N] [generation defaults...]\n",
            argv0);
    } else {
        fprintf(stderr,
            "usage: %s <image.png> <out.glb> [options]\n"
            "   or: %s --image <image.png> --output <out.glb> [options]\n",
            argv0, argv0);
    }
    fprintf(stderr,
        "\n"
        "  -i, --image PATH        input image                  (image->3D)\n"
        "  -o, --output PATH       output .glb                  (default model.glb)\n"
        "  -m, --models DIR        GGUF model directory\n"
        "      --gpu N             GPU index, <0 = CPU          (default 0)\n"
        "  -s, --seed N            RNG seed                     (default 42)\n"
        "      --res 512|1024|1536 geometry resolution\n"
        "      --max-tokens N      HR token budget              (default 49152)\n"
        "      --bg-removal MODE   threshold | birefnet\n"
        "      --birefnet          alias for --bg-removal birefnet\n"
        "      --no-texture        geometry only\n"
        "      --xatlas            xatlas UV unwrap (default)\n"
        "      --box-uv            voxel-native box projection (faster)\n"
        "      --decim GRID        legacy cluster-grid decimation (default: quadric\n"
        "                          simplify to 300K faces @1024 / 150K @512; 0 = none)\n"
        "      --atlas PX          UV atlas size (default 2048 @1024 / 1024 @512)\n"
        "      --f32               f32 sparse-conv compute\n"
        "      --no-fa             disable FlashAttention\n"
        "      --require-gpu       refuse CPU fallback\n"
        "      --gss F  --gsh F    guidance strengths\n"
        "      --host H  --port P  trellis-server bind address\n"
        "      --voxply            also dump the voxel point cloud as .ply\n"
        "      --dump-slat         dump the structured latent to disk\n"
        "  -h, --help              show this help\n");
}

bool parse_args(int argc, char** argv, TrellisParams& p) {
    int positional = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "[trellis] %s needs a value\n", name); return nullptr; }
            return argv[++i];
        };
        auto need = [&](const char* name) -> const char* {
            const char* v = next(name);
            return v;
        };

        if      (a == "-h" || a == "--help")    { p.help = true; return false; }
        else if (a == "-i" || a == "--image")   { const char* v = need(a.c_str()); if (!v) return false; p.image = v; }
        else if (a == "-o" || a == "--output")  { const char* v = need(a.c_str()); if (!v) return false; p.output = v; }
        else if (a == "-m" || a == "--models")  { const char* v = need(a.c_str()); if (!v) return false; p.models = v; }
        else if (a == "--gpu")                  { const char* v = need(a.c_str()); if (!v) return false; p.gpu = atoi(v); }
        else if (a == "-s" || a == "--seed")    { const char* v = need(a.c_str()); if (!v) return false; p.seed = (uint32_t)atoi(v); }
        else if (a == "--res")                  { const char* v = need(a.c_str()); if (!v) return false; p.set_res(atoi(v)); }
        else if (a == "--max-tokens")           { const char* v = need(a.c_str()); if (!v) return false; p.max_tokens = atoi(v); }
        else if (a == "--bg-removal")           { const char* v = need(a.c_str()); if (!v) return false; p.birefnet = (std::strcmp(v, "birefnet") == 0); }
        else if (a == "--birefnet")             { p.birefnet = true; }
        else if (a == "--no-texture")           { p.texture = false; }
        else if (a == "--xatlas")               { p.xatlas = true; }
        else if (a == "--box-uv")               { p.xatlas = false; }
        else if (a == "--decim")                { const char* v = need(a.c_str()); if (!v) return false; p.decim = atoi(v); }
        else if (a == "--atlas" || a == "--tex"){ const char* v = need(a.c_str()); if (!v) return false; p.tex = atoi(v); }
        else if (a == "--f32")                  { p.f32 = true; }
        else if (a == "--no-fa")                { p.no_fa = true; }
        else if (a == "--require-gpu")          { p.require_gpu = true; }
        else if (a == "--gss")                  { const char* v = need(a.c_str()); if (!v) return false; p.gss = (float)atof(v); }
        else if (a == "--gsh")                  { const char* v = need(a.c_str()); if (!v) return false; p.gsh = (float)atof(v); }
        else if (a == "--host")                 { const char* v = need(a.c_str()); if (!v) return false; p.host = v; }
        else if (a == "--port")                 { const char* v = need(a.c_str()); if (!v) return false; p.port = atoi(v); }
        else if (a == "--voxply")               { p.voxply = true; }
        else if (a == "--dump-slat")            { p.dump_slat = true; }
        else if (!a.empty() && a[0] == '-')     { fprintf(stderr, "[trellis] unknown option: %s\n", a.c_str()); return false; }
        else if (positional == 0)               { p.image  = a; positional = 1; }
        else if (positional == 1)               { p.output = a; positional = 2; }
        else                                    { fprintf(stderr, "[trellis] unexpected argument: %s\n", a.c_str()); return false; }
    }
    return true;
}

}  // namespace trellis
