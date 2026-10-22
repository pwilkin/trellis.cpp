#include "trellis_args.h"

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <stdexcept>

namespace trellis {

bool parse_camera_arg(const std::string& name, const char* value, TrellisParams& p, std::string& error) {
    char* end = nullptr;
    errno = 0;
    if (name == "extend-pixel") {
        const long n = std::strtol(value, &end, 10);
        if (end == value || *end || errno == ERANGE || n < 0 || n > INT_MAX) {
            error = "extend-pixel must be a nonnegative integer";
            return false;
        }
        p.extend_pixel = (int)n;
        return true;
    }
    const float n = std::strtof(value, &end);
    if (end == value || *end || errno == ERANGE || !std::isfinite(n) || n <= 0 ||
        (name == "fov" && n >= 180)) {
        error = name == "fov" ? "fov must be finite and between 0 and 180 degrees"
                              : "mesh-scale must be finite and positive";
        return false;
    }
    if (name == "fov") p.fov_deg = n;
    else p.mesh_scale = n;
    return true;
}

bool parse_faces_arg(const char* value, TrellisParams& p, std::string& error) {
    if (std::string(value) == "auto") { p.faces = -1; return true; }
    char* end = nullptr;
    errno = 0;
    const long n = std::strtol(value, &end, 10);
    if (end == value || *end || errno == ERANGE || n < kQemFacesMin || n > INT_MAX) {
        error = "faces must be an integer >= " + std::to_string(kQemFacesMin) + " (or auto)";
        return false;
    }
    p.faces = (int)n;
    return true;
}

bool check_faces_compat(const TrellisParams& p, std::string& error) {
    if (p.faces > 0 && p.decim >= 0) {
        error = "faces sets the QEM decimation target and cannot be combined with --decim";
        return false;
    }
    return true;
}

const char* model_family_name(ModelFamily f) {
    return f == ModelFamily::Pixal3D ? "pixal3d" : "trellis";
}

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
        "      --copyright TEXT    glTF asset.copyright metadata\n"
        "  -m, --models DIR        GGUF model directory\n"
        "      --model FAMILY      trellis (default) | pixal3d — which flow weights the model\n"
        "                          directory holds. pixal3d swaps the DINOv3 cross-attention for\n"
        "                          view-aligned projection conditioning; the samplers, decoders\n"
        "                          and every postprocessing stage are shared.\n"
        "      --fov DEG           pixal3d: horizontal field of view of the input image, which\n"
        "                          fixes the projection camera (default 49.13, Pixal3D's own).\n"
        "                          Upstream estimates this with MoGe-2; that model is not ported,\n"
        "                          so a wrong FOV shows up as geometry drifting off the silhouette.\n"
        "      --mesh-scale F      pixal3d: object scale inside the unit grid   (default 1.0)\n"
        "      --extend-pixel N    pixal3d: push the camera's virtual image border outward by N\n"
        "                          pixels of a 512 frame (default 0). Rarely useful: background\n"
        "                          removal already reframes around the subject, and extending\n"
        "                          past the cutout projects onto pixels that do not exist.\n"
        "      --no-naf            pixal3d: skip NAF guided upsampling (needs no naf.gguf, but\n"
        "                          the shape/texture stages then lose their high-frequency\n"
        "                          projection branch)\n"
        "      --gpu N             GPU index, <0 = CPU          (default 0)\n"
        "      --backend NAME      force a ggml backend: Vulkan | HTP | CPU | ...\n"
        "                          (HTP is the Qualcomm Hexagon NPU. Default: auto.\n"
        "                          Run trellis-devices to list what this build sees.\n"
        "                          Needed on Snapdragon, where the NPU and the Adreno\n"
        "                          GPU both register as GPU devices.)\n"
        "  -t, --threads N         CPU backend threads (default: all cores. ggml's own\n"
        "                          default is 4 regardless of core count.)\n"
        "      --sched on|off      run graphs through the multi-backend scheduler so ops\n"
        "                          the chosen backend cannot execute fall back to another\n"
        "                          (default: on unless the backend is the CPU). Required\n"
        "                          for --backend HTP, whose op coverage is partial.\n"
        "      --vulkan-fallback  with HTP, try Vulkan before CPU for unsupported ops\n"
        "                          (same as TRELLIS_VULKAN_FALLBACK=1)\n"
        "      --no-vulkan-fallback\n"
        "                          disable Vulkan fallback even if the environment enables it\n"
        "  -v, --verbose           per-stage timings, graph node counts, and a heartbeat\n"
        "                          while a backend compute is running (TRELLIS_VERBOSE=1;\n"
        "                          =2 also dumps the per-op histogram of each graph)\n"
        "  -s, --seed N            RNG seed                     (default 42)\n"
        "      --steps N           flow sampler steps (default 12). Lower is faster and\n"
        "                          rougher; useful for backend bring-up, where a CPU and\n"
        "                          an NPU run at the same N stay directly comparable.\n"
        "      --res 512|1024|1536 geometry resolution\n"
        "      --max-tokens N      HR token budget              (default 49152)\n"
        "      --bg-removal MODE   threshold | birefnet   (default: auto -- a pre-matted\n"
        "                          image keeps its alpha; otherwise BiRefNet when its model\n"
        "                          is present. The plain threshold matte cuts out specular\n"
        "                          highlights, which the flow then turns into holes.)\n"
        "      --birefnet          alias for --bg-removal birefnet\n"
        "      --no-texture        geometry only\n"
        "      --xatlas            xatlas UV unwrap (default)\n"
        "      --box-uv            voxel-native box projection (faster)\n"
        "      --band N            narrow-band DC remesh band width (default: auto —\n"
        "                          res/512, i.e. 1 @512 / 2 @1024, which suppresses the\n"
        "                          res-1024 outer-skin speckle; N forces that width)\n"
        "      --faces N           QEM decimation face target, >= 1000 (default 300K @1024\n"
        "                          and @1536 / 150K @512). Lower = lighter asset, faster\n"
        "                          UV unwrap; e.g. 50000 for a game prop\n"
        "      --decim GRID        legacy cluster-grid decimation instead of QEM\n"
        "                          (0 = keep the full-res mesh); not combinable with --faces\n"
        "      --atlas PX          UV atlas size (default 2048 @1024 / 1024 @512)\n"
        "      --tex-res N         texture PBR resolution 512/1024 (default: auto — drops\n"
        "                          a dense res-1024 decode to a clean res-512 PBR volume)\n"
        "      --webp on|off       encode GLB textures as WebP (default: on when built with\n"
        "                          WebP support; off = PNG)\n"
        "      --normal-map on|off bake a tangent-space normal map from the dense remeshed\n"
        "                          shell onto the decimated mesh (3rd GLB texture; default off)\n"
        "      --dump-bg           also write the background-removal cutout as <out>_cutout.png\n"
        "      --dump-post PATH    write the raw decoded mesh + sparse PBR volume to PATH and\n"
        "                          exit (skips remesh/decimate/UV/bake/GLB) -- for external\n"
        "                          post-processing pipelines\n"
        "      --bg-only           background removal only: write the cutout and skip the rest\n"
        "      --retopo GRID FIRST_FACES FINAL_FACES  native quad export (Linux, retopo build)\n"
        "      --retopo-atlas PX   quad atlas size: 1024, 2048, 4096 (default 4096)\n"
        "      --retopo-no-weld-fill  skip initial weld/fill before tetra remesh\n"
        "      --retopo-dual-pbr  decode 1024 PBR for quad export with 512 primary PBR\n"
        "      --retopo-workdir DIR   large intermediates (default .codex/retopo/runs)\n"
        "      --f32               f32 sparse-conv compute\n"
        "      --fa-fast           force F16 K/V + fast accumulation (default on HTP)\n"
        "      --fa-f32            use BF16 K/V + F32 accumulation (HTP opt-out)\n"
        "      --no-fa             disable FlashAttention\n"
        "      --require-gpu       refuse CPU fallback\n"
        "      --threads N         CPU backend threads      (default all cores)\n"
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
        auto integer = [&](const char* value, int& target) -> bool {
            try {
                size_t used=0;
                const int parsed=std::stoi(value,&used);
                if (used!=std::strlen(value)) throw std::invalid_argument("trailing characters");
                target=parsed;
                return true;
            } catch (...) {
                fprintf(stderr,"[trellis] invalid integer: %s\n",value);
                return false;
            }
        };

        if      (a == "-h" || a == "--help")    { p.help = true; return false; }
        else if (a == "-i" || a == "--image")   { const char* v = need(a.c_str()); if (!v) return false; p.image = v; }
        else if (a == "-o" || a == "--output")  { const char* v = need(a.c_str()); if (!v) return false; p.output = v; }
        else if (a == "--copyright")            { const char* v = need(a.c_str()); if (!v) return false; p.copyright = v; }
        else if (a == "-m" || a == "--models")  { const char* v = need(a.c_str()); if (!v) return false; p.models = v; }
        else if (a == "--model")                { const char* v = need(a.c_str()); if (!v) return false;
                                                  if      (std::strcmp(v, "trellis") == 0) p.family = ModelFamily::Trellis;
                                                  else if (std::strcmp(v, "pixal3d") == 0) p.family = ModelFamily::Pixal3D;
                                                  else { fprintf(stderr, "[trellis] unknown model family: %s (trellis|pixal3d)\n", v); return false; } }
        else if (a == "--fov" || a == "--mesh-scale" || a == "--extend-pixel") {
            const char* v = need(a.c_str()); if (!v) return false;
            std::string error;
            if (!parse_camera_arg(a.substr(2), v, p, error)) {
                fprintf(stderr, "[trellis] %s\n", error.c_str());
                return false;
            }
        }
        else if (a == "--no-naf")               { p.naf = false; }
        else if (a == "--gpu")                  { const char* v = need(a.c_str()); if (!v) return false; p.gpu = atoi(v); }
        else if (a == "--backend")              { const char* v = need(a.c_str()); if (!v) return false; p.backend = v; }
        else if (a == "-t" || a == "--threads") { const char* v = need(a.c_str()); if (!v) return false; p.threads = atoi(v); }
        else if (a == "-v" || a == "--verbose") { p.verbose = true; }
        else if (a == "--steps")                { const char* v = need(a.c_str()); if (!v) return false; p.steps = atoi(v); }
        else if (a == "--sched")                { const char* v = need(a.c_str()); if (!v) return false;
                                                  if (strcmp(v, "on") == 0 || strcmp(v, "1") == 0) p.sched = 1;
                                                  else if (strcmp(v, "off") == 0 || strcmp(v, "0") == 0) p.sched = 0;
                                                  else { fprintf(stderr, "[trellis] --sched expects on or off\n"); return false; } }
        else if (a == "--vulkan-fallback")      { p.vulkan_fallback = 1; }
        else if (a == "--no-vulkan-fallback")   { p.vulkan_fallback = 0; }
        else if (a == "-s" || a == "--seed")    { const char* v = need(a.c_str()); if (!v) return false; p.seed = (uint32_t)atoi(v); }
        else if (a == "--res")                  { const char* v = need(a.c_str()); if (!v) return false; p.set_res(atoi(v)); }
        else if (a == "--max-tokens")           { const char* v = need(a.c_str()); if (!v) return false; p.max_tokens = atoi(v); }
        else if (a == "--bg-removal")           { const char* v = need(a.c_str()); if (!v) return false; p.birefnet = (std::strcmp(v, "birefnet") == 0) ? 1 : 0; }
        else if (a == "--birefnet")             { p.birefnet = 1; }
        else if (a == "--no-texture")           { p.texture = false; }
        else if (a == "--xatlas")               { p.xatlas = true; }
        else if (a == "--box-uv")               { p.xatlas = false; }
        else if (a == "--dump-post")            { const char* v = need(a.c_str()); if (!v) return false; p.dump_post = v; }
        else if (a == "--band")                 { const char* v = need(a.c_str()); if (!v) return false; p.band = atoi(v); }
        else if (a == "--decim")                { const char* v = need(a.c_str()); if (!v) return false; p.decim = atoi(v); }
        else if (a == "--faces")                { const char* v = need(a.c_str()); if (!v) return false;
                                                  std::string error;
                                                  if (!parse_faces_arg(v, p, error)) { fprintf(stderr, "[trellis] %s\n", error.c_str()); return false; } }
        else if (a == "--atlas" || a == "--tex"){ const char* v = need(a.c_str()); if (!v) return false; p.tex = atoi(v); }
        else if (a == "--tex-res")              { const char* v = need(a.c_str()); if (!v) return false; p.tex_res = atoi(v); }
        else if (a == "--webp")                 { const char* v = need(a.c_str()); if (!v) return false;
                                                  p.webp = (std::strcmp(v,"off")==0 || std::strcmp(v,"0")==0 || std::strcmp(v,"false")==0) ? 0
                                                         : (std::strcmp(v,"on")==0 || std::strcmp(v,"1")==0 || std::strcmp(v,"true")==0) ? 1 : -1; }
        else if (a == "--normal-map")           { const char* v = need(a.c_str()); if (!v) return false;
                                                  if      (std::strcmp(v,"on")==0  || std::strcmp(v,"1")==0 || std::strcmp(v,"true")==0)  p.normal_map = 1;
                                                  else if (std::strcmp(v,"off")==0 || std::strcmp(v,"0")==0 || std::strcmp(v,"false")==0) p.normal_map = 0;
                                                  else { fprintf(stderr, "[trellis] --normal-map must be on or off\n"); return false; } }
        else if (a == "--dump-bg")              { p.dump_bg = true; }
        else if (a == "--bg-only")              { p.bg_only = true; p.dump_bg = true; }
        else if (a == "--retopo")               { const char* g=need(a.c_str()); if (!g) return false;
                                                  const char* f=need(a.c_str()); if (!f) return false;
                                                  const char* q=need(a.c_str()); if (!q) return false;
                                                  p.retopo=true;
                                                  if (!integer(g,p.retopo_grid) || !integer(f,p.retopo_first_faces) ||
                                                      !integer(q,p.retopo_final_faces)) return false; }
        else if (a == "--retopo-atlas")         { const char* v=need(a.c_str()); if (!v) return false;
                                                  if (!integer(v,p.retopo_atlas)) return false; }
        else if (a == "--retopo-no-weld-fill")  { p.retopo_no_weld_fill=true; }
        else if (a == "--retopo-dual-pbr")      { p.retopo_dual_pbr=true; }
        else if (a == "--retopo-workdir")       { const char* v=need(a.c_str()); if (!v) return false; p.retopo_workdir=v; }
        else if (a == "--f32")                  { p.f32 = true; }
        else if (a == "--fa-fast")              { p.fa_fast = 1; }
        else if (a == "--fa-f32")               { p.fa_fast = 0; }
        else if (a == "--no-fa")                { p.no_fa = true; }
        else if (a == "--require-gpu")          { p.require_gpu = true; }
        else if (a == "--threads")              { const char* v = need(a.c_str()); if (!v) return false; p.threads = atoi(v); }
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
    if (p.retopo && ((p.retopo_grid!=512 && p.retopo_grid!=1024 && p.retopo_grid!=1536) ||
                     p.retopo_first_faces<0 || p.retopo_final_faces<=0 ||
                     (p.retopo_atlas!=1024 && p.retopo_atlas!=2048 && p.retopo_atlas!=4096) ||
                     p.retopo_workdir.empty())) {
        fprintf(stderr,"[trellis] invalid retopo settings\n");
        return false;
    }
    {
        std::string error;
        if (!check_faces_compat(p, error)) { fprintf(stderr, "[trellis] %s\n", error.c_str()); return false; }
    }
    if (!p.retopo && p.retopo_no_weld_fill) {
        fprintf(stderr,"[trellis] --retopo-no-weld-fill requires --retopo\n");
        return false;
    }
    if (p.retopo_dual_pbr && (!p.retopo || p.family != ModelFamily::Trellis ||
                              !p.cascade || p.hr_res != 1024 || p.tex_res != 512)) {
        fprintf(stderr,"[trellis] --retopo-dual-pbr requires --model trellis, --retopo, --res 1024, and --tex-res 512\n");
        return false;
    }
    return true;
}

}  // namespace trellis
