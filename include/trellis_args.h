#pragma once
#include <cstdint>
#include <string>

namespace trellis {

// Cross-module runtime flags. Set from parsed params at the start of trellis_run;
// the modules that own them read them with an environment fallback so test
// binaries (which don't parse args) keep their historical TRELLIS_* behavior.
extern bool g_sparse_cast_f32;  // defined in sparse.cpp        (TRELLIS_F32)
extern bool g_no_fa;            // defined in dit.cpp           (TRELLIS_NOFA)
extern int  g_fa_fast;          // defined in dit.cpp: -1 auto/env, 0 BF16/F32, 1 F16/fast
extern bool g_require_gpu;      // defined in trellis_model.cpp (TRELLIS_REQUIRE_GPU)
extern int  g_cpu_threads;      // defined in trellis_model.cpp (TRELLIS_THREADS)
extern std::string g_backend;   // defined in trellis_model.cpp (TRELLIS_BACKEND)

// Which family of flow weights the GGUF directory holds. Both share the TRELLIS.2 DiT,
// sampler and decoders; they differ only in how the image conditions the flow — see pixal3d.h.
enum class ModelFamily {
    Trellis,   // TRELLIS.2: cross-attention over every DINOv3 token
    Pixal3D,   // Pixal3D: 5 global tokens + per-token view-aligned projection
};

const char* model_family_name(ModelFamily f);

// Every knob for one TRELLIS.2 image->3D run. Resolved as default -> environment
// (the historical TRELLIS_* / GSS / GSH names) -> CLI flag, with the CLI winning.
// trellis-cli and trellis-server share the parser: the server runs it once for its
// launch defaults, then per request to apply overrides (resolution, bg removal, ...).
struct TrellisParams {
    std::string image;                                          // input image (image->3D)
    std::string output = "model.glb";                           // output .glb
    std::string copyright;                                      // glTF asset.copyright metadata
    std::string models = "models";              // GGUF dir; override with --models DIR
    std::string host   = "127.0.0.1";                           // trellis-server only
    int      port = 8080;                                       // trellis-server only
    int      gpu  = 0;                                          // >=0 GPU index, <0 CPU
    std::string backend;        // force a ggml backend by name ("Vulkan", "HTP",
                                // "CPU", ...); empty = auto-select. --gpu N then indexes
                                // within that backend's devices.
    int threads = 0;            // CPU backend thread count; 0 = auto (all cores).
                                // ggml's own default is GGML_DEFAULT_N_THREADS == 4.
    uint32_t seed = 0;

    ModelFamily family = ModelFamily::Trellis;   // --model trellis|pixal3d
    // Pixal3D only. The projection needs the camera the image was "taken" with: upstream
    // estimates the horizontal FOV with MoGe-2 and derives the distance from it in closed form.
    // MoGe-2 is not ported, so the FOV is a flag; 0 keeps Pixal3D's own default (49.13 deg).
    float fov_deg    = 0.0f;
    float mesh_scale = 1.0f;
    // Pushes the camera's virtual image border outward, so the projection grid spans more than
    // the frame. The reference's own `extend_pixel`, which upstream never exposes. Rarely useful
    // here: background removal already reframes around the subject's alpha bbox, and grid cells
    // pushed past the cutout sample the border clamp — unconditioned, and the model fills them
    // arbitrarily.
    int extend_pixel = 0;
    // NAF guided upsampling of the DINOv3 feature map (the shape/texture stages' second proj
    // branch). Off falls back to sampling the bare feature map twice, which halves the effective
    // proj input — accepted only as a way to run without naf.gguf.
    bool naf = true;

    bool cascade    = true;     // 1024 cascade (default); --res 512 selects the light path
    int  hr_res     = 1024;     // HR cascade target resolution (1024 / 1536)
    int  max_tokens = 49152;    // HR token budget (backoff floors at 1024)

    int birefnet = -1;          // bg removal: 1 BiRefNet, 0 white-threshold, -1 auto
                                // (auto: keep a pre-matted image's alpha; else BiRefNet when
                                // birefnet.gguf is present; threshold as last resort. The
                                // threshold matte reads specular highlights [min(RGB)>=232]
                                // as background and the model then generates holes there.)
    bool texture  = true;       // texture flow + UV bake (else geometry-only)
    bool xatlas   = true;       // xatlas UV unwrap (else voxel-native box projection)
    int  band     = 0;          // narrow-band DC remesh band width (remesh_dc.h).
                                //   0 = auto: scale with resolution (res/512) so the
                                //   smoothing offset is resolution-independent — 1 @512,
                                //   2 @1024 — which suppresses the res-1024 "outer-skin"
                                //   speckle (issue #22). >0 forces that width (e.g. 1 for
                                //   the thin-wall reference look, 2 for a thicker shell).
    int  decim    = -1;         // decimation cluster grid   (-1 => per-cascade default)
    int  faces    = -1;         // QEM decimation face target (-1 => per-cascade default,
                                //   see qem_face_target()). Only the default QEM path
                                //   (decim < 0) uses it; --decim N and --decim 0 bypass QEM.
    int  tex      = -1;         // UV atlas size in px        (-1 => per-cascade default)
    int  tex_res  = -1;         // texture PBR resolution: -1 => auto (drop dense res-1024 tex to
                                //   512, whose clean coarse PBR bakes onto the res-1024 mesh
                                //   without the partial-coverage "skin" speckle); else force 512/1024
    int  webp     = -1;         // GLB texture encoding: -1 auto (WebP if built with it), 1 on, 0 off (PNG)
    int  normal_map = 0;        // bake a tangent-space normal map from the remeshed shell onto the
                                //   decimated mesh and add it to the GLB as a third texture
                                //   (--normal-map on|off, server "normal_map"; default off)
    bool f32      = false;      // f32 sparse-conv compute
    bool no_fa    = false;      // disable FlashAttention (manual softmax)
    int  fa_fast  = -1;         // FA precision: -1 auto (fast on HTP), 0 BF16/F32, 1 F16/fast
    bool require_gpu = false;   // refuse CPU fallback if no GPU is usable
    float gss = 7.5f;           // sparse-structure guidance strength
    float gsh = 7.5f;           // shape-SLAT guidance strength
    bool voxply = false;        // dump out/myvox.ply              (debug)
    bool dump_slat = false;     // dump /tmp/hr_slat.bin           (debug)
    bool dump_bg = false;       // also write the bg-removal cutout as <out>_cutout.png
    std::string dump_post;      // write the raw decoded mesh + sparse PBR volume to this
                                // path and exit (skips remesh/decimate/UV/bake/GLB) --
                                // machine-readable output for external post-processing
                                // pipelines (binary: i32 V,F,Mv,res; f32 verts[V*3];
                                // i32 faces[F*3]; i32 coords[Mv*3]; f32 pbr6[Mv*6];
                                // same layout as the TRELLIS_DUMP_POST debug env)
    bool bg_only = false;       // background removal only: write the cutout and skip the rest

    int  steps    = 0;          // flow sampler steps; 0 = model default (12). Lowering this
                                //   trades output quality for turnaround and is meant for
                                //   backend bring-up: a CPU and an NPU run at the same low
                                //   step count are still directly comparable to each other.
    int  sched = -1;            // multi-backend scheduler: -1 auto (on unless the primary
                                // backend is the CPU), 0 off, 1 on. Required for partial
                                // op-coverage accelerators like the Hexagon NPU.
    int  vulkan_fallback = -1;   // add Vulkan between HTP and CPU: -1 environment/default,
                                // 0 disabled, 1 enabled
    bool verbose = false;       // --verbose: per-stage timings, graph shapes, and a
                                // heartbeat while a backend compute is in flight

    bool retopo = false;
    int retopo_grid = 0;
    int retopo_first_faces = 0;
    int retopo_final_faces = 0;
    int retopo_atlas = 4096;
    bool retopo_no_weld_fill = false;
    bool retopo_dual_pbr = false;
    std::string retopo_workdir = ".codex/retopo/runs";

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
bool parse_camera_arg(const std::string& name, const char* value, TrellisParams& p, std::string& error);

// ---- QEM face target (--faces / server field "faces") ----
// The default postprocess quadric-simplifies the remeshed surface to a face budget before
// UV unwrap and bake. Lower budgets give lighter game/web assets and a faster xatlas pass;
// the texture still carries the surface detail the geometry drops.
constexpr int kQemFacesCascade = 300000;   // default @1024 / @1536 (reference parity)
constexpr int kQemFacesLight   = 150000;   // default @512
constexpr int kQemFacesMin     = 1000;     // below this QEM collapses thin parts outright

// The face budget this run decimates to: --faces when given, else the per-cascade default.
inline int qem_face_target(const TrellisParams& p) {
    return p.faces > 0 ? p.faces : (p.cascade ? kQemFacesCascade : kQemFacesLight);
}

// Parse a face target ("50000"; "auto" restores the default). Shared by the CLI's --faces
// and trellis-server's "faces" field so both accept and reject exactly the same values.
bool parse_faces_arg(const char* value, TrellisParams& p, std::string& error);

// --faces only steers the QEM path; reject it alongside --decim (cluster grid or 0 = keep
// full mesh) rather than silently ignoring it. Call after all params are applied.
bool check_faces_compat(const TrellisParams& p, std::string& error);

}  // namespace trellis
