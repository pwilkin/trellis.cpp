// trellis-cli: image -> 3D mesh (.glb), the TRELLIS.2 geometry pipeline in GGML.
//   trellis-cli <image.png> <out.glb> [gpu] [models_dir] [seed]
// Models are loaded/freed per stage to keep VRAM modest.
#include "trellis_model.h"
#include "trellis_debug.h"
#include "trellis_sched.h"
#include "preprocess.h"
#include "dinov3.h"
#include "pixal3d.h"
#include "flow_runner.h"
#include "ss_decoder.h"
#include "shape_decoder.h"
#include "dual_grid.h"
#include "mesh_glb.h"
#include "uv_bake.h"
#include "tri_bvh.h"
#include "remesh_dc.h"
#include "mesh_shape.h"
#include "stb_image_write.h"
#include "trellis_run.h"
#include "retopo_process.h"
#include "ggml.h"   // proj_in_channels is read straight off the checkpoint's proj_linear weight

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <vector>
#include <string>
#include <chrono>
#include <set>
#include <array>
#include <cmath>
#include <stdexcept>
#include <cctype>
#include <filesystem>
#include <system_error>
#ifdef __linux__
#include <sys/wait.h>
#include <spawn.h>
#include <unistd.h>
#endif

using std::vector;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static bool backend_is_htp(const std::string& backend) {
    if (backend.size() != 3) return false;
    return std::tolower((unsigned char)backend[0]) == 'h' &&
           std::tolower((unsigned char)backend[1]) == 't' &&
           std::tolower((unsigned char)backend[2]) == 'p';
}
// [dbg] overall stats of a flat tensor — used to compare LR vs HR shape-SLAT in decode space.
static void slat_stats(const char* tag, const vector<float>& v) {
    if (v.empty()) { printf("      [stats] %s EMPTY\n", tag); return; }
    double s = 0, s2 = 0; float mn = 0, mx = 0; bool first = true; size_t bad = 0;
    for (float x : v) {
        if (std::isnan(x) || std::isinf(x)) { bad++; continue; }
        s += x; s2 += (double)x * x;
        if (first) { mn = mx = x; first = false; } else { if (x < mn) mn = x; if (x > mx) mx = x; }
    }
    size_t n = v.size() - bad; double mean = n ? s / n : 0, var = n ? s2 / n - mean * mean : 0;
    printf("      [stats] %s n=%zu mean=%.4f std=%.4f min=%.3f max=%.3f nan/inf=%zu\n",
           tag, v.size(), mean, var > 0 ? std::sqrt(var) : 0.0, mn, mx, bad);
}

static const float SHAPE_MEAN[32]={0.781296f,0.018091f,-0.495192f,-0.558457f,1.060530f,0.093252f,1.518149f,-0.933218f,-0.732996f,2.604095f,-0.118341f,-2.143904f,0.495076f,-2.179512f,-2.130751f,-0.996944f,0.261421f,-2.217463f,1.260067f,-0.150213f,3.790713f,1.481266f,-1.046058f,-1.523667f,-0.059621f,2.220780f,1.621212f,0.877230f,0.567247f,-3.175944f,-3.186688f,1.578665f};
static const float SHAPE_STD[32]={5.972266f,4.706852f,5.445010f,5.209927f,5.320220f,4.547237f,5.020802f,5.444004f,5.226681f,5.683095f,4.831436f,5.286469f,5.652043f,5.367606f,5.525084f,4.730578f,4.805265f,5.124013f,5.530808f,5.619001f,5.103930f,5.417670f,5.269677f,5.547194f,5.634698f,5.235274f,6.110351f,5.511298f,6.237273f,4.879207f,5.347008f,5.405691f};
static const float TEX_MEAN[32]={3.501659f,2.212398f,2.226094f,0.251093f,-0.026248f,-0.687364f,0.439898f,-0.928075f,0.029398f,-0.339596f,-0.869527f,1.038479f,-0.972385f,0.126042f,-1.129303f,0.455149f,-1.209521f,2.069067f,0.544735f,2.569128f,-0.323407f,2.293000f,-1.925608f,-1.217717f,1.213905f,0.971588f,-0.023631f,0.106750f,2.021786f,0.250524f,-0.662387f,-0.768862f};
static const float TEX_STD[32]={2.665652f,2.743913f,2.765121f,2.595319f,3.037293f,2.291316f,2.144656f,2.911822f,2.969419f,2.501689f,2.154811f,3.163343f,2.621215f,2.381943f,3.186697f,3.021588f,2.295916f,3.234985f,3.233086f,2.260140f,2.874801f,2.810596f,3.292720f,2.674999f,2.680878f,2.372054f,2.451546f,2.353556f,2.995195f,2.379849f,2.786195f,2.775190f};

int trellis_run(const trellis::TrellisParams& cfg) {
    namespace fs = std::filesystem;
    fs::path retopo_driver, retopo_run;
    if (cfg.retopo) {
#if !defined(__linux__) && !defined(_WIN32)
        fprintf(stderr,"[trellis] native retopo launch is supported on Linux and Windows\n");
        return 1;
#else
        if (!cfg.texture || cfg.bg_only || std::getenv("TRELLIS_STOP_AFTER_POST_DUMP")) {
            fprintf(stderr,"[trellis] retopo needs textured generation through the POST stage\n");
            return 1;
        }
        if ((cfg.retopo_grid!=512 && cfg.retopo_grid!=1024 && cfg.retopo_grid!=1536) ||
            cfg.retopo_first_faces<0 || cfg.retopo_final_faces<=0 ||
            (cfg.retopo_atlas!=1024 && cfg.retopo_atlas!=2048 && cfg.retopo_atlas!=4096)) {
            fprintf(stderr,"[trellis] invalid retopo settings\n");
            return 1;
        }
        std::error_code ec;
#ifdef _WIN32
        retopo_driver=trellis::retopo_process::executable_directory()/"trellis-retopo-atlas.exe";
        const bool driver_available=fs::is_regular_file(retopo_driver,ec);
#else
        retopo_driver=fs::canonical("/proc/self/exe",ec).parent_path()/"trellis-retopo-atlas";
        const bool driver_available=!ec && access(retopo_driver.c_str(),X_OK)==0;
#endif
        if (ec || !driver_available) {
            fprintf(stderr,"[trellis] retopo driver unavailable: %s\n",retopo_driver.c_str());
            return 1;
        }
        const fs::path root=fs::absolute(cfg.retopo_workdir);
        fs::create_directories(root,ec);
        if (ec) {
            fprintf(stderr,"[trellis] cannot create retopo workdir: %s\n",ec.message().c_str());
            return 1;
        }
#ifdef _WIN32
        std::random_device random;
        bool created=false;
        for (int attempt=0;attempt<64 && !created;++attempt) {
            const uint64_t nonce=(uint64_t(random())<<32)|random();
            retopo_run=root/("run-"+std::to_string(nonce));
            created=fs::create_directory(retopo_run,ec);
            if (ec && ec!=std::errc::file_exists) break;
            ec.clear();
        }
        if (!created) {
            fprintf(stderr,"[trellis] cannot create retopo run directory: %s\n",ec.message().c_str());
            return 1;
        }
#else
        std::string pattern=(root/"run-XXXXXX").string();
        std::vector<char> mutable_pattern(pattern.begin(),pattern.end());
        mutable_pattern.push_back('\0');
        const char* created=mkdtemp(mutable_pattern.data());
        if (!created) {
            fprintf(stderr,"[trellis] cannot create retopo run directory\n");
            return 1;
        }
        retopo_run=created;
#endif
#endif
    }
    // Unbuffered, not line-buffered: MSVCRT treats _IOLBF as full buffering, which
    // swallows stage progress when piped (e.g. under Lemonade) if the process crashes.
    setvbuf(stdout, nullptr, _IONBF, 0);
    uint32_t run_seed = cfg.seed;
    if (run_seed == 0) {
        std::random_device rd;
        run_seed = (uint32_t(rd()) << 16) ^ uint32_t(rd()) ^
                   uint32_t(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        if (run_seed == 0) run_seed = 1;
        fprintf(stderr, "[trellis] generating model with seed %u (auto)\n", run_seed);
    } else {
        fprintf(stderr, "[trellis] generating model with seed %u\n", run_seed);
    }
    // Publish the cross-module flags this run wants (modules read them with an env fallback).
    const bool F32 = cfg.f32; trellis::g_sparse_cast_f32 = F32;  // f16 default (rope bug was the real issue)
    trellis::g_no_fa = cfg.no_fa;
    // CLI flags win over environment switches and the HTP-specific fast default.
    std::string requested_backend = cfg.backend;
    if (requested_backend.empty()) {
        if (const char* e = std::getenv("TRELLIS_BACKEND")) requested_backend = e;
    }
    const bool fa_f32_env = std::getenv("TRELLIS_FA_F32") != nullptr;
    const bool fa_fast_env = std::getenv("TRELLIS_FA_FAST") != nullptr;
    const bool fa_auto_htp = cfg.fa_fast < 0 && !fa_f32_env && !fa_fast_env &&
                             backend_is_htp(requested_backend);
    if (cfg.fa_fast >= 0) {
        trellis::g_fa_fast = cfg.fa_fast;
    } else if (fa_f32_env) {
        trellis::g_fa_fast = 0;
    } else if (fa_fast_env) {
        trellis::g_fa_fast = 1;
    } else {
        // An empty requested backend is genuinely auto-selected later by Model::load;
        // leave the policy unresolved so sdpa can inspect that model's actual backend.
        trellis::g_fa_fast = requested_backend.empty() ? -1 : (fa_auto_htp ? 1 : 0);
    }
    trellis::g_require_gpu = cfg.require_gpu;
    trellis::g_cpu_threads = cfg.threads;
    trellis::g_backend = cfg.backend;
    trellis::set_verbose(cfg.verbose);
    trellis::set_sched_mode(cfg.sched);
    trellis::set_vulkan_fallback_mode(cfg.vulkan_fallback);
    if (!cfg.no_fa) {
        if (trellis::g_fa_fast < 0) {
            fprintf(stderr, "[trellis] FlashAttention precision: auto (F16 fast on HTP; BF16/F32 otherwise)\n");
        } else {
            fprintf(stderr, "[trellis] FlashAttention precision: %s%s\n",
                    trellis::g_fa_fast ? "F16 fast" : "BF16 K/V + F32 accumulation",
                    fa_auto_htp ? " (HTP default)" : "");
        }
    }
    const std::string& img = cfg.image;
    const std::string& outglb = cfg.output;
    const std::string& M = cfg.models;
    const int gpu = cfg.gpu;
    const bool cascade = cfg.cascade;   // 1024 cascade is the TRELLIS default; --res 512 forces the light path

    // --model pixal3d. Everything below the conditioning — sampler schedules, guidance, the SS /
    // shape / texture decoders, the remesh and the bake — is shared with TRELLIS.2; only how the
    // image enters each DiT changes. See pixal3d.h.
    const bool pix = cfg.family == trellis::ModelFamily::Pixal3D;
    // Both families live in ONE model directory. The flows and NAF are family-specific and carry
    // a `pixal3d_` prefix; the decoders, DINOv3 and BiRefNet are byte-identical between the two
    // and keep their plain names, so adding Pixal3D to an existing TRELLIS.2 set is 5 new files
    // rather than a second copy of everything.
    const std::string FP = pix ? "/pixal3d_" : "/";
    trellis::CameraParams cam;
    if (pix) {
        constexpr float PIXAL3D_DEFAULT_FOV = 0.8575560450553894f;   // radians (~49.13 deg)
        const float fov = cfg.fov_deg > 0.0f ? cfg.fov_deg * 3.14159265358979f / 180.0f
                                             : PIXAL3D_DEFAULT_FOV;
        // The distance is derived at 512 on purpose: the projection is resolution-independent
        // once normalized, so one camera serves both the 512 and the 1024 stages.
        cam = trellis::pixal3d_camera(fov, cfg.mesh_scale, 512, cfg.extend_pixel);
        printf("[trellis] model family: pixal3d (fov %.2f deg, distance %.4f, mesh scale %.2f, extend %d px)\n",
               fov * 180.0f / 3.14159265358979f, cam.distance, cam.mesh_scale, cfg.extend_pixel);
        if (cfg.fov_deg <= 0.0f)
            printf("      (using Pixal3D's default FOV — MoGe-2 estimation is not ported; pass"
                   " --fov if the object's perspective is noticeably wider or flatter)\n");
    }

    std::mt19937 rng(run_seed); std::normal_distribution<float> randn(0.f, 1.f);
    auto noise = [&](size_t n){ vector<float> v(n); for (auto& x : v) x = randn(rng); return v; };
    double t0 = now();

    bool birefnet = cfg.birefnet == 1;
    if (cfg.birefnet < 0) {
        // Auto bg-removal. A pre-matted image keeps its own alpha (threshold path uses it
        // as-is). Otherwise prefer the neural matte: the white-threshold rule reads specular
        // highlights (min(RGB)>=232) as background, and conditioning the flow on an object
        // with punched-out highlights makes it generate HOLES right there (issue #1
        // follow-ups: helmet-crest and axe-edge highlights on assets/goblin.png).
        if (trellis::image_has_alpha(img)) {
            birefnet = false;
        } else {
            FILE* bf = fopen((M + "/birefnet.gguf").c_str(), "rb");
            if (bf) { fclose(bf); birefnet = true; }
            else printf("      (birefnet.gguf not found -- falling back to threshold matte;"
                        " bright highlights may punch holes)\n");
        }
    }
    vector<float> chw, chw1024;
    std::vector<unsigned char> cutout; int cut_sz = 0;   // the bg-removal result (for --dump-bg / --bg-only)
    if (birefnet) {
        printf("[1/6] preprocess %s (BiRefNet bg removal, %s)\n", img.c_str(), cascade ? "1024 cascade" : "512");
        // Full BiRefNet (Swin-L backbone + deformable-conv decoder) runs on the GPU. Cutout computed
        // once, normalized for 512 and 1024.
        // BiRefNet's small convolutions are slower and currently nondeterministic on HTP.
        // Keep other explicitly selected backends on their existing path.
        const std::string bg_saved_backend = trellis::g_backend;
        const bool bg_on_cpu = backend_is_htp(requested_backend);
        if (bg_on_cpu) trellis::g_backend = "CPU";
        trellis::Model bm = trellis::Model::load(M + "/birefnet.gguf",
                                                 bg_on_cpu ? -1 : gpu);
        trellis::g_backend = bg_saved_backend;
        cutout = trellis::birefnet_cutout(img, bm, gpu < 0 ? 0 : gpu, cut_sz);
        bm.free();
        if (cutout.empty()) return 1;
        chw = trellis::normalize_cutout(cutout, cut_sz, 512);
        if (cascade) chw1024 = trellis::normalize_cutout(cutout, cut_sz, 1024);
    } else {
        printf("[1/6] preprocess %s (%s)\n", img.c_str(), cascade ? "1024 cascade" : "512");
        cutout = trellis::threshold_cutout(img, cut_sz);
        if (cutout.empty()) return 1;
        chw = trellis::normalize_cutout(cutout, cut_sz, 512);
        if (cascade) chw1024 = trellis::normalize_cutout(cutout, cut_sz, 1024);
    }

    // --dump-bg: write the bg-removal cutout next to the output; --bg-only: stop here.
    if (cfg.dump_bg || cfg.bg_only) {
        const std::string cut_png = outglb.substr(0, outglb.find_last_of('.')) + "_cutout.png";
        const int cut_channels = cut_sz > 0 && cutout.size() == (size_t)cut_sz * cut_sz * 4 ? 4 : 3;
        if (cut_sz > 0 && stbi_write_png(cut_png.c_str(), cut_sz, cut_sz, cut_channels,
                                        cutout.data(), cut_sz*cut_channels))
            printf("      bg-removal cutout -> %s\n", cut_png.c_str());
        else
            fprintf(stderr, "      [warn] could not write bg-removal cutout to %s\n", cut_png.c_str());
        if (cfg.bg_only) { printf("[bg-only] done (%.1fs)\n", now() - t0); return 0; }
    }

    // Raw [0,1] guides for NAF. The DINOv3 branch wants the ImageNet-normalized tensor; NAF's
    // image encoder wants the unnormalized one, so both are kept.
    vector<float> guide, guide1024;
    if (pix && cfg.naf) {
        guide = trellis::cutout_to_chw01(cutout, cut_sz, 512);
        if (cascade) guide1024 = trellis::cutout_to_chw01(cutout, cut_sz, 1024);
    }

    printf("[2/6] DINOv3 conditioning\n");
    vector<float> dino, dino1024;       // full token stream: 5 global + (S/16)^2 patches
    { trellis::Model m = trellis::Model::load(M + "/dinov3.gguf", gpu);
      dino = trellis::dinov3_encode(m, chw, 512);
      if (cascade) dino1024 = trellis::dinov3_encode(m, chw1024, 1024);
      m.free(); }
    // TRELLIS.2 cross-attends over every token. Pixal3D cross-attends over the 5 global tokens
    // (cls + registers) only and routes the patch grid through the projection branch instead, so
    // the cross-attention context is just a prefix of the same tensor.
    constexpr size_t N_GLOBAL_FLOAT = 5 * 1024;
    vector<float> cond    = pix ? vector<float>(dino.begin(), dino.begin() + N_GLOBAL_FLOAT) : dino;
    vector<float> cond1024;
    if (cascade)
        cond1024 = pix ? vector<float>(dino1024.begin(), dino1024.begin() + N_GLOBAL_FLOAT) : dino1024;
    const int Lc = (int)(cond.size() / 1024);
    vector<float> neg(cond.size(), 0.0f);
    const int Lc1024 = cascade ? (int)(cond1024.size() / 1024) : 0;
    vector<float> neg1024(cond1024.size(), 0.0f);
    printf("      cond tokens=%d%s\n", Lc, cascade ? (" / 1024-cond tokens=" + std::to_string(Lc1024)).c_str() : "");
    slat_stats("cond_512 (DINOv3@512)", cond);
    if (cascade) slat_stats("cond_1024 (DINOv3@1024)", cond1024);

    // Per-stage projection conditioning. `proj_ch` is read off the stage's own proj_linear rather
    // than guessed from a config, which also doubles as the check that a --model pixal3d run is
    // pointed at Pixal3D weights (and a --model trellis run is not).
    auto proj_ch_of = [&](const trellis::Model& m) -> int {
        ggml_tensor* w = m.try_get("blocks.0.cross_attn.proj_linear.weight");
        if (pix && !w)
            throw std::runtime_error("--model pixal3d but the checkpoint has no cross_attn.proj_linear "
                                     "(these are TRELLIS.2 weights)");
        if (!pix && w)
            throw std::runtime_error("--model trellis but the checkpoint has cross_attn.proj_linear "
                                     "(these are Pixal3D weights; pass --model pixal3d)");
        return w ? (int)w->ne[0] : 0;
    };
    // grid_res: the stage's projection grid. S / naf_out: the DINOv3 image size the stage was
    // trained on and its NAF target, both taken from the Pixal3D stage configs.
    auto build_proj = [&](int grid_res, int S, int naf_out, int proj_ch,
                          const vector<std::array<int,3>>* cds) {
        const vector<float>& dn = (S == 1024) ? dino1024 : dino;
        const vector<float>& gd = (S == 1024) ? guide1024 : guide;
        const bool want_naf = (proj_ch == 2048) && cfg.naf;
        trellis::ProjCond pc;
        if (want_naf) {
            trellis::Model nm = trellis::Model::load(M + FP + "naf.gguf", gpu);
            pc = trellis::pixal3d_proj_cond(dn, S, grid_res, proj_ch, cam, cds, &nm, &gd, naf_out);
            nm.free();
        } else {
            pc = trellis::pixal3d_proj_cond(dn, S, grid_res, proj_ch, cam, cds, nullptr, nullptr, naf_out);
        }
        printf("      proj cond: grid %d^3, image %d, %d ch%s\n", grid_res, S, proj_ch,
               proj_ch == 2048 ? (want_naf ? ", NAF upsampled" : ", NAF DISABLED") : "");
        return pc;
    };

    printf("[3/6] sparse-structure flow + decode\n");
    vector<std::array<int,3>> coords;
    {
        trellis::Model m = trellis::Model::load(M + FP + "ss_flow.gguf", gpu);
        trellis::DiTParams p; p.in_ch = 8; p.out_ch = 8; p.d_cond = 1024; p.cast_f32 = F32;
        p.proj_ch = proj_ch_of(m); p.proj_mode = p.proj_ch > 0;
        // The sparse-structure DiT is dense at 16^3, and Pixal3D's SS stage projects a 16^3 grid,
        // so its proj tokens line up one-to-one with the DiT tokens in the same x-major order.
        trellis::ProjCond pc;
        if (pix) pc = build_proj(16, 512, 0, p.proj_ch, nullptr);
        trellis::DitRunner* run = trellis::make_dense_runner(m, p, 16, Lc);
        trellis::FlowFwd fwd = [&](const vector<float>& x, float ts, const trellis::FlowCond& c){ return run->forward(x, ts, c); };
        trellis::SamplerParams sp; sp.steps=(cfg.steps>0?cfg.steps:12); sp.guidance_strength=cfg.gss; sp.guidance_rescale=0.7f; sp.gi0=0.6f; sp.gi1=1.0f; sp.rescale_t=5.0f;
        vector<float> z = trellis::sample_flow(fwd, noise(8*4096),
                              trellis::FlowCond(cond.data(), pix ? pc.proj.data()     : nullptr),
                              trellis::FlowCond(neg.data(),  nullptr), sp);  // [8,4096] ne0=8
        delete run; m.free();
        // transpose [8,L] -> torch [8,16,16,16] memory (c*4096 + sp)
        vector<float> zdec(8*4096);
        for (int c = 0; c < 8; ++c) for (int sp2 = 0; sp2 < 4096; ++sp2) zdec[(size_t)c*4096 + sp2] = z[c + 8*sp2];
        // This small decoder currently miscomputes a CONT/view-to-ADD chain on HTP.
        // Its CPU cost is negligible; other backends retain their existing placement.
        const std::string saved_backend = trellis::g_backend;
        const bool ss_dec_on_cpu = backend_is_htp(requested_backend);
        if (ss_dec_on_cpu) trellis::g_backend = "CPU";
        trellis::Model d = trellis::Model::load(M + "/ss_dec.gguf", ss_dec_on_cpu ? -1 : gpu);
        trellis::g_backend = saved_backend;
        vector<float> logits = trellis::ss_decode(d, zdec); d.free();
        coords = trellis::ss_coords(logits, 64, 32);
    }
    if (cfg.voxply) { FILE*f=fopen("out/myvox.ply","wb"); fprintf(f,"ply\nformat binary_little_endian 1.0\nelement vertex %zu\nproperty float x\nproperty float y\nproperty float z\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n",coords.size()); for(auto&c:coords){float p[3]={(c[0]+0.5f)/32-0.5f,(c[1]+0.5f)/32-0.5f,(c[2]+0.5f)/32-0.5f}; fwrite(p,4,3,f);} fclose(f); }
    printf("      active voxels @res32 = %d\n", (int)coords.size());
    if (coords.empty()) { fprintf(stderr, "no voxels produced\n"); return 1; }

    bool do_tex = cfg.texture;
    // Pixal3D publishes a single (1024) texture flow, where TRELLIS.2 publishes both. The light
    // --res 512 path has no texture model to run at all, so it degrades to geometry only.
    if (do_tex && pix && !cascade) {
        FILE* tf = fopen((M + FP + "tex_flow_512.gguf").c_str(), "rb");
        if (tf) fclose(tf);
        else { printf("      (pixal3d ships no res-512 texture flow -- writing geometry only)\n"); do_tex = false; }
    }

    // one shape SLAT flow run -> normalized [32,n] (sparse, CFG 7.5, gi[0.6,1], rescale_t 3)
    // `grid_res` is the resolution the sparse coords are expressed in — the same grid Pixal3D
    // projects for this stage — and S / naf_out are the stage's DINOv3 image size and NAF target.
    // They are ignored in TRELLIS.2 mode.
    auto shape_flow = [&](const std::string& path, const vector<std::array<int,3>>& cds,
                          const float* cnd, const float* ncnd, int lc,
                          int grid_res, int S, int naf_out) {
        const int n = (int)cds.size();
        trellis::Model m = trellis::Model::load(path, gpu);
        trellis::DiTParams p; p.in_ch = 32; p.out_ch = 32; p.d_cond = 1024; p.cast_f32 = F32;
        p.proj_ch = proj_ch_of(m); p.proj_mode = p.proj_ch > 0;
        trellis::ProjCond pc;
        if (pix) pc = build_proj(grid_res, S, naf_out, p.proj_ch, &cds);
        trellis::DitRunner* run = trellis::make_sparse_runner(m, p, cds, lc);
        trellis::FlowFwd fwd = [&](const vector<float>& x, float ts, const trellis::FlowCond& c){ return run->forward(x, ts, c); };
        trellis::SamplerParams sp; sp.steps=(cfg.steps>0?cfg.steps:12); sp.guidance_strength=cfg.gsh; sp.guidance_rescale=0.5f; sp.gi0=0.6f; sp.gi1=1.0f; sp.rescale_t=3.0f;
        vector<float> sn = trellis::sample_flow(fwd, noise((size_t)32*n),
                               trellis::FlowCond(cnd,  pix ? pc.proj.data()     : nullptr),
                               trellis::FlowCond(ncnd, nullptr), sp);   // [32,n]
        delete run; m.free();
        return sn;
    };

    vector<float> slat_norm, slat_dn;        // normalized (for tex concat) and denormalized (for decode)
    vector<std::array<int,3>> shc;           // coords where the shape SLAT lives + is decoded
    int RES = 512;                           // final grid resolution
    const float* cond_dec = cond.data();     // cond used by HR shape + tex flows
    const float* neg_dec  = neg.data();
    int Lc_dec = Lc;
    vector<float> lr_norm, lr_dn;            // LR (res-512) shape slat @res32 — reused for the res-512 tex path
    if (cascade) {
        // Cascade target resolution: 1024 (default, '1024_cascade') or e.g. 1536 ('1536_cascade').
        // Both reuse the SAME shape_flow_1024 / tex_flow_1024 / cond_1024 — only the HR quantization
        // grid (res//16) and the final decode resolution change. The reference floors the backoff at
        // 1024 and caps the HR token count at max_num_tokens (49152) to stay within VRAM; the backoff
        // steps the grid down by -128 while the unique res//16 token count would exceed it.
        // Verified end-to-end: a clean reconstruction reaches grid 96 (res1536, ~15k tokens) and
        // decodes a coherent ~5.4M-vert mesh — the c26fc76 FA holds at the cascade's token counts.
        // (Caveat: a *bad TRELLIS seed* can yield a degenerate ~4x-bloated SLAT — ~44k tokens, garbage
        //  geometry, and it can NaN the HR FA at grid 80. That's a reconstruction-quality problem, not
        //  an FA limit: re-roll the seed rather than lowering this cap.)
        const int hr_target = cfg.hr_res;
        const int max_tok   = cfg.max_tokens;
        printf("[4/7] shape SLAT flow (LR 512 -> upsample -> HR %d cascade, max_tok=%d)\n", hr_target, max_tok);
        // (1) LR shape flow @res32 with cond_512
        lr_norm = shape_flow(M + FP + "shape_flow_512.gguf", coords, cond.data(), neg.data(), Lc,
                             /*grid_res=*/32, /*S=*/512, /*naf_out=*/512);
        lr_dn.resize(lr_norm.size());
        for (size_t n = 0; n < coords.size(); ++n) for (int c = 0; c < 32; ++c)
            lr_dn[(size_t)c + 32*n] = lr_norm[(size_t)c + 32*n]*SHAPE_STD[c] + SHAPE_MEAN[c];
        slat_stats("LR slat (res32, decodes OK via upsample)", lr_dn);
        // (2) decoder.upsample(LR slat, 4) -> res512 coords
        vector<std::array<int,3>> hr_coords;
        { trellis::Model m = trellis::Model::load(M + "/shape_dec.gguf", gpu);
          hr_coords = trellis::shape_upsample(m, lr_dn, coords); m.free(); }
        // (3) quantize res512 -> res(hr_res//16) with the reference's adaptive token-budget backoff:
        //     start at hr_target, step -128 toward the 1024 floor while the unique token count
        //     would exceed max_num_tokens. grid = hr_res//16 is integral since 128/16 = 8
        //     (1536->96, 1408->88, ..., 1024->64).
        //
        //     The two families quantize DIFFERENTLY, and it matters far more than it looks.
        //     TRELLIS.2 floors `u * grid` — a cell quantizer, and the coords only feed RoPE, where
        //     a half-cell offset is a smooth reparameterization. Pixal3D rounds `u * (grid - 1)`,
        //     because for it the token index ALSO selects which projection-grid node the token
        //     samples the image at, and that grid is the endpoint-inclusive linspace(-1, 1, grid):
        //     rounding to the nearest node is what keeps the pixel-aligned sample registered.
        //     (Pixal3D's own sample_shape_slat_cascade still carries the TRELLIS.2 form, but run()
        //     never calls it — that dead helper is what this port originally followed.)
        //
        //     Using the wrong one is not a smooth warp but a step function: at grid 64 it moves
        //     25% of coordinates a full cell along each axis, in stripes of period 8, so 58% of
        //     tokens are misregistered on at least one axis. A cell is 16 res-1024 voxels — one
        //     ViT-L/16 patch at S=1024. The silhouette survives on the global tokens; the fine
        //     detail does not, and the decode comes out speckled.
        int hr_res = hr_target;
        for (;;) {
            const int gi = hr_res / 16;          // integral grid (ref's hr_resolution//16)
            const float g = pix ? (float)(gi - 1) : (float)gi;
            auto qz = [&](float c) { return pix ? (int)lrintf((c + 0.5f) / 512.f * g)
                                               : (int)((c + 0.5f) / 512.f * g); };
            std::set<std::array<int,3>> q;
            for (auto& c : hr_coords) q.insert({ qz((float)c[0]), qz((float)c[1]), qz((float)c[2]) });
            if ((int)q.size() < max_tok || hr_res <= 1024) {
                shc.assign(q.begin(), q.end());
                printf("      upsampled coords @res512=%d -> quantized @res%d (grid %d) = %d tokens\n",
                       (int)hr_coords.size(), hr_res, gi, (int)shc.size());
                break;
            }
            printf("      res%d (grid %d) -> %d tokens >= %d, backing off -128\n",
                   hr_res, gi, (int)q.size(), max_tok);
            hr_res -= 128;
        }
        // (4) HR shape flow @res(hr_res//16) with cond_1024. The projection grid follows the same
        //     backoff as the token grid — Pixal3D overrides its cond model's grid_resolution to
        //     hr_res//16 for exactly this reason — while the NAF target stays at the stage's 512.
        // The HR stage is the only one conditioned on DINOv3 at 1024. If its feature map were read
        // with a different spatial layout than the 512 one — which the LR stage uses successfully —
        // nothing else in the pipeline would show it: the features would still be well-scaled,
        // well-correlated across the proj halves, and finite.
        if (pix)
            printf("      [stats] proj: DINOv3@1024 vs @512 agreement on the same points = %.3f\n",
                   trellis::pixal3d_cross_res_agreement(dino1024, 1024, dino, 512,
                                                        hr_res / 16, cam, shc));
        slat_norm = shape_flow(M + FP + "shape_flow_1024.gguf", shc, cond1024.data(), neg1024.data(), Lc1024,
                               /*grid_res=*/hr_res / 16, /*S=*/1024, /*naf_out=*/512);
        RES = hr_res; cond_dec = cond1024.data(); neg_dec = neg1024.data(); Lc_dec = Lc1024;
    } else {
        printf("[4/7] shape SLAT flow (512)\n");
        shc = coords;
        slat_norm = shape_flow(M + FP + "shape_flow_512.gguf", coords, cond.data(), neg.data(), Lc,
                               /*grid_res=*/32, /*S=*/512, /*naf_out=*/512);
    }
    const int N = (int)shc.size();
    slat_dn.resize(slat_norm.size());
    for (int n = 0; n < N; ++n) for (int c = 0; c < 32; ++c)
        slat_dn[(size_t)c + 32*n] = slat_norm[(size_t)c + 32*n]*SHAPE_STD[c] + SHAPE_MEAN[c];
    slat_stats(cascade ? "HR slat" : "slat (res32)", slat_dn);
    if (cascade && cfg.dump_slat) {   // dump HR slat for the reference-decoder diff
        FILE* f = fopen("/tmp/hr_slat.bin", "wb");
        if (f) {
            int n = N, res = RES; fwrite(&n, 4, 1, f); fwrite(&res, 4, 1, f);
            for (auto& c : shc) { int xyz[3] = { c[0], c[1], c[2] }; fwrite(xyz, 4, 3, f); }
            fwrite(slat_dn.data(), 4, slat_dn.size(), f); fclose(f);
            printf("      [dump] /tmp/hr_slat.bin: N=%d res=%d feats=%zu\n", n, res, slat_dn.size());
        }
    }

    printf("[5/7] FlexiDualGrid shape decode -> mesh @res%d\n", RES);
    trellis::Mesh mesh;
    trellis::ShapeOut so;
    {
        trellis::Model m = trellis::Model::load(M + "/shape_dec.gguf", gpu);
        so = trellis::shape_decode(m, slat_dn, shc, RES); m.free();
        printf("      decoded voxels @res%d = %d\n", so.res, (int)so.coords.size());
        mesh = trellis::dual_grid_to_mesh(so);
        // feats7 is consumed by dual_grid_to_mesh and is never referenced again.
        // Release it before the enormous host-side topology passes; on the
        // 31M-voxel torture case this returns ~830 MiB immediately.
        const size_t shape_feat_bytes = so.feats7.capacity() * sizeof(float);
        std::vector<float>().swap(so.feats7);
        if (shape_feat_bytes >= (size_t)128 * 1024 * 1024) {
            printf("      [host-mem] released shape feats7 %.2f GiB before mesh postprocess\n",
                   shape_feat_bytes / (1024.0*1024.0*1024.0));
            fflush(stdout);
        }
    }
    printf("      mesh V=%d F=%d\n", mesh.V(), mesh.F());
    {   // reference postprocess fills small holes BEFORE the remesh (max_hole_perimeter=3e-2):
        // open cracks make the narrow-band in/out test ambiguous and leak holes into the
        // remeshed surface (ours had ~30x the reference's boundary edges without this).
        const int nh = trellis::fill_holes(mesh.verts, mesh.faces, 3e-2f);
        if (nh) printf("      filled %d small holes -> V=%d F=%d\n", nh, mesh.V(), mesh.F());
    }
    if (const char* dp = std::getenv("TRELLIS_DUMP_DECMESH")) {   // pre-remesh decoded mesh (int V,F,verts,faces) for remesh A/B
        FILE* f = fopen(dp, "wb"); if (f) { int v=mesh.V(), fc=mesh.F();
            fwrite(&v,4,1,f); fwrite(&fc,4,1,f); fwrite(mesh.verts.data(),4,mesh.verts.size(),f); fwrite(mesh.faces.data(),4,mesh.faces.size(),f); fclose(f);
            printf("      [dump] pre-remesh mesh -> %s\n", dp); fflush(stdout); }
    }

    vector<float> colors, pbr6, dual_pbr6;   // colors = base RGB (PLY); pbr6 = per-vertex [V*6] for UV bake
    const std::string retopo_dual_post=(retopo_run/"dual-high.post").string();
    const char* dual_pbr_path=cfg.retopo_dual_pbr ? retopo_dual_post.c_str() : std::getenv("TRELLIS_DUMP_DUAL_PBR_POST");
    trellis::ShapeOut so_tex;                                    // res-512 tex-guide decode (mixed-res path)
    const vector<std::array<int,3>>* pbr_coords = &so.coords;    // coords/res the bake samples the PBR at
    int pbr_res = so.res;
    if (do_tex) {
        // Texture PBR resolution. A dense res-1024 decode paints its outermost voxel layer with a
        // partial-coverage "skin" (dark/incoherent), and the bake, snapping the decimated mesh through
        // that layer, samples it inconsistently -> salt-and-pepper / colour-patch speckle. (Verified on
        // the reference decoder: it produces the same skin at res-1024.) The res-512 decode never
        // resolves that layer, so its clean, coherent PBR baked onto the res-1024 mesh keeps the
        // geometry detail without the speckle. Auto: drop to 512 above ~9M voxels; --tex-res forces it.
        constexpr int DENSE_TEX = 9000000;
        const int tex_res = cfg.tex_res > 0 ? cfg.tex_res
                          : (cascade && (int)so.coords.size() > DENSE_TEX ? 512 : RES);
        // Pixal3D does not publish a res-512 texture flow; its texture stays at cascade resolution.
        const bool mixed = cascade && tex_res != RES && !pix;
        if (dual_pbr_path && (!*dual_pbr_path || !mixed || RES != 1024)) {
            fprintf(stderr, "[trellis] dual PBR requires TRELLIS.2 with 1024 geometry and 512 primary PBR\n");
            return 1;
        }
        printf("[6/7] texture SLAT flow + PBR decode%s\n", mixed ? "  (res-512 texture on res-1024 mesh)" : "");

        if (mixed) {   // decode a res-512 shape (from the LR slat) to guide the res-512 tex decode
            trellis::Model m = trellis::Model::load(M + "/shape_dec.gguf", gpu);
            so_tex = trellis::shape_decode(m, lr_dn, coords, 512); m.free();
            pbr_coords = &so_tex.coords; pbr_res = so_tex.res;
            // The texture decoder needs only coords+subdivision masks from this
            // guide; its 7-channel shape features are dead weight from here on.
            std::vector<float>().swap(so_tex.feats7);
            // Mixed mode now samples PBR from so_tex.coords, so the giant HR
            // coordinate table can also be returned to the OS before tex flow.
            if (!dual_pbr_path) std::vector<std::array<int,3>>().swap(so.coords);
            printf("      res-512 tex-guide decode: %d voxels\n", (int)so_tex.coords.size());
            printf("      [host-mem] mixed texture: %s + tex-guide feats7\n",
                   dual_pbr_path ? "retained HR shape coords for dual PBR" : "released HR shape coords");
            fflush(stdout);
        }
        auto decode_texture = [&](bool use_mixed) {
            const std::string tflow = M + FP + (use_mixed ? "tex_flow_512.gguf" : (cascade ? "tex_flow_1024.gguf" : "tex_flow_512.gguf"));
            const vector<std::array<int,3>>& tcoords = use_mixed ? coords : shc;
            const vector<float>& tslat = use_mixed ? lr_norm : slat_norm;
            const float* tcond = use_mixed ? cond.data() : cond_dec;
            const float* tneg = use_mixed ? neg.data() : neg_dec;
            const int tlc = use_mixed ? Lc : Lc_dec;
            const int tN = (int)tcoords.size();
            const auto& tsubs = use_mixed ? so_tex.subs : so.subs;
            const int Mv = (int)(use_mixed ? so_tex.coords.size() : so.coords.size());
            const bool tex_lr = use_mixed || !cascade;
            const int tgrid = tex_lr ? 32 : RES / 16;
            const int tS = tex_lr ? 512 : 1024;
            const int tnaf = tex_lr ? 256 : 1024;
            vector<float> texlat;

            trellis::Model m = trellis::Model::load(tflow, gpu);
            trellis::DiTParams p; p.in_ch = 64; p.out_ch = 32; p.d_cond = 1024; p.cast_f32 = F32;
            p.proj_ch = proj_ch_of(m); p.proj_mode = p.proj_ch > 0;
            trellis::ProjCond pc;
            if (pix) pc = build_proj(tgrid, tS, tnaf, p.proj_ch, &tcoords);
            trellis::DitRunner* run = trellis::make_sparse_runner(m, p, tcoords, tlc);
            trellis::FlowFwd fwd = [&](const vector<float>& st, float ts, const trellis::FlowCond& c) {
                vector<float> x64((size_t)64 * tN);
                for (int n = 0; n < tN; ++n) {
                    for (int k = 0; k < 32; ++k) x64[(size_t)k + 64*n]      = st[(size_t)k + 32*n];
                    for (int k = 0; k < 32; ++k) x64[(size_t)32 + k + 64*n] = tslat[(size_t)k + 32*n];
                }
                return run->forward(x64, ts, c);
            };
            trellis::SamplerParams sp; sp.steps=(cfg.steps>0?cfg.steps:12); sp.guidance_strength=1.0f; sp.guidance_rescale=0.0f; sp.gi0=0.6f; sp.gi1=0.9f; sp.rescale_t=3.0f;
            texlat = trellis::sample_flow(fwd, noise((size_t)32*tN),
                         trellis::FlowCond(tcond, pix ? pc.proj.data()     : nullptr),
                         trellis::FlowCond(tneg,  nullptr), sp);  // [32,tN]
            delete run; m.free();
            for (int n = 0; n < tN; ++n) for (int c = 0; c < 32; ++c) texlat[(size_t)c + 32*n] = texlat[(size_t)c + 32*n]*TEX_STD[c] + TEX_MEAN[c];
            trellis::Model decoder = trellis::Model::load(M + "/tex_dec.gguf", gpu);
            vector<float> pbr = trellis::tex_decode(decoder, texlat, tcoords, tsubs); decoder.free();
            vector<float> decoded((size_t)Mv * 6);
            auto cl = [](float v){ return v < 0 ? 0.f : (v > 1 ? 1.f : v); };
            for (int i = 0; i < Mv; ++i)
                for (int k = 0; k < 6; ++k)
                    decoded[(size_t)i*6 + k] = cl(pbr[(size_t)k + 6*i] * 0.5f + 0.5f);
            return decoded;
        };
        const auto texture_rng = rng;
        pbr6 = decode_texture(mixed);
        const int Mv = (int)pbr_coords->size();
        colors.resize((size_t)Mv * 3);
        for (int i = 0; i < Mv; ++i)
            for (int k = 0; k < 3; ++k)
                colors[(size_t)i*3 + k] = pbr6[(size_t)i*6 + k];
        printf("      PBR voxels=%d @res%d\n", Mv, pbr_res);
        if (dual_pbr_path) {
            rng = texture_rng;
            dual_pbr6 = decode_texture(false);
            printf("      secondary PBR voxels=%zu @res%d\n", dual_pbr6.size()/6, so.res);
        }
        // Texture decode has consumed the subdivision guides.  Release their
        // backing storage before the raw 100M+ face mesh enters weld/hole/BVH.
        std::vector<std::vector<uint8_t>>().swap(so.subs);
        std::vector<std::vector<uint8_t>>().swap(so_tex.subs);
        // `colors` is per-VOXEL but consumed per-VERTEX (weld, vertex-color GLB, PLY),
        // relying on dual_grid_to_mesh's vertex==voxel correspondence. fill_holes adds
        // cap vertices beyond Mv -- pad them (neutral grey; the caps are sub-voxel and
        // the real texture comes from the voxel-volume bake, not these colors). In
        // mixed-res mode the correspondence never held at all (res-512 voxels vs
        // res-1024 vertices): drop the colors instead of reading out of bounds.
        // (Windows surfaced this as a silent crash at [7/7]; Linux read garbage.)
        if (colors.size() != (size_t)mesh.V() * 3) {
            if (pbr_res == so.res && colors.size() < (size_t)mesh.V() * 3)
                colors.resize((size_t)mesh.V() * 3, 0.5f);
            else
                colors.clear();
        }
    }

    if (!cfg.dump_post.empty()) {
        // Machine-readable output for external post-processing pipelines
        // (simplification budgets / UV / baking done by the caller, e.g.
        // QtMeshEditor): the CLEANED geometry mesh + the sparse PBR volume it
        // would be baked from, then exit before decimate/UV/bake/GLB.
        //
        // The reference cleanup chain runs first (same as the GLB path):
        // weld -> fill_small_holes -> narrow-band DC remesh -> clean ->
        // drop floaters. The raw dual-grid decode is non-manifold at
        // sub-voxel scale with inconsistently-wound patches — downstream
        // consumers hit unwrap/simplify/culling pathologies on it — while
        // the remeshed shell is watertight with consistent winding. Falls
        // back to the raw decoded mesh if the remesh produces nothing.
        //
        // Binary layout matches the TRELLIS_DUMP_POST debug env:
        // i32 V,F,Mv,res; f32 verts[V*3]; i32 faces[F*3]; i32 coords[Mv*3];
        // f32 pbr6[Mv*6] (base_color.rgb, metallic, roughness, alpha in [0,1]).
        trellis::weld_vertices(mesh.verts, mesh.faces, nullptr,
                               1.0f / ((float)so.res * 8.0f));
        trellis::fill_small_holes(mesh.faces);
        trellis::TriBvh dbvh = trellis::TriBvh::build(mesh.verts.data(), mesh.V(),
                                                      mesh.faces.data(), mesh.F());
        const int dband = cfg.band > 0 ? cfg.band : std::max(1, so.res / 512);
        trellis::Mesh rm = trellis::remesh_narrow_band_dc(mesh.verts.data(), mesh.V(),
                                                          mesh.faces.data(), mesh.F(),
                                                          dbvh, so.res, dband);
        if (rm.F() > 0) {
            trellis::clean_mesh(rm.V(), rm.faces);
            trellis::drop_small_components(rm.verts, rm.faces, 0.02f);
        }
        const std::vector<float>&   dverts = rm.F() > 0 ? rm.verts : mesh.verts;
        const std::vector<int32_t>& dfaces = rm.F() > 0 ? rm.faces : mesh.faces;
        FILE* dfp = fopen(cfg.dump_post.c_str(), "wb");
        if (!dfp) { fprintf(stderr, "cannot write %s\n", cfg.dump_post.c_str()); return 1; }
        const int dV = (int)dverts.size() / 3, dFc = (int)dfaces.size() / 3;
        const int Mv = pbr6.empty() ? 0 : (int)pbr_coords->size(), res = pbr_res;
        fwrite(&dV,4,1,dfp); fwrite(&dFc,4,1,dfp); fwrite(&Mv,4,1,dfp); fwrite(&res,4,1,dfp);
        fwrite(dverts.data(),4,(size_t)dV*3,dfp);
        fwrite(dfaces.data(),4,(size_t)dFc*3,dfp);
        if (Mv) {
            for (auto& c : *pbr_coords) { int xyz[3] = {c[0],c[1],c[2]}; fwrite(xyz,4,3,dfp); }
            fwrite(pbr6.data(),4,(size_t)Mv*6,dfp);
        }
        fclose(dfp);
        printf("[7/7] cleaned dump -> %s (V=%d F=%d%s, PBR=%d @res%d)\n",
               cfg.dump_post.c_str(), dV, dFc,
               rm.F() > 0 ? " remeshed" : " raw (remesh empty)", Mv, res);
        printf("done in %.1fs -> %s\n", now() - t0, cfg.dump_post.c_str());
        return 0;
    }

    printf("[7/7] write %s\n", outglb.c_str());
    bool textured = false;
    if (!pbr6.empty()) {   // UV-baked textured GLB (PBR material)
        // UV method: xatlas unwrap by default — unique chart space per face, no projection
        // overlap. --box-uv selects the voxel-native 6-way box projection: O(F) and seconds vs
        // xatlas's ~superlinear chart-compute (grid 384 -> 382K faces took >9min pre-decimation),
        // occlusion-aware bucket assignment + depth-tested raster keep its bleed low.
        const bool boxuv = !cfg.xatlas;
        const int T = cfg.tex >= 0 ? cfg.tex : (cascade ? 2048 : 1024);
        auto dump_post = [&](const char* path) {
            FILE* dfp = fopen(path,"wb");
            if (!dfp) return false;
            const int dV=mesh.V(), dFc=mesh.F(), Mv=(int)pbr_coords->size(), res=pbr_res;
            bool ok=fwrite(&dV,4,1,dfp)==1 && fwrite(&dFc,4,1,dfp)==1 &&
                    fwrite(&Mv,4,1,dfp)==1 && fwrite(&res,4,1,dfp)==1 &&
                    fwrite(mesh.verts.data(),4,(size_t)dV*3,dfp)==(size_t)dV*3 &&
                    fwrite(mesh.faces.data(),4,(size_t)dFc*3,dfp)==(size_t)dFc*3;
            for (const auto& c : *pbr_coords) {
                int xyz[3]={c[0],c[1],c[2]};
                ok=fwrite(xyz,4,3,dfp)==3 && ok;
            }
            ok=fwrite(pbr6.data(),4,(size_t)Mv*6,dfp)==(size_t)Mv*6 && ok;
            ok=fclose(dfp)==0 && ok;
            return ok;
        };
        if (const char* dp=std::getenv("TRELLIS_DUMP_POST")) {
            if (!dump_post(dp)) {
                fprintf(stderr,"[trellis] failed writing POST dump %s\n",dp);
                return 1;
            }
            printf("      [dump] post-stage inputs -> %s\n",dp);
        }
        fs::path retopo_post;
        if (cfg.retopo) {
#if defined(__linux__) || defined(_WIN32)
            retopo_post=retopo_run/"source.post";
            if (!dump_post(retopo_post.c_str())) {
                fprintf(stderr,"[trellis] failed writing retopo POST %s\n",retopo_post.c_str());
                return 1;
            }
            printf("      [retopo] POST and intermediates -> %s\n",retopo_run.c_str());
#endif
        }
        if (dual_pbr_path) {
            FILE* dfp = fopen(dual_pbr_path, "wb");
            if (!dfp) {
                fprintf(stderr, "[trellis] cannot open dual PBR dump %s\n", dual_pbr_path);
                return 1;
            }
            int dV = mesh.V(), dFc = mesh.F(), Mv = (int)so.coords.size(), res = so.res;
            bool ok = fwrite(&dV,4,1,dfp)==1 && fwrite(&dFc,4,1,dfp)==1 &&
                      fwrite(&Mv,4,1,dfp)==1 && fwrite(&res,4,1,dfp)==1 &&
                      fwrite(mesh.verts.data(),4,(size_t)dV*3,dfp)==(size_t)dV*3 &&
                      fwrite(mesh.faces.data(),4,(size_t)dFc*3,dfp)==(size_t)dFc*3;
            for (const auto& c : so.coords) {
                int xyz[3] = {c[0],c[1],c[2]};
                ok = fwrite(xyz,4,3,dfp)==3 && ok;
            }
            ok = fwrite(dual_pbr6.data(),4,dual_pbr6.size(),dfp)==dual_pbr6.size() && ok;
            ok = fclose(dfp)==0 && ok;
            if (!ok) {
                fprintf(stderr, "[trellis] failed writing dual PBR dump %s\n", dual_pbr_path);
                return 1;
            }
            printf("      [dump] secondary PBR on the same geometry -> %s\n", dual_pbr_path);
        }
        if (std::getenv("TRELLIS_STOP_AFTER_POST_DUMP")) return 0;
        if (cfg.retopo) {
#if defined(__linux__) || defined(_WIN32)
            const fs::path quad_glb=retopo_run/"model.glb";
            std::vector<std::string> args={retopo_driver.string(),"--from-post",retopo_post.string(),
                quad_glb.string(),std::to_string(cfg.retopo_grid),
                std::to_string(cfg.retopo_first_faces),std::to_string(cfg.retopo_final_faces),
                std::to_string(cfg.retopo_atlas)};
            if (cfg.retopo_no_weld_fill) args.emplace_back("--no-weld-fill");
#ifdef _WIN32
            std::vector<std::pair<std::wstring,std::wstring>> child_env={
                {L"TEMP",retopo_run.wstring()},{L"TMP",retopo_run.wstring()},
                {L"TRELLIS_RETOPO_SEED",std::to_wstring(run_seed)},
                {L"TRELLIS_RETOPO_COPYRIGHT",trellis::retopo_process::wide(cfg.copyright)},
                {L"TRELLIS_RETOPO_PBR_POST",L""}};
            if (cfg.retopo_dual_pbr)
                child_env.emplace_back(L"TRELLIS_RETOPO_PBR_POST",
                    trellis::retopo_process::wide(retopo_dual_post));
            int status=1;
            try {
                status=trellis::retopo_process::run(retopo_driver,
                    {args.begin()+1,args.end()},child_env);
            } catch (const std::exception& error) {
                fprintf(stderr,"[trellis] cannot launch retopo driver: %s\n",error.what());
                return 1;
            }
            const bool stage_ok=status==0;
#else
            std::vector<char*> raw;
            for (auto& arg:args) raw.push_back(arg.data());
            raw.push_back(nullptr);
            extern char **environ;
            std::vector<std::string> child_env;
            for (char** entry=environ;*entry;++entry) {
                const std::string value(*entry);
                if (value.rfind("TMPDIR=",0)!=0 &&
                    value.rfind("TRELLIS_RETOPO_SEED=",0)!=0 &&
                    value.rfind("TRELLIS_RETOPO_COPYRIGHT=",0)!=0 &&
                    value.rfind("TRELLIS_RETOPO_PBR_POST=",0)!=0)
                    child_env.push_back(value);
            }
            child_env.push_back("TMPDIR="+retopo_run.string());
            child_env.push_back("TRELLIS_RETOPO_SEED="+std::to_string(run_seed));
            child_env.push_back("TRELLIS_RETOPO_COPYRIGHT="+cfg.copyright);
            if (cfg.retopo_dual_pbr) child_env.push_back("TRELLIS_RETOPO_PBR_POST="+retopo_dual_post);
            std::vector<char*> env_raw;
            for (auto& value:child_env) env_raw.push_back(value.data());
            env_raw.push_back(nullptr);
            pid_t child=0;
            const int launch=posix_spawn(&child,retopo_driver.c_str(),nullptr,nullptr,
                                         raw.data(),env_raw.data());
            if (launch!=0) {
                fprintf(stderr,"[trellis] cannot launch retopo driver: %d\n",launch);
                return 1;
            }
            int status=0;
            const bool stage_ok=waitpid(child,&status,0)>=0 && WIFEXITED(status) &&
                                WEXITSTATUS(status)==0;
#endif
            if (!stage_ok ||
                !fs::is_regular_file(quad_glb) ||
                !fs::is_regular_file(retopo_run/"model.retopo-accepted.json")) {
                fprintf(stderr,"[trellis] retopo failed; diagnostics in %s\n",retopo_run.c_str());
                return 1;
            }
            std::error_code ec;
            fs::path destination=fs::absolute(outglb);
            fs::create_directories(destination.parent_path(),ec);
            if (ec) {
                fprintf(stderr,"[trellis] cannot create output directory: %s\n",ec.message().c_str());
                return 1;
            }
#ifdef _WIN32
            const fs::path staged=destination.string()+".retopo-staged-"+
                                  std::to_string(GetCurrentProcessId());
#else
            const fs::path staged=destination.string()+".retopo-staged-"+std::to_string(getpid());
#endif
            fs::copy_file(quad_glb,staged,fs::copy_options::overwrite_existing,ec);
            if (ec) {
                fprintf(stderr,"[trellis] cannot stage retopo GLB: %s\n",ec.message().c_str());
                fs::remove(staged);
                return 1;
            }
#ifdef _WIN32
            if (!MoveFileExW(staged.c_str(),destination.c_str(),
                             MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))
                ec=std::error_code(GetLastError(),std::system_category());
#else
            fs::rename(staged,destination,ec);
#endif
            if (ec) {
                fprintf(stderr,"[trellis] cannot publish retopo GLB: %s\n",ec.message().c_str());
                fs::remove(staged);
                return 1;
            }
            printf("done in %.1fs -> %s (quad audit: %s)\n",now()-t0,
                   outglb.c_str(),(retopo_run/"model.retopo-accepted.json").c_str());
            return 0;
#endif
        }
        trellis::weld_vertices(mesh.verts, mesh.faces, colors.empty() ? nullptr : &colors,
                               1.0f / ((float)so.res * 8.0f));
        trellis::fill_small_holes(mesh.faces);
        // Reference production pipeline: rebuild the noisy dual-grid mesh as
        // the narrow-band offset shell (watertight manifold), then quadric
        // simplify to the face target. The BVH over the original hole-filled
        // mesh serves both the remesh UDF and the bake's texel snap.
        trellis::TriBvh bvh = trellis::TriBvh::build(mesh.verts.data(), mesh.V(),
                                                     mesh.faces.data(), mesh.F());
        // The narrow-band remesh offset is eps = band*scale/res, i.e. it shrinks with
        // resolution. At res-512 (band=1) that offset absorbs the decoder's sub-voxel
        // "outer-skin" noise; at res-1024 the SAME band=1 halves the world-space offset,
        // so the noise survives as the issue-#22 speckle. Default band (cfg.band==0)
        // scales with resolution to keep the offset resolution-independent (512->1,
        // 1024->2, 1536->3); an explicit --band / per-request band forces that value.
        int remesh_band = cfg.band > 0 ? cfg.band : std::max(1, so.res / 512);
        trellis::Mesh rm = trellis::remesh_narrow_band_dc(mesh.verts.data(), mesh.V(),
                                                          mesh.faces.data(), mesh.F(),
                                                          bvh, so.res, remesh_band);
        // Clean the narrow-band DC output (drop degenerate faces, unify winding), then drop
        // decode floaters (the reference is a single watertight component; ours shattered into
        // 50+ pieces). The faithful QEM simplifier below handles surface smoothing via its
        // quadric + skinny-triangle cost, so no Taubin pre-pass is needed (and none is applied,
        // which keeps the mesh aligned to the voxel PBR volume for correct texture sampling).
        if (rm.F() > 0) {
            trellis::clean_mesh(rm.V(), rm.faces);
            if (!std::getenv("TRELLIS_RETOPO_KEEP_COMPONENTS")) {
                int ndrop = trellis::drop_small_components(rm.verts, rm.faces, 0.02f);
                printf("  remesh postproc: dropped %d floater comps -> V=%d F=%d\n", ndrop, rm.V(), rm.F());
                fflush(stdout);
            }
        }
        const std::vector<float>& sverts = rm.F() > 0 ? rm.verts : mesh.verts;
        const std::vector<int32_t>& sfaces = rm.F() > 0 ? rm.faces : mesh.faces;
        std::vector<float> dv, dp; std::vector<int32_t> df;
        if (cfg.decim > 0) {
            trellis::decimate_cluster(sverts, (int)sverts.size()/3, sfaces, (int)sfaces.size()/3, {}, cfg.decim, dv, df, dp);
        } else if (cfg.decim == 0) {
            dv = sverts; df = sfaces;
        } else {
            trellis::decimate_qem(sverts, (int)sverts.size()/3, sfaces, (int)sfaces.size()/3,
                                  cascade ? 300000 : 150000, dv, df);
            if (!std::getenv("TRELLIS_QEM_PRESERVE_TOPOLOGY") &&
                !std::getenv("TRELLIS_QEM_COLLISION_GUARD")) {
                trellis::weld_vertices(dv, df, nullptr, 1.0f / ((float)so.res * 8.0f));
                trellis::fill_small_holes(df);
            }
            // Second component pass on the decimated mesh: a hallucinated ground plane
            // survives the dense-mesh drop (it decimates to a large flat slab) but is a
            // small fraction here and disconnected from the body. Ref is a single component.
            if (!std::getenv("TRELLIS_RETOPO_KEEP_COMPONENTS")) {
                int ndrop2 = trellis::drop_small_components(dv, df, 0.03f);
                if (ndrop2) { printf("  decimated postproc: dropped %d more comps -> F=%d\n", ndrop2, (int)df.size()/3); fflush(stdout); }
            }
        }
        if (std::getenv("TRELLIS_RETOPO_SHAPE_REPAIR")) {
            float minimum_angle = 1.0f;
            if (const char* setting = std::getenv("TRELLIS_RETOPO_SHAPE_MIN_ANGLE")) {
                char* end = nullptr;
                minimum_angle = std::strtof(setting, &end);
                if (end == setting || *end || !std::isfinite(minimum_angle) ||
                    minimum_angle <= 0.0f || minimum_angle > 30.0f) {
                    fprintf(stderr,"[trellis] invalid TRELLIS_RETOPO_SHAPE_MIN_ANGLE\n");
                    return 1;
                }
            }
            float maximum_move = 0.001f;
            if (const char* setting = std::getenv("TRELLIS_RETOPO_SHAPE_MAX_MOVE")) {
                char* end = nullptr;
                maximum_move = std::strtof(setting, &end);
                if (end == setting || *end || !std::isfinite(maximum_move) ||
                    maximum_move <= 0.0f || maximum_move > 0.01f) {
                    fprintf(stderr,"[trellis] invalid TRELLIS_RETOPO_SHAPE_MAX_MOVE\n");
                    return 1;
                }
            }
            auto flipped = trellis::flip_skinny_triangles(dv,df,minimum_angle);
            auto smoothed = trellis::smooth_skinny_triangles(dv,df,minimum_angle,maximum_move);
            printf("  shape repair: flips=%d moved=%d collision_rejections=%d max_displacement=%.9g\n",
                   flipped.flips,smoothed.moved,
                   flipped.collision_rejections+smoothed.collision_rejections,
                   smoothed.maximum_displacement);
        }
        const int dV = (int)dv.size()/3, dF = (int)df.size()/3;
        // Texels are shaded straight from the per-voxel PBR volume (trilinear sampling, the
        // reference bake behavior) rather than from decimation-averaged vertex colors, so
        // full material detail survives simplification.
        trellis::VoxelPbr vox{pbr_coords, &pbr6, pbr_res, &bvh};
        const std::vector<float> no_vp;
        trellis::BakedMesh bm = boxuv ? trellis::uv_box_project(dv, dV, df, dF, no_vp, T, &vox)
                                      : trellis::uv_bake(dv, dV, df, dF, no_vp, T, &vox);
        if (!boxuv && !bm.ok()) bm = trellis::uv_chart_project(dv, dV, df, dF, no_vp, T, &vox);
        if (bm.ok()) {
            trellis::write_glb_textured(outglb.c_str(), bm.verts.data(), (int64_t)bm.verts.size()/3, bm.uv.data(),
                                        bm.faces.data(), (int64_t)bm.faces.size()/3, bm.base.data(), bm.mr.data(), bm.T,
                                        /*double_sided=*/rm.F() == 0, run_seed,
                                        cfg.copyright.empty() ? nullptr : cfg.copyright.c_str(),
                                        /*use_webp=*/cfg.webp != 0);
            std::string tex = outglb.substr(0, outglb.find_last_of('.')) + "_base.png";
            stbi_write_png(tex.c_str(), bm.T, bm.T, 4, bm.base.data(), bm.T*4);
            textured = true;
            printf("      textured GLB (atlas %d, +%s)\n", bm.T, tex.c_str());
        } else printf("      uv_bake failed; falling back to vertex colors\n");
    }
    if (cfg.retopo && !textured) {
        fprintf(stderr,"[trellis] textured POST unavailable for retopo\n");
        return 1;
    }
    if (!textured)
        trellis::write_glb(outglb.c_str(), mesh.verts.data(), mesh.V(), mesh.faces.data(), mesh.F(),
                           colors.empty() ? nullptr : colors.data(), run_seed,
                           cfg.copyright.empty() ? nullptr : cfg.copyright.c_str());
    std::string ply = outglb.substr(0, outglb.find_last_of('.')) + ".ply";
    trellis::write_ply(ply.c_str(), mesh.verts.data(), mesh.V(), mesh.faces.data(), mesh.F(), colors.empty() ? nullptr : colors.data());
    printf("done in %.1fs -> %s (+ %s)\n", now() - t0, outglb.c_str(), ply.c_str());
    return 0;
}
