// trellis-cli: image -> 3D mesh (.glb), the TRELLIS.2 geometry pipeline in GGML.
//   trellis-cli <image.png> <out.glb> [gpu] [models_dir] [seed]
// Models are loaded/freed per stage to keep VRAM modest.
#include "trellis_model.h"
#include "preprocess.h"
#include "dinov3.h"
#include "flow_runner.h"
#include "ss_decoder.h"
#include "shape_decoder.h"
#include "dual_grid.h"
#include "mesh_glb.h"
#include "uv_bake.h"
#include "tri_bvh.h"
#include "remesh_dc.h"
#include "stb_image_write.h"
#include "trellis_run.h"

#include <cstdio>
#include <random>
#include <vector>
#include <string>
#include <chrono>
#include <set>
#include <array>
#include <cmath>

using std::vector;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
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
    trellis::g_require_gpu = cfg.require_gpu;
    const std::string& img = cfg.image;
    const std::string& outglb = cfg.output;
    const std::string& M = cfg.models;
    const int gpu = cfg.gpu;
    const bool cascade = cfg.cascade;   // 1024 cascade is the TRELLIS default; --res 512 forces the light path
    std::mt19937 rng(run_seed); std::normal_distribution<float> randn(0.f, 1.f);
    auto noise = [&](size_t n){ vector<float> v(n); for (auto& x : v) x = randn(rng); return v; };
    double t0 = now();

    const bool birefnet = cfg.birefnet;
    vector<float> chw, chw1024;
    if (birefnet) {
        printf("[1/6] preprocess %s (BiRefNet bg removal, %s)\n", img.c_str(), cascade ? "1024 cascade" : "512");
        // Full BiRefNet (Swin-L backbone + deformable-conv decoder) runs on the GPU. Cutout computed
        // once, normalized for 512 and 1024.
        trellis::Model bm = trellis::Model::load(M + "/birefnet.gguf", gpu);
        int sz = 0; std::vector<unsigned char> cut = trellis::birefnet_cutout(img, bm, gpu < 0 ? 0 : gpu, sz);
        bm.free();
        if (cut.empty()) return 1;
        chw = trellis::normalize_cutout(cut, sz, 512);
        if (cascade) chw1024 = trellis::normalize_cutout(cut, sz, 1024);
    } else {
        printf("[1/6] preprocess %s (%s)\n", img.c_str(), cascade ? "1024 cascade" : "512");
        chw = trellis::preprocess_image(img, 512);
        if (chw.empty()) return 1;
        chw1024 = cascade ? trellis::preprocess_image(img, 1024) : vector<float>();
    }

    printf("[2/6] DINOv3 conditioning\n");
    vector<float> cond, cond1024;
    { trellis::Model m = trellis::Model::load(M + "/dinov3.gguf", gpu);
      cond = trellis::dinov3_encode(m, chw, 512);
      if (cascade) cond1024 = trellis::dinov3_encode(m, chw1024, 1024);
      m.free(); }
    const int Lc = (int)(cond.size() / 1024);
    vector<float> neg(cond.size(), 0.0f);
    const int Lc1024 = cascade ? (int)(cond1024.size() / 1024) : 0;
    vector<float> neg1024(cond1024.size(), 0.0f);
    printf("      cond tokens=%d%s\n", Lc, cascade ? (" / 1024-cond tokens=" + std::to_string(Lc1024)).c_str() : "");
    slat_stats("cond_512 (DINOv3@512)", cond);
    if (cascade) slat_stats("cond_1024 (DINOv3@1024)", cond1024);

    printf("[3/6] sparse-structure flow + decode\n");
    vector<std::array<int,3>> coords;
    {
        trellis::Model m = trellis::Model::load(M + "/ss_flow.gguf", gpu);
        trellis::DiTParams p; p.in_ch = 8; p.out_ch = 8; p.d_cond = 1024; p.cast_f32 = F32;
        trellis::DitRunner* run = trellis::make_dense_runner(m, p, 16, Lc);
        trellis::FlowFwd fwd = [&](const vector<float>& x, float ts, const float* c){ return run->forward(x, ts, c); };
        trellis::SamplerParams sp; sp.steps=12; sp.guidance_strength=cfg.gss; sp.guidance_rescale=0.7f; sp.gi0=0.6f; sp.gi1=1.0f; sp.rescale_t=5.0f;
        vector<float> z = trellis::sample_flow(fwd, noise(8*4096), cond.data(), neg.data(), sp);  // [8,4096] ne0=8
        delete run; m.free();
        // transpose [8,L] -> torch [8,16,16,16] memory (c*4096 + sp)
        vector<float> zdec(8*4096);
        for (int c = 0; c < 8; ++c) for (int sp2 = 0; sp2 < 4096; ++sp2) zdec[(size_t)c*4096 + sp2] = z[c + 8*sp2];
        trellis::Model d = trellis::Model::load(M + "/ss_dec.gguf", gpu);
        vector<float> logits = trellis::ss_decode(d, zdec); d.free();
        coords = trellis::ss_coords(logits, 64, 32);
    }
    if (cfg.voxply) { FILE*f=fopen("out/myvox.ply","wb"); fprintf(f,"ply\nformat binary_little_endian 1.0\nelement vertex %zu\nproperty float x\nproperty float y\nproperty float z\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n",coords.size()); for(auto&c:coords){float p[3]={(c[0]+0.5f)/32-0.5f,(c[1]+0.5f)/32-0.5f,(c[2]+0.5f)/32-0.5f}; fwrite(p,4,3,f);} fclose(f); }
    printf("      active voxels @res32 = %d\n", (int)coords.size());
    if (coords.empty()) { fprintf(stderr, "no voxels produced\n"); return 1; }

    const bool do_tex = cfg.texture;

    // one shape SLAT flow run -> normalized [32,n] (sparse, CFG 7.5, gi[0.6,1], rescale_t 3)
    auto shape_flow = [&](const std::string& path, const vector<std::array<int,3>>& cds,
                          const float* cnd, const float* ncnd, int lc) {
        const int n = (int)cds.size();
        trellis::Model m = trellis::Model::load(path, gpu);
        trellis::DiTParams p; p.in_ch = 32; p.out_ch = 32; p.d_cond = 1024; p.cast_f32 = F32;
        trellis::DitRunner* run = trellis::make_sparse_runner(m, p, cds, lc);
        trellis::FlowFwd fwd = [&](const vector<float>& x, float ts, const float* c){ return run->forward(x, ts, c); };
        trellis::SamplerParams sp; sp.steps=12; sp.guidance_strength=cfg.gsh; sp.guidance_rescale=0.5f; sp.gi0=0.6f; sp.gi1=1.0f; sp.rescale_t=3.0f;
        vector<float> sn = trellis::sample_flow(fwd, noise((size_t)32*n), cnd, ncnd, sp);   // [32,n]
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
        lr_norm = shape_flow(M + "/shape_flow_512.gguf", coords, cond.data(), neg.data(), Lc);
        lr_dn.resize(lr_norm.size());
        for (size_t n = 0; n < coords.size(); ++n) for (int c = 0; c < 32; ++c)
            lr_dn[(size_t)c + 32*n] = lr_norm[(size_t)c + 32*n]*SHAPE_STD[c] + SHAPE_MEAN[c];
        slat_stats("LR slat (res32, decodes OK via upsample)", lr_dn);
        // (2) decoder.upsample(LR slat, 4) -> res512 coords
        vector<std::array<int,3>> hr_coords;
        { trellis::Model m = trellis::Model::load(M + "/shape_dec.gguf", gpu);
          hr_coords = trellis::shape_upsample(m, lr_dn, coords); m.free(); }
        // (3) quantize res512 -> res(hr_res//16) with the reference's adaptive token-budget backoff
        //     (sample_shape_slat_cascade): start at hr_target, step -128 toward the 1024 floor while
        //     the unique token count would exceed max_num_tokens. grid = hr_res//16 is integral since
        //     128/16 = 8 (1536->96, 1408->88, ..., 1024->64).
        int hr_res = hr_target;
        for (;;) {
            const int gi = hr_res / 16;          // integral grid (ref's hr_resolution//16)
            const float g = (float)gi;
            std::set<std::array<int,3>> q;
            for (auto& c : hr_coords) q.insert({ (int)((c[0]+0.5f)/512.f*g), (int)((c[1]+0.5f)/512.f*g), (int)((c[2]+0.5f)/512.f*g) });
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
        // (4) HR shape flow @res(hr_res//16) with cond_1024
        slat_norm = shape_flow(M + "/shape_flow_1024.gguf", shc, cond1024.data(), neg1024.data(), Lc1024);
        RES = hr_res; cond_dec = cond1024.data(); neg_dec = neg1024.data(); Lc_dec = Lc1024;
    } else {
        printf("[4/7] shape SLAT flow (512)\n");
        shc = coords;
        slat_norm = shape_flow(M + "/shape_flow_512.gguf", coords, cond.data(), neg.data(), Lc);
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
    }
    printf("      mesh V=%d F=%d\n", mesh.V(), mesh.F());

    vector<float> colors, pbr6;   // colors = base RGB (PLY); pbr6 = per-vertex [V*6] for UV bake
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
        const bool mixed = cascade && tex_res != RES;   // res-1024 geometry + res-512 texture
        printf("[6/7] texture SLAT flow + PBR decode%s\n", mixed ? "  (res-512 texture on res-1024 mesh)" : "");

        if (mixed) {   // decode a res-512 shape (from the LR slat) to guide the res-512 tex decode
            trellis::Model m = trellis::Model::load(M + "/shape_dec.gguf", gpu);
            so_tex = trellis::shape_decode(m, lr_dn, coords, 512); m.free();
            pbr_coords = &so_tex.coords; pbr_res = so_tex.res;
            printf("      res-512 tex-guide decode: %d voxels\n", (int)so_tex.coords.size());
        }
        // tex flow + decode inputs: HR path (shc/slat_norm/cond_dec/so.subs) vs res-512 mixed path
        // (coords/lr_norm/cond_512/so_tex.subs). The tex decoder upsamples via the guide subdivision.
        const std::string tflow = M + (mixed ? "/tex_flow_512.gguf" : (cascade ? "/tex_flow_1024.gguf" : "/tex_flow_512.gguf"));
        const vector<std::array<int,3>>& tcoords = mixed ? coords : shc;
        const vector<float>& tslat = mixed ? lr_norm : slat_norm;
        const float* tcond = mixed ? cond.data() : cond_dec;
        const float* tneg  = mixed ? neg.data()  : neg_dec;
        const int    tlc   = mixed ? Lc : Lc_dec;
        const int    tN    = (int)tcoords.size();
        const std::vector<std::vector<uint8_t>>& tsubs = mixed ? so_tex.subs : so.subs;

        vector<float> texlat;
        {
            trellis::Model m = trellis::Model::load(tflow, gpu);
            trellis::DiTParams p; p.in_ch = 64; p.out_ch = 32; p.d_cond = 1024; p.cast_f32 = F32;
            trellis::DitRunner* run = trellis::make_sparse_runner(m, p, tcoords, tlc);
            // state is the 32-ch noise; each forward concat [noise(32) ; shape_slat_norm(32)] -> 64ch
            trellis::FlowFwd fwd = [&](const vector<float>& st, float ts, const float* c) {
                vector<float> x64((size_t)64 * tN);
                for (int n = 0; n < tN; ++n) {
                    for (int k = 0; k < 32; ++k) x64[(size_t)k + 64*n]      = st[(size_t)k + 32*n];
                    for (int k = 0; k < 32; ++k) x64[(size_t)32 + k + 64*n] = tslat[(size_t)k + 32*n];
                }
                return run->forward(x64, ts, c);
            };
            trellis::SamplerParams sp; sp.steps=12; sp.guidance_strength=1.0f; sp.guidance_rescale=0.0f; sp.gi0=0.6f; sp.gi1=0.9f; sp.rescale_t=3.0f;
            texlat = trellis::sample_flow(fwd, noise((size_t)32*tN), tcond, tneg, sp);  // [32,tN]
            delete run; m.free();
            for (int n = 0; n < tN; ++n) for (int c = 0; c < 32; ++c) texlat[(size_t)c + 32*n] = texlat[(size_t)c + 32*n]*TEX_STD[c] + TEX_MEAN[c];
        }
        {
            trellis::Model m = trellis::Model::load(M + "/tex_dec.gguf", gpu);
            vector<float> pbr = trellis::tex_decode(m, texlat, tcoords, tsubs); m.free();   // [6,Mv] pre-scale
            const int Mv = (int)pbr_coords->size();
            colors.resize((size_t)Mv * 3); pbr6.resize((size_t)Mv * 6);
            auto cl = [](float v){ return v < 0 ? 0.f : (v > 1 ? 1.f : v); };
            for (int i = 0; i < Mv; ++i) {
                for (int k = 0; k < 6; ++k) pbr6[(size_t)i*6 + k] = cl(pbr[(size_t)k + 6*i] * 0.5f + 0.5f);
                for (int k = 0; k < 3; ++k) colors[(size_t)i*3 + k] = pbr6[(size_t)i*6 + k];
            }
            printf("      PBR voxels=%d @res%d\n", Mv, pbr_res);
        }
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
        if (const char* dp = std::getenv("TRELLIS_DUMP_POST")) {
            FILE* dfp = fopen(dp, "wb");
            if (dfp) {   // geometry mesh + the PBR volume the bake samples (may be res-512 in mixed mode)
                int dV = mesh.V(), dFc = mesh.F(), Mv = (int)pbr_coords->size(), res = pbr_res;
                fwrite(&dV,4,1,dfp); fwrite(&dFc,4,1,dfp); fwrite(&Mv,4,1,dfp); fwrite(&res,4,1,dfp);
                fwrite(mesh.verts.data(),4,(size_t)dV*3,dfp);
                fwrite(mesh.faces.data(),4,(size_t)dFc*3,dfp);
                for (auto& c : *pbr_coords) { int xyz[3] = {c[0],c[1],c[2]}; fwrite(xyz,4,3,dfp); }
                fwrite(pbr6.data(),4,(size_t)Mv*6,dfp);
                fclose(dfp);
                printf("      [dump] post-stage inputs -> %s\n", dp);
            }
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
        trellis::Mesh rm = trellis::remesh_narrow_band_dc(mesh.verts.data(), mesh.V(),
                                                          mesh.faces.data(), mesh.F(),
                                                          bvh, so.res);
        // Clean the narrow-band DC output (drop degenerate faces, unify winding), then drop
        // decode floaters (the reference is a single watertight component; ours shattered into
        // 50+ pieces). The faithful QEM simplifier below handles surface smoothing via its
        // quadric + skinny-triangle cost, so no Taubin pre-pass is needed (and none is applied,
        // which keeps the mesh aligned to the voxel PBR volume for correct texture sampling).
        if (rm.F() > 0) {
            trellis::clean_mesh(rm.V(), rm.faces);
            int ndrop = trellis::drop_small_components(rm.verts, rm.faces, 0.02f);
            printf("  remesh postproc: dropped %d floater comps -> V=%d F=%d\n", ndrop, rm.V(), rm.F());
            fflush(stdout);
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
            trellis::weld_vertices(dv, df, nullptr, 1.0f / ((float)so.res * 8.0f));
            trellis::fill_small_holes(df);
            // Second component pass on the decimated mesh: a hallucinated ground plane
            // survives the dense-mesh drop (it decimates to a large flat slab) but is a
            // small fraction here and disconnected from the body. Ref is a single component.
            int ndrop2 = trellis::drop_small_components(dv, df, 0.03f);
            if (ndrop2) { printf("  decimated postproc: dropped %d more comps -> F=%d\n", ndrop2, (int)df.size()/3); fflush(stdout); }
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
                                        /*double_sided=*/rm.F() == 0, run_seed);
            std::string tex = outglb.substr(0, outglb.find_last_of('.')) + "_base.png";
            stbi_write_png(tex.c_str(), bm.T, bm.T, 4, bm.base.data(), bm.T*4);
            textured = true;
            printf("      textured GLB (atlas %d, +%s)\n", bm.T, tex.c_str());
        } else printf("      uv_bake failed; falling back to vertex colors\n");
    }
    if (!textured)
        trellis::write_glb(outglb.c_str(), mesh.verts.data(), mesh.V(), mesh.faces.data(), mesh.F(), colors.empty() ? nullptr : colors.data(), run_seed);
    std::string ply = outglb.substr(0, outglb.find_last_of('.')) + ".ply";
    trellis::write_ply(ply.c_str(), mesh.verts.data(), mesh.V(), mesh.faces.data(), mesh.F(), colors.empty() ? nullptr : colors.data());
    printf("done in %.1fs -> %s (+ %s)\n", now() - t0, outglb.c_str(), ply.c_str());
    return 0;
}
