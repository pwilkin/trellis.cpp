// BiRefNet Swin-L backbone in GGML. Segmented per-stage graphs; window partition / shifted-window
// roll / attention mask are precomputed on the host as gather/scatter index arrays + an additive
// mask, applied via ggml_get_rows. Validated against tools/ref_birefnet.py dumps.
#include "birefnet.h"
#include "deform_conv.h"
#include "trellis_model.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace trellis {
using T = ggml_tensor;

static const int WS = 12;          // window size (swin_v1_l)
static const int HEADS[4] = {6, 12, 24, 48};
static const int DEPTH[4] = {2, 2, 18, 2};

static ggml_context* mkctx(size_t nodes = 16384) {
    size_t meta = ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false) + (1 << 20);
    return ggml_init({ meta, nullptr, true });
}

// build + run a graph; returns each output read to host f32 (read BEFORE freeing the gallocr).
static std::vector<std::vector<float>> run_graph(const Model& m, ggml_context* c, std::vector<T*> outs,
                      std::vector<std::pair<T*, const void*>> ins, size_t nodes = 16384) {
    ggml_cgraph* g = ggml_new_graph_custom(c, nodes, false);
    for (T* o : outs) { ggml_set_output(o); ggml_build_forward_expand(g, o); }
    ggml_gallocr_t a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    if (!ggml_gallocr_alloc_graph(a, g)) throw std::runtime_error("birefnet alloc");
    for (auto& [t, d] : ins) ggml_backend_tensor_set(t, d, 0, ggml_nbytes(t));
    if (ggml_backend_graph_compute(m.backend, g) != GGML_STATUS_SUCCESS) throw std::runtime_error("birefnet compute");
    std::vector<std::vector<float>> r;
    for (T* o : outs) r.push_back(tensor_to_f32(o));
    ggml_gallocr_free(a);
    return r;
}

static T* ln(ggml_context* c, T* x, T* w, T* b, float eps = 1e-5f) {  // LayerNorm over ne0
    x = ggml_norm(c, x, eps);
    x = ggml_mul(c, x, w);
    return ggml_add(c, x, b);
}
static T* lin(ggml_context* c, const Model& m, const std::string& p, T* x) {
    T* y = ggml_mul_mat(c, m.get(p + ".weight"), x);
    if (m.has(p + ".bias")) y = ggml_add(c, y, m.get(p + ".bias"));
    return y;
}

// ---- host window index bookkeeping ----
struct WinIdx {
    int Hp, Wp, nWh, nWw, nW, N;
    std::vector<int32_t> gather;   // [nW*WS*WS] padded-grid token -> real token (or N for zero pad)
    std::vector<int32_t> scatter;  // [N] real token -> window-major position
    std::vector<float>   mask;     // [WS*WS(k) * WS*WS(q) * nW] additive (shifted only), else empty
};
static int wrap(int a, int n) { return ((a % n) + n) % n; }

static WinIdx build_winidx(int H, int W, int shift) {
    WinIdx wi;
    wi.N = H * W;
    wi.Hp = ((H + WS - 1) / WS) * WS; wi.Wp = ((W + WS - 1) / WS) * WS;
    wi.nWh = wi.Hp / WS; wi.nWw = wi.Wp / WS; wi.nW = wi.nWh * wi.nWw;
    const int ws2 = WS * WS, N = wi.N;
    wi.gather.resize((size_t)wi.nW * ws2);
    for (int w = 0; w < wi.nW; ++w) {
        int wr = w / wi.nWw, wc = w % wi.nWw;
        for (int p = 0; p < ws2; ++p) {
            int pr = p / WS, pc = p % WS;
            int gh = wr * WS + pr, gw = wc * WS + pc;
            int oh = wrap(gh + shift, wi.Hp), ow = wrap(gw + shift, wi.Wp);
            wi.gather[(size_t)w * ws2 + p] = (oh < H && ow < W) ? (oh * W + ow) : N;
        }
    }
    wi.scatter.resize(N);
    for (int n = 0; n < N; ++n) {
        int oh = n / W, ow = n % W;
        int sh = wrap(oh - shift, wi.Hp), sw = wrap(ow - shift, wi.Wp);
        int wr = sh / WS, wc = sw / WS, pr = sh % WS, pc = sw % WS;
        wi.scatter[n] = (wr * wi.nWw + wc) * ws2 + (pr * WS + pc);
    }
    if (shift > 0) {
        std::vector<int> img((size_t)wi.Hp * wi.Wp);
        auto region = [&](int x, int len) { return x < len - WS ? 0 : (x < len - shift ? 1 : 2); };
        for (int gh = 0; gh < wi.Hp; ++gh) for (int gw = 0; gw < wi.Wp; ++gw)
            img[(size_t)gh * wi.Wp + gw] = region(gh, wi.Hp) * 3 + region(gw, wi.Wp);
        std::vector<int> mw((size_t)wi.nW * ws2);
        for (int w = 0; w < wi.nW; ++w) { int wr = w / wi.nWw, wc = w % wi.nWw;
            for (int p = 0; p < ws2; ++p) { int pr = p / WS, pc = p % WS;
                mw[(size_t)w * ws2 + p] = img[(size_t)(wr*WS+pr) * wi.Wp + (wc*WS+pc)]; } }
        wi.mask.assign((size_t)ws2 * ws2 * wi.nW, 0.0f);   // [k, q, nW]
        for (int w = 0; w < wi.nW; ++w) for (int q = 0; q < ws2; ++q) for (int k = 0; k < ws2; ++k)
            wi.mask[(size_t)k + ws2 * (q + (size_t)ws2 * w)] =
                (mw[(size_t)w*ws2 + q] != mw[(size_t)w*ws2 + k]) ? -100.0f : 0.0f;
    }
    return wi;
}

// one Swin block as its OWN graph (host [C,N] -> host [C,N]) to bound per-graph memory. smask_host =
// soft_max_ext additive mask: non-shifted -> rel-pos bias [ws2,ws2,NH] (broadcast); shifted -> full
// bias+shiftmask [ws2,ws2,NH,nW].
static std::vector<float> swin_block_run(const Model& m, const std::string& p,
        const std::vector<float>& xin, int C, int N, int NH, const WinIdx& wi, bool shifted,
        const std::vector<float>& smask_host) {
    const int ws2 = WS * WS, HD = C / NH, Mwin = wi.nW * ws2;
    const float scale = 1.0f / std::sqrt((float)HD);
    // host-pad: append a zero column so gather's pad-index (=N) reads zeros. (Avoid ggml_pad: its
    // CUDA kernel launches a grid dim = ne1 = N+1, which exceeds the 65535 limit at stage 0 -> the
    // whole window attention runs in ONE GPU graph per block.)
    std::vector<float> xpad((size_t)C * (N + 1), 0.0f);
    std::memcpy(xpad.data(), xin.data(), (size_t)C * N * sizeof(float));
    std::vector<float> cmaskd(N + 1, 1.0f); cmaskd[N] = 0.0f;       // zero the pad column post-norm
    ggml_context* c = mkctx();
    T* x  = ggml_new_tensor_2d(c, GGML_TYPE_F32, C, N + 1);       ggml_set_input(x);
    T* xb = ggml_new_tensor_2d(c, GGML_TYPE_F32, C, N);           ggml_set_input(xb);   // residual (unpadded)
    T* cmask = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, N + 1);    ggml_set_input(cmask);
    T* gi = ggml_new_tensor_1d(c, GGML_TYPE_I32, Mwin);           ggml_set_input(gi);
    T* si = ggml_new_tensor_1d(c, GGML_TYPE_I32, N);              ggml_set_input(si);
    T* sm = shifted ? ggml_new_tensor_4d(c, GGML_TYPE_F32, ws2, ws2, NH, wi.nW)
                    : ggml_new_tensor_3d(c, GGML_TYPE_F32, ws2, ws2, NH);  ggml_set_input(sm);
    T* h = ln(c, x, m.get(p + ".norm1.weight"), m.get(p + ".norm1.bias"));  // LN over the padded [C,N+1]
    h = ggml_mul(c, h, cmask);                                      // norm1 THEN pad-zero (matches ref order)
    h = ggml_get_rows(c, h, gi);                                    // [C, Mwin] (gathers real + zero-pad cols)
    T* qkv = lin(c, m, p + ".attn.qkv", h);                         // [3C, Mwin]
    T* q = ggml_view_2d(c, qkv, C, Mwin, qkv->nb[1], 0);
    T* k = ggml_view_2d(c, qkv, C, Mwin, qkv->nb[1], (size_t)C * ggml_element_size(qkv));
    T* v = ggml_view_2d(c, qkv, C, Mwin, qkv->nb[1], (size_t)2 * C * ggml_element_size(qkv));
    auto heads = [&](T* t) {
        t = ggml_cont(c, t);
        t = ggml_reshape_4d(c, t, HD, NH, ws2, wi.nW);
        t = ggml_cont(c, ggml_permute(c, t, 0, 2, 1, 3));
        return ggml_reshape_3d(c, t, HD, ws2, NH * wi.nW);
    };
    q = heads(q); k = heads(k); v = heads(v);
    T* kq = ggml_mul_mat(c, k, q);                                  // [ws2(k), ws2(q), NH*nW]
    kq = ggml_reshape_4d(c, kq, ws2, ws2, NH, wi.nW);
    kq = ggml_soft_max_ext(c, kq, sm, scale, 0.0f);                 // softmax(kq*scale + mask)
    kq = ggml_reshape_3d(c, kq, ws2, ws2, NH * wi.nW);
    T* vt = ggml_cont(c, ggml_permute(c, v, 1, 0, 2, 3));           // [ws2(k), HD, NH*nW]
    T* o = ggml_mul_mat(c, vt, kq);                                 // [HD, ws2(q), NH*nW]
    o = ggml_reshape_4d(c, o, HD, ws2, NH, wi.nW);
    o = ggml_cont(c, ggml_permute(c, o, 0, 2, 1, 3));
    o = ggml_reshape_2d(c, o, C, Mwin);
    o = lin(c, m, p + ".attn.proj", o);
    o = ggml_get_rows(c, o, si);                                    // scatter -> [C, N]
    T* xr = ggml_add(c, xb, o);                                     // residual
    T* h2 = ln(c, xr, m.get(p + ".norm2.weight"), m.get(p + ".norm2.bias"));
    h2 = lin(c, m, p + ".mlp.fc1", h2);
    h2 = ggml_gelu_erf(c, h2);
    h2 = lin(c, m, p + ".mlp.fc2", h2);
    T* outp = ggml_add(c, xr, h2);
    std::vector<float> r = run_graph(m, c, {outp},
        { {x, xpad.data()}, {xb, xin.data()}, {cmask, cmaskd.data()}, {gi, wi.gather.data()},
          {si, wi.scatter.data()}, {sm, smask_host.data()} })[0];
    ggml_free(c);
    return r;
}

BBOut swin_backbone(const Model& m, const std::vector<float>& chw, int S) {
    BBOut out;
    std::vector<float> tok; int C, H, W;
    {   // patch embed: conv4x4 s4 -> [C,N] -> LN
        ggml_context* c = mkctx();
        T* img = ggml_new_tensor_4d(c, GGML_TYPE_F32, S, S, 3, 1); ggml_set_input(img);
        T* pe = ggml_conv_2d(c, m.get("bb.patch_embed.proj.weight"), img, 4, 4, 0, 0, 1, 1);  // [W,H,192]
        pe = ggml_add(c, pe, ggml_reshape_3d(c, m.get("bb.patch_embed.proj.bias"), 1, 1, 192));
        int Wp = S / 4, Hp = S / 4;
        T* tc = ggml_cont(c, ggml_permute(c, pe, 1, 2, 0, 3));       // [192, W, H]
        T* t = ggml_reshape_2d(c, tc, 192, Wp * Hp);                 // [192, N]
        t = ln(c, t, m.get("bb.patch_embed.norm.weight"), m.get("bb.patch_embed.norm.bias"));
        (void)tc;
        tok = run_graph(m, c, {t}, { {img, chw.data()} })[0];
        ggml_free(c);
        C = 192; H = Hp; W = Wp;
    }

    for (int s = 0; s < 4; ++s) {
        const int NH = HEADS[s], N = H * W, ws2 = WS * WS;
        WinIdx wi0 = build_winidx(H, W, 0), wi1 = build_winidx(H, W, WS / 2);
        std::string p0 = "bb.layers." + std::to_string(s) + ".blocks.";
        std::vector<int32_t> idx((size_t)ws2 * ws2);
        ggml_backend_tensor_get(m.get(p0 + "0.attn.relative_position_index"), idx.data(), 0,
                                ggml_nbytes(m.get(p0 + "0.attn.relative_position_index")));
        // each block: own graph, host [C,N] in/out
        for (int b = 0; b < DEPTH[s]; ++b) {
            bool sh = (b % 2) == 1;
            std::vector<float> tab = tensor_to_f32(m.get(p0 + std::to_string(b) + ".attn.relative_position_bias_table"));
            std::vector<float> bias((size_t)ws2 * ws2 * NH);   // [k,q,h]
            for (int hh = 0; hh < NH; ++hh) for (int q = 0; q < ws2; ++q) for (int k = 0; k < ws2; ++k)
                bias[(size_t)k + ws2*(q + (size_t)ws2*hh)] = tab[(size_t)hh + (size_t)NH*idx[(size_t)q*ws2 + k]];
            std::vector<float> smask;
            if (!sh) smask = std::move(bias);
            else {   // bias[k,q,h] + shiftmask[k,q,w] -> full [k,q,h,w]
                smask.resize((size_t)ws2 * ws2 * NH * wi1.nW);
                for (int w = 0; w < wi1.nW; ++w) for (int hh = 0; hh < NH; ++hh)
                    for (int q = 0; q < ws2; ++q) for (int k = 0; k < ws2; ++k)
                        smask[(size_t)k + ws2*(q + (size_t)ws2*(hh + (size_t)NH*w))] =
                            bias[(size_t)k + ws2*(q + (size_t)ws2*hh)] +
                            wi1.mask[(size_t)k + ws2*(q + (size_t)ws2*w)];
            }
            tok = swin_block_run(m, p0 + std::to_string(b), tok, C, N, NH, sh ? wi1 : wi0, sh, smask);
        }
        // norm{s}(tok) -> out{s}; and PatchMerging(tok) -> next stage input
        std::vector<float> xout;
        {   // final per-stage LayerNorm
            ggml_context* c = mkctx();
            T* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, C, N); ggml_set_input(x);
            T* xo = ln(c, x, m.get("bb.norm" + std::to_string(s) + ".weight"), m.get("bb.norm" + std::to_string(s) + ".bias"));
            xout = run_graph(m, c, {xo}, { {x, tok.data()} }, 4096)[0];
            ggml_free(c);
        }
        out.C[s] = C; out.H[s] = H; out.W[s] = W;
        out.f[s].resize((size_t)C * H * W);
        for (int n = 0; n < N; ++n) for (int cc = 0; cc < C; ++cc)
            out.f[s][(size_t)cc * H * W + n] = xout[(size_t)cc + (size_t)C * n];
        if (s < 3) {   // PatchMerging on the (pre-norm) block output `tok`
            int Hh = H/2, Wh = W/2, Nh = Hh*Wh;
            std::vector<int32_t> mg[4];
            for (int q = 0; q < 4; ++q) mg[q].resize(Nh);
            for (int hr = 0; hr < Hh; ++hr) for (int wr = 0; wr < Wh; ++wr) {
                int np = hr*Wh + wr;
                mg[0][np]=(2*hr)*W+(2*wr); mg[1][np]=(2*hr+1)*W+(2*wr);
                mg[2][np]=(2*hr)*W+(2*wr+1); mg[3][np]=(2*hr+1)*W+(2*wr+1);
            }
            ggml_context* c = mkctx();
            T* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, C, N); ggml_set_input(x);
            T* g0=ggml_new_tensor_1d(c,GGML_TYPE_I32,Nh); T* g1=ggml_new_tensor_1d(c,GGML_TYPE_I32,Nh);
            T* g2=ggml_new_tensor_1d(c,GGML_TYPE_I32,Nh); T* g3=ggml_new_tensor_1d(c,GGML_TYPE_I32,Nh);
            ggml_set_input(g0); ggml_set_input(g1); ggml_set_input(g2); ggml_set_input(g3);
            std::string dp = "bb.layers." + std::to_string(s) + ".downsample.";
            T* x0=ggml_get_rows(c,x,g0); T* x1=ggml_get_rows(c,x,g1);
            T* x2=ggml_get_rows(c,x,g2); T* x3=ggml_get_rows(c,x,g3);
            T* cat = ggml_concat(c, ggml_concat(c,x0,x1,0), ggml_concat(c,x2,x3,0), 0);   // [4C,Nh]
            cat = ln(c, cat, m.get(dp+"norm.weight"), m.get(dp+"norm.bias"));
            T* nx = ggml_mul_mat(c, m.get(dp+"reduction.weight"), cat);                    // [2C,Nh]
            tok = run_graph(m, c, {nx}, { {x,tok.data()}, {g0,mg[0].data()}, {g1,mg[1].data()}, {g2,mg[2].data()}, {g3,mg[3].data()} }, 4096)[0];
            ggml_free(c);
            C *= 2; H /= 2; W /= 2;
        }
    }
    return out;
}

// ============================ squeeze + decoder ============================
// Host feature map in torch [C,H,W] order (== ggml [W,H,C] same bytes).
struct Feat { std::vector<float> d; int C = 0, H = 0, W = 0; };

// conv2d via ggml; torch[C,H,W] -> [OC,Ho,Wo] (stride 1). bias added if present.
static Feat conv2d(const Model& m, const std::string& p, const Feat& x, int pad) {
    Feat o;
    ggml_context* c = mkctx();
    T* gx = ggml_new_tensor_3d(c, GGML_TYPE_F32, x.W, x.H, x.C); ggml_set_input(gx);
    T* w = m.get(p + ".weight");
    int OC = (int)w->ne[3];
    T* y = ggml_conv_2d(c, w, gx, 1, 1, pad, pad, 1, 1);            // [Wo,Ho,OC]
    if (m.has(p + ".bias")) y = ggml_add(c, y, ggml_reshape_3d(c, m.get(p + ".bias"), 1, 1, OC));
    o.C = OC; o.H = (int)y->ne[1]; o.W = (int)y->ne[0];
    o.d = run_graph(m, c, {y}, { {gx, x.d.data()} })[0];
    ggml_free(c);
    return o;
}
static void bn(const Model& m, const std::string& p, Feat& x) {     // folded BN: per-channel scale/shift
    std::vector<float> sc = tensor_to_f32(m.get(p + ".scale")), sh = tensor_to_f32(m.get(p + ".shift"));
    long HW = (long)x.H * x.W;
    for (int ch = 0; ch < x.C; ++ch) { float a = sc[ch], b = sh[ch];
        for (long i = 0; i < HW; ++i) x.d[(size_t)ch*HW + i] = x.d[(size_t)ch*HW + i]*a + b; }
}
static void relu(Feat& x) { for (auto& v : x.d) if (v < 0) v = 0; }

static Feat interp(const Feat& x, int Ho, int Wo) {                 // bilinear, align_corners=True
    Feat o; o.C = x.C; o.H = Ho; o.W = Wo; o.d.resize((size_t)x.C * Ho * Wo);
    double sy = Ho > 1 ? (double)(x.H - 1) / (Ho - 1) : 0.0, sx = Wo > 1 ? (double)(x.W - 1) / (Wo - 1) : 0.0;
    for (int ho = 0; ho < Ho; ++ho) { double fy = ho * sy; int y0 = (int)fy; int y1 = std::min(y0+1, x.H-1); double ly = fy - y0;
        for (int wo = 0; wo < Wo; ++wo) { double fx = wo * sx; int x0 = (int)fx; int x1 = std::min(x0+1, x.W-1); double lx = fx - x0;
            for (int ch = 0; ch < x.C; ++ch) { const float* s = &x.d[(size_t)ch*x.H*x.W];
                double v = (1-ly)*(1-lx)*s[(size_t)y0*x.W+x0] + (1-ly)*lx*s[(size_t)y0*x.W+x1]
                         + ly*(1-lx)*s[(size_t)y1*x.W+x0] + ly*lx*s[(size_t)y1*x.W+x1];
                o.d[(size_t)ch*Ho*Wo + (size_t)ho*Wo + wo] = (float)v; } } }
    return o;
}
static Feat concat_ch(std::vector<const Feat*> fs) {               // along channel
    Feat o; o.H = fs[0]->H; o.W = fs[0]->W; o.C = 0;
    for (auto f : fs) o.C += f->C;
    o.d.resize((size_t)o.C * o.H * o.W); size_t off = 0;
    for (auto f : fs) { std::memcpy(&o.d[off], f->d.data(), f->d.size()*sizeof(float)); off += f->d.size(); }
    return o;
}
static Feat from_bb(const BBOut& b, int i) { Feat f; f.C=b.C[i]; f.H=b.H[i]; f.W=b.W[i]; f.d=b.f[i]; return f; }

// one DeformableConv2d + bn + relu (a _ASPPModuleDeformable)
static Feat deform_module(const Model& m, const std::string& p, const Feat& x, int gpu) {
    std::string ac = p + ".atrous_conv";
    int K = (int)m.get(ac + ".regular_conv.weight")->ne[0];
    Feat off = conv2d(m, ac + ".offset_conv", x, K/2);             // [2K^2,H,W]
    Feat mod = conv2d(m, ac + ".modulator_conv", x, K/2);         // [K^2,H,W]
    for (auto& v : mod.d) v = 2.0f / (1.0f + std::exp(-v));        // 2*sigmoid
    T* w = m.get(ac + ".regular_conv.weight"); int OC = (int)w->ne[3];
    std::vector<float> wt = tensor_to_f32(w);                      // [KW,KH,Cin,OC] == [OC,Cin,K,K] C-order
    Feat o; o.C = OC; o.H = x.H; o.W = x.W; o.d.resize((size_t)OC * x.H * x.W);
    deform_conv2d_run(x.d.data(), x.C, x.H, x.W, off.d.data(), mod.d.data(), wt.data(), nullptr, OC, K, o.d.data(), gpu);
    bn(m, p + ".bn", o); relu(o);
    return o;
}
static Feat global_pool_branch(const Model& m, const std::string& p, const Feat& x) {
    long HW = (long)x.H * x.W;
    std::vector<float> a(x.C); for (int ch = 0; ch < x.C; ++ch) { double s = 0; for (long i=0;i<HW;++i) s += x.d[(size_t)ch*HW+i]; a[ch] = (float)(s/HW); }
    T* w = m.get(p + ".1.weight"); int OC = (int)w->ne[3]; std::vector<float> wt = tensor_to_f32(w);  // [1,1,Cin,OC]
    Feat o; o.C = OC; o.H = 1; o.W = 1; o.d.assign(OC, 0.f);
    for (int oc = 0; oc < OC; ++oc) { double s = 0; for (int ic = 0; ic < x.C; ++ic) s += wt[(size_t)ic + (size_t)x.C*oc]*a[ic]; o.d[oc] = (float)s; }
    bn(m, p + ".2", o); relu(o);
    Feat r; r.C = OC; r.H = x.H; r.W = x.W; r.d.resize((size_t)OC * x.H * x.W);
    for (int oc = 0; oc < OC; ++oc) for (long i = 0; i < HW; ++i) r.d[(size_t)oc*HW + i] = o.d[oc];
    return r;
}
static Feat aspp_deformable(const Model& m, const std::string& p, const Feat& x, int gpu) {
    Feat x1 = deform_module(m, p + ".aspp1", x, gpu);
    Feat d0 = deform_module(m, p + ".aspp_deforms.0", x, gpu);
    Feat d1 = deform_module(m, p + ".aspp_deforms.1", x, gpu);
    Feat d2 = deform_module(m, p + ".aspp_deforms.2", x, gpu);
    Feat g  = global_pool_branch(m, p + ".global_avg_pool", x);
    Feat cat = concat_ch({&x1, &d0, &d1, &d2, &g});               // 1280
    Feat o = conv2d(m, p + ".conv1", cat, 0); bn(m, p + ".bn1", o); relu(o);
    return o;
}
static Feat basic_dec_blk(const Model& m, const std::string& p, const Feat& x, int gpu) {
    Feat o = conv2d(m, p + ".conv_in", x, 1); bn(m, p + ".bn_in", o); relu(o);
    o = aspp_deformable(m, p + ".dec_att", o, gpu);
    o = conv2d(m, p + ".conv_out", o, 1); bn(m, p + ".bn_out", o);
    return o;
}
static Feat simple_convs(const Model& m, const std::string& p, const Feat& x) {
    return conv2d(m, p + ".conv_out", conv2d(m, p + ".conv1", x, 1), 1);   // two 3x3, no act between
}
static Feat image2patches(const Feat& img, int Href, int Wref) {  // 'b c (hg h)(wg w)->b (c hg wg) h w'
    int hg = img.H / Href, wg = img.W / Wref;
    Feat o; o.C = img.C * hg * wg; o.H = Href; o.W = Wref; o.d.resize((size_t)o.C * Href * Wref);
    for (int c = 0; c < img.C; ++c) for (int ih = 0; ih < hg; ++ih) for (int iw = 0; iw < wg; ++iw) {
        int oc = c*hg*wg + ih*wg + iw;
        for (int y = 0; y < Href; ++y) for (int x = 0; x < Wref; ++x)
            o.d[(size_t)oc*Href*Wref + (size_t)y*Wref + x] =
                img.d[(size_t)c*img.H*img.W + (size_t)(ih*Href+y)*img.W + (iw*Wref+x)];
    }
    return o;
}
static void gdt_attn(const Model& m, int level, Feat& p, int gpu) { (void)gpu;
    std::string g = "decoder.gdt_convs_" + std::to_string(level);
    Feat h = conv2d(m, g + ".0", p, 1); bn(m, g + ".1", h); relu(h);   // ->16
    Feat at = conv2d(m, "decoder.gdt_convs_attn_" + std::to_string(level) + ".0", h, 0);  // 1x1 ->1
    for (auto& v : at.d) v = 1.0f / (1.0f + std::exp(-v));
    long HW = (long)p.H * p.W;
    for (int c = 0; c < p.C; ++c) for (long i = 0; i < HW; ++i) p.d[(size_t)c*HW + i] *= at.d[i];
}

std::vector<float> birefnet_matte(const Model& m, const std::vector<float>& chw1024, int gpu) {
    Feat img; img.C = 3; img.H = 1024; img.W = 1024; img.d = chw1024;
    // ---- backbone twice (mul_scl_ipt='cat') ----
    BBOut full = swin_backbone(m, chw1024, 1024);
    Feat img512 = interp(img, 512, 512);
    BBOut half = swin_backbone(m, img512.d, 512);
    Feat xs[4];
    for (int i = 0; i < 4; ++i) {
        Feat f = from_bb(full, i), h = from_bb(half, i);
        Feat hi = interp(h, f.H, f.W);
        xs[i] = concat_ch({&f, &hi});                 // doubled channels
    }
    // ---- cxt cat on x4 + squeeze ----
    Feat c1 = interp(xs[0], xs[3].H, xs[3].W), c2 = interp(xs[1], xs[3].H, xs[3].W), c3 = interp(xs[2], xs[3].H, xs[3].W);
    Feat x4c = concat_ch({&c1, &c2, &c3, &xs[3]});    // 5760@32
    Feat x4 = basic_dec_blk(m, "squeeze_module.0", x4c, gpu);   // 3072@32

    Feat x1 = xs[0], x2 = xs[1], x3 = xs[2];
    // ---- decoder ----
    auto ipt = [&](const std::string& blk, int ref, const Feat& add_to) {
        Feat pat = image2patches(img, ref, ref);
        return simple_convs(m, "decoder." + blk, pat);
    };
    // block4
    { Feat e = ipt("ipt_blk5", x4.H, x4); x4 = concat_ch({&x4, &e}); }   // 3456@32
    Feat p4 = basic_dec_blk(m, "decoder.decoder_block4", x4, gpu);       // 1536@32
    gdt_attn(m, 4, p4, gpu);
    Feat _p = interp(p4, x3.H, x3.W);                                    // 1536@64
    { Feat lat = conv2d(m, "decoder.lateral_block4.conv", x3, 0); for (size_t i=0;i<_p.d.size();++i) _p.d[i]+=lat.d[i]; }
    { Feat e = ipt("ipt_blk4", x3.H, _p); _p = concat_ch({&_p, &e}); }   // 1920@64
    Feat p3 = basic_dec_blk(m, "decoder.decoder_block3", _p, gpu);       // 768@64
    gdt_attn(m, 3, p3, gpu);
    _p = interp(p3, x2.H, x2.W);                                         // 768@128
    { Feat lat = conv2d(m, "decoder.lateral_block3.conv", x2, 0); for (size_t i=0;i<_p.d.size();++i) _p.d[i]+=lat.d[i]; }
    { Feat e = ipt("ipt_blk3", x2.H, _p); _p = concat_ch({&_p, &e}); }   // 960@128
    Feat p2 = basic_dec_blk(m, "decoder.decoder_block2", _p, gpu);       // 384@128
    gdt_attn(m, 2, p2, gpu);
    _p = interp(p2, x1.H, x1.W);                                         // 384@256
    { Feat lat = conv2d(m, "decoder.lateral_block2.conv", x1, 0); for (size_t i=0;i<_p.d.size();++i) _p.d[i]+=lat.d[i]; }
    { Feat e = ipt("ipt_blk2", x1.H, _p); _p = concat_ch({&_p, &e}); }   // 480@256
    _p = basic_dec_blk(m, "decoder.decoder_block1", _p, gpu);            // 192@256
    _p = interp(_p, 1024, 1024);                                         // 192@1024
    { Feat e = ipt("ipt_blk1", 1024, _p); _p = concat_ch({&_p, &e}); }   // 240@1024
    Feat outp = conv2d(m, "decoder.conv_out1.0", _p, 0);                 // 1@1024
    return outp.d;
}

} // namespace trellis
