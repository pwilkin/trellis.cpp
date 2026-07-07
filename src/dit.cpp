#include "dit.h"
#include "trellis_model.h"
#include "ggml.h"

#include <cmath>
#include <cstdlib>
#include <string>

namespace trellis {

using T = ggml_tensor;

static bool g_cast_f32 = false;   // set per build_dit_dense call
bool g_no_fa = false;             // --no-fa; set by trellis_run

static T* lin(ggml_context* c, const Model& m, const std::string& p, T* x) {
    T* w = m.get(p + ".weight");
    if (g_cast_f32 && w->type == GGML_TYPE_F16) w = ggml_cast(c, w, GGML_TYPE_F32);
    T* y = ggml_mul_mat(c, w, x);
    if (T* b = m.try_get(p + ".bias")) y = ggml_add(c, y, b);
    return y;
}

// LayerNorm over ne0. weight/bias optional (affine).
static T* layernorm(ggml_context* c, T* x, float eps, T* w = nullptr, T* b = nullptr) {
    x = ggml_norm(c, x, eps);
    if (w) x = ggml_mul(c, x, w);
    if (b) x = ggml_add(c, x, b);
    return x;
}

// MultiHeadRMSNorm: ggml_rms_norm(x) already == F.normalize(x)*sqrt(head_dim); then * gamma.
static T* rms_gamma(ggml_context* c, T* x, T* gamma, float eps) {
    x = ggml_rms_norm(c, x, eps);
    return ggml_mul(c, x, gamma);   // gamma cast to f32 by caller
}

// x: [head_dim, n_heads, L]; cos/sin: [1, head_dim/2, 1, L]. Interleaved-pair rotation.
static T* apply_rope(ggml_context* c, T* x, T* cos, T* sin) {
    const int64_t hd = x->ne[0], nh = x->ne[1], L = x->ne[2];
    const int64_t half = hd / 2;
    T* x5 = ggml_reshape_4d(c, x, 2, half, nh, L);              // [2, half, nh, L]
    T* x0 = ggml_cont(c, ggml_view_4d(c, x5, 1, half, nh, L, x5->nb[1], x5->nb[2], x5->nb[3], 0));
    T* x1 = ggml_cont(c, ggml_view_4d(c, x5, 1, half, nh, L, x5->nb[1], x5->nb[2], x5->nb[3], x5->nb[0]));
    T* r  = ggml_concat(c, ggml_scale(c, x1, -1.0f), x0, 0);    // [-x1, x0] along pair-elem
    T* out = ggml_add(c, ggml_mul(c, x5, cos), ggml_mul(c, r, sin));
    return ggml_reshape_3d(c, out, hd, nh, L);
}

// An FA padding mask [Lk_pad, Lq] (F16): 0 for real keys (< Lk_real), a large negative for the
// zero-padded tail. WITHOUT it, ggml's CUDA FlashAttention folds the (zero) padded keys into the
// softmax; on the >=1024-token HR flow that path NaNs a subset of queries (props, <1024 tokens, dodge
// it). WITH it the kernel masks/skips the padded KV tiles -> correct softmax, no NaN. Built once per
// flow (same N every block) and threaded into every attention; -30000 (not -inf) so 0*mask can't NaN.
static T* build_pad_mask(ggml_context* c, int64_t Lk_real, int64_t Lq) {
    const int64_t KQ = 256;
    const int64_t Lk_pad = ((Lk_real + KQ - 1) / KQ) * KQ;
    T* sh = ggml_arange(c, 0.5f - (float)Lk_real, (float)Lk_pad - (float)Lk_real + 0.5f, 1.0f); // [Lk_pad]
    T* col = ggml_scale(c, ggml_step(c, sh), -30000.0f);            // 0 (keep) | -30000 (mask) per key
    T* colh = ggml_cast(c, ggml_reshape_2d(c, col, Lk_pad, 1), GGML_TYPE_F16);
    return ggml_repeat(c, colh, ggml_new_tensor_2d(c, GGML_TYPE_F16, Lk_pad, Lq)); // [Lk_pad,Lq] F16, contig
}

// SDPA over heads. q:[hd,nh,Lq]  k,v:[hd,nh,Lk] -> [d_model, Lq].  `mask`: optional [Lk_pad,Lq] F16.
static T* sdpa(ggml_context* c, T* q, T* k, T* v, int d_model, T* mask = nullptr) {
    const float scale = 1.0f / std::sqrt((float)q->ne[0]);
    // FlashAttention: a fused, tiled SDPA that never materialises the [Lk,Lq,nh] score
    // matrix — O(N) memory instead of O(N^2). That score buffer is exactly what OOMs the
    // 1024 cascade (~18 GB single alloc); FA keeps it small so 1024 fits the 16 GB card.
    // ggml_flash_attn_ext wants [head_dim, n_tok, n_head, n_batch] with low-prec K/V.
    // Two HR-scale gotchas, both fatal to the 1024 cascade (NaN SLAT -> decoder collapse):
    //  (1) KV-length padding. ggml's CUDA FA reads keys in tiles of FATTN_KQ_STRIDE=256.
    //      With a null mask and an unpadded last tile (Lk % 256 != 0), the tail positions
    //      are uninitialised garbage that poison the softmax -> ~NaN. The LR flow (8.5k tok)
    //      happened to dodge it; the HR flow (52.7k tok) hits it ~70%. Fix: **zero-pad K/V's
    //      key dim up to a 256 multiple** — zero keys add exp(0-rowmax)~0 (numerically exact),
    //      remove the garbage tile, AND unlock the aligned (Lk%256==0) kernel path.
    //  (2) Use **BF16** (not F16) for K/V so HR activations can't overflow F16's +-65504 range;
    //      BF16 shares F32's exponent range and matches the reference's bf16 checkpoints.
    // (--no-fa forces the original soft_max path, for A/B + fallback.)
    const bool no_fa = g_no_fa;
    if (!no_fa) {
        const int64_t KQ_STRIDE = 256;
        auto prep_kv = [&](T* x) {                              // -> [hd, Lk_pad, nh] BF16
            T* p = ggml_cont(c, ggml_permute(c, x, 0, 2, 1, 3));   // [hd, Lk, nh] F32
            const int64_t pad = (KQ_STRIDE - (p->ne[1] % KQ_STRIDE)) % KQ_STRIDE;
            if (pad) p = ggml_pad(c, p, 0, (int)pad, 0, 0);        // zero-pad key dim
            return ggml_cast(c, p, GGML_TYPE_BF16);
        };
        T* qf = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));  // [hd, Lq, nh]
        if (qf->type != GGML_TYPE_F32) qf = ggml_cast(c, qf, GGML_TYPE_F32);
        T* kf = prep_kv(k);
        T* vf = prep_kv(v);
        T* out = ggml_flash_attn_ext(c, qf, kf, vf, mask, scale, 0.0f, 0.0f);  // [hd, nh, Lq]
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        return ggml_reshape_2d(c, out, d_model, out->ne[2]);   // [d_model, Lq]
    }
    T* q2 = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));       // [hd, Lq, nh]
    T* k2 = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));       // [hd, Lk, nh]
    T* kq = ggml_mul_mat(c, k2, q2);                            // [Lk, Lq, nh]
    kq = ggml_soft_max_ext(c, kq, nullptr, scale, 0.0f);
    T* v2 = ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3));       // [Lk, hd, nh]
    T* kqv = ggml_mul_mat(c, v2, kq);                           // [hd, Lq, nh]
    kqv = ggml_cont(c, ggml_permute(c, kqv, 0, 2, 1, 3));       // [hd, nh, Lq]
    return ggml_reshape_2d(c, kqv, d_model, kqv->ne[2]);        // [d_model, Lq]
}

static T* gamma32(ggml_context* c, const Model& m, const std::string& key) {
    T* g = m.get(key);
    return g->type == GGML_TYPE_F32 ? g : ggml_cast(c, g, GGML_TYPE_F32);
}

static T* self_attn(ggml_context* c, const Model& m, const std::string& pre, T* h,
                    T* cos, T* sin, const DiTParams& p, T* mask = nullptr) {
    const int hd = p.head_dim, nh = p.n_heads;
    const int64_t L = h->ne[1];
    T* qkv = lin(c, m, pre + ".to_qkv", h);                     // [3*d_model, L]
    qkv = ggml_reshape_4d(c, qkv, hd, nh, 3, L);
    auto pick = [&](int s) {
        T* t = ggml_view_4d(c, qkv, hd, nh, 1, L, qkv->nb[1], qkv->nb[2], qkv->nb[3], s * qkv->nb[2]);
        return ggml_reshape_3d(c, ggml_cont(c, t), hd, nh, L);
    };
    T* q = pick(0); T* k = pick(1); T* v = pick(2);
    q = rms_gamma(c, q, gamma32(c, m, pre + ".q_rms_norm.gamma"), p.rms_eps);
    k = rms_gamma(c, k, gamma32(c, m, pre + ".k_rms_norm.gamma"), p.rms_eps);
    q = apply_rope(c, q, cos, sin);
    k = apply_rope(c, k, cos, sin);
    return lin(c, m, pre + ".to_out", sdpa(c, q, k, v, p.d_model, mask));
}

static T* cross_attn(ggml_context* c, const Model& m, const std::string& pre, T* h, T* cond,
                     const DiTParams& p, T* mask = nullptr) {
    const int hd = p.head_dim, nh = p.n_heads;
    const int64_t L = h->ne[1], Lc = cond->ne[1];
    T* q = lin(c, m, pre + ".to_q", h);
    q = ggml_reshape_3d(c, q, hd, nh, L);
    T* kv = lin(c, m, pre + ".to_kv", cond);                    // [2*d_model, Lc]
    kv = ggml_reshape_4d(c, kv, hd, nh, 2, Lc);
    auto pick = [&](int s) {
        T* t = ggml_view_4d(c, kv, hd, nh, 1, Lc, kv->nb[1], kv->nb[2], kv->nb[3], s * kv->nb[2]);
        return ggml_reshape_3d(c, ggml_cont(c, t), hd, nh, Lc);
    };
    T* k = pick(0); T* v = pick(1);
    q = rms_gamma(c, q, gamma32(c, m, pre + ".q_rms_norm.gamma"), p.rms_eps);
    k = rms_gamma(c, k, gamma32(c, m, pre + ".k_rms_norm.gamma"), p.rms_eps);
    return lin(c, m, pre + ".to_out", sdpa(c, q, k, v, p.d_model, mask));
}

// x*(1+scale)+shift, scale/shift: [d_model] broadcast over L
static T* modulate(ggml_context* c, T* x, T* scale, T* shift) {
    return ggml_add(c, ggml_add(c, x, ggml_mul(c, x, scale)), shift);
}

static T* block(ggml_context* c, const Model& m, int i, T* h, T* mod, T* cond,
                T* cos, T* sin, const DiTParams& p, std::map<std::string, T*>* inter = nullptr,
                T* self_mask = nullptr, T* cross_mask = nullptr) {
    const std::string b = "blocks." + std::to_string(i);
    const int dm = p.d_model;
    auto dbg = [&](const char* n, T* t) { if (inter && i == 0) { (*inter)[n] = t; ggml_set_name(t, n); } return t; };
    T* mb = ggml_add(c, m.get(b + ".modulation"), mod);        // [6*d_model]
    auto ch = [&](int j) { return ggml_view_1d(c, mb, dm, (size_t)j * dm * ggml_element_size(mb)); };
    T* shift_msa = ch(0), *scale_msa = ch(1), *gate_msa = ch(2);
    T* shift_mlp = ch(3), *scale_mlp = ch(4), *gate_mlp = ch(5);

    T* hh = layernorm(c, h, p.ln_eps);
    hh = modulate(c, hh, scale_msa, shift_msa);
    hh = self_attn(c, m, b + ".self_attn", hh, cos, sin, p, self_mask);
    dbg("blk0_msa", hh);
    h = ggml_add(c, h, ggml_mul(c, hh, gate_msa));

    hh = layernorm(c, h, p.ln_eps, m.get(b + ".norm2.weight"), m.get(b + ".norm2.bias"));
    hh = cross_attn(c, m, b + ".cross_attn", hh, cond, p, cross_mask);
    dbg("blk0_cross", hh);
    h = ggml_add(c, h, hh);

    hh = layernorm(c, h, p.ln_eps);
    hh = modulate(c, hh, scale_mlp, shift_mlp);
    hh = lin(c, m, b + ".mlp.mlp.0", hh);
    hh = ggml_gelu(c, hh);                                      // GELU(approximate=tanh)
    hh = lin(c, m, b + ".mlp.mlp.2", hh);
    dbg("blk0_mlp", hh);
    h = ggml_add(c, h, ggml_mul(c, hh, gate_mlp));
    return h;
}

ggml_tensor* build_dit_dense(ggml_context* c, const Model& m, const DiTParams& p,
                             T* h0, T* tfreq, T* cond, T* cos, T* sin,
                             std::map<std::string, T*>* inter) {
    g_cast_f32 = p.cast_f32;
    auto keep = [&](const char* n, T* t) { if (inter) (*inter)[n] = t; ggml_set_name(t, n); return t; };

    T* h = lin(c, m, "input_layer", h0);                       // [d_model, L]
    keep("after_input_layer", h);

    T* te = lin(c, m, "t_embedder.mlp.0", tfreq);
    te = ggml_silu(c, te);
    te = lin(c, m, "t_embedder.mlp.2", te);                    // [d_model]
    T* mod = lin(c, m, "adaLN_modulation.1", ggml_silu(c, te));// [6*d_model]
    keep("t_emb_mod", mod);

    // Padding masks for the two attentions, built ONCE (token counts are fixed across blocks) and
    // shared by every block — they tell the CUDA FlashAttention to exclude the zero-padded key tiles
    // (else the >=1024-token flow NaNs a subset of queries). self: Lk=L (the latent); cross: Lk=Lc.
    T* self_mask  = build_pad_mask(c, h0->ne[1], h0->ne[1]);
    T* cross_mask = build_pad_mask(c, cond->ne[1], h0->ne[1]);
    for (int i = 0; i < p.n_blocks; ++i) {
        h = block(c, m, i, h, mod, cond, cos, sin, p, inter, self_mask, cross_mask);
        if (i == 0) keep("after_block0", h);
        if (i == 1) keep("after_block1", h);
        if (i == p.n_blocks - 1) keep("after_block29", h);
    }
    h = layernorm(c, h, p.final_ln_eps);
    keep("prefinal", h);
    h = lin(c, m, "out_layer", h);                             // [out_ch, L]
    keep("output", h);
    return h;
}

} // namespace trellis
