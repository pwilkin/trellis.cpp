#include "dit.h"
#include "trellis_model.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

namespace trellis {

using T = ggml_tensor;

static bool g_cast_f32 = false;   // set per build_dit_dense call

// Budget for one query chunk's [Lk, nq, nh] score tile in the exact (non-FA) SDPA path.
// Bounds the peak regardless of Lq, which is what made FA necessary in the first place.
static constexpr int64_t kAttnChunkBytes = 1024ll * 1024 * 1024;
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
// The rotated pair [x_even*cos - x_odd*sin, x_odd*cos + x_even*sin] is scattered back into the
// output with two ggml_set_rows (single flat-grid dispatch each) rather than ggml_concat: ggml's
// concat launches one kernel per ne[3] slice, so the old concat over the [2,half,nh,L] pair tensor
// fired L (token-count) dispatches per call -> ~30M concat launches over a flow. q/k are F32 here
// (mul_mat output), which ggml_set_rows requires. Even/odd row indices come from ggml_arange.
static T* apply_rope(ggml_context* c, T* x, T* cos, T* sin) {
    const int64_t hd = x->ne[0], nh = x->ne[1], L = x->ne[2];
    const int64_t half = hd / 2;
    T* x5 = ggml_reshape_4d(c, x, 2, half, nh, L);              // [2, half, nh, L]
    T* x0 = ggml_cont(c, ggml_view_4d(c, x5, 1, half, nh, L, x5->nb[1], x5->nb[2], x5->nb[3], 0));         // even
    T* x1 = ggml_cont(c, ggml_view_4d(c, x5, 1, half, nh, L, x5->nb[1], x5->nb[2], x5->nb[3], x5->nb[0])); // odd
    T* ev = ggml_sub(c, ggml_mul(c, x0, cos), ggml_mul(c, x1, sin));   // [1,half,nh,L] rotated even
    T* od = ggml_add(c, ggml_mul(c, x1, cos), ggml_mul(c, x0, sin));   // [1,half,nh,L] rotated odd
    T* ce = ggml_cast(c, ggml_arange(c, 0.0f, (float)hd, 2.0f), GGML_TYPE_I32);  // [0,2,..,hd-2]
    T* co = ggml_cast(c, ggml_arange(c, 1.0f, (float)hd, 2.0f), GGML_TYPE_I32);  // [1,3,..,hd-1]
    T* out = ggml_scale(c, ggml_reshape_4d(c, x, 1, hd, nh, L), 0.0f);  // allocated [1,hd,nh,L] scratch
    out = ggml_set_rows(c, out, ev, ce);
    out = ggml_set_rows(c, out, od, co);
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
    // The QUERY dim must be padded to GGML_KQ_MASK_PAD too, not just the key dim: CUDA's
    // flash_attn_mask_to_KV_max walks the mask in query tiles (mask += jt*ncols1*s31) to find
    // which KV tiles a tile can skip, so an unpadded Lq makes the last tile read past the end.
    // The garbage it finds there corrupts that tile's KV_max -> wrong keys survive the softmax.
    // Every row here is identical (the mask depends only on key index), so the extra rows are free.
    constexpr int64_t KQ_MASK_PAD = 64;   // llama.cpp's GGML_KQ_MASK_PAD; not exported by ggml
    const int64_t Lq_pad = ((Lq + KQ_MASK_PAD - 1) / KQ_MASK_PAD) * KQ_MASK_PAD;
    return ggml_repeat(c, colh, ggml_new_tensor_2d(c, GGML_TYPE_F16, Lk_pad, Lq_pad)); // [Lk_pad,Lq_pad] F16
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
    // set_prec(GGML_PREC_F32) below is LOAD-BEARING, and only became so in ggml 2d6d0b0c.
    // Before it, only fattn-wmma-f16.cu read ggml_flash_attn_ext_get_prec; TILE (what this
    // shape gets on gfx1151), VEC and MMA silently ignored it and accumulated VKQ in half2.
    // Summing ~15k weighted V terms in F16 stagnates -- small addends round away once the
    // running sum is large -- biasing low, growing with KV tile count, and it wrecked the
    // latent: mean -0.8048 vs the reference's -0.0353+-0.0064 (~120 sigma out, n=5 a side).
    // With the fix TILE honours prec and the bias drops to -0.00235, matching the exact path.
    // Do NOT assume a stock ggml has this: check with GGML_FA_DEBUG=1 (prints kernel + prec)
    // and trellis-test-shape-flow, whose `output` mean-delta is the canary (-0.0024 good,
    // -0.017 = prec is being ignored again). --no-fa falls back to the exact chunked path.
    const bool no_fa = g_no_fa;
    if (!no_fa) {
        // TRELLIS_FA_FAST=1: F16 K/V + default (F16) accumulation — the shapes
        // the Vulkan coopmat FA shaders are specialized for. A/B only: F16 K/V
        // can overflow on HR activations (the reason BF16+F32 is the default).
        static const bool fa_fast = std::getenv("TRELLIS_FA_FAST") != nullptr;
        const int64_t KQ_STRIDE = 256;
        auto prep_kv = [&](T* x) {                              // -> [hd, Lk_pad, nh] BF16
            T* p = ggml_cont(c, ggml_permute(c, x, 0, 2, 1, 3));   // [hd, Lk, nh] F32
            const int64_t pad = (KQ_STRIDE - (p->ne[1] % KQ_STRIDE)) % KQ_STRIDE;
            if (pad) p = ggml_pad(c, p, 0, (int)pad, 0, 0);        // zero-pad key dim
            return ggml_cast(c, p, fa_fast ? GGML_TYPE_F16 : GGML_TYPE_BF16);
        };
        T* qf = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));  // [hd, Lq, nh]
        if (qf->type != GGML_TYPE_F32) qf = ggml_cast(c, qf, GGML_TYPE_F32);
        T* kf = prep_kv(k);
        T* vf = prep_kv(v);
        // TRELLIS_FA_NOMASK=1: drop the mask. If the output is UNCHANGED, the mask is being
        // ignored and the zero-padded keys are diluting the softmax (exp(0-rowmax) is only
        // negligible when rowmax >> 0), which shrinks every output toward zero.
        static const bool fa_nomask = std::getenv("TRELLIS_FA_NOMASK") != nullptr;
        T* out = ggml_flash_attn_ext(c, qf, kf, vf, fa_nomask ? nullptr : mask, scale, 0.0f, 0.0f);  // [hd, nh, Lq]
        if (!fa_fast) ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        return ggml_reshape_2d(c, out, d_model, out->ne[2]);   // [d_model, Lq]
    }
    // Exact SDPA, chunked over QUERIES. The whole reason FA exists here is the [Lk, Lq, nh]
    // score matrix -- at the HR flow that is 15104*15006*12*4 = 10.9 TB, so it cannot be
    // materialised whole. But attention is independent per query: a query range needs no halo
    // (it reads all of K/V, which stay whole), so the scores can be built a slice at a time and
    // the peak is [Lk, nq, nh] instead. K/V are the UNPADDED originals, so no pad mask is
    // needed and none of FA's tile machinery is involved -- this is the path the golden-tensor
    // test matches to 8e-5 on the output mean.
    T* q2 = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));       // [hd, Lq, nh]
    T* k2 = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));       // [hd, Lk, nh]
    T* v2 = ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3));       // [Lk, hd, nh]
    const int64_t hd = q2->ne[0], Lq = q2->ne[1], nh = q2->ne[2], Lk = k2->ne[1];

    int64_t budget = kAttnChunkBytes;
    if (const char* e = getenv("TRELLIS_ATTN_CHUNK_MB")) budget = atoll(e) * 1024 * 1024;
    const int64_t per_q = Lk * nh * 4;                          // one query's score column
    int64_t nq = std::max<int64_t>(1, budget / std::max<int64_t>(per_q, 1));
    // Floor on the chunk size: each chunk adds ~8 nodes and a whole 30-block DiT with two
    // attentions per block is built as ONE graph, so an unbounded chunk count exhausts the
    // ggml context (ggml_new_object: not enough space). Hitting this cap costs memory, not
    // correctness -- query chunking is bit-exact either way (no reduction crosses queries).
    constexpr int64_t kMaxAttnChunks = 32;
    if (nq * kMaxAttnChunks < Lq) nq = (Lq + kMaxAttnChunks - 1) / kMaxAttnChunks;
    if (nq >= Lq) nq = Lq;                                      // small attn: single chunk, no concat

    T* out = nullptr;
    for (int64_t q0 = 0; q0 < Lq; q0 += nq) {
        const int64_t n = std::min(nq, Lq - q0);
        T* qc = (n == Lq) ? q2 : ggml_cont(c, ggml_view_3d(c, q2, hd, n, nh,
                                q2->nb[1], q2->nb[2], (size_t)q0 * q2->nb[1]));   // [hd, n, nh]
        T* kq = ggml_mul_mat(c, k2, qc);                        // [Lk, n, nh]
        kq = ggml_soft_max_ext(c, kq, nullptr, scale, 0.0f);
        T* kqv = ggml_mul_mat(c, v2, kq);                       // [hd, n, nh]
        kqv = ggml_cont(c, ggml_permute(c, kqv, 0, 2, 1, 3));   // [hd, nh, n]
        T* o = ggml_reshape_2d(c, kqv, d_model, n);             // [d_model, n]
        out = out ? ggml_concat(c, out, o, 1) : o;
    }
    return out;                                                 // [d_model, Lq]
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
