#include "flow_runner.h"
#include "trellis_model.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>

namespace trellis {

static void timestep_embedding(float t, std::vector<float>& out) {
    out.resize(256);
    for (int j = 0; j < 128; ++j) {
        float f = std::exp(-std::log(10000.f) * j / 128.f);
        out[j] = std::cos(t * f);
        out[128 + j] = std::sin(t * f);
    }
}

DitRunner::DitRunner(const Model& m, const DiTParams& p, int N, int n_cond,
                     const std::vector<float>& rcos, const std::vector<float>& rsin)
    : m_(m), p_(p), N_(N), Lc_(n_cond) {
    const int half = p_.head_dim / 2;
    size_t meta = ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(32768, false) + (1 << 20);
    ctx_ = ggml_init({ meta, nullptr, true });
    gh0_  = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, p_.in_ch, N_);   ggml_set_input(gh0_);
    gtf_  = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, 256);            ggml_set_input(gtf_);
    gcond_= ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, p_.d_cond, Lc_); ggml_set_input(gcond_);
    gcos_ = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, 1, half, 1, N_); ggml_set_input(gcos_);
    gsin_ = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, 1, half, 1, N_); ggml_set_input(gsin_);
    dbg_nan_ = std::getenv("TRELLIS_DBG_NAN") != nullptr;
    gout_ = build_dit_dense(ctx_, m_, p_, gh0_, gtf_, gcond_, gcos_, gsin_, dbg_nan_ ? &inter_ : nullptr);
    g_ = ggml_new_graph_custom(ctx_, 32768, false);
    ggml_build_forward_expand(g_, gout_);
    ggml_set_output(gout_);
    if (dbg_nan_) for (auto& [nm, t] : inter_) { ggml_build_forward_expand(g_, t); ggml_set_output(t); }
    alloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m_.backend));
    if (!ggml_gallocr_alloc_graph(alloc_, g_)) throw std::runtime_error("DitRunner: alloc failed");
    rcos_ = rcos; rsin_ = rsin;   // keep; re-upload each forward (gallocr reuses input buffers across runs)
}

DitRunner::~DitRunner() {
    if (alloc_) ggml_gallocr_free(alloc_);
    if (ctx_)   ggml_free(ctx_);
}

std::vector<float> DitRunner::forward(const std::vector<float>& xt, float t_scaled, const float* cond) {
    std::vector<float> tf; timestep_embedding(t_scaled, tf);
    ggml_backend_tensor_set(gh0_,  xt.data(), 0, xt.size() * 4);
    ggml_backend_tensor_set(gtf_,  tf.data(), 0, tf.size() * 4);
    ggml_backend_tensor_set(gcond_, cond,     0, (size_t)p_.d_cond * Lc_ * 4);
    ggml_backend_tensor_set(gcos_, rcos_.data(), 0, rcos_.size() * 4);   // re-upload (buffers reused across runs)
    ggml_backend_tensor_set(gsin_, rsin_.data(), 0, rsin_.size() * 4);
    if (ggml_backend_graph_compute(m_.backend, g_) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("DitRunner: compute failed");
    std::vector<float> outv = tensor_to_f32(gout_);
    size_t out_bad = 0; for (float x : outv) if (!std::isfinite(x)) out_bad++;
    // Dump the per-layer breakdown for the FIRST forward whose OUTPUT goes NaN (the failing low-t
    // step), not just the very first forward (which is clean) — that's where to look for the cause.
    if (dbg_nan_ && !dbg_done_ && out_bad > 0) {
        dbg_done_ = true;
        fprintf(stderr, "      [dit-nan] *** first NaN forward: t_scaled=%.2f  out_nan=%zu/%zu ***\n",
                t_scaled, out_bad, outv.size());
        const char* order[] = { "after_input_layer", "after_block0", "after_block1",
                                "blk0_msa", "blk0_cross", "blk0_mlp",
                                "after_block29", "prefinal", "output" };
        for (const char* nm : order) {
            auto it = inter_.find(nm); if (it == inter_.end()) continue;
            std::vector<float> v = tensor_to_f32(it->second);
            size_t bad = 0; double amax = 0; for (float x : v) {
                if (std::isnan(x) || std::isinf(x)) bad++;
                else if (std::fabs(x) > amax) amax = std::fabs(x); }
            fprintf(stderr, "      [dit-nan] %-18s nan/inf=%zu/%zu  max|finite|=%.2f\n",
                    nm, bad, v.size(), amax);
        }
    }
    return outv;
}

// 3D interleaved-pair RoPE cos/sin tables: data[token*half + pair].
static void fill_rope(const DiTParams& p, int N, const std::function<void(int,int&,int&,int&)>& coord,
                      std::vector<float>& rcos, std::vector<float>& rsin) {
    const int half = p.head_dim / 2, fd = half / 3;     // 64, 21
    std::vector<float> freqs(fd);
    for (int j = 0; j < fd; ++j) freqs[j] = 1.0f / std::pow(10000.f, (float)j / fd);
    rcos.assign((size_t)N * half, 0.f); rsin.assign((size_t)N * half, 0.f);
    for (int tok = 0; tok < N; ++tok) {
        int cx, cy, cz; coord(tok, cx, cy, cz);
        for (int pp = 0; pp < half; ++pp) {
            float ang = 0;
            if (pp < fd) ang = cx * freqs[pp];
            else if (pp < 2*fd) ang = cy * freqs[pp - fd];
            else if (pp < 3*fd) ang = cz * freqs[pp - 2*fd];
            rcos[(size_t)tok * half + pp] = std::cos(ang);
            rsin[(size_t)tok * half + pp] = std::sin(ang);
        }
    }
}

DitRunner* make_dense_runner(const Model& m, const DiTParams& p, int R, int n_cond) {
    std::vector<float> rcos, rsin;
    fill_rope(p, R*R*R, [R](int tok, int& cx, int& cy, int& cz) {
        cx = tok / (R*R); cy = (tok / R) % R; cz = tok % R; }, rcos, rsin);
    return new DitRunner(m, p, R*R*R, n_cond, rcos, rsin);
}

DitRunner* make_sparse_runner(const Model& m, const DiTParams& p,
                              const std::vector<std::array<int,3>>& coords, int n_cond) {
    std::vector<float> rcos, rsin;
    fill_rope(p, (int)coords.size(), [&coords](int tok, int& cx, int& cy, int& cz) {
        cx = coords[tok][0]; cy = coords[tok][1]; cz = coords[tok][2]; }, rcos, rsin);
    return new DitRunner(m, p, (int)coords.size(), n_cond, rcos, rsin);
}

std::vector<float> sample_flow(const FlowFwd& fwd, std::vector<float> sample,
                               const float* cond, const float* neg_cond, const SamplerParams& sp,
                               std::vector<std::vector<float>>* trace) {
    const float sm = sp.sigma_min;
    const size_t Nst = sample.size();
    std::vector<float> ts(sp.steps + 1);
    for (int i = 0; i <= sp.steps; ++i) {
        float t = 1.0f - (float)i / sp.steps;
        ts[i] = sp.rescale_t * t / (1.0f + (sp.rescale_t - 1.0f) * t);
    }
    std::vector<float> pos, neg, pred(Nst);
    static const bool dbg_step = std::getenv("TRELLIS_DBG_STEP") != nullptr;
    static const bool no_fix   = std::getenv("TRELLIS_NOFIX") != nullptr;  // robustness guards ON by default
    auto fstats = [](const std::vector<float>& v, size_t& bad, double& mx) {
        bad = 0; mx = 0; for (float x : v) { if (!std::isfinite(x)) bad++; else if (std::fabs(x) > mx) mx = std::fabs(x); }
    };
    for (int i = 0; i < sp.steps; ++i) {
        const float t = ts[i], tprev = ts[i + 1];
        const float gs = (sp.gi0 <= t && t <= sp.gi1) ? sp.guidance_strength : 1.0f;
        const float tscaled = 1000.0f * t;
        if (gs == 1.0f) {
            pred = fwd(sample, tscaled, cond);
        } else if (gs == 0.0f) {
            pred = fwd(sample, tscaled, neg_cond);
        } else {
            pos = fwd(sample, tscaled, cond);
            neg = fwd(sample, tscaled, neg_cond);
            for (size_t k = 0; k < Nst; ++k) pred[k] = gs * pos[k] + (1 - gs) * neg[k];
            if (sp.guidance_rescale > 0.0f) {
                const float a = 1 - sm, b = sm + (1 - sm) * t;
                double mp = 0, mc = 0;
                std::vector<float> x0p(Nst), x0c(Nst);
                for (size_t k = 0; k < Nst; ++k) { x0p[k] = a*sample[k] - b*pos[k]; x0c[k] = a*sample[k] - b*pred[k]; mp += x0p[k]; mc += x0c[k]; }
                mp /= Nst; mc /= Nst;
                double vp = 0, vc = 0;
                for (size_t k = 0; k < Nst; ++k) { vp += (x0p[k]-mp)*(x0p[k]-mp); vc += (x0c[k]-mc)*(x0c[k]-mc); }
                float ratio = vc > 0 ? (float)(std::sqrt(vp/(Nst-1)) / std::sqrt(vc/(Nst-1))) : 1.0f;
                // OOD inputs (e.g. a thin figure at HR) can make vc tiny -> ratio explodes -> the
                // rescaled velocity blows the latent past representable range over the 12 steps ->
                // all-NaN SLAT. Clamp ratio to a sane band: it sits at ~1.0 for in-distribution
                // props (a no-op there), and only bites on the pathological tail. (TRELLIS_NOFIX=1
                // restores the raw behaviour for A/B.)
                if (!no_fix) { if (!std::isfinite(ratio)) ratio = 1.0f; ratio = fminf(fmaxf(ratio, 0.2f), 5.0f); }
                float gr = sp.guidance_rescale;
                for (size_t k = 0; k < Nst; ++k) { float x0r = x0c[k]*ratio; float x0 = gr*x0r + (1-gr)*x0c[k]; pred[k] = (a*sample[k] - x0) / b; }
            }
        }
        // Safety net: never integrate a non-finite velocity (one poisoned tap would spread to the
        // whole latent on the next attention). A no-op when everything is finite.
        if (!no_fix) for (size_t k = 0; k < Nst; ++k) if (!std::isfinite(pred[k])) pred[k] = 0.0f;
        for (size_t k = 0; k < Nst; ++k) sample[k] -= (t - tprev) * pred[k];
        if (dbg_step) { size_t pb, sb; double pm, sm2; fstats(pred, pb, pm); fstats(sample, sb, sm2);
            fprintf(stderr, "      [flow-step %2d] t=%.3f gs=%.1f  pred[nan=%zu max=%.3g]  sample[nan=%zu max=%.3g]\n", i, t, gs, pb, pm, sb, sm2); }
        if (trace) trace->push_back(sample);
    }
    return sample;
}

} // namespace trellis
