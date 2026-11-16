#include "flow_runner.h"
#include "trellis_model.h"
#include "trellis_debug.h"
#include "trellis_sched.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace trellis {

static bool stdout_is_tty() {
#if defined(_WIN32)
    return _isatty(_fileno(stdout)) != 0;
#else
    return isatty(fileno(stdout)) != 0;
#endif
}

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
    // Query-chunked FlashAttention creates several small mask/FA nodes per attention instead
    // of one giant quadratic mask. Give the graph generous HOST metadata headroom; this does
    // not reserve an equivalent amount of VRAM.
    static constexpr size_t kDitGraphNodes = 65536;
    size_t meta = ggml_tensor_overhead() * kDitGraphNodes +
                  ggml_graph_overhead_custom(kDitGraphNodes, false) + (1 << 20);
    ctx_ = ggml_init({ meta, nullptr, true });
    gh0_  = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, p_.in_ch, N_);   ggml_set_input(gh0_);
    gtf_  = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, 256);            ggml_set_input(gtf_);
    gcond_= ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, p_.d_cond, Lc_); ggml_set_input(gcond_);
    if (p_.proj_mode) {
        if (p_.proj_ch <= 0) throw std::runtime_error("DitRunner: proj mode needs proj_ch");
        gproj_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, p_.proj_ch, N_); ggml_set_input(gproj_);
    }
    gcos_ = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, 1, half, 1, N_); ggml_set_input(gcos_);
    gsin_ = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, 1, half, 1, N_); ggml_set_input(gsin_);
    dbg_nan_ = std::getenv("TRELLIS_DBG_NAN") != nullptr;
    {
        ScopedTimer _t("dit.build_graph");
        gout_ = build_dit_dense(ctx_, m_, p_, gh0_, gtf_, gcond_, gproj_, gcos_, gsin_, dbg_nan_ ? &inter_ : nullptr);
        g_ = ggml_new_graph_custom(ctx_, kDitGraphNodes, false);
        ggml_build_forward_expand(g_, gout_);
        ggml_set_output(gout_);
        if (dbg_nan_) for (auto& [nm, t] : inter_) { ggml_build_forward_expand(g_, t); ggml_set_output(t); }
    }
    log_graph("dit.graph", g_);
    {
        ScopedTimer _t("dit.alloc_graph");
        exec_ = std::make_unique<GraphExec>(m_);
        if (!exec_->alloc(g_)) {
            exec_.reset();
            ggml_free(ctx_); ctx_ = nullptr;
            throw std::runtime_error("DitRunner: alloc failed");
        }
    }
    if (getenv("TRELLIS_DBG_ALLOC"))
        fprintf(stderr, "      [dit-alloc] N=%d nodes=%d buffer = %.2f GB\n",
                N_, ggml_graph_n_nodes(g_), exec_->buffer_size() / 1e9);
    rcos_ = rcos; rsin_ = rsin;   // keep; re-upload each forward (gallocr reuses input buffers across runs)
}

DitRunner::~DitRunner() {
    exec_.reset();
    if (ctx_)   ggml_free(ctx_);
}

std::vector<float> DitRunner::forward(const std::vector<float>& xt, float t_scaled, const FlowCond& c) {
    std::vector<float> tf; timestep_embedding(t_scaled, tf);
    ggml_backend_tensor_set(gh0_,  xt.data(), 0, xt.size() * 4);
    ggml_backend_tensor_set(gtf_,  tf.data(), 0, tf.size() * 4);
    ggml_backend_tensor_set(gcond_, c.cond,   0, (size_t)p_.d_cond * Lc_ * 4);
    if (gproj_) {
        const size_t nb = (size_t)p_.proj_ch * N_ * 4;
        // Re-written every forward: gallocr may hand the same buffer to both CFG branches.
        if (c.proj) ggml_backend_tensor_set(gproj_, c.proj, 0, nb);
        else        ggml_backend_tensor_memset(gproj_, 0, 0, nb);
    }
    ggml_backend_tensor_set(gcos_, rcos_.data(), 0, rcos_.size() * 4);   // re-upload (buffers reused across runs)
    ggml_backend_tensor_set(gsin_, rsin_.data(), 0, rsin_.size() * 4);
    if (exec_->compute(g_, "dit.forward") != GGML_STATUS_SUCCESS)
        throw std::runtime_error("DitRunner: compute failed");
    std::vector<float> outv;
    { ScopedTimer _t("dit.readback"); outv = tensor_to_f32(gout_); }
    size_t out_bad = 0; for (float x : outv) if (!std::isfinite(x)) out_bad++;
    // Dump the per-layer breakdown for the FIRST forward whose OUTPUT goes NaN (the failing low-t
    // step), not just the very first forward (which is clean) — that's where to look for the cause.
    if (verbose()) {
        double mn = 0, mx = 0, sum = 0, sq = 0;
        size_t bad = 0;
        for (size_t i = 0; i < outv.size(); ++i) {
            const float x = outv[i];
            if (!std::isfinite(x)) { bad++; continue; }
            if (i == 0 || x < mn) mn = x;
            if (i == 0 || x > mx) mx = x;
            sum += x; sq += (double) x * x;
        }
        const double n    = (double) (outv.size() - bad);
        const double mean = n > 0 ? sum / n : 0.0;
        const double std  = n > 0 ? std::sqrt(std::max(0.0, sq / n - mean * mean)) : 0.0;
        fprintf(stderr, "      [dit-fwd %2d] t=%.3f mean=%.4f std=%.4f min=%.3f max=%.3f nan/inf=%zu"
                        "  out=%p buf=%s\n",
                n_fwd_++, t_scaled, mean, std, mn, mx, bad,
                (void*) gout_->data,
                gout_->buffer ? ggml_backend_buffer_name(gout_->buffer) : "(none)");
    }

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
    // A forward with NO finite output is a broken backend, not a stray NaN: the sampler's
    // safety net would zero every velocity and hand the decoder the initial noise, which then
    // fails stages later with nothing pointing back here (issue #50). Stop at the source.
    if (!outv.empty() && out_bad == outv.size() && !std::getenv("TRELLIS_NOFIX")) {
        const char* be = m_.backend ? ggml_backend_name(m_.backend) : "?";
        throw std::runtime_error(std::string("flow: every output of this forward is NaN/Inf on ") + be +
            " -- the model is miscomputing on this device. Run with --verbose and TRELLIS_DBG_NAN=1"
            " to see the first bad layer, and try --backend CPU to confirm.");
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

// One guided velocity: guidance interval already folded into `gs`, then CFG and CFG-rescale,
// exactly as the reference ClassifierFreeGuidanceSamplerMixin. `neg_cached` (when non-null) is
// the negative branch's velocity at this (x_t, t), reused instead of running a forward;
// `neg_out` (when non-null) receives the negative velocity this call computed or reused.
// The arithmetic order is the historical sample_flow's -- the N=1 guarantee depends on it.
static void guided_velocity(const FlowFwd& fwd, const std::vector<float>& sample, float t, float gs,
                            const FlowCond& cond, const FlowCond& neg_cond, const SamplerParams& sp,
                            const std::vector<float>* neg_cached, std::vector<float>& pred,
                            std::vector<float>* neg_out, int& n_fwd) {
    static const bool no_fix = std::getenv("TRELLIS_NOFIX") != nullptr;
    const float sm = sp.sigma_min;
    const size_t Nst = sample.size();
    const float tscaled = 1000.0f * t;
    if (gs == 1.0f) {
        pred = fwd(sample, tscaled, cond);
        ++n_fwd;
        return;
    }
    if (gs == 0.0f) {
        if (neg_cached) pred = *neg_cached;
        else { pred = fwd(sample, tscaled, neg_cond); ++n_fwd; }
        if (neg_out && neg_out != neg_cached) *neg_out = pred;
        return;
    }
    std::vector<float> pos = fwd(sample, tscaled, cond);
    ++n_fwd;
    std::vector<float> neg_local;
    const std::vector<float>* negp = neg_cached;
    if (!negp) { neg_local = fwd(sample, tscaled, neg_cond); ++n_fwd; negp = &neg_local; }
    const std::vector<float>& neg = *negp;
    pred.resize(Nst);
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
    if (neg_out && neg_out != neg_cached) {
        if (negp == &neg_local) *neg_out = std::move(neg_local);
        else *neg_out = neg;
    }
}

// The Euler loop shared by the single- and multi-view samplers: timestep schedule, guidance
// interval, safety net, progress, trace. `velocity` fills `pred` for one step.
using StepVelocity = std::function<void(const std::vector<float>& sample, float t, float gs,
                                        std::vector<float>& pred, int& n_fwd)>;
static std::vector<float> euler_loop(std::vector<float> sample, const SamplerParams& sp,
                                     std::vector<std::vector<float>>* trace, int n_views,
                                     const StepVelocity& velocity) {
    const size_t Nst = sample.size();
    std::vector<float> ts(sp.steps + 1);
    for (int i = 0; i <= sp.steps; ++i) {
        float t = 1.0f - (float)i / sp.steps;
        ts[i] = sp.rescale_t * t / (1.0f + (sp.rescale_t - 1.0f) * t);
    }
    std::vector<float> pred(Nst);
    static const bool dbg_step = std::getenv("TRELLIS_DBG_STEP") != nullptr;
    static const bool no_fix   = std::getenv("TRELLIS_NOFIX") != nullptr;  // robustness guards ON by default
    const auto tflow0 = std::chrono::steady_clock::now();
    int n_fwd = 0;
    auto fstats = [](const std::vector<float>& v, size_t& bad, double& mx) {
        bad = 0; mx = 0; for (float x : v) { if (!std::isfinite(x)) bad++; else if (std::fabs(x) > mx) mx = std::fabs(x); }
    };
    // Progress. The 1024 cascade's HR pass is ~16 min of silence otherwise, which reads as
    // a hang. On a TTY redraw one line; when redirected to a log, emit a line per step (12
    // steps, so it stays readable). ETA from the mean step so far -- steps are near-uniform
    // except where the guidance interval drops a forward, so it settles after step 2.
    const bool tty = stdout_is_tty();
    auto progress = [&](int done) {
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - tflow0).count();
        char bar[21];
        const int fill = sp.steps ? done * 20 / sp.steps : 20;
        for (int k = 0; k < 20; ++k) bar[k] = k < fill ? '#' : '.';
        bar[20] = 0;
        char eta[32];
        if (done) snprintf(eta, sizeof eta, "~%.0fs left", el / done * (sp.steps - done));
        else      snprintf(eta, sizeof eta, "starting");     // no rate yet -- don't invent an ETA
        printf("%s      [flow] [%s] %2d/%d  %5.1fs  %-12s%s",
               tty ? "\r" : "", bar, done, sp.steps, el, eta, tty ? "" : "\n");
        fflush(stdout);
    };
    progress(0);
    for (int i = 0; i < sp.steps; ++i) {
        const float t = ts[i], tprev = ts[i + 1];
        const float gs = (sp.gi0 <= t && t <= sp.gi1) ? sp.guidance_strength : 1.0f;
        velocity(sample, t, gs, pred, n_fwd);
        // Safety net: never integrate a non-finite velocity (one poisoned tap would spread to the
        // whole latent on the next attention). A no-op when everything is finite.
        if (!no_fix) for (size_t k = 0; k < Nst; ++k) if (!std::isfinite(pred[k])) pred[k] = 0.0f;
        for (size_t k = 0; k < Nst; ++k) sample[k] -= (t - tprev) * pred[k];
        progress(i + 1);
        if (dbg_step) { size_t pb, sb; double pm, sm2; fstats(pred, pb, pm); fstats(sample, sb, sm2);
            if (tty) printf("\n");
            fprintf(stderr, "      [flow-step %2d] t=%.3f gs=%.1f  pred[nan=%zu max=%.3g]  sample[nan=%zu max=%.3g]\n", i, t, gs, pb, pm, sb, sm2); }
        if (trace) trace->push_back(sample);
    }
    if (tty) printf("\n");
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - tflow0).count();
    if (n_views > 1) printf("      [flow] %d steps, %d forwards, %d views, %.1fs\n", sp.steps, n_fwd, n_views, secs);
    else             printf("      [flow] %d steps, %d forwards, %.1fs\n", sp.steps, n_fwd, secs);
    fflush(stdout);
    return sample;
}

std::vector<float> sample_flow(const FlowFwd& fwd, std::vector<float> sample,
                               const FlowCond& cond, const FlowCond& neg_cond, const SamplerParams& sp,
                               std::vector<std::vector<float>>* trace) {
    return euler_loop(std::move(sample), sp, trace, 1,
        [&](const std::vector<float>& x, float t, float gs, std::vector<float>& pred, int& n_fwd) {
            guided_velocity(fwd, x, t, gs, cond, neg_cond, sp, nullptr, pred, nullptr, n_fwd);
        });
}

std::vector<float> sample_flow_mv(const FlowFwd& fwd, std::vector<float> sample,
                                  const MultiViewCond& mv, const SamplerParams& sp,
                                  std::vector<std::vector<float>>* trace) {
    const int V = (int)mv.cond.size();
    if (V <= 0) throw std::runtime_error("sample_flow_mv: no views");
    // N=1 is the single-view sampler, bit for bit: same calls, same arithmetic, same log line.
    if (V == 1) return sample_flow(fwd, std::move(sample), mv.cond[0], mv.neg, sp, trace);
    if (!mv.weights || mv.n_tok <= 0 || sample.size() % (size_t)mv.n_tok != 0)
        throw std::runtime_error("sample_flow_mv: weights / token count do not match the sample");
    static const bool no_fix = std::getenv("TRELLIS_NOFIX") != nullptr;
    const size_t Nst = sample.size();
    const size_t N = (size_t)mv.n_tok;
    const size_t C = Nst / N;
    std::vector<float> view_pred, neg;
    return euler_loop(std::move(sample), sp, trace, V,
        [&](const std::vector<float>& x, float t, float gs, std::vector<float>& acc, int& n_fwd) {
            // Every view's negative is the same zero tensor at the same (x_t, t): one forward
            // serves all of them, so a guided step costs V+1 forwards instead of 2V.
            const bool need_neg = gs != 1.0f;
            if (need_neg) { neg = fwd(x, 1000.0f * t, mv.neg); ++n_fwd; }
            acc.assign(Nst, 0.0f);
            for (int v = 0; v < V; ++v) {
                guided_velocity(fwd, x, t, gs, mv.cond[(size_t)v], mv.neg, sp,
                                need_neg ? &neg : nullptr, view_pred, nullptr, n_fwd);
                // Per-view safety net, so one bad view cannot poison the blend.
                if (!no_fix) for (float& p : view_pred) if (!std::isfinite(p)) p = 0.0f;
                const float* w = mv.weights + (size_t)v * N;
                for (size_t n = 0; n < N; ++n) {
                    const float wn = w[n];
                    float* a = acc.data() + n * C;
                    const float* p = view_pred.data() + n * C;
                    for (size_t c = 0; c < C; ++c) a[c] += p[c] * wn;
                }
            }
        });
}

} // namespace trellis
