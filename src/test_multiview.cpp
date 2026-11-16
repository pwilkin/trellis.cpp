// Multi-view conditioning tests (docs/spec/30-multiview.md). No GPU, no model weights.
//
//   trellis-test-multiview [weights_dir] [step_dir]
//
//   T1  (weights_dir, from tools/ref_multiview_weights.py) view_weights_sparse/_dense vs the
//       reference formula (cardinal ref-* cases) and the generalised formula (all others).
//   T2  sample_flow_mv over a deterministic fake forward: V=1 == sample_flow bit for bit
//       (CFG on/off, rescale on/off), identical conds == single view, one-hot weights == the
//       selected view, forward count 12*V + guided steps with the shared negative.
//   T3  (step_dir, from tools/ref_multiview_step.py) full multi-view trajectories vs the
//       reference sampler maths over the same fake model.
//   CLI --view / --front-axis / --mv-stages parsing and resolve_views.
#include "flow_runner.h"
#include "multiview.h"
#include "npy.h"
#include "trellis_args.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace trellis;
using std::string;
using std::vector;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

// ---------------------------------------------------------------- fake forward
// v(x, ts, cond)[k] = (0.5 + ts/2000) * x[k] + 0.7 * mean(cond) + 0.1 * sin(0.37 * k)
struct Fake {
    size_t cond_len;
    int calls = 0;
    vector<float> operator()(const vector<float>& x, float ts, const FlowCond& c) {
        ++calls;
        double m = 0;
        for (size_t i = 0; i < cond_len; ++i) m += c.cond[i];
        const float mean = (float)(m / (double)cond_len);
        const float a = 0.5f + ts / 2000.0f;
        vector<float> v(x.size());
        for (size_t k = 0; k < x.size(); ++k)
            v[k] = a * x[k] + 0.7f * mean + 0.1f * std::sin(0.37f * (float)k);
        return v;
    }
};

static vector<float> randv(size_t n, unsigned seed) {
    vector<float> v(n);
    unsigned s = seed * 2654435761u + 1;
    for (auto& x : v) { s = s * 1664525u + 1013904223u; x = ((s >> 8) / 16777216.0f) * 2.0f - 1.0f; }
    return v;
}

static double max_rel(const vector<float>& a, const float* b, size_t n) {
    double md = 0, mb = 0;
    for (size_t i = 0; i < n; ++i) { md = std::max(md, (double)std::fabs(a[i] - b[i])); mb = std::max(mb, (double)std::fabs(b[i])); }
    return md / std::max(mb, 1e-12);
}

// ---------------------------------------------------------------- T1
static void test_weights(const string& dir) {
    std::ifstream f(dir + "/cases.txt");
    if (!f) { fprintf(stderr, "T1: cannot open %s/cases.txt\n", dir.c_str()); ++g_fail; return; }
    string line;
    int n = 0, nref = 0;
    double worst = 0, worst_sum = 0;
    string worst_case;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream is(line);
        string name, kind, preset, vs;
        int grid, mres; float yaw, tau;
        is >> name >> kind >> grid >> mres >> preset >> yaw >> tau >> vs;
        vector<ViewSpec> views;
        std::stringstream vss(vs); string tok;
        while (std::getline(vss, tok, ';')) {
            ViewSpec v; sscanf(tok.c_str(), "%f,%f", &v.az_deg, &v.el_deg); views.push_back(v);
        }
        BlendParams bp; bp.temperature = tau; bp.yaw_offset = yaw;
        if (!parse_front_axis(preset, bp.front)) { CHECK(false, "bad preset %s", preset.c_str()); continue; }
        if (bp.ref_norm()) ++nref;
        vector<float> w;
        if (kind == "sparse") {
            npy::Array c = npy::load(dir + "/" + name + "_coords.npy");
            vector<std::array<int,3>> coords((size_t)c.shape[0]);
            for (size_t i = 0; i < coords.size(); ++i)
                for (int a = 0; a < 3; ++a) coords[i][a] = (int)c.data[i * 3 + a];
            w = view_weights_sparse(coords, grid, views, bp, mres);
        } else {
            w = view_weights_dense(grid, views, bp);
        }
        npy::Array g = npy::load(dir + "/" + name + "_w.npy");
        if ((int64_t)w.size() != g.numel()) { CHECK(false, "%s: size %zu vs %lld", name.c_str(), w.size(), (long long)g.numel()); continue; }
        double md = 0;
        for (size_t i = 0; i < w.size(); ++i) md = std::max(md, (double)std::fabs(w[i] - g.data[i]));
        if (md > worst) { worst = md; worst_case = line; }
        const int V = (int)views.size();
        const size_t N = w.size() / V;
        for (size_t t = 0; t < N; ++t) {
            double s = 0; for (int v = 0; v < V; ++v) s += w[(size_t)v * N + t];
            worst_sum = std::max(worst_sum, std::fabs(s - 1.0));
        }
        if (V == 1) for (float x : w) CHECK(x == 1.0f, "%s: single view weight %g != 1", name.c_str(), x);
        ++n;
    }
    printf("T1 weights: %d cases (%d ref-normalised), max |dw| = %.3e, max |sum-1| = %.3e%s\n",
           n, nref, worst, worst_sum, worst > 1e-5 ? (" worst: " + worst_case).c_str() : "");
    CHECK(n > 0, "T1: no cases");
    CHECK(worst < 1e-5, "T1: max |dw| %.3e >= 1e-5", worst);
    CHECK(worst_sum < 1e-6, "T1: rows do not sum to 1 (%.3e)", worst_sum);
}

// ---------------------------------------------------------------- T2
static void test_sampler_fake() {
    const int C = 8, N = 512, L = 4, D = 16;
    const size_t cl = (size_t)L * D;
    vector<float> c0 = randv(cl, 1), c1 = randv(cl, 2), c2 = randv(cl, 3), neg(cl, 0.0f);
    vector<float> x0 = randv((size_t)C * N, 9);

    for (int cfg = 0; cfg < 2; ++cfg) for (int rs = 0; rs < 2; ++rs) {
        SamplerParams sp; sp.steps = 12; sp.guidance_strength = cfg ? 7.5f : 1.0f;
        sp.guidance_rescale = rs ? 0.7f : 0.0f; sp.gi0 = 0.6f; sp.gi1 = 1.0f; sp.rescale_t = 5.0f;
        Fake f1{cl}, f2{cl};
        FlowFwd fw1 = [&](const vector<float>& x, float t, const FlowCond& c) { return f1(x, t, c); };
        FlowFwd fw2 = [&](const vector<float>& x, float t, const FlowCond& c) { return f2(x, t, c); };
        vector<float> a = sample_flow(fw1, x0, FlowCond(c0.data()), FlowCond(neg.data()), sp);
        MultiViewCond mv; mv.cond = { FlowCond(c0.data()) }; mv.neg = FlowCond(neg.data());
        vector<float> b = sample_flow_mv(fw2, x0, mv, sp);
        CHECK(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0,
              "T2a: V=1 not bit-identical (cfg=%d rescale=%d)", cfg, rs);
        CHECK(f1.calls == f2.calls, "T2a: forward counts differ %d vs %d", f1.calls, f2.calls);
    }
    printf("T2a V=1 == sample_flow (memcmp, CFG/rescale on/off): %s\n", g_fail ? "see failures" : "OK");

    SamplerParams sp; sp.steps = 12; sp.guidance_strength = 7.5f; sp.guidance_rescale = 0.7f;
    sp.gi0 = 0.6f; sp.gi1 = 1.0f; sp.rescale_t = 5.0f;
    // T2b: three identical conds, arbitrary weights summing to 1 -> single-view result.
    {
        Fake fs{cl}, fm{cl};
        FlowFwd fws = [&](const vector<float>& x, float t, const FlowCond& c) { return fs(x, t, c); };
        FlowFwd fwm = [&](const vector<float>& x, float t, const FlowCond& c) { return fm(x, t, c); };
        vector<float> a = sample_flow(fws, x0, FlowCond(c1.data()), FlowCond(neg.data()), sp);
        vector<ViewSpec> views = { {"", 0, 0}, {"", 120, 0}, {"", 240, 30} };
        vector<std::array<int,3>> coords(N);
        for (int n = 0; n < N; ++n) coords[n] = { n % 8, (n / 8) % 8, n / 64 };
        BlendParams bp;
        vector<float> w = view_weights_sparse(coords, 8, views, bp);
        MultiViewCond mv; mv.cond = { FlowCond(c1.data()), FlowCond(c1.data()), FlowCond(c1.data()) };
        mv.neg = FlowCond(neg.data()); mv.weights = w.data(); mv.n_tok = N;
        vector<float> b = sample_flow_mv(fwm, x0, mv, sp);
        const double r = max_rel(b, a.data(), a.size());
        printf("T2b V=3 identical conds vs single view: max rel %.3e\n", r);
        CHECK(r < 1e-5, "T2b: %.3e", r);
    }
    // T2c: one-hot weights pick one view.
    {
        Fake fs{cl}, fm{cl};
        FlowFwd fws = [&](const vector<float>& x, float t, const FlowCond& c) { return fs(x, t, c); };
        FlowFwd fwm = [&](const vector<float>& x, float t, const FlowCond& c) { return fm(x, t, c); };
        vector<float> a = sample_flow(fws, x0, FlowCond(c2.data()), FlowCond(neg.data()), sp);
        vector<float> w((size_t)2 * N, 0.0f);
        for (int n = 0; n < N; ++n) w[(size_t)N + n] = 1.0f;     // all weight on view 1 (= c2)
        MultiViewCond mv; mv.cond = { FlowCond(c0.data()), FlowCond(c2.data()) };
        mv.neg = FlowCond(neg.data()); mv.weights = w.data(); mv.n_tok = N;
        vector<float> b = sample_flow_mv(fwm, x0, mv, sp);
        const double r = max_rel(b, a.data(), a.size());
        printf("T2c V=2 one-hot weights vs selected view: max rel %.3e\n", r);
        CHECK(r < 1e-6, "T2c: %.3e", r);
    }
    // T2d: forward count = steps*V + guided steps (shared negative), texture-like gs=1 = steps*V.
    for (int V : {2, 4}) for (float gs : {7.5f, 1.0f}) {
        SamplerParams s2 = sp; s2.guidance_strength = gs;
        int guided = 0;
        for (int i = 0; i < s2.steps; ++i) {
            float t = 1.0f - (float)i / s2.steps; t = s2.rescale_t * t / (1.0f + (s2.rescale_t - 1.0f) * t);
            if (s2.gi0 <= t && t <= s2.gi1 && gs != 1.0f) ++guided;
        }
        Fake fm{cl};
        FlowFwd fwm = [&](const vector<float>& x, float t, const FlowCond& c) { return fm(x, t, c); };
        vector<float> w((size_t)V * N, 1.0f / V);
        MultiViewCond mv; mv.neg = FlowCond(neg.data()); mv.weights = w.data(); mv.n_tok = N;
        for (int v = 0; v < V; ++v) mv.cond.push_back(FlowCond(v % 2 ? c1.data() : c0.data()));
        sample_flow_mv(fwm, x0, mv, s2);
        const int want = s2.steps * V + guided;
        printf("T2d V=%d gs=%.1f: %d forwards (want %d = %d*%d + %d guided)\n", V, gs, fm.calls, want, s2.steps, V, guided);
        CHECK(fm.calls == want, "T2d: %d forwards, want %d", fm.calls, want);
    }
}

// ---------------------------------------------------------------- T3
static void test_step_parity(const string& dir) {
    const char* cases[] = { "dense_cfg_rescale", "dense_cfg", "sparse_cfg", "sparse_nocfg", "sparse_gs0" };
    for (const char* name : cases) {
        const string d = dir + "/" + name;
        std::ifstream pf(d + "/params.txt");
        if (!pf) { CHECK(false, "T3: missing %s", d.c_str()); continue; }
        int C, N, V, steps; float gs, gr, gi0, gi1, rt;
        pf >> C >> N >> V >> steps >> gs >> gr >> gi0 >> gi1 >> rt;
        npy::Array x0 = npy::load(d + "/x0.npy"), conds = npy::load(d + "/conds.npy");
        npy::Array w = npy::load(d + "/w.npy"), out = npy::load(d + "/out.npy");
        const size_t cl = (size_t)(conds.numel() / V);
        vector<float> neg(cl, 0.0f);
        Fake fm{cl};
        FlowFwd fwm = [&](const vector<float>& x, float t, const FlowCond& c) { return fm(x, t, c); };
        SamplerParams sp; sp.steps = steps; sp.guidance_strength = gs; sp.guidance_rescale = gr;
        sp.gi0 = gi0; sp.gi1 = gi1; sp.rescale_t = rt;
        MultiViewCond mv; mv.neg = FlowCond(neg.data()); mv.weights = w.data.data(); mv.n_tok = N;
        for (int v = 0; v < V; ++v) mv.cond.push_back(FlowCond(conds.data.data() + (size_t)v * cl));
        vector<vector<float>> trace;
        vector<float> got = sample_flow_mv(fwm, x0.data, mv, sp, &trace);
        double worst_step = 0;
        for (int i = 0; i < steps; ++i) {
            npy::Array s = npy::load(d + "/step_" + std::to_string(i) + ".npy");
            worst_step = std::max(worst_step, max_rel(trace[i], s.data.data(), s.data.size()));
        }
        const double r = max_rel(got, out.data.data(), out.data.size());
        printf("T3 %-18s C=%d N=%d V=%d steps=%d gs=%.1f gr=%.1f: final max rel %.3e, worst step %.3e\n",
               name, C, N, V, steps, gs, gr, r, worst_step);
        CHECK(r < 1e-5 && worst_step < 1e-5, "T3 %s: %.3e / %.3e", name, r, worst_step);
    }
}

// ---------------------------------------------------------------- CLI
static bool parse(vector<string> args, TrellisParams& p) {
    vector<char*> argv; static char prog[] = "trellis-cli";
    argv.push_back(prog);
    for (auto& a : args) argv.push_back(a.data());
    p = TrellisParams();
    return parse_args((int)argv.size(), argv.data(), p);
}

static void test_cli() {
    struct S { const char* in; bool ok; const char* path; float az, el; };
    const S specs[] = {
        { "a.png", true, "a.png", 0, 0 },
        { "a.png:180", true, "a.png", 180, 0 },
        { "a.png:-45:12.5", true, "a.png", -45, 12.5f },
        { "a.png:back", true, "a.png", 180, 0 },
        { "a.png:top", true, "a.png", 0, 90 },
        { "a.png:left:20", true, "a.png", 270, 20 },
        { "C:\\dir\\a.png", true, "C:\\dir\\a.png", 0, 0 },
        { "C:\\dir\\a.png:180", true, "C:\\dir\\a.png", 180, 0 },
        { "C:\\dir\\a.png:90:-30", true, "C:\\dir\\a.png", 90, -30 },
        { "/x/y:z/a.png:270", true, "/x/y:z/a.png", 270, 0 },
        { ":90", false, "", 0, 0 },
        { "a.png:10:95", false, "", 0, 0 },
        { "", false, "", 0, 0 },
    };
    for (const S& s : specs) {
        ViewSpec v; string err;
        const bool ok = parse_view_spec(s.in, v, err);
        CHECK(ok == s.ok, "parse_view_spec('%s') ok=%d want %d (%s)", s.in, ok, s.ok, err.c_str());
        if (ok && s.ok)
            CHECK(v.image == s.path && v.az_deg == s.az && v.el_deg == s.el,
                  "parse_view_spec('%s') -> '%s' %g %g", s.in, v.image.c_str(), v.az_deg, v.el_deg);
    }

    TrellisParams p, base;
    CHECK(parse({ "a.png", "o.glb" }, base), "plain parse");
    CHECK(base.views.empty() && base.blend_temp == 2.0f && base.front_axis == "-y" && base.mv_stages == 7u,
          "defaults changed");
    vector<ViewSpec> vs; string err;
    CHECK(resolve_views(base, vs, err) && vs.size() == 1 && vs[0].image == "a.png", "resolve single");

    CHECK(parse({ "f.png", "o.glb", "--view", "b.png:180", "--view", "r.png:right:10",
                  "--blend-temp", "1.5", "--front-axis", "ref-z", "--yaw-offset", "-30",
                  "--mv-stages", "shape,tex" }, p), "mv parse");
    CHECK(resolve_views(p, vs, err) && vs.size() == 3, "resolve 3 views: %s", err.c_str());
    if (vs.size() == 3)
        CHECK(vs[0].image == "f.png" && vs[1].az_deg == 180 && vs[2].az_deg == 90 && vs[2].el_deg == 10,
              "resolved view order");
    CHECK(p.blend_temp == 1.5f && p.front_axis == "ref-z" && p.yaw_offset == -30 && p.mv_stages == 6u,
          "mv field values");

    CHECK(parse({ "--view", "f.png:0", "--view", "b.png:180", "-o", "x.glb" }, p), "views only");
    CHECK(resolve_views(p, vs, err) && vs.size() == 2 && vs[0].image == "f.png", "views-only resolve");

    CHECK(!parse({ "a.png", "--blend-temp", "0" }, p), "blend-temp 0 rejected");
    CHECK(!parse({ "a.png", "--blend-temp", "x" }, p), "blend-temp x rejected");
    CHECK(!parse({ "a.png", "--front-axis", "z" }, p), "front-axis z rejected");
    CHECK(!parse({ "a.png", "--mv-stages", "ss,foo" }, p), "mv-stages foo rejected");
    CHECK(!parse({ "a.png", "--view", "b.png:0:100" }, p), "elevation 100 rejected");
    CHECK(parse({ "-o", "x.glb" }, p) && !resolve_views(p, vs, err), "no view rejected");
    {
        vector<string> a = { "a.png" };
        for (int i = 0; i < kMaxViews; ++i) { a.push_back("--view"); a.push_back("b.png:" + std::to_string(i * 30)); }
        CHECK(parse(a, p) && !resolve_views(p, vs, err), "13 views rejected");
    }
    CHECK(parse({ "a.png", "--model", "pixal3d", "--view", "b.png:180" }, p) && !resolve_views(p, vs, err),
          "pixal3d multi-view rejected");
    CHECK(parse({ "a.png", "--model", "pixal3d" }, p) && resolve_views(p, vs, err), "pixal3d single ok");
    unsigned st = 0;
    CHECK(parse_mv_stages("all", st, err) && st == 7u, "all");
    CHECK(parse_mv_stages("none", st, err) && st == 0u, "none");
    CHECK(parse_mv_stages("ss", st, err) && st == 1u, "ss");

    // Directions: the default preset puts the front camera on grid -y and az=90 on grid +x.
    BlendParams bp;
    auto d0 = view_dir_grid(0, 0, bp), d90 = view_dir_grid(90, 0, bp), dt = view_dir_grid(0, 90, bp);
    CHECK(d0[0] == 0 && d0[1] == -1 && d0[2] == 0, "front dir %g %g %g", d0[0], d0[1], d0[2]);
    CHECK(d90[0] == 1 && d90[1] == 0 && d90[2] == 0, "right dir %g %g %g", d90[0], d90[1], d90[2]);
    CHECK(dt[0] == 0 && dt[1] == 0 && dt[2] == 1, "top dir %g %g %g", dt[0], dt[1], dt[2]);
    printf("CLI parse / resolve_views / directions: %s\n", g_fail ? "see failures" : "OK");
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    test_cli();
    test_sampler_fake();
    if (argc > 1) test_weights(argv[1]); else printf("T1 skipped (no weights dir)\n");
    if (argc > 2) test_step_parity(argv[2]); else printf("T3 skipped (no step dir)\n");
    printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL OK", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
