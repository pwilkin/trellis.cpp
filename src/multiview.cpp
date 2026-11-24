#include "multiview.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace trellis {

namespace {

constexpr double kPi = 3.14159265358979323846;

bool parse_float(const std::string& s, float& out) {
    if (s.empty()) return false;
    const char* b = s.c_str();
    char* end = nullptr;
    errno = 0;
    const float v = std::strtof(b, &end);
    if (end == b || *end || errno == ERANGE || !std::isfinite(v)) return false;
    // strtof accepts "inf"/"nan"/hex; isfinite rejects the first two. Reject leading blanks too.
    if (std::isspace((unsigned char)b[0])) return false;
    out = v;
    return true;
}

bool view_name(const std::string& s, float& az, float& el) {
    struct N { const char* n; float az, el; };
    static const N names[] = { {"front", 0, 0}, {"back", 180, 0}, {"left", 270, 0},
                               {"right", 90, 0}, {"top", 0, 90}, {"bottom", 0, -90} };
    for (const N& n : names)
        if (s == n.n) { az = n.az; el = n.el; return true; }
    return false;
}

// One axis of the preset: grid component = sign * viewer[idx].
struct Ax { int idx; int sign; };
struct Preset { Ax g[3]; };

Preset preset_of(FrontAxis f) {
    // viewer index: 0 = X (right), 1 = Y (up), 2 = Z (toward the front camera)
    switch (f) {
        case FrontAxis::NegY: return { { {0, +1}, {2, -1}, {1, +1} } };   // gx=X  gy=-Z gz=Y
        case FrontAxis::PosY: return { { {0, -1}, {2, +1}, {1, +1} } };   // gx=-X gy=Z  gz=Y
        case FrontAxis::PosX: return { { {2, +1}, {0, +1}, {1, +1} } };   // gx=Z  gy=X  gz=Y
        case FrontAxis::NegX: return { { {2, -1}, {0, -1}, {1, +1} } };   // gx=-Z gy=-X gz=Y
        case FrontAxis::RefZ: return { { {2, +1}, {1, -1}, {0, +1} } };   // gx=Z  gy=-Y gz=X
        case FrontAxis::RefX: return { { {0, +1}, {1, +1}, {2, +1} } };   // gx=X  gy=Y  gz=Z
    }
    return preset_of(FrontAxis::NegY);
}

// softmax over V scores at one token, written view-major into w[v*N + n].
void softmax_token(const double* score, int V, double tau, float* w, size_t N, size_t n) {
    double mx = -1e300;
    for (int v = 0; v < V; ++v) mx = std::max(mx, tau * score[v]);
    double e[kMaxViews], sum = 0;
    for (int v = 0; v < V; ++v) { e[v] = std::exp(tau * score[v] - mx); sum += e[v]; }
    for (int v = 0; v < V; ++v) w[(size_t)v * N + n] = (float)(e[v] / sum);
}

std::vector<std::array<double,3>> grid_dirs(const std::vector<ViewSpec>& views, const BlendParams& bp) {
    std::vector<std::array<double,3>> d(views.size());
    const Preset P = preset_of(bp.front);
    for (size_t v = 0; v < views.size(); ++v) {
        const auto dv = view_dir_viewer(views[v].az_deg, views[v].el_deg, bp.yaw_offset);
        for (int a = 0; a < 3; ++a) d[v][a] = P.g[a].sign * dv[P.g[a].idx];
    }
    return d;
}

}  // namespace

bool parse_front_axis(const std::string& s, FrontAxis& out) {
    if      (s == "-y")    out = FrontAxis::NegY;
    else if (s == "+y")    out = FrontAxis::PosY;
    else if (s == "+x")    out = FrontAxis::PosX;
    else if (s == "-x")    out = FrontAxis::NegX;
    else if (s == "ref-z") out = FrontAxis::RefZ;
    else if (s == "ref-x") out = FrontAxis::RefX;
    else return false;
    return true;
}

const char* front_axis_name(FrontAxis f) {
    switch (f) {
        case FrontAxis::NegY: return "-y";
        case FrontAxis::PosY: return "+y";
        case FrontAxis::PosX: return "+x";
        case FrontAxis::NegX: return "-x";
        case FrontAxis::RefZ: return "ref-z";
        case FrontAxis::RefX: return "ref-x";
    }
    return "?";
}

bool parse_view_spec(const std::string& s, ViewSpec& out, std::string& error) {
    ViewSpec v;
    std::string path = s;
    // Rightmost piece: an azimuth (number or name), or an elevation when the piece before it is
    // also an azimuth. A piece that is neither (e.g. "\\dir\\a.png" after a drive letter) is path.
    auto split_last = [](const std::string& p, std::string& head, std::string& tail) {
        const size_t c = p.find_last_of(':');
        if (c == std::string::npos) return false;
        head = p.substr(0, c); tail = p.substr(c + 1);
        return true;
    };
    std::string head1, t1;
    if (split_last(path, head1, t1)) {
        float f1, naz, nel;
        const bool num1 = parse_float(t1, f1), name1 = view_name(t1, naz, nel);
        if (num1 || name1) {
            std::string head2, t2;
            float f2;
            bool two = false;
            if (num1 && split_last(head1, head2, t2)) {
                if (view_name(t2, naz, nel))      { v.az_deg = naz; v.el_deg = f1; two = true; }
                else if (parse_float(t2, f2))     { v.az_deg = f2;  v.el_deg = f1; two = true; }
            }
            if (two) path = head2;
            else {
                if (name1) { v.az_deg = naz; v.el_deg = nel; } else v.az_deg = f1;
                path = head1;
            }
        }
    }
    if (path.empty()) { error = "view spec has no image path: " + s; return false; }
    if (v.el_deg < -90.0f || v.el_deg > 90.0f) {
        error = "view elevation must be within [-90, 90] degrees: " + s;
        return false;
    }
    v.image = path;
    out = v;
    return true;
}

std::array<double,3> view_dir_viewer(float az_deg, float el_deg, float yaw_offset) {
    const double th = ((double)az_deg + (double)yaw_offset) * kPi / 180.0;
    const double ph = (double)el_deg * kPi / 180.0;
    std::array<double,3> d = { std::cos(ph) * std::sin(th), std::sin(ph), std::cos(ph) * std::cos(th) };
    for (double& c : d) if (std::fabs(c) < 1e-9) c = 0.0;   // cardinal views exact
    return d;
}

std::array<float,3> view_dir_grid(float az_deg, float el_deg, const BlendParams& bp) {
    const auto d = grid_dirs({ ViewSpec{ "", az_deg, el_deg } }, bp)[0];
    return { (float)d[0], (float)d[1], (float)d[2] };
}

std::vector<float> view_weights_sparse(const std::vector<std::array<int,3>>& coords, int grid,
                                       const std::vector<ViewSpec>& views, const BlendParams& bp,
                                       int ref_model_res) {
    const size_t N = coords.size();
    const int V = (int)views.size();
    if (V <= 1) return std::vector<float>(N, 1.0f);
    const auto d = grid_dirs(views, bp);
    const bool ref = bp.ref_norm();
    const double R = ref ? (double)(ref_model_res > 0 ? ref_model_res : grid) : (double)grid;
    std::vector<float> w((size_t)V * N);
    double score[kMaxViews];
    for (size_t n = 0; n < N; ++n) {
        double p[3];
        for (int a = 0; a < 3; ++a)
            p[a] = ref ? (double)(float)coords[n][a] / R * 2.0 - 1.0
                       : ((double)coords[n][a] + 0.5) / R * 2.0 - 1.0;
        for (int v = 0; v < V; ++v) score[v] = p[0] * d[v][0] + p[1] * d[v][1] + p[2] * d[v][2];
        softmax_token(score, V, bp.temperature, w.data(), N, n);
    }
    return w;
}

std::vector<float> view_weights_dense(int R, const std::vector<ViewSpec>& views, const BlendParams& bp) {
    const size_t N = (size_t)R * R * R;
    const int V = (int)views.size();
    if (V <= 1) return std::vector<float>(N, 1.0f);
    const auto d = grid_dirs(views, bp);
    const bool ref = bp.ref_norm();
    auto pos = [&](int i) -> double {
        if (ref) return R > 1 ? -1.0 + 2.0 * i / (R - 1) : -1.0;
        return ((double)i + 0.5) / R * 2.0 - 1.0;
    };
    std::vector<float> w((size_t)V * N);
    double score[kMaxViews];
    for (int x = 0; x < R; ++x)
        for (int y = 0; y < R; ++y)
            for (int z = 0; z < R; ++z) {
                const size_t n = (size_t)x * R * R + (size_t)y * R + z;
                const double p[3] = { pos(x), pos(y), pos(z) };
                for (int v = 0; v < V; ++v) score[v] = p[0] * d[v][0] + p[1] * d[v][1] + p[2] * d[v][2];
                softmax_token(score, V, bp.temperature, w.data(), N, n);
            }
    return w;
}

std::vector<double> mean_view_weights(const std::vector<float>& w, int V) {
    std::vector<double> m(V, 0.0);
    if (V <= 0 || w.empty()) return m;
    const size_t N = w.size() / V;
    for (int v = 0; v < V; ++v) {
        double s = 0;
        for (size_t n = 0; n < N; ++n) s += w[(size_t)v * N + n];
        m[v] = N ? s / N : 0.0;
    }
    return m;
}

}  // namespace trellis
