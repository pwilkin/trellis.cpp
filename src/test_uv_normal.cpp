// Normal-map bake check without models: a dense lat-long sphere as the "remeshed
// shell" and a coarse UV sphere as the decimated mesh. Three runs:
//   1. plain sphere       -> the baked normal must be flat, (0,0,1), everywhere
//   2. bumps along lon    -> ts.x must anti-correlate with cos(k*lon)  (tangent = +u)
//   3. bumps along lat    -> ts.y must correlate with cos(k*lat)       (bitangent = -v,
//                            glTF +Y up the image)
// Texels within 30 deg of the poles are skipped (degenerate tangents there).
#include "uv_bake.h"
#include "tri_bvh.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace trellis;

static const float PI = 3.14159265358979f;

struct Sphere { std::vector<float> verts, uv; std::vector<int32_t> faces; };

// r(lon,lat) = R + A*sin(k_lon*lon) + A*sin(k_lat*lat)
static Sphere make_sphere(int nlon, int nlat, float R, float A, int k_lon, int k_lat) {
    Sphere s;
    for (int j = 0; j <= nlat; ++j) for (int i = 0; i <= nlon; ++i) {
        const float u = (float)i / nlon, v = (float)j / nlat;
        const float lon = 2.f * PI * u, lat = -PI / 2 + PI * v;
        const float r = R + (k_lon ? A * std::sin(k_lon * lon) : 0.f) + (k_lat ? A * std::sin(k_lat * lat) : 0.f);
        s.verts.push_back(r * std::cos(lat) * std::cos(lon));
        s.verts.push_back(r * std::cos(lat) * std::sin(lon));
        s.verts.push_back(r * std::sin(lat));
        s.uv.push_back(u); s.uv.push_back(v);
    }
    auto id = [&](int i, int j) { return j * (nlon + 1) + i; };
    for (int j = 0; j < nlat; ++j) for (int i = 0; i < nlon; ++i) {
        // outward winding (CCW seen from outside)
        s.faces.push_back(id(i, j)); s.faces.push_back(id(i + 1, j)); s.faces.push_back(id(i + 1, j + 1));
        s.faces.push_back(id(i, j)); s.faces.push_back(id(i + 1, j + 1)); s.faces.push_back(id(i, j + 1));
    }
    return s;
}

struct Stats { double mean_z = 0, corr_x = 0, corr_y = 0; long n = 0; long flat = 0; };

static Stats run(const char* name, int k_lon, int k_lat, int T) {
    const float R = 0.4f, A = 0.004f;
    Sphere hi = make_sphere(512, 256, R, A, k_lon, k_lat);
    Sphere lo = make_sphere(32, 16, R, 0.f, 0, 0);
    TriBvh bvh = TriBvh::build(hi.verts.data(), (int64_t)hi.verts.size() / 3, hi.faces.data(), (int64_t)hi.faces.size() / 3);
    NormalSource src{&bvh, hi.verts.data(), hi.faces.data(), 0.f};

    BakedMesh bm;
    bm.verts = lo.verts; bm.uv = lo.uv; bm.faces = lo.faces; bm.T = T;
    const std::vector<int32_t> owner = rasterize_owner(bm);
    bake_normals(bm, owner, src);

    Stats st;
    // Pearson correlation accumulators
    double sx = 0, sy = 0, scl = 0, sct = 0, sxx = 0, syy = 0, scll = 0, sctt = 0, sxcl = 0, syct = 0;
    for (int y = 0; y < T; ++y) for (int x = 0; x < T; ++x) {
        const size_t t = (size_t)y * T + x;
        if (owner[t] < 0) continue;
        const float u = (x + 0.5f) / T, v = (y + 0.5f) / T;
        const float lon = 2.f * PI * u, lat = -PI / 2 + PI * v;
        if (std::fabs(lat) > PI / 3) continue;
        const double nx = bm.nrm[4 * t] / 127.5 - 1.0, ny = bm.nrm[4 * t + 1] / 127.5 - 1.0, nz = bm.nrm[4 * t + 2] / 127.5 - 1.0;
        const double cl = std::cos(k_lon * lon), ct = std::cos(k_lat * lat);
        st.mean_z += nz; ++st.n;
        if (std::fabs(nx) < 0.1 && std::fabs(ny) < 0.1) ++st.flat;
        sx += nx; sy += ny; scl += cl; sct += ct; sxx += nx * nx; syy += ny * ny; scll += cl * cl; sctt += ct * ct; sxcl += nx * cl; syct += ny * ct;
    }
    const double n = (double)st.n;
    st.mean_z /= n;
    auto corr = [&](double sa, double sb, double saa, double sbb, double sab) {
        const double cov = sab / n - (sa / n) * (sb / n);
        const double va = saa / n - (sa / n) * (sa / n), vb = sbb / n - (sb / n) * (sb / n);
        return (va > 1e-12 && vb > 1e-12) ? cov / std::sqrt(va * vb) : 0.0;
    };
    st.corr_x = corr(sx, scl, sxx, scll, sxcl);
    st.corr_y = corr(sy, sct, syy, sctt, syct);
    long wpos = 0; for (size_t i = 0; i < bm.tangents.size() / 4; ++i) wpos += bm.tangents[4 * i + 3] > 0;
    printf("%-10s texels=%ld mean_z=%.4f flat=%.1f%% corr(x,cos k*lon)=%+.3f corr(y,cos k*lat)=%+.3f  tangent w>0: %ld/%zu\n",
           name, st.n, st.mean_z, 100.0 * st.flat / n, st.corr_x, st.corr_y, wpos, bm.tangents.size() / 4);
    return st;
}

int main() {
    const int T = 256;
    int fails = 0;
    auto expect = [&](bool ok, const char* what) { if (!ok) { printf("  FAIL: %s\n", what); ++fails; } };

    const Stats a = run("plain", 0, 0, T);
    expect(a.n > 0.5 * T * T, "plain: sphere UVs should cover most of the atlas");
    expect(a.mean_z > 0.99, "plain: baked normal should be flat (mean z > 0.99)");
    expect(a.flat > 0.98 * a.n, "plain: 98% of texels within 0.1 of (0,0,1)");

    const Stats b = run("bump-lon", 16, 0, T);
    expect(b.corr_x < -0.9, "bump-lon: ts.x must anti-correlate with cos(k*lon) (tangent = +u)");
    expect(std::fabs(b.corr_y) < 0.3, "bump-lon: ts.y should be unaffected");

    const Stats c = run("bump-lat", 0, 16, T);
    expect(c.corr_y > 0.9, "bump-lat: ts.y must correlate with cos(k*lat) (bitangent = -v, glTF +Y up)");
    expect(std::fabs(c.corr_x) < 0.3, "bump-lat: ts.x should be unaffected");

    printf(fails ? "trellis-test-uv-normal: %d FAILED\n" : "trellis-test-uv-normal: OK\n", fails);
    return fails ? 1 : 0;
}
