// Voxeliser tests (mesh_to_flexible_dual_grid, docs/spec/31-mesh-refine.md). CPU only.
//
//   trellis-test-voxelize [mesh.glb grid]
//
// Analytic shapes at grid 64:
//   box     the exact set of surface-crossing grid edges is known: every flagged edge must be
//           one of them and every one of them must be flagged; every voxel lies within one
//           voxel of the surface; every surface sample's voxel is present.
//   sphere  every surface sample's voxel is present; no voxel is farther than one voxel (plus
//           half a diagonal) from the sphere.
// With a mesh argument: normalise it and voxelise it at `grid`, report counts and timing.
#include "mesh_voxelize.h"
#include "mesh_io.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace trellis;

static constexpr double kPi = 3.14159265358979323846;   // M_PI is not standard (MSVC)

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static void add_quad(std::vector<float>& v, std::vector<int32_t>& f, const float* a, const float* b, const float* c, const float* d) {
    const int32_t base = (int32_t)v.size() / 3;
    for (const float* p : { a, b, c, d }) { v.push_back(p[0]); v.push_back(p[1]); v.push_back(p[2]); }
    f.insert(f.end(), { base, base + 1, base + 2, base, base + 2, base + 3 });
}

static void test_box(int G) {
    const float lo[3] = { -0.3013f, -0.2071f, -0.2533f }, hi[3] = { 0.2917f, 0.1987f, 0.2471f };
    std::vector<float> v; std::vector<int32_t> f;
    auto P = [&](int i, int j, int k, float* out) { out[0] = i ? hi[0] : lo[0]; out[1] = j ? hi[1] : lo[1]; out[2] = k ? hi[2] : lo[2]; };
    float c[8][3];
    for (int i = 0; i < 8; ++i) P(i & 1, (i >> 1) & 1, (i >> 2) & 1, c[i]);
    add_quad(v, f, c[0], c[2], c[6], c[4]); add_quad(v, f, c[1], c[5], c[7], c[3]);   // x faces
    add_quad(v, f, c[0], c[4], c[5], c[1]); add_quad(v, f, c[2], c[3], c[7], c[6]);   // y faces
    add_quad(v, f, c[0], c[1], c[3], c[2]); add_quad(v, f, c[4], c[6], c[7], c[5]);   // z faces
    const DualGrid dg = mesh_to_flexible_dual_grid(v.data(), (int64_t)v.size() / 3, f.data(), (int64_t)f.size() / 3, G, true);
    // Expected flagged edges: for the face pair normal to axis a, the edge along a at the max
    // corner (b, c) = ((j+1)/G, (k+1)/G) of voxel (floor((face+0.5)G), j, k) crosses the face
    // when both corners fall strictly inside the face rectangle.
    std::set<std::array<int,4>> want;   // (x, y, z, axis)
    for (int a = 0; a < 3; ++a) {
        const int b = (a + 1) % 3, cc = (a + 2) % 3;
        for (float face : { lo[a], hi[a] }) {
            const int ia = (int)std::floor((face + 0.5f) * G);
            for (int j = 0; j < G - 1; ++j) for (int k = 0; k < G - 1; ++k) {
                const float yb = (float)(j + 1) / G - 0.5f, zc = (float)(k + 1) / G - 0.5f;
                if (yb > lo[b] && yb < hi[b] && zc > lo[cc] && zc < hi[cc]) {
                    std::array<int,4> e; e[a] = ia; e[b] = j; e[cc] = k; e[3] = a;
                    want.insert(e);
                }
            }
        }
    }
    std::set<std::array<int,4>> got;
    for (size_t i = 0; i < dg.coords.size(); ++i)
        for (int a = 0; a < 3; ++a)
            if (dg.intersected[i * 3 + a]) got.insert({ dg.coords[i][0], dg.coords[i][1], dg.coords[i][2], a });
    size_t missing = 0, extra = 0;
    for (auto& e : want) if (!got.count(e)) ++missing;
    for (auto& e : got) if (!want.count(e)) ++extra;
    printf("box  @%d: %zu voxels, %zu flagged edges (expected %zu): missing %zu, extra %zu\n",
           G, dg.coords.size(), got.size(), want.size(), missing, extra);
    CHECK(missing == 0 && extra == 0, "box intersected flags: missing %zu extra %zu", missing, extra);
    // Every voxel within one voxel of the box surface (distance from the voxel centre).
    double worst = 0;
    for (auto& cd : dg.coords) {
        double p[3], dout = 0, din = 1e9;
        for (int a = 0; a < 3; ++a) p[a] = (cd[a] + 0.5) / G - 0.5;
        bool inside = true;
        for (int a = 0; a < 3; ++a) {
            const double e = std::max(lo[a] - p[a], p[a] - hi[a]);
            if (e > 0) { dout += e * e; inside = false; }
            din = std::min(din, std::min(p[a] - lo[a], hi[a] - p[a]));
        }
        const double d = (inside ? din : std::sqrt(dout)) * G;
        worst = std::max(worst, d);
    }
    printf("box  @%d: max voxel-centre distance to surface %.3f voxels\n", G, worst);
    CHECK(worst <= 1.0 + 0.8661, "box voxel %.3f voxels from the surface", worst);
    std::set<std::array<int,3>> cs(dg.coords.begin(), dg.coords.end());
    CHECK(cs.size() == dg.coords.size(), "duplicate voxels");
    // Surface samples.
    size_t miss = 0, n = 0;
    srand(1);
    for (int a = 0; a < 3; ++a) for (float face : { lo[a], hi[a] }) for (int s = 0; s < 4000; ++s) {
        float p[3];
        for (int q = 0; q < 3; ++q) p[q] = lo[q] + (hi[q] - lo[q]) * (rand() / (float)RAND_MAX);
        p[a] = face;
        std::array<int,3> vx;
        for (int q = 0; q < 3; ++q) vx[q] = std::min(G - 1, (int)std::floor((p[q] + 0.5f) * G));
        ++n; if (!cs.count(vx)) ++miss;
    }
    printf("box  @%d: %zu/%zu surface samples outside the voxel set\n", G, miss, n);
    CHECK(miss == 0, "box: %zu surface samples not covered", miss);
}

// Dual vertices on the box: a voxel crossed by a single face plane must put its dual vertex on
// that plane (the plane QEF and the regulariser toward the crossing points both sit on it); the
// voxels at the 8 corners must land on the corner.
static void test_box_dual(int G) {
    const float lo[3] = { -0.3013f, -0.2071f, -0.2533f }, hi[3] = { 0.2917f, 0.1987f, 0.2471f };
    std::vector<float> v; std::vector<int32_t> f;
    float c[8][3];
    for (int i = 0; i < 8; ++i) { c[i][0] = (i & 1) ? hi[0] : lo[0]; c[i][1] = ((i >> 1) & 1) ? hi[1] : lo[1]; c[i][2] = ((i >> 2) & 1) ? hi[2] : lo[2]; }
    add_quad(v, f, c[0], c[2], c[6], c[4]); add_quad(v, f, c[1], c[5], c[7], c[3]);
    add_quad(v, f, c[0], c[4], c[5], c[1]); add_quad(v, f, c[2], c[3], c[7], c[6]);
    add_quad(v, f, c[0], c[1], c[3], c[2]); add_quad(v, f, c[4], c[6], c[7], c[5]);
    const DualGrid dg = mesh_to_flexible_dual_grid(v.data(), (int64_t)v.size() / 3, f.data(), (int64_t)f.size() / 3, G, false);
    CHECK(dg.dual.size() == dg.coords.size() * 3, "dual size");
    double worst_face = 0, worst_corner = 0, worst_range = 0;
    size_t nface = 0, ncorner = 0;
    for (size_t i = 0; i < dg.coords.size(); ++i) {
        double w[3];
        for (int a = 0; a < 3; ++a) {
            const double off = dg.dual[i * 3 + a];
            worst_range = std::max(worst_range, std::max(-off, off - 1.0));
            w[a] = (dg.coords[i][a] + off) / G - 0.5;   // world position
        }
        // which box planes pass through this voxel?
        int planes = 0, which = -1; float val = 0;
        int corner_hits = 0;
        for (int a = 0; a < 3; ++a) for (float p : { lo[a], hi[a] }) {
            const int ia = (int)std::floor((p + 0.5f) * G);
            if (ia == dg.coords[i][a]) {
                // and the face rectangle overlaps the voxel on the other two axes
                const int b = (a + 1) % 3, cc = (a + 2) % 3;
                const double b0 = (double)dg.coords[i][b] / G - 0.5, b1 = b0 + 1.0 / G;
                const double c0 = (double)dg.coords[i][cc] / G - 0.5, c1 = c0 + 1.0 / G;
                if (b1 > lo[b] && b0 < hi[b] && c1 > lo[cc] && c0 < hi[cc]) { ++planes; which = a; val = p; }
                ++corner_hits;
            }
        }
        if (planes == 1) {
            worst_face = std::max(worst_face, std::fabs(w[which] - val) * G); ++nface;
        } else if (planes == 3 && corner_hits == 3) {
            double d = 0;
            for (int a = 0; a < 3; ++a) {
                const float p = (std::floor((lo[a] + 0.5f) * G) == dg.coords[i][a]) ? lo[a] : hi[a];
                d = std::max(d, std::fabs(w[a] - p) * G);
            }
            worst_corner = std::max(worst_corner, d); ++ncorner;
        }
    }
    printf("box dual @%d: %zu single-face voxels max |off-plane| %.2e voxels; %zu corner voxels max |d| %.2e; "
           "offsets outside [0,1] by %.2e\n", G, nface, worst_face, ncorner, worst_corner, worst_range);
    CHECK(nface > 0 && worst_face < 1e-3, "box dual: off-plane %.3e", worst_face);
    // The 1e-2 regulariser pulls a corner toward the mean crossing point: ~1e-2 voxels, not 0.
    CHECK(ncorner == 8 && worst_corner < 2e-2, "box dual: corners %zu, %.3e", ncorner, worst_corner);
    CHECK(worst_range < 1e-4, "box dual: offset out of the voxel by %.3e", worst_range);
}

static void test_sphere(int G) {
    const int nu = 96, nv = 48;
    const double R = 0.37, cx = 0.013, cy = -0.021, cz = 0.007;
    std::vector<float> v; std::vector<int32_t> f;
    for (int j = 0; j <= nv; ++j) for (int i = 0; i < nu; ++i) {
        const double th = kPi * j / nv, ph = 2 * kPi * i / nu;
        v.push_back((float)(cx + R * std::sin(th) * std::cos(ph)));
        v.push_back((float)(cy + R * std::sin(th) * std::sin(ph)));
        v.push_back((float)(cz + R * std::cos(th)));
    }
    for (int j = 0; j < nv; ++j) for (int i = 0; i < nu; ++i) {
        const int a = j * nu + i, b = j * nu + (i + 1) % nu, c = (j + 1) * nu + (i + 1) % nu, d = (j + 1) * nu + i;
        f.insert(f.end(), { a, b, c, a, c, d });
    }
    const auto t0 = std::chrono::steady_clock::now();
    const DualGrid dg = mesh_to_flexible_dual_grid(v.data(), (int64_t)v.size() / 3, f.data(), (int64_t)f.size() / 3, G, true);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::set<std::array<int,3>> cs(dg.coords.begin(), dg.coords.end());
    // The tessellated sphere deviates from the true one by up to R(1 - cos(pi/nv)) ~ 8e-4.
    double worst = 0;
    for (auto& cd : dg.coords) {
        const double p[3] = { (cd[0] + 0.5) / G - 0.5 - cx, (cd[1] + 0.5) / G - 0.5 - cy, (cd[2] + 0.5) / G - 0.5 - cz };
        worst = std::max(worst, std::fabs(std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) - R) * G);
    }
    size_t miss = 0, n = 0;
    srand(2);
    for (int s = 0; s < 20000; ++s) {
        // sample on the triangles themselves (the voxeliser sees the tessellation)
        const size_t t = (size_t)(rand() % (int)(f.size() / 3));
        double r1 = std::sqrt(rand() / (double)RAND_MAX), r2 = rand() / (double)RAND_MAX;
        const double w0 = 1 - r1, w1 = r1 * (1 - r2), w2 = r1 * r2;
        std::array<int,3> vx;
        for (int q = 0; q < 3; ++q) {
            const double p = w0 * v[f[t * 3] * 3 + q] + w1 * v[f[t * 3 + 1] * 3 + q] + w2 * v[f[t * 3 + 2] * 3 + q];
            vx[q] = std::min(G - 1, (int)std::floor((p + 0.5) * G));
        }
        ++n; if (!cs.count(vx)) ++miss;
    }
    printf("sphere @%d: %zu voxels (%.1f ms), max centre distance %.3f voxels, %zu/%zu samples uncovered\n",
           G, dg.coords.size(), ms, worst, miss, n);
    CHECK(miss == 0, "sphere: %zu surface samples not covered", miss);
    CHECK(worst <= 1.0 + 0.8661 + 0.1, "sphere voxel %.3f voxels from the surface", worst);
}

// Parity vs the o-voxel oracle (tools/ref_voxelize): `parity <mesh|box|sphere> <grid> <dir>` writes
// <dir>/in.bin; run `trellis-ref-voxelize <dir>/in.bin <grid> <dir>/ref.bin`; run parity again
// to compare. Coordinate sets and intersected flags must match exactly; dual offsets within 1e-3
// voxel for >= 99.5% of voxels and < 0.05 for all.
static bool load_shape(const std::string& what, std::vector<float>& v, std::vector<int32_t>& f) {
    if (what == "box" || what == "sphere") {
        if (what == "box") {
            const float lo[3] = { -0.3013f, -0.2071f, -0.2533f }, hi[3] = { 0.2917f, 0.1987f, 0.2471f };
            float c[8][3];
            for (int i = 0; i < 8; ++i) { c[i][0] = (i & 1) ? hi[0] : lo[0]; c[i][1] = ((i >> 1) & 1) ? hi[1] : lo[1]; c[i][2] = ((i >> 2) & 1) ? hi[2] : lo[2]; }
            add_quad(v, f, c[0], c[2], c[6], c[4]); add_quad(v, f, c[1], c[5], c[7], c[3]);
            add_quad(v, f, c[0], c[4], c[5], c[1]); add_quad(v, f, c[2], c[3], c[7], c[6]);
            add_quad(v, f, c[0], c[1], c[3], c[2]); add_quad(v, f, c[4], c[6], c[7], c[5]);
        } else {
            const int nu = 96, nv = 48; const double R = 0.37;
            for (int j = 0; j <= nv; ++j) for (int i = 0; i < nu; ++i) {
                const double th = kPi * j / nv, ph = 2 * kPi * i / nu;
                v.push_back((float)(0.013 + R * std::sin(th) * std::cos(ph)));
                v.push_back((float)(-0.021 + R * std::sin(th) * std::sin(ph)));
                v.push_back((float)(0.007 + R * std::cos(th)));
            }
            for (int j = 0; j < nv; ++j) for (int i = 0; i < nu; ++i) {
                const int a = j * nu + i, b = j * nu + (i + 1) % nu, c = (j + 1) * nu + (i + 1) % nu, d = (j + 1) * nu + i;
                f.insert(f.end(), { a, b, c, a, c, d });
            }
        }
        return true;
    }
    Mesh m; std::string err; MeshNorm nm;
    if (!load_mesh(what, m, err) || !normalize_mesh_trellis(m, nm)) { fprintf(stderr, "%s\n", err.c_str()); return false; }
    v = std::move(m.verts); f = std::move(m.faces);
    return true;
}

static int parity(const std::string& what, int G, const std::string& dir) {
    std::vector<float> v; std::vector<int32_t> f;
    if (!load_shape(what, v, f)) return 1;
    const std::string in = dir + "/in.bin", ref = dir + "/ref.bin";
    FILE* rf = fopen(ref.c_str(), "rb");
    if (!rf) {
        FILE* o = fopen(in.c_str(), "wb");
        if (!o) { perror(in.c_str()); return 1; }
        const int64_t V = (int64_t)v.size() / 3, F = (int64_t)f.size() / 3;
        fwrite(&V, 8, 1, o); fwrite(&F, 8, 1, o);
        fwrite(v.data(), 4, v.size(), o); fwrite(f.data(), 4, f.size(), o); fclose(o);
        printf("wrote %s (V=%lld F=%lld); now run trellis-ref-voxelize %s %d %s\n", in.c_str(),
               (long long)V, (long long)F, in.c_str(), G, ref.c_str());
        return 0;
    }
    int64_t N = 0;
    if (fread(&N, 8, 1, rf) != 1) return 1;
    std::vector<int32_t> rc((size_t)N * 3); std::vector<float> rd((size_t)N * 3); std::vector<uint8_t> ri((size_t)N * 3);
    if (fread(rc.data(), 4, rc.size(), rf) != rc.size() || fread(rd.data(), 4, rd.size(), rf) != rd.size() ||
        fread(ri.data(), 1, ri.size(), rf) != ri.size()) { fclose(rf); return 1; }
    fclose(rf);
    const auto t0 = std::chrono::steady_clock::now();
    const float fw = getenv("REF_FW") ? (float)atof(getenv("REF_FW")) : 1.0f;
    const float bw = getenv("REF_BW") ? (float)atof(getenv("REF_BW")) : 0.2f;
    const float rw = getenv("REF_RW") ? (float)atof(getenv("REF_RW")) : 1e-2f;
    const DualGrid dg = mesh_to_flexible_dual_grid(v.data(), (int64_t)v.size() / 3, f.data(), (int64_t)f.size() / 3, G, false, fw, bw, rw);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const size_t n = dg.coords.size();
    bool same_order = n == (size_t)N;
    for (size_t i = 0; same_order && i < n; ++i)
        for (int a = 0; a < 3; ++a) if (dg.coords[i][a] != rc[i * 3 + a]) { same_order = false; break; }
    // compare by coordinate (order-independent)
    std::map<std::array<int,3>, size_t> ri_of;
    for (size_t i = 0; i < (size_t)N; ++i) ri_of[{ rc[i * 3], rc[i * 3 + 1], rc[i * 3 + 2] }] = i;
    size_t missing = 0, inter_diff = 0, within = 0;
    double worst = 0;
    std::vector<double> errs;
    for (size_t i = 0; i < n; ++i) {
        auto it = ri_of.find(dg.coords[i]);
        if (it == ri_of.end()) { ++missing; continue; }
        const size_t j = it->second;
        for (int a = 0; a < 3; ++a) if ((dg.intersected[i * 3 + a] != 0) != (ri[j * 3 + a] != 0)) { ++inter_diff; break; }
        double e = 0;
        for (int a = 0; a < 3; ++a) e = std::max(e, (double)std::fabs(dg.dual[i * 3 + a] - rd[j * 3 + a]));
        worst = std::max(worst, e);
        if (e < 1e-3) ++within;
    }
    const size_t extra = (size_t)N + missing - n;
    const double frac = n ? (double)within / n : 0;
    printf("parity %s @%d: ours %zu voxels (%.2fs), ref %lld; same order %s; missing %zu extra %zu; "
           "intersected diff %zu; dual |d| < 1e-3 for %.4f%%, max %.3e\n",
           what.c_str(), G, n, secs, (long long)N, same_order ? "yes" : "no", missing, extra, inter_diff,
           100 * frac, worst);
    const bool ok = missing == 0 && extra == 0 && inter_diff == 0 && frac >= 0.995 && worst < 0.05;
    printf("%s\n", ok ? "PARITY OK" : "PARITY FAILED");
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc > 4 && std::string(argv[1]) == "parity") return parity(argv[2], atoi(argv[3]), argv[4]);
    test_box(64);
    test_box_dual(64);
    test_sphere(64);
    test_sphere(256);
    if (argc > 2) {
        Mesh m; std::string err; MeshNorm nm;
        if (!load_mesh(argv[1], m, err) || !normalize_mesh_trellis(m, nm)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        const int G = atoi(argv[2]);
        const auto t0 = std::chrono::steady_clock::now();
        const DualGrid dg = mesh_to_flexible_dual_grid(m.verts.data(), m.V(), m.faces.data(), m.F(), G, true);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::set<std::array<int,3>> q;
        for (auto& c : dg.coords) q.insert({ c[0] / 16, c[1] / 16, c[2] / 16 });
        printf("%s @%d: V=%d F=%d -> %zu voxels in %.2fs, %zu cells @%d\n", argv[1], G, m.V(), m.F(),
               dg.coords.size(), s, q.size(), G / 16);
    }
    printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL OK", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
