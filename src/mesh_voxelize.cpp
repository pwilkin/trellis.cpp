// Port of o-voxel's mesh_to_flexible_dual_grid (TRELLIS.2 o-voxel/src/convert/
// flexible_dual_grid.cpp). The arithmetic mirrors the original: float vertices, the scan-line
// done in double, float voxel size, voxels in first-touch order.
#include "mesh_voxelize.h"

#include <algorithm>
#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <unordered_map>

namespace trellis {

namespace {

struct D3 { double v[3]; double& operator[](int i) { return v[i]; } double operator[](int i) const { return v[i]; } };

template <typename T>
inline T lerp_s(double a, double b, double t, T va, T vb) {
    if (a == b) return va;   // avoid divide by zero
    const double alpha = (t - a) / (b - a);
    return (1 - alpha) * va + alpha * vb;
}

// 4x4 symmetric QEF in float, row-major, as Eigen::Matrix4f.
struct Q4 { float m[16]; };
inline void q_add(Q4& a, const Q4& b, float w = 1.0f) { for (int i = 0; i < 16; ++i) a.m[i] += w * b.m[i]; }

struct Acc {
    std::unordered_map<uint64_t, uint32_t> index;
    std::vector<std::array<int,3>> coords;
    std::vector<std::array<uint8_t,3>> inter;
    std::vector<std::array<float,3>> means;   // sum of intersection points
    std::vector<float> cnt;
    std::vector<Q4> qefs;
    bool qef = false;
    int grid;
    int find(int x, int y, int z) const {
        if (x < 0 || y < 0 || z < 0 || x >= grid || y >= grid || z >= grid) return -1;
        auto it = index.find(key(x, y, z));
        return it == index.end() ? -1 : (int)it->second;
    }
    uint64_t key(int x, int y, int z) const {
        return ((uint64_t)(uint32_t)x * (uint64_t)grid + (uint64_t)(uint32_t)y) * (uint64_t)grid + (uint64_t)(uint32_t)z;
    }
    // Registers an edge crossing at `p` for voxel c (o-voxel's intersect_qef bookkeeping).
    uint32_t touch(const std::array<int,3>& c, const float* p, const Q4& Q) {
        const uint64_t k = key(c[0], c[1], c[2]);
        auto it = index.find(k);
        if (it != index.end()) {
            const uint32_t i = it->second;
            if (qef) {
                for (int a = 0; a < 3; ++a) means[i][a] += p[a];
                cnt[i] += 1;
                q_add(qefs[i], Q);
            }
            return i;
        }
        const uint32_t i = (uint32_t)coords.size();
        index.emplace(k, i);
        coords.push_back(c);
        inter.push_back({0, 0, 0});
        if (qef) { means.push_back({ p[0], p[1], p[2] }); cnt.push_back(1); qefs.push_back(Q); }
        return i;
    }
};

// ---- small dense solves, mirroring Eigen's ColPivHouseholderQR<MatrixXf>::solve (Eigen 3.4) ----
// Same pivoting, Householder convention, norm downdates and nonzero-pivot cut-off, in float, so
// rank-deficient cells resolve the way the reference's colPivHouseholderQr() does (up to the
// summation order of Eigen's vectorised kernels).
template <int N>
static void colpiv_qr_solve(const float (&Ain)[N][N], const float (&bin)[N], float (&x)[N]) {
    float qr[N][N];
    for (int r = 0; r < N; ++r) for (int c = 0; c < N; ++c) qr[r][c] = Ain[r][c];
    float hcoef[N], normU[N], normD[N];
    int trans[N];
    const float eps = std::numeric_limits<float>::epsilon();
    auto colnorm = [&](int c, int r0) { float s2 = 0; for (int r = r0; r < N; ++r) s2 += qr[r][c] * qr[r][c]; return std::sqrt(s2); };
    float maxn = 0;
    for (int c = 0; c < N; ++c) { normU[c] = normD[c] = colnorm(c, 0); maxn = std::max(maxn, normU[c]); }
    const float thr_helper = (maxn * eps) * (maxn * eps) / (float)N;
    const float downdate_thr = std::sqrt(eps);
    int nonzero = N;
    for (int k = 0; k < N; ++k) {
        int big = k; float bv = normU[k];
        for (int c = k + 1; c < N; ++c) if (normU[c] > bv) { bv = normU[c]; big = c; }
        const float big_sq = bv * bv;
        if (nonzero == N && big_sq < thr_helper * (float)(N - k)) nonzero = k;
        trans[k] = big;
        if (big != k) {
            for (int r = 0; r < N; ++r) std::swap(qr[r][k], qr[r][big]);
            std::swap(normU[k], normU[big]); std::swap(normD[k], normD[big]);
        }
        // makeHouseholderInPlace on qr[k..N-1][k]
        float tail_sq = 0;
        for (int r = k + 1; r < N; ++r) tail_sq += qr[r][k] * qr[r][k];
        const float c0 = qr[k][k];
        float tau, beta;
        if (tail_sq <= std::numeric_limits<float>::min()) {
            tau = 0; beta = c0;
            for (int r = k + 1; r < N; ++r) qr[r][k] = 0;
        } else {
            beta = std::sqrt(c0 * c0 + tail_sq);
            if (c0 >= 0) beta = -beta;
            for (int r = k + 1; r < N; ++r) qr[r][k] = qr[r][k] / (c0 - beta);
            tau = (beta - c0) / beta;
        }
        hcoef[k] = tau;
        qr[k][k] = beta;
        // applyHouseholderOnTheLeft to the columns right of k, rows k..N-1
        if (k + 1 < N) {
            if (N - k == 1) {
                for (int c = k + 1; c < N; ++c) qr[k][c] *= (1 - tau);
            } else if (tau != 0) {
                for (int c = k + 1; c < N; ++c) {
                    float tmp = 0;
                    for (int r = k + 1; r < N; ++r) tmp += qr[r][k] * qr[r][c];
                    tmp += qr[k][c];
                    qr[k][c] -= tau * tmp;
                    for (int r = k + 1; r < N; ++r) qr[r][c] -= tau * qr[r][k] * tmp;
                }
            }
        }
        for (int j = k + 1; j < N; ++j) {
            if (normU[j] != 0) {
                float t = std::fabs(qr[k][j]) / normU[j];
                t = (1 + t) * (1 - t);
                t = t < 0 ? 0 : t;
                const float r = normU[j] / normD[j];
                const float t2 = t * r * r;
                if (t2 <= downdate_thr) { normD[j] = colnorm(j, k + 1); normU[j] = normD[j]; }
                else normU[j] *= std::sqrt(t);
            }
        }
    }
    int perm[N];
    for (int i = 0; i < N; ++i) perm[i] = i;
    for (int k = 0; k < N; ++k) std::swap(perm[k], perm[trans[k]]);
    if (nonzero == 0) { for (int i = 0; i < N; ++i) x[i] = 0; return; }
    float c[N];
    for (int i = 0; i < N; ++i) c[i] = bin[i];
    for (int k = 0; k < nonzero; ++k) {      // Q^T b, reflectors in order
        const float tau = hcoef[k];
        if (N - k == 1) { c[k] *= (1 - tau); continue; }
        if (tau == 0) continue;
        float tmp = 0;                         // essential^T * bottom, then + the top entry
        for (int r = k + 1; r < N; ++r) tmp += qr[r][k] * c[r];
        tmp += c[k];
        c[k] -= tau * tmp;
        for (int r = k + 1; r < N; ++r) c[r] -= tau * qr[r][k] * tmp;
    }
    for (int i = nonzero - 1; i >= 0; --i) {  // upper-triangular back substitution, column-oriented
        if (c[i] == 0.0f) continue;           // as Eigen's triangular_solve_vector
        c[i] /= qr[i][i];
        for (int r = 0; r < i; ++r) c[r] -= c[i] * qr[r][i];
    }
    for (int i = 0; i < N; ++i) x[i] = 0;
    for (int i = 0; i < nonzero; ++i) x[perm[i]] = c[i];
}

// 4-term dot product in the order of Eigen's vectorised redux over a Packet4f: the lane products,
// then (l0 + l2) + (l1 + l3). Eigen evaluates `p.transpose() * Q * p` as a row vector (each entry a
// dot with a column of Q) dotted with p, both through this reduction.
static inline float dot4(const float a[4], const float b[4]) {
    const float m0 = a[0] * b[0], m1 = a[1] * b[1], m2 = a[2] * b[2], m3 = a[3] * b[3];
    return (m0 + m2) + (m1 + m3);
}
static inline float quad_form(const Q4& Q, const float p[4]) {   // p^T Q p
    float row[4];
    for (int c = 0; c < 4; ++c) {
        const float col[4] = { Q.m[0 * 4 + c], Q.m[1 * 4 + c], Q.m[2 * 4 + c], Q.m[3 * 4 + c] };
        row[c] = dot4(p, col);
    }
    return dot4(row, p);
}

static Q4 plane_qef(const float* v0, const float* v1, const float* v2, float n_out[3]) {
    const float e0[3] = { v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2] };
    const float e1[3] = { v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2] };
    float n[3] = { e0[1] * e1[2] - e0[2] * e1[1], e0[2] * e1[0] - e0[0] * e1[2], e0[0] * e1[1] - e0[1] * e1[0] };
    const float z = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
    if (z > 0) { const float s = std::sqrt(z); n[0] /= s; n[1] /= s; n[2] /= s; }
    const float pl[4] = { n[0], n[1], n[2], -(n[0] * v0[0] + n[1] * v0[1] + n[2] * v0[2]) };
    Q4 Q;
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) Q.m[i * 4 + j] = pl[i] * pl[j];
    if (n_out) { n_out[0] = n[0]; n_out[1] = n[1]; n_out[2] = n[2]; }
    return Q;
}

}  // namespace

DualGrid mesh_to_flexible_dual_grid(const float* verts, int64_t V, const int32_t* faces, int64_t F,
                                    int grid, bool coords_only, float face_weight,
                                    float boundary_weight, float regularization_weight) {
    if (grid <= 0 || grid > (1 << 20)) throw std::runtime_error("mesh_to_flexible_dual_grid: bad grid");
    // The Python wrapper: aabb [-0.5, 0.5]^3, vertices - aabb_min, voxel_size = (max-min)/grid.
    const float vs = 1.0f / (float)grid;
    const int gmin = 0, gmax = grid;
    Acc acc; acc.grid = grid; acc.qef = !coords_only;
    acc.index.reserve((size_t)std::min<int64_t>(F * 4, (int64_t)1 << 26));
    std::vector<float> sv((size_t)V * 3);   // shifted vertices (vertices - aabb_min)
    for (int64_t i = 0; i < V * 3; ++i) sv[(size_t)i] = verts[i] + 0.5f;
    for (int64_t i = 0; i < F * 3; ++i)
        if (faces[i] < 0 || faces[i] >= V) throw std::runtime_error("mesh_to_flexible_dual_grid: face index out of range");

    // ---- intersect_qef: three scan-line sweeps ----
    const Q4 Qzero{};
    for (int64_t f = 0; f < F; ++f) {
        const float* tri[3] = { &sv[(size_t)faces[f * 3] * 3], &sv[(size_t)faces[f * 3 + 1] * 3], &sv[(size_t)faces[f * 3 + 2] * 3] };
        const Q4 Q = acc.qef ? plane_qef(tri[0], tri[1], tri[2], nullptr) : Qzero;
        for (int ax2 = 0; ax2 < 3; ++ax2) {
            const int ax0 = (ax2 + 1) % 3, ax1 = (ax2 + 2) % 3;
            // (x, y, z) = (ax0, ax1, ax2), sorted by y
            D3 t[3];
            for (int k = 0; k < 3; ++k) { t[k][0] = tri[k][ax0]; t[k][1] = tri[k][ax1]; t[k][2] = tri[k][ax2]; }
            std::sort(t, t + 3, [](const D3& a, const D3& b) { return a[1] < b[1]; });
            auto cell = [&](double c) { return std::clamp((int)(c / vs), gmin, gmax - 1); };
            const int start = cell(t[0][1]), mid = cell(t[1][1]), end = cell(t[2][1]);
            auto half = [&](int row_start, int row_end, const D3& t0, const D3& t1, const D3& t2) {
                for (int y_idx = row_start; y_idx < row_end; ++y_idx) {
                    const double y = (double)((float)(y_idx + 1) * vs);
                    double t3x = lerp_s(t0[1], t1[1], y, t0[0], t1[0]);
                    double t3z = lerp_s(t0[1], t1[1], y, t0[2], t1[2]);
                    double t4x = lerp_s(t0[1], t2[1], y, t0[0], t2[0]);
                    double t4z = lerp_s(t0[1], t2[1], y, t0[2], t2[2]);
                    if (t3x > t4x) { std::swap(t3x, t4x); std::swap(t3z, t4z); }
                    const int line_start = cell(t3x), line_end = cell(t4x);
                    for (int x_idx = line_start; x_idx < line_end; ++x_idx) {
                        const double x = (double)((float)(x_idx + 1) * vs);
                        const double z = lerp_s(t3x, t4x, x, t3z, t4z);
                        const int z_idx = (int)(z / vs);
                        if (z_idx < gmin || z_idx >= gmax) continue;
                        float ip[3];
                        ip[ax0] = (float)x; ip[ax1] = (float)y; ip[ax2] = (float)z;
                        for (int dx = 0; dx < 2; ++dx)
                            for (int dy = 0; dy < 2; ++dy) {
                                std::array<int,3> c;
                                c[ax0] = x_idx + dx; c[ax1] = y_idx + dy; c[ax2] = z_idx;
                                const uint32_t i = acc.touch(c, ip, Q);
                                if (dx == 0 && dy == 0) acc.inter[i][ax2] = 1;
                            }
                    }
                }
            };
            half(start, mid, t[0], t[1], t[2]);
            half(mid, end, t[2], t[1], t[0]);
        }
    }

    DualGrid dg;
    dg.grid = grid;
    const size_t N = acc.coords.size();
    dg.intersected.resize(N * 3);
    for (size_t i = 0; i < N; ++i)
        for (int a = 0; a < 3; ++a) dg.intersected[i * 3 + a] = acc.inter[i][a];
    if (coords_only) { dg.coords = std::move(acc.coords); return dg; }

    // ---- face_qef: triangle / voxel-box overlap (the plane Q is added unweighted; face_weight
    //      only enables the pass, as in o-voxel) ----
    if (face_weight > 0.0f) {
        for (int64_t f = 0; f < F; ++f) {
            const float* v0 = &sv[(size_t)faces[f * 3] * 3];
            const float* v1 = &sv[(size_t)faces[f * 3 + 1] * 3];
            const float* v2 = &sv[(size_t)faces[f * 3 + 2] * 3];
            float n[3];
            const Q4 Q = plane_qef(v0, v1, v2, n);
            const float e0[3] = { v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2] };
            const float e1[3] = { v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2] };
            const float e2[3] = { v0[0] - v2[0], v0[1] - v2[1], v0[2] - v2[2] };
            int bmin[3], bmax[3];
            for (int a = 0; a < 3; ++a) {
                const float lo = std::min(std::min(v0[a], v1[a]), v2[a]) / vs;
                const float hi = std::max(std::max(v0[a], v1[a]), v2[a]) / vs;
                bmin[a] = std::max((int)lo, gmin);
                bmax[a] = std::min((int)(hi + 1), gmax);
            }
            const float c[3] = { n[0] > 0 ? vs : 0.0f, n[1] > 0 ? vs : 0.0f, n[2] > 0 ? vs : 0.0f };
            const float d1 = n[0] * (c[0] - v0[0]) + n[1] * (c[1] - v0[1]) + n[2] * (c[2] - v0[2]);
            const float d2 = n[0] * (vs - c[0] - v0[0]) + n[1] * (vs - c[1] - v0[1]) + n[2] * (vs - c[2] - v0[2]);
            // 2D edge tests on the three axis planes: (ia, ib) = (x,y) sign by n.z, (y,z) by n.x, (z,x) by n.y
            struct Edge2 { float nx, ny, d; };
            auto setup = [&](int ia, int ib, int nc, Edge2 out[3]) {
                const int mul = n[nc] < 0.0f ? -1 : 1;
                const float* es[3] = { e0, e1, e2 };
                const float* vv[3] = { v0, v1, v2 };
                for (int k = 0; k < 3; ++k) {
                    out[k].nx = -mul * es[k][ib];
                    out[k].ny =  mul * es[k][ia];
                    out[k].d = -(out[k].nx * vv[k][ia] + out[k].ny * vv[k][ib]) +
                               (std::max(out[k].nx, 0.0f) * vs + std::max(out[k].ny, 0.0f) * vs);
                }
            };
            Edge2 exy[3], eyz[3], ezx[3];
            setup(0, 1, 2, exy); setup(1, 2, 0, eyz); setup(2, 0, 1, ezx);
            for (int z = bmin[2]; z < bmax[2]; ++z)
                for (int y = bmin[1]; y < bmax[1]; ++y)
                    for (int x = bmin[0]; x < bmax[0]; ++x) {
                        const float p[3] = { vs * (float)x, vs * (float)y, vs * (float)z };
                        const float np = n[0] * p[0] + n[1] * p[1] + n[2] * p[2];
                        if ((np + d1) * (np + d2) > 0.0f) continue;
                        bool ok = true;
                        for (int k = 0; k < 3 && ok; ++k) if (exy[k].nx * p[0] + exy[k].ny * p[1] + exy[k].d < 0) ok = false;
                        for (int k = 0; k < 3 && ok; ++k) if (eyz[k].nx * p[1] + eyz[k].ny * p[2] + eyz[k].d < 0) ok = false;
                        for (int k = 0; k < 3 && ok; ++k) if (ezx[k].nx * p[2] + ezx[k].ny * p[0] + ezx[k].d < 0) ok = false;
                        if (!ok) continue;
                        const int i = acc.find(x, y, z);
                        if (i >= 0) q_add(acc.qefs[(size_t)i], Q);
                    }
        }
    }

    // ---- boundry_qef: edges used by exactly one face, DDA through the voxels ----
    if (boundary_weight > 0.0f) {
        std::unordered_map<uint64_t, int> ecount;
        ecount.reserve((size_t)F * 3);
        for (int64_t f = 0; f < F; ++f)
            for (int k = 0; k < 3; ++k) {
                uint32_t a = (uint32_t)faces[f * 3 + k], b = (uint32_t)faces[f * 3 + (k + 1) % 3];
                if (a > b) std::swap(a, b);
                ecount[((uint64_t)a << 32) | b]++;
            }
        std::vector<uint64_t> bnd;
        for (const auto& e : ecount) if (e.second == 1) bnd.push_back(e.first);
        std::sort(bnd.begin(), bnd.end());   // o-voxel iterates a std::map of (v0, v1)
        for (uint64_t e : bnd) {
            const float* v0 = &sv[(size_t)(e >> 32) * 3];
            const float* v1 = &sv[(size_t)(e & 0xffffffffu) * 3];
            double dir[3] = { (double)(v1[0] - v0[0]), (double)(v1[1] - v0[1]), (double)(v1[2] - v0[2]) };
            const double len = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
            if (len < 1e-6) continue;
            for (double& d : dir) d /= len;
            float A[3][3];
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) A[i][j] = (i == j ? 1.0f : 0.0f) - (float)(dir[i] * dir[j]);
            // b = -A * v0, rounded as Eigen 3.4 evaluates a Matrix3f * Vector3f product with NEON: rows
            // 0-1 through a Packet2f multiply-add chain (fused), row 2 as a scalar p0 + (p1 + p2).
            // std::fma is exactly rounded everywhere, so this is deterministic across platforms.
            float b[3], Av0[3];
            for (int i = 0; i < 2; ++i)
                b[i] = std::fma(-A[i][2], v0[2], std::fma(-A[i][1], v0[1], -A[i][0] * v0[0]));
            b[2] = -A[2][0] * v0[0] + (-A[2][1] * v0[1] + -A[2][2] * v0[2]);
            for (int i = 0; i < 3; ++i) Av0[i] = A[i][0] * v0[0] + A[i][1] * v0[1] + A[i][2] * v0[2];
            const float cc = v0[0] * Av0[0] + v0[1] * Av0[1] + v0[2] * Av0[2];
            Q4 Q{};
            for (int i = 0; i < 3; ++i) { for (int j = 0; j < 3; ++j) Q.m[i * 4 + j] = A[i][j]; Q.m[i * 4 + 3] = b[i]; Q.m[12 + i] = b[i]; }
            Q.m[15] = cc;
            int cur[3], step[3];
            double tmax[3], tdel[3];
            for (int a = 0; a < 3; ++a) {
                cur[a] = (int)std::floor(v0[a] / vs);
                step[a] = dir[a] > 0 ? 1 : -1;
                if (dir[a] == 0.0) { tmax[a] = tdel[a] = std::numeric_limits<double>::infinity(); }
                else {
                    const float border = vs * (float)(cur[a] + (step[a] > 0 ? 1 : 0));
                    tmax[a] = (double)(border - v0[a]) / dir[a];   // float subtraction, as o-voxel
                    tdel[a] = (double)vs / std::fabs(dir[a]);
                }
            }
            std::vector<std::array<int,3>> path;
            path.push_back({ cur[0], cur[1], cur[2] });
            for (;;) {
                int axis;
                if (tmax[0] < tmax[1]) axis = tmax[0] < tmax[2] ? 0 : 2;
                else                   axis = tmax[1] < tmax[2] ? 1 : 2;
                if (tmax[axis] > len) break;
                cur[axis] += step[axis];
                tmax[axis] += tdel[axis];
                path.push_back({ cur[0], cur[1], cur[2] });
            }
            for (const auto& c : path) {
                const int i = acc.find(c[0], c[1], c[2]);
                if (i >= 0) q_add(acc.qefs[(size_t)i], Q, boundary_weight);
            }
        }
    }

    // Debug: TRELLIS_DUMP_QEF=<path> writes i64 N, then per voxel i32 coord[3], f32 Q[16] (before
    // the regulariser), f32 mean_sum[3], f32 cnt -- for checking the solve against Eigen offline.
    if (const char* qp = std::getenv("TRELLIS_DUMP_QEF")) {
        if (FILE* qf = std::fopen(qp, "wb")) {
            const int64_t n64 = (int64_t)N;
            std::fwrite(&n64, 8, 1, qf);
            for (size_t i = 0; i < N; ++i) {
                std::fwrite(acc.coords[i].data(), 4, 3, qf); std::fwrite(acc.qefs[i].m, 4, 16, qf);
                std::fwrite(acc.means[i].data(), 4, 3, qf); std::fwrite(&acc.cnt[i], 4, 1, qf);
            }
            std::fclose(qf);
        }
    }
    // ---- solve: unconstrained, then 6 faces, 12 edges, 8 corners of the voxel box ----
    dg.dual.resize(N * 3);
    for (size_t i = 0; i < N; ++i) {
        const auto& cd = acc.coords[i];
        Q4 Q = acc.qefs[i];
        const float mn[3] = { cd[0] * vs, cd[1] * vs, cd[2] * vs };
        const float mx[3] = { (cd[0] + 1) * vs, (cd[1] + 1) * vs, (cd[2] + 1) * vs };
        if (regularization_weight > 0.0f) {
            const float p[3] = { acc.means[i][0] / acc.cnt[i], acc.means[i][1] / acc.cnt[i], acc.means[i][2] / acc.cnt[i] };
            Q4 R{};
            for (int a = 0; a < 3; ++a) { R.m[a * 4 + a] = 1; R.m[a * 4 + 3] = -p[a]; R.m[12 + a] = -p[a]; }
            R.m[15] = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
            q_add(Q, R, regularization_weight * acc.cnt[i]);
        }
        auto q = [&](int r, int c) { return Q.m[r * 4 + c]; };
        float A3[3][3], b3[3], v[3];
        for (int r = 0; r < 3; ++r) { for (int c = 0; c < 3; ++c) A3[r][c] = q(r, c); b3[r] = -q(r, 3); }
        colpiv_qr_solve<3>(A3, b3, v);
        auto inside = [&](const float* p) {
            return p[0] >= mn[0] && p[0] <= mx[0] && p[1] >= mn[1] && p[1] <= mx[1] && p[2] >= mn[2] && p[2] <= mx[2];
        };
        if (!inside(v)) {
            float best = std::numeric_limits<float>::infinity();
            auto consider = [&](const float p4[4]) {
                const float err = quad_form(Q, p4);
                if (err < best) { best = err; v[0] = p4[0]; v[1] = p4[1]; v[2] = p4[2]; }
            };
            for (int fa = 0; fa < 3; ++fa) {            // one fixed axis
                const int a1 = (fa + 1) % 3, a2 = (fa + 2) % 3;
                const float A2[2][2] = { { q(a1, a1), q(a1, a2) }, { q(a2, a1), q(a2, a2) } };
                const float B[2][2] = { { q(a1, fa), q(a1, 3) }, { q(a2, fa), q(a2, 3) } };
                for (float fixed : { mn[fa], mx[fa] }) {
                    const float bb[2] = { -(B[0][0] * fixed + B[0][1]), -(B[1][0] * fixed + B[1][1]) };
                    float xs[2];
                    colpiv_qr_solve<2>(A2, bb, xs);
                    if (xs[0] >= mn[a1] && xs[0] <= mx[a1] && xs[1] >= mn[a2] && xs[1] <= mx[a2]) {
                        float p4[4]; p4[fa] = fixed; p4[a1] = xs[0]; p4[a2] = xs[1]; p4[3] = 1.0f;
                        consider(p4);
                    }
                }
            }
            for (int fr = 0; fr < 3; ++fr) {            // one free axis
                const int a1 = (fr + 1) % 3, a2 = (fr + 2) % 3;
                const float a = q(fr, fr);
                const float b[3] = { q(fr, a1), q(fr, a2), q(fr, 3) };
                const float c1s[2] = { mn[a1], mx[a1] }, c2s[2] = { mn[a2], mx[a2] };
                for (int u = 0; u < 2; ++u) for (int w = 0; w < 2; ++w) {
                    const float qq[3] = { c1s[u], c2s[w], 1.0f };
                    const float x = -(b[0] * qq[0] + b[1] * qq[1] + b[2] * qq[2]) / a;
                    if (x >= mn[fr] && x <= mx[fr]) {
                        float p4[4]; p4[fr] = x; p4[a1] = c1s[u]; p4[a2] = c2s[w]; p4[3] = 1.0f;
                        consider(p4);
                    }
                }
            }
            for (int xc = 0; xc < 2; ++xc) for (int yc = 0; yc < 2; ++yc) for (int zc = 0; zc < 2; ++zc) {
                const float p4[4] = { xc ? mn[0] : mx[0], yc ? mn[1] : mx[1], zc ? mn[2] : mx[2], 1.0f };
                consider(p4);
            }
        }
        // the pipeline's feats: dual_vertices * resolution - voxel_indices
        for (int a = 0; a < 3; ++a) dg.dual[i * 3 + a] = v[a] * (float)grid - (float)cd[a];
    }
    dg.coords = std::move(acc.coords);
    return dg;
}

}  // namespace trellis
