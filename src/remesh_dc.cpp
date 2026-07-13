#include "remesh_dc.h"
#include "tri_bvh.h"
#include <algorithm>
#include <atomic>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <cmath>
#include <cstdio>
#include <functional>
#include <thread>
#include <unordered_map>
#include <vector>

namespace trellis {

namespace {

inline int ctz64(uint64_t v) {
#ifdef _MSC_VER
    unsigned long i;
    _BitScanForward64(&i, v);
    return (int)i;
#else
    return __builtin_ctzll(v);
#endif
}

inline uint64_t key3(int x, int y, int z) {
    return ((uint64_t)(uint32_t)x << 42) | ((uint64_t)(uint32_t)y << 21) | (uint32_t)z;
}

void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn) {
    const int nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> ts;
    const int64_t chunk = (n + nt - 1) / nt;
    for (int t = 0; t < nt; ++t) {
        const int64_t b = t * chunk, e = std::min(n, b + chunk);
        if (b >= e) break;
        ts.emplace_back(fn, b, e);
    }
    for (auto& t : ts) t.join();
}

}  // namespace

Mesh remesh_narrow_band_dc(const float* iverts, int64_t iV, const int32_t* ifaces, int64_t iF,
                           const TriBvh& bvh, int res, int band) {
    (void)iV;
    Mesh out;
    if (iF == 0 || bvh.empty() || res <= 0) return out;

    // Reference domain: the world cube is inflated by (res+3·band)/res so the
    // offset shell never touches the boundary; eps is the offset distance.
    const float scale = (float)(res + 3 * band) / (float)res;
    const float cell = scale / (float)res;
    const float eps = (float)band * cell;
    const float keep = 0.87f * cell;

    // Candidate cells: conservative dilation of every triangle's AABB by the
    // band-plus-crossing radius, marked in a res^3 bitset.
    const int64_t nbits = (int64_t)res * res * res;
    std::vector<uint64_t> cand((size_t)((nbits + 63) / 64), 0);
    auto bit_set = [&cand, res](int x, int y, int z) {
        const int64_t i = ((int64_t)x * res + y) * res + z;
        cand[(size_t)(i >> 6)] |= 1ull << (i & 63);
    };
    auto bit_get = [&cand, res](int x, int y, int z) -> bool {
        const int64_t i = ((int64_t)x * res + y) * res + z;
        return (cand[(size_t)(i >> 6)] >> (i & 63)) & 1;
    };
    // A cell is active iff UDF(center) < (band+0.87)·cell, and its closest
    // surface point lies inside some triangle-marked cell, so the per-axis
    // index distance is < band+1.37, i.e. ≤ band+1.
    const int dil = band + 1;
    {
        const int F = (int)iF;
        std::vector<std::vector<uint64_t>> parts;
        const int nt = std::max(1u, std::thread::hardware_concurrency());
        parts.assign(nt, {});
        std::vector<std::thread> ts;
        const int chunk = (F + nt - 1) / nt;
        for (int t = 0; t < nt; ++t) {
            const int b = t * chunk, e = std::min(F, b + chunk);
            if (b >= e) break;
            parts[t].assign(cand.size(), 0);
            ts.emplace_back([&, t, b, e]() {
                auto& bits = parts[t];
                auto setb = [&bits, res](int x, int y, int z) {
                    const int64_t i = ((int64_t)x * res + y) * res + z;
                    bits[(size_t)(i >> 6)] |= 1ull << (i & 63);
                };
                for (int f = b; f < e; ++f) {
                    float bmin[3] = {1e30f, 1e30f, 1e30f}, bmax[3] = {-1e30f, -1e30f, -1e30f};
                    for (int j = 0; j < 3; ++j) {
                        const float* p = &iverts[3 * ifaces[3*f+j]];
                        for (int k = 0; k < 3; ++k) {
                            bmin[k] = std::min(bmin[k], p[k]);
                            bmax[k] = std::max(bmax[k], p[k]);
                        }
                    }
                    int c0[3], c1[3];
                    for (int k = 0; k < 3; ++k) {
                        c0[k] = std::max(0, (int)std::floor((bmin[k] / scale + 0.5f) * res) - dil);
                        c1[k] = std::min(res - 1, (int)std::floor((bmax[k] / scale + 0.5f) * res) + dil);
                    }
                    for (int x = c0[0]; x <= c1[0]; ++x)
                        for (int y = c0[1]; y <= c1[1]; ++y)
                            for (int z = c0[2]; z <= c1[2]; ++z) setb(x, y, z);
                }
            });
        }
        for (auto& th : ts) th.join();
        for (auto& bits : parts)
            if (!bits.empty())
                for (size_t i = 0; i < cand.size(); ++i) cand[i] |= bits[i];
    }

    // Active voxels: |UDF(center) - eps| < 0.87*cell (spec 27 §4.1).
    std::vector<int> acoord;
    {
        std::vector<int64_t> cand_cells;
        for (int64_t w = 0; w < (int64_t)cand.size(); ++w) {
            uint64_t bits = cand[w];
            while (bits) {
                const int b = ctz64(bits);
                bits &= bits - 1;
                cand_cells.push_back((w << 6) | b);
            }
        }
        std::vector<uint8_t> act(cand_cells.size(), 0);
        parallel_for((int64_t)cand_cells.size(), [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) {
                const int64_t c = cand_cells[i];
                const int x = (int)(c / ((int64_t)res * res)), y = (int)((c / res) % res), z = (int)(c % res);
                const float p[3] = { ((x + 0.5f) / res - 0.5f) * scale,
                                     ((y + 0.5f) / res - 0.5f) * scale,
                                     ((z + 0.5f) / res - 0.5f) * scale };
                const TriBvh::Hit h = bvh.closest(p, eps + keep);
                if (h.face < 0) continue;
                const float f = std::sqrt(h.dist2) - eps;
                if (std::fabs(f) < keep) act[i] = 1;
            }
        });
        for (size_t i = 0; i < cand_cells.size(); ++i) {
            if (!act[i]) continue;
            const int64_t c = cand_cells[i];
            acoord.push_back((int)(c / ((int64_t)res * res)));
            acoord.push_back((int)((c / res) % res));
            acoord.push_back((int)(c % res));
        }
        cand.clear(); cand.shrink_to_fit();
    }
    const int64_t Na = (int64_t)acoord.size() / 3;
    if (Na < 100) {
        fprintf(stderr, "  remesh: only %lld active voxels; skipping remesh\n", (long long)Na);
        return out;
    }

    std::unordered_map<uint64_t, int> vox;
    vox.reserve((size_t)Na * 2);
    for (int64_t i = 0; i < Na; ++i) vox.emplace(key3(acoord[3*i], acoord[3*i+1], acoord[3*i+2]), (int)i);

    // f = UDF - eps at the grid VERTICES (corner mapping v/res, spec 27 §4.3).
    std::vector<int> vcoord;
    std::unordered_map<uint64_t, int> vmap;
    vmap.reserve((size_t)Na * 3);
    for (int64_t i = 0; i < Na; ++i)
        for (int dx = 0; dx < 2; ++dx) for (int dy = 0; dy < 2; ++dy) for (int dz = 0; dz < 2; ++dz) {
            const int x = acoord[3*i] + dx, y = acoord[3*i+1] + dy, z = acoord[3*i+2] + dz;
            if (vmap.emplace(key3(x, y, z), (int)(vcoord.size() / 3)).second) {
                vcoord.push_back(x); vcoord.push_back(y); vcoord.push_back(z);
            }
        }
    const int64_t Nv = (int64_t)vcoord.size() / 3;
    std::vector<float> fvert((size_t)Nv);
    parallel_for(Nv, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) {
            const float p[3] = { ((float)vcoord[3*i]   / res - 0.5f) * scale,
                                 ((float)vcoord[3*i+1] / res - 0.5f) * scale,
                                 ((float)vcoord[3*i+2] / res - 0.5f) * scale };
            // Crossing edges always have their far endpoint under eps+cell
            // (f changes at most one cell-length per edge), so this bound
            // never clips a value that feeds the crossing interpolation.
            const TriBvh::Hit h = bvh.closest(p, eps + 2 * cell);
            fvert[i] = (h.face >= 0 ? std::sqrt(h.dist2) : eps + 2 * cell) - eps;
        }
    });
    auto fval = [&](int x, int y, int z) -> float {
        auto it = vmap.find(key3(x, y, z));
        return it == vmap.end() ? 1e9f : fvert[it->second];
    };

    // Dual vertices: plain mean of edge crossings, cell-center fallback; per
    // voxel, ownership of the 3 "far" edges records crossing direction
    // (spec 27 §4.4).
    std::vector<float> dual((size_t)Na * 3);
    std::vector<int8_t> owned((size_t)Na * 3, 0);
    parallel_for(Na, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) {
            const int vx = acoord[3*i], vy = acoord[3*i+1], vz = acoord[3*i+2];
            double sum[3] = {0, 0, 0};
            int cnt = 0;
            for (int axis = 0; axis < 3; ++axis)
                for (int u = 0; u < 2; ++u) for (int v = 0; v < 2; ++v) {
                    int a0[3] = {vx, vy, vz}, a1[3];
                    a0[(axis + 1) % 3] += u;
                    a0[(axis + 2) % 3] += v;
                    a1[0] = a0[0]; a1[1] = a0[1]; a1[2] = a0[2];
                    a1[axis] += 1;
                    const float v1 = fval(a0[0], a0[1], a0[2]);
                    const float v2 = fval(a1[0], a1[1], a1[2]);
                    const bool c12 = v1 < 0 && v2 >= 0, c21 = v1 >= 0 && v2 < 0;
                    if (c12 || c21) {
                        const float t = -v1 / (v2 - v1);
                        double pt[3] = {(double)a0[0], (double)a0[1], (double)a0[2]};
                        pt[axis] += t;
                        for (int k = 0; k < 3; ++k) sum[k] += pt[k];
                        ++cnt;
                    }
                    if (u == 1 && v == 1) owned[3*i + axis] = c12 ? 1 : (c21 ? -1 : 0);
                }
            if (cnt) for (int k = 0; k < 3; ++k) dual[3*i+k] = (float)(sum[k] / cnt);
            else { dual[3*i] = vx + 0.5f; dual[3*i+1] = vy + 0.5f; dual[3*i+2] = vz + 0.5f; }
        }
    });

    // Quad assembly per owned crossing edge (spec 27 §4.5); winding from the
    // crossing direction. The reference's "planar diagonal" selection is a
    // latent no-op upstream (always diagonal q0-q2 unless triangle q0q1q2 is
    // exactly degenerate) — reproduced faithfully here as split 1 always.
    static const int OFF[3][4][3] = {
        {{0,0,0}, {0,0,1}, {0,1,1}, {0,1,0}},
        {{0,0,0}, {1,0,0}, {1,0,1}, {0,0,1}},
        {{0,0,0}, {0,1,0}, {1,1,0}, {1,0,0}},
    };
    std::vector<int32_t> qfaces;
    qfaces.reserve((size_t)Na * 6);
    std::vector<uint8_t> used((size_t)Na, 0);
    for (int64_t i = 0; i < Na; ++i) {
        const int vx = acoord[3*i], vy = acoord[3*i+1], vz = acoord[3*i+2];
        for (int axis = 0; axis < 3; ++axis) {
            const int dir = owned[3*i + axis];
            if (!dir) continue;
            int q[4];
            bool ok = true;
            for (int k = 0; k < 4 && ok; ++k) {
                auto it = vox.find(key3(vx + OFF[axis][k][0], vy + OFF[axis][k][1], vz + OFF[axis][k][2]));
                if (it == vox.end()) ok = false;
                else q[k] = it->second;
            }
            if (!ok) continue;
            static const int S1N[6] = {0, 1, 2, 0, 2, 3};
            static const int S1P[6] = {0, 2, 1, 0, 3, 2};
            const int* sp = dir > 0 ? S1P : S1N;
            for (int k = 0; k < 6; ++k) qfaces.push_back(q[sp[k]]);
            for (int k = 0; k < 4; ++k) used[q[k]] = 1;
        }
    }
    if (qfaces.empty()) return out;

    // Compact used dual vertices; map back to world coordinates.
    std::vector<int32_t> remap((size_t)Na, -1);
    int nv2 = 0;
    for (int64_t i = 0; i < Na; ++i)
        if (used[i]) {
            remap[i] = nv2++;
            out.verts.push_back((dual[3*i]   / res - 0.5f) * scale);
            out.verts.push_back((dual[3*i+1] / res - 0.5f) * scale);
            out.verts.push_back((dual[3*i+2] / res - 0.5f) * scale);
        }
    out.faces.resize(qfaces.size());
    for (size_t k = 0; k < qfaces.size(); ++k) out.faces[k] = remap[qfaces[k]];
    printf("  remesh_dc: %lld active voxels -> V=%d F=%d (eps=%.4g)\n",
           (long long)Na, out.V(), out.F(), eps);
    fflush(stdout);
    return out;
}

}  // namespace trellis
