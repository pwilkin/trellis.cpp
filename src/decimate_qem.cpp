// CPU port of CuMesh's QEM edge-collapse simplifier (refs/CuMesh/src/simplify.cu +
// dtypes.cuh QEM). Faithful to the reference: Garland-Heckbert vertex quadrics, an
// edge-collapse cost of QEM(v_new) + lambda_edge_length*|e|^2 + lambda_skinny*skinny*|e|^2
// (skinny = mean over incident tris of 1 - normalized-triangle-quality), flip rejection,
// boundary-weighted collapse target, parallel independent-set collapse per round, and the
// threshold-ladder driver (thresh=1e-8, x10 when a round removes <1% of faces, until the
// face target is met). This replaces the meshopt/FQMS stand-in whose non-adaptive, sliny
// output was the geometry-quality gap vs the reference (issues #1/#2).
#include "uv_bake.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <limits>
#include <unordered_map>

namespace trellis {

// Self-contained CUDA/HIP port (src/decimate_qem.cu). Guarded prototype: only present when
// a GPU backend was compiled in; falls through to the CPU path below on false / failure.
#ifdef TRELLIS_HAVE_GPU_DECIMATE
bool decimate_qem_gpu(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                      int target_faces, std::vector<float>& ov, std::vector<int32_t>& of);
#endif

// Self-contained Vulkan compute port (src/decimate_qem_vk.cpp). Runs the same four per-round
// kernels (qem/cost/propagate/collapse) on a headless Vulkan device; the host builds CSR
// adjacency/edges/boundary and does the stream compaction between rounds. Only present when a
// Vulkan backend was compiled in; requires 64-bit shader atomics and falls through to the CPU
// path below on false / failure (no device, missing atomics, alloc/submit error).
#ifdef TRELLIS_HAVE_VK_DECIMATE
bool decimate_qem_vk(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                     int target_faces, std::vector<float>& ov, std::vector<int32_t>& of);
#endif

namespace {

struct QEM {
    float e[10];
    void zero() { for (int i = 0; i < 10; ++i) e[i] = 0.f; }
    // add plane (a,b,c,d) as outer product p p^T (upper triangle of symmetric 4x4)
    void add_plane(float a, float b, float c, float d) {
        e[0]+=a*a; e[1]+=a*b; e[2]+=a*c; e[3]+=a*d;
        e[4]+=b*b; e[5]+=b*c; e[6]+=b*d;
        e[7]+=c*c; e[8]+=c*d; e[9]+=d*d;
    }
    // v^T Q v for v=(x,y,z,1)
    float evaluate(float x, float y, float z) const {
        return e[0]*x*x + 2*e[1]*x*y + 2*e[2]*x*z + 2*e[3]*x
             + e[4]*y*y + 2*e[5]*y*z + 2*e[6]*y
             + e[7]*z*z + 2*e[8]*z    + e[9];
    }
};

struct V3 { float x, y, z; };
inline V3   sub(const V3& a, const V3& b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline V3   cross(const V3& a, const V3& b) { return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
inline float dot(const V3& a, const V3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline float nrm2(const V3& a) { return dot(a, a); }

inline uint64_t pack_cost(int id, float c) { uint32_t b; std::memcpy(&b, &c, 4); return ((uint64_t)b << 32) | (uint32_t)id; }

// One simplify round. Mutates verts (positions of kept vertices) + faces (compacted),
// and V/F. Returns nothing; caller reads V/F. Independent-set collapse: an edge collapses
// only if it is the strict minimum-cost edge over every face incident to either endpoint,
// so no two collapses in a round share a face or vertex (matches the CUDA propagate+own check).
void simplify_round(std::vector<float>& verts, int& V, std::vector<int32_t>& faces, int& F,
                    float lam_len, float lam_skinny, float thresh) {
    auto vat = [&](int i) -> V3 { return {verts[3*i], verts[3*i+1], verts[3*i+2]}; };

    // vertex -> incident face adjacency (CSR)
    std::vector<int> off(V + 1, 0);
    for (int f = 0; f < F; ++f) for (int k = 0; k < 3; ++k) off[faces[3*f+k] + 1]++;
    for (int i = 0; i < V; ++i) off[i+1] += off[i];
    std::vector<int> v2f(off[V]);
    { std::vector<int> cur(off.begin(), off.end() - 1);
      for (int f = 0; f < F; ++f) for (int k = 0; k < 3; ++k) { int v = faces[3*f+k]; v2f[cur[v]++] = f; } }

    // unique undirected edges + boundary vertices (edge used by a single face)
    std::unordered_map<uint64_t,int> ecount;
    ecount.reserve((size_t)F * 2);
    auto ekey = [](int a, int b) -> uint64_t { if (a > b) { int t=a; a=b; b=t; } return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b; };
    for (int f = 0; f < F; ++f) {
        int a = faces[3*f], b = faces[3*f+1], c = faces[3*f+2];
        ecount[ekey(a,b)]++; ecount[ekey(b,c)]++; ecount[ekey(c,a)]++;
    }
    std::vector<uint8_t> boundary((size_t)V, 0);
    std::vector<uint64_t> edges; edges.reserve(ecount.size());
    for (auto& kv : ecount) {
        int a = (int)(kv.first >> 32), b = (int)(kv.first & 0xffffffffu);
        if (kv.second == 1) { boundary[a] = boundary[b] = 1; }
        edges.push_back(kv.first);
    }
    const int E = (int)edges.size();

    // per-vertex QEM = sum of incident face plane quadrics (normalized normals)
    std::vector<QEM> qem((size_t)V);
    for (int i = 0; i < V; ++i) {
        qem[i].zero();
        for (int j = off[i]; j < off[i+1]; ++j) {
            int f = v2f[j];
            V3 a = vat(faces[3*f]), b = vat(faces[3*f+1]), c = vat(faces[3*f+2]);
            V3 n = cross(sub(b,a), sub(c,a));
            float ln = std::sqrt(nrm2(n));
            if (ln > 1e-20f) { n.x/=ln; n.y/=ln; n.z/=ln; }
            qem[i].add_plane(n.x, n.y, n.z, -dot(n, a));
        }
    }

    // shape-metric accumulation over the faces incident to `keep` (skip those that will be
    // removed with `other`); returns false if any face flips normal.
    auto process = [&](int keep, int other, const V3& vn, float& skinny, int& ntri) -> bool {
        for (int j = off[keep]; j < off[keep+1]; ++j) {
            int f = v2f[j]; int x = faces[3*f], y = faces[3*f+1], z = faces[3*f+2];
            if (x == other || y == other || z == other) continue;
            V3 a = vat(x), b = vat(y), c = vat(z);
            V3 na = (x==keep)?vn:a, nb = (y==keep)?vn:b, nc = (z==keep)?vn:c;
            V3 on = cross(sub(b,a), sub(c,a));
            V3 ne1 = sub(nb,na), ne2 = sub(nc,na), nn = cross(ne1, ne2);
            if (dot(on, nn) < 0.f) return false;               // flip
            float narea = 0.5f * std::sqrt(nrm2(nn));
            float denom = nrm2(sub(nc,nb)) + nrm2(ne1) + nrm2(ne2);
            if (denom < 1e-12f) denom = 1e-12f;
            float sm = 4.0f * 1.7320508f * narea / denom;       // 1 = equilateral, 0 = degenerate
            skinny += 1.0f - std::min(std::max(sm, 0.0f), 1.0f);
            ntri++;
        }
        return true;
    };

    // edge collapse cost + collapse target
    std::vector<float> cost((size_t)E);
    std::vector<V3> vnew((size_t)E);
    for (int t = 0; t < E; ++t) {
        int e0 = (int)(edges[t] >> 32), e1 = (int)(edges[t] & 0xffffffffu);
        V3 v0 = vat(e0), v1 = vat(e1);
        float w0 = 0.5f;
        if (boundary[e0] && !boundary[e1]) w0 = 1.0f;
        else if (!boundary[e0] && boundary[e1]) w0 = 0.0f;
        V3 v = { v0.x*w0 + v1.x*(1-w0), v0.y*w0 + v1.y*(1-w0), v0.z*w0 + v1.z*(1-w0) };
        vnew[t] = v;
        float el2 = nrm2(sub(v1, v0));
        float c = (QEM{{qem[e0].e[0]+qem[e1].e[0], qem[e0].e[1]+qem[e1].e[1], qem[e0].e[2]+qem[e1].e[2],
                        qem[e0].e[3]+qem[e1].e[3], qem[e0].e[4]+qem[e1].e[4], qem[e0].e[5]+qem[e1].e[5],
                        qem[e0].e[6]+qem[e1].e[6], qem[e0].e[7]+qem[e1].e[7], qem[e0].e[8]+qem[e1].e[8],
                        qem[e0].e[9]+qem[e1].e[9]}}).evaluate(v.x, v.y, v.z);
        c += lam_len * el2;
        float skinny = 0.f; int ntri = 0; bool ok = true;
        ok = process(e0, e1, v, skinny, ntri) && process(e1, e0, v, skinny, ntri);
        if (!ok) { cost[t] = std::numeric_limits<float>::infinity(); continue; }
        if (ntri > 0) skinny /= ntri;
        c += lam_skinny * skinny * el2;
        cost[t] = c;
    }

    // propagate each edge's (cost,id) to all faces touching either endpoint (min-reduce)
    std::vector<uint64_t> prop((size_t)F, UINT64_MAX);
    for (int t = 0; t < E; ++t) {
        uint64_t p = pack_cost(t, cost[t]);
        int e0 = (int)(edges[t] >> 32), e1 = (int)(edges[t] & 0xffffffffu);
        for (int j = off[e0]; j < off[e0+1]; ++j) { int f = v2f[j]; if (p < prop[f]) prop[f] = p; }
        for (int j = off[e1]; j < off[e1+1]; ++j) { int f = v2f[j]; if (p < prop[f]) prop[f] = p; }
    }

    // collapse edges that win every incident face and are under threshold
    std::vector<uint8_t> vdead((size_t)V, 0), fdead((size_t)F, 0);
    for (int t = 0; t < E; ++t) {
        if (!(cost[t] <= thresh)) continue;
        uint64_t p = pack_cost(t, cost[t]);
        int e0 = (int)(edges[t] >> 32), e1 = (int)(edges[t] & 0xffffffffu);
        bool own = true;
        for (int j = off[e0]; j < off[e0+1] && own; ++j) if (prop[v2f[j]] != p) own = false;
        for (int j = off[e1]; j < off[e1+1] && own; ++j) if (prop[v2f[j]] != p) own = false;
        if (!own) continue;
        verts[3*e0] = vnew[t].x; verts[3*e0+1] = vnew[t].y; verts[3*e0+2] = vnew[t].z;
        vdead[e1] = 1;
        for (int j = off[e0]; j < off[e0+1]; ++j) { int f = v2f[j]; int* ff = &faces[3*f]; if (ff[0]==e1||ff[1]==e1||ff[2]==e1) fdead[f] = 1; }
        for (int j = off[e1]; j < off[e1+1]; ++j) { int f = v2f[j]; int* ff = &faces[3*f]; for (int k = 0; k < 3; ++k) if (ff[k]==e1) ff[k] = e0; }
    }

    // compact vertices (drop collapsed) + faces (drop deleted/degenerate)
    std::vector<int> vmap((size_t)V, -1); int nV = 0;
    for (int i = 0; i < V; ++i) if (!vdead[i]) vmap[i] = nV++;
    std::vector<float> nv((size_t)nV * 3);
    for (int i = 0; i < V; ++i) if (vmap[i] >= 0) { nv[3*vmap[i]] = verts[3*i]; nv[3*vmap[i]+1] = verts[3*i+1]; nv[3*vmap[i]+2] = verts[3*i+2]; }
    std::vector<int32_t> nf; nf.reserve(faces.size());
    for (int f = 0; f < F; ++f) {
        if (fdead[f]) continue;
        int a = vmap[faces[3*f]], b = vmap[faces[3*f+1]], c = vmap[faces[3*f+2]];
        if (a < 0 || b < 0 || c < 0 || a == b || b == c || a == c) continue;
        nf.push_back(a); nf.push_back(b); nf.push_back(c);
    }
    // Set V/F from the NEW buffers BEFORE swapping (after faces.swap(nf), nf aliases the
    // old, larger buffer -- reading its size there set F too high and walked past the mesh).
    V = nV; F = (int)nf.size() / 3;
    verts.swap(nv); faces.swap(nf);
}

} // namespace

void decimate_qem(const std::vector<float>& in_verts, int V0, const std::vector<int32_t>& in_faces, int F0,
                  int target_faces, std::vector<float>& ov, std::vector<int32_t>& of) {
    std::vector<float> verts = in_verts;
    std::vector<int32_t> faces = in_faces;
    int V = V0, F = F0;
    if (F <= target_faces) { ov = verts; of = faces; return; }

#ifdef TRELLIS_HAVE_GPU_DECIMATE
    // Run the whole simplification on the GPU when a CUDA/HIP backend is built in; on any
    // failure (no device, alloc/kernel error) fall through to the validated CPU path.
    if (decimate_qem_gpu(in_verts, V0, in_faces, F0, target_faces, ov, of)) return;
    // Any message above (e.g. "device kernel image is invalid" when the kernel was
    // built for a different GPU arch — issue #14) is non-fatal: the mesh is still
    // decimated correctly on the CPU below, just slower.
    fprintf(stderr, "[decimate] GPU decimation unavailable; falling back to the CPU path (output is unaffected)\n");
#endif

#ifdef TRELLIS_HAVE_VK_DECIMATE
    // Same, on a headless Vulkan compute device (used in Vulkan-only builds with no CUDA/HIP
    // kernel). Falls through to the CPU path on any failure or when the device lacks 64-bit atomics.
    if (decimate_qem_vk(in_verts, V0, in_faces, F0, target_faces, ov, of)) return;
#endif

    float thresh = 1e-8f;
    const float lam_len = 1e-2f, lam_skinny = 1e-3f;
    int prevF = F, stalls = 0;
    for (int round = 0; round < 400 && F > target_faces; ++round) {
        simplify_round(verts, V, faces, F, lam_len, lam_skinny, thresh);
        if (F <= target_faces) break;
        int removed = prevF - F;
        if (removed <= 0) { if (++stalls >= 2) { thresh *= 10.0f; stalls = 0; } }
        else { stalls = 0; if ((float)removed / prevF < 1e-2f) thresh *= 10.0f; }
        prevF = F;
        if (thresh > 1e12f) break;   // fully collapsed within tolerance; stop escalating
    }

    // final compaction to referenced vertices only (drop any isolated leftovers)
    std::vector<int> used((size_t)V, -1); int nV = 0;
    for (int f = 0; f < F; ++f) for (int k = 0; k < 3; ++k) { int v = faces[3*f+k]; if (used[v] < 0) used[v] = nV++; }
    ov.assign((size_t)nV * 3, 0.f);
    for (int i = 0; i < V; ++i) if (used[i] >= 0) { ov[3*used[i]] = verts[3*i]; ov[3*used[i]+1] = verts[3*i+1]; ov[3*used[i]+2] = verts[3*i+2]; }
    of.resize((size_t)F * 3);
    for (int f = 0; f < F; ++f) for (int k = 0; k < 3; ++k) of[3*f+k] = used[faces[3*f+k]];
    printf("  decimate_qem(target=%d): V %d->%d, F %d->%d (thresh=%.1e)\n", target_faces, V0, nV, F0, F, (double)thresh);
    fflush(stdout);
}

} // namespace trellis
