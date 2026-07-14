// Self-contained CUDA/HIP port of CuMesh's QEM edge-collapse simplifier
// (refs/CuMesh/src/simplify.cu + connectivity.cu + dtypes.cuh QEM), mirroring the
// validated CPU port in src/decimate_qem.cpp exactly:
//   * per-vertex Garland-Heckbert quadrics from incident face planes (guarded normalize),
//   * edge-collapse cost = QEM(v_new) + lam_len*|e|^2 + lam_skinny*skinny*|e|^2, INFINITY on flip,
//   * boundary-weighted midpoint (w0 = 0.5, or 1.0/0.0 if exactly one endpoint is boundary),
//   * parallel independent-set collapse via 64-bit atomicMin cost propagation + per-face ownership,
//   * threshold-ladder driver (thresh=1e-8, x10 when a round removes <1% of faces, until F<=target).
// All rounds run on the GPU (adjacency, edges, boundary, qem, cost, propagate, collapse, compaction).
// hipcub/cub provide DeviceScan (compaction/CSR offsets), DeviceRadixSort + DeviceRunLengthEncode
// (edge dedup). Returns false with no side effects on any failure so the caller falls back to CPU.

#include <vector>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <limits>

#if defined(__HIP__) || defined(__HIP_PLATFORM_AMD__) || defined(GGML_USE_HIP)
  #include <hip/hip_runtime.h>
  #include <hipcub/hipcub.hpp>
  namespace gpucub = hipcub;
  #define gpuError_t              hipError_t
  #define gpuSuccess              hipSuccess
  #define gpuMalloc               hipMalloc
  #define gpuFree                 hipFree
  #define gpuMemcpy               hipMemcpy
  #define gpuMemcpyAsync          hipMemcpyAsync
  #define gpuMemsetAsync          hipMemsetAsync
  #define gpuMemcpyHostToDevice   hipMemcpyHostToDevice
  #define gpuMemcpyDeviceToHost   hipMemcpyDeviceToHost
  #define gpuStream_t             hipStream_t
  #define gpuStreamCreate         hipStreamCreate
  #define gpuStreamDestroy        hipStreamDestroy
  #define gpuStreamSynchronize    hipStreamSynchronize
  #define gpuGetLastError         hipGetLastError
  #define gpuGetErrorString       hipGetErrorString
  #define gpuGetDeviceCount       hipGetDeviceCount
#else
  #include <cuda_runtime.h>
  #include <cub/cub.cuh>
  namespace gpucub = cub;
  #define gpuError_t              cudaError_t
  #define gpuSuccess              cudaSuccess
  #define gpuMalloc               cudaMalloc
  #define gpuFree                 cudaFree
  #define gpuMemcpy               cudaMemcpy
  #define gpuMemcpyAsync          cudaMemcpyAsync
  #define gpuMemsetAsync          cudaMemsetAsync
  #define gpuMemcpyHostToDevice   cudaMemcpyHostToDevice
  #define gpuMemcpyDeviceToHost   cudaMemcpyDeviceToHost
  #define gpuStream_t             cudaStream_t
  #define gpuStreamCreate         cudaStreamCreate
  #define gpuStreamDestroy        cudaStreamDestroy
  #define gpuStreamSynchronize    cudaStreamSynchronize
  #define gpuGetLastError         cudaGetLastError
  #define gpuGetErrorString       cudaGetErrorString
  #define gpuGetDeviceCount       cudaGetDeviceCount
#endif

// ROCm's clang marks hip runtime calls [[nodiscard]]; the buffer teardown paths below
// intentionally fire-and-forget the free()/streamDestroy() results (errors there cannot be
// acted on and must not mask the real failure). Silence just that class of warning on clang.
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-value"
#pragma clang diagnostic ignored "-Wunused-result"
#endif

namespace trellis {
namespace {

constexpr int BLK = 256;
static inline int NB(long long n) { return (int)((n + BLK - 1) / BLK); }

#define GPU_OK(x) do { gpuError_t e_ = (x); if (e_ != gpuSuccess) { \
    fprintf(stderr, "[decimate_qem_gpu] %s:%d: %s\n", __FILE__, __LINE__, gpuGetErrorString(e_)); \
    return false; } } while (0)
#define GPU_LAST() do { gpuError_t e_ = gpuGetLastError(); if (e_ != gpuSuccess) { \
    fprintf(stderr, "[decimate_qem_gpu] %s:%d: kernel: %s\n", __FILE__, __LINE__, gpuGetErrorString(e_)); \
    return false; } } while (0)

struct QEM { float e[10]; };

// ------- small device vector helpers (mirror dtypes.cuh Vec3f semantics) -------
__device__ __forceinline__ float3 vsub(const float3& a, const float3& b) { return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); }
__device__ __forceinline__ float3 vcross(const float3& a, const float3& b) { return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x); }
__device__ __forceinline__ float  vdot(const float3& a, const float3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }

__device__ __forceinline__ void qem_add_plane(QEM& q, float a, float b, float c, float d) {
    q.e[0]+=a*a; q.e[1]+=a*b; q.e[2]+=a*c; q.e[3]+=a*d;
    q.e[4]+=b*b; q.e[5]+=b*c; q.e[6]+=b*d;
    q.e[7]+=c*c; q.e[8]+=c*d; q.e[9]+=d*d;
}
__device__ __forceinline__ float qem_eval(const QEM& q, float x, float y, float z) {
    return q.e[0]*x*x + 2.f*q.e[1]*x*y + 2.f*q.e[2]*x*z + 2.f*q.e[3]*x
         + q.e[4]*y*y + 2.f*q.e[5]*y*z + 2.f*q.e[6]*y
         + q.e[7]*z*z + 2.f*q.e[8]*z    + q.e[9];
}

// ---------------------------------- kernels ----------------------------------
__global__ void k_count_faces(const int3* faces, int F, int* cnt) {
    int t = blockIdx.x*blockDim.x + threadIdx.x; if (t >= F) return;
    int3 f = faces[t];
    atomicAdd(&cnt[f.x], 1); atomicAdd(&cnt[f.y], 1); atomicAdd(&cnt[f.z], 1);
}
__global__ void k_fill_v2f(const int3* faces, int F, int* v2f, const int* off, int* cnt) {
    int t = blockIdx.x*blockDim.x + threadIdx.x; if (t >= F) return;
    int3 f = faces[t];
    v2f[off[f.x] + atomicAdd(&cnt[f.x], 1)] = t;
    v2f[off[f.y] + atomicAdd(&cnt[f.y], 1)] = t;
    v2f[off[f.z] + atomicAdd(&cnt[f.z], 1)] = t;
}
__global__ void k_expand_edges(const int3* faces, int F, uint64_t* edges) {
    int t = blockIdx.x*blockDim.x + threadIdx.x; if (t >= F) return;
    int3 f = faces[t]; int b = t*3;
    int xy0 = f.x < f.y ? f.x : f.y, xy1 = f.x < f.y ? f.y : f.x;
    int yz0 = f.y < f.z ? f.y : f.z, yz1 = f.y < f.z ? f.z : f.y;
    int zx0 = f.z < f.x ? f.z : f.x, zx1 = f.z < f.x ? f.x : f.z;
    edges[b+0] = ((uint64_t)(uint32_t)xy0 << 32) | (uint32_t)xy1;
    edges[b+1] = ((uint64_t)(uint32_t)yz0 << 32) | (uint32_t)yz1;
    edges[b+2] = ((uint64_t)(uint32_t)zx0 << 32) | (uint32_t)zx1;
}
__global__ void k_boundary(const uint64_t* edges, const int* ecnt, int E, uint8_t* boundary) {
    int t = blockIdx.x*blockDim.x + threadIdx.x; if (t >= E) return;
    if (ecnt[t] == 1) {
        uint64_t e = edges[t];
        boundary[(int)(e >> 32)] = 1;
        boundary[(int)(e & 0xffffffffu)] = 1;
    }
}
__global__ void k_qem(const float3* V, const int3* faces, const int* v2f, const int* off, int Vn, QEM* qems) {
    int t = blockIdx.x*blockDim.x + threadIdx.x; if (t >= Vn) return;
    QEM q; for (int i = 0; i < 10; ++i) q.e[i] = 0.f;
    for (int j = off[t]; j < off[t+1]; ++j) {
        int3 fv = faces[v2f[j]];
        float3 a = V[fv.x], b = V[fv.y], c = V[fv.z];
        float3 n = vcross(vsub(b,a), vsub(c,a));
        float ln = sqrtf(vdot(n,n));
        if (ln > 1e-20f) { n.x/=ln; n.y/=ln; n.z/=ln; }
        qem_add_plane(q, n.x, n.y, n.z, -vdot(n,a));
    }
    qems[t] = q;
}
// mirror CPU `process`: accumulate skinny-metric over one face incident to `keep`,
// skipping faces that also contain `other`; false if the face flips normal.
__device__ __forceinline__ bool proc_tri(int f, int keep, int other, const float3* V, const int3* faces,
                                          const float3& vn, float& skinny, int& ntri) {
    int3 fv = faces[f];
    if (fv.x == other || fv.y == other || fv.z == other) return true;   // removed with `other`
    float3 a = V[fv.x], b = V[fv.y], c = V[fv.z];
    float3 na = (fv.x == keep) ? vn : a;
    float3 nb = (fv.y == keep) ? vn : b;
    float3 nc = (fv.z == keep) ? vn : c;
    float3 on  = vcross(vsub(b,a), vsub(c,a));
    float3 ne1 = vsub(nb,na), ne2 = vsub(nc,na);
    float3 nn  = vcross(ne1, ne2);
    if (vdot(on, nn) < 0.f) return false;                                // flip
    float narea = 0.5f * sqrtf(vdot(nn,nn));
    float3 ne0 = vsub(nc,nb);
    float denom = vdot(ne0,ne0) + vdot(ne1,ne1) + vdot(ne2,ne2);
    if (denom < 1e-12f) denom = 1e-12f;
    float sm = 4.0f * 1.7320508f * narea / denom;
    skinny += 1.0f - fminf(fmaxf(sm, 0.0f), 1.0f);
    ntri++;
    return true;
}
__global__ void k_cost(const float3* V, const int3* faces, const int* v2f, const int* off,
                        const uint64_t* edges, const uint8_t* boundary, const QEM* qems,
                        int E, float lam_len, float lam_skinny, float* cost) {
    int t = blockIdx.x*blockDim.x + threadIdx.x; if (t >= E) return;
    uint64_t e = edges[t]; int e0 = (int)(e >> 32), e1 = (int)(e & 0xffffffffu);
    float3 v0 = V[e0], v1 = V[e1];
    float w0 = 0.5f;
    if (boundary[e0] && !boundary[e1]) w0 = 1.0f;
    else if (!boundary[e0] && boundary[e1]) w0 = 0.0f;
    float3 v = make_float3(v0.x*w0 + v1.x*(1.f-w0), v0.y*w0 + v1.y*(1.f-w0), v0.z*w0 + v1.z*(1.f-w0));
    QEM q; for (int i = 0; i < 10; ++i) q.e[i] = qems[e0].e[i] + qems[e1].e[i];
    float c = qem_eval(q, v.x, v.y, v.z);
    float3 d = vsub(v1, v0); float el2 = vdot(d, d);
    c += lam_len * el2;
    float skinny = 0.f; int ntri = 0;
    for (int j = off[e0]; j < off[e0+1]; ++j)
        if (!proc_tri(v2f[j], e0, e1, V, faces, v, skinny, ntri)) { cost[t] = INFINITY; return; }
    for (int j = off[e1]; j < off[e1+1]; ++j)
        if (!proc_tri(v2f[j], e1, e0, V, faces, v, skinny, ntri)) { cost[t] = INFINITY; return; }
    if (ntri > 0) skinny /= ntri;
    c += lam_skinny * skinny * el2;
    cost[t] = c;
}
__global__ void k_propagate(const uint64_t* edges, const int* v2f, const int* off,
                            const float* cost, int E, unsigned long long* prop) {
    int t = blockIdx.x*blockDim.x + threadIdx.x; if (t >= E) return;
    uint64_t e = edges[t]; int e0 = (int)(e >> 32), e1 = (int)(e & 0xffffffffu);
    unsigned long long p = ((unsigned long long)__float_as_uint(cost[t]) << 32) | (unsigned int)t;
    for (int j = off[e0]; j < off[e0+1]; ++j) atomicMin(&prop[v2f[j]], p);
    for (int j = off[e1]; j < off[e1+1]; ++j) atomicMin(&prop[v2f[j]], p);
}
__global__ void k_collapse(float3* V, int3* faces, const uint64_t* edges, const int* v2f, const int* off,
                           const float* cost, const unsigned long long* prop, const uint8_t* boundary,
                           int E, float thresh, int* vdead, uint8_t* fdead) {
    int t = blockIdx.x*blockDim.x + threadIdx.x; if (t >= E) return;
    float c = cost[t];
    if (!(c <= thresh)) return;                                          // excludes INF/NaN
    uint64_t e = edges[t]; int e0 = (int)(e >> 32), e1 = (int)(e & 0xffffffffu);
    unsigned long long p = ((unsigned long long)__float_as_uint(c) << 32) | (unsigned int)t;
    for (int j = off[e0]; j < off[e0+1]; ++j) if (prop[v2f[j]] != p) return;
    for (int j = off[e1]; j < off[e1+1]; ++j) if (prop[v2f[j]] != p) return;
    // owns every incident face -> collapse (independent-set: disjoint faces/vertices, so writes race-free)
    float3 v0 = V[e0], v1 = V[e1];
    float w0 = 0.5f;
    if (boundary[e0] && !boundary[e1]) w0 = 1.0f;
    else if (!boundary[e0] && boundary[e1]) w0 = 0.0f;
    V[e0] = make_float3(v0.x*w0 + v1.x*(1.f-w0), v0.y*w0 + v1.y*(1.f-w0), v0.z*w0 + v1.z*(1.f-w0));
    vdead[e1] = 1;
    for (int j = off[e0]; j < off[e0+1]; ++j) {
        int fid = v2f[j]; int3 fv = faces[fid];
        if (fv.x == e1 || fv.y == e1 || fv.z == e1) fdead[fid] = 1;      // shared faces removed
    }
    for (int j = off[e1]; j < off[e1+1]; ++j) {
        int fid = v2f[j]; int3 fv = faces[fid];
        if      (fv.x == e1) fv.x = e0;
        else if (fv.y == e1) fv.y = e0;
        else if (fv.z == e1) fv.z = e0;
        faces[fid] = fv;                                                 // retarget e1 -> e0
    }
}
__global__ void k_vkeep(const int* vdead, int Vn, int* vkeep) {
    int t = blockIdx.x*blockDim.x + threadIdx.x; if (t >= Vn) return;
    vkeep[t] = vdead[t] ? 0 : 1;
}
__global__ void k_compact_verts(const int* vdead, const int* vscan, const float3* oldV, int Vn, float3* newV) {
    int t = blockIdx.x*blockDim.x + threadIdx.x; if (t >= Vn) return;
    if (!vdead[t]) newV[vscan[t]] = oldV[t];
}
__global__ void k_fkeep(const int3* faces, const uint8_t* fdead, const int* vdead, const int* vscan,
                        int Fn, int* fkeep) {
    int t = blockIdx.x*blockDim.x + threadIdx.x; if (t >= Fn) return;
    if (fdead[t]) { fkeep[t] = 0; return; }
    int3 fv = faces[t];
    if (vdead[fv.x] || vdead[fv.y] || vdead[fv.z]) { fkeep[t] = 0; return; }
    int a = vscan[fv.x], b = vscan[fv.y], c = vscan[fv.z];
    fkeep[t] = (a == b || b == c || a == c) ? 0 : 1;                     // drop degenerate
}
__global__ void k_compact_faces(const int3* faces, const int* fkeep, const int* fscan, const int* vscan,
                                int Fn, int3* newF) {
    int t = blockIdx.x*blockDim.x + threadIdx.x; if (t >= Fn) return;
    if (!fkeep[t]) return;
    int3 fv = faces[t];
    newF[fscan[t]] = make_int3(vscan[fv.x], vscan[fv.y], vscan[fv.z]);
}

// ------------------------------- cub wrappers -------------------------------
static bool cub_exclusive_sum(const int* in, int* out, int n, gpuStream_t s) {
    size_t bytes = 0;
    GPU_OK(gpucub::DeviceScan::ExclusiveSum(nullptr, bytes, in, out, n, s));
    void* tmp = nullptr; GPU_OK(gpuMalloc(&tmp, bytes ? bytes : 1));
    gpuError_t e = gpucub::DeviceScan::ExclusiveSum(tmp, bytes, in, out, n, s);
    gpuFree(tmp);
    GPU_OK(e);
    return true;
}
static bool cub_sort_keys(const uint64_t* in, uint64_t* out, int n, gpuStream_t s) {
    size_t bytes = 0;
    GPU_OK(gpucub::DeviceRadixSort::SortKeys(nullptr, bytes, in, out, n, 0, 64, s));
    void* tmp = nullptr; GPU_OK(gpuMalloc(&tmp, bytes ? bytes : 1));
    gpuError_t e = gpucub::DeviceRadixSort::SortKeys(tmp, bytes, in, out, n, 0, 64, s);
    gpuFree(tmp);
    GPU_OK(e);
    return true;
}
static bool cub_rle(const uint64_t* in, uint64_t* uniq, int* counts, int* num_runs, int n, gpuStream_t s) {
    size_t bytes = 0;
    GPU_OK(gpucub::DeviceRunLengthEncode::Encode(nullptr, bytes, in, uniq, counts, num_runs, n, s));
    void* tmp = nullptr; GPU_OK(gpuMalloc(&tmp, bytes ? bytes : 1));
    gpuError_t e = gpucub::DeviceRunLengthEncode::Encode(tmp, bytes, in, uniq, counts, num_runs, n, s);
    gpuFree(tmp);
    GPU_OK(e);
    return true;
}

// One simplify round entirely on the GPU. Reallocates d_verts/d_faces to the compacted
// buffers and updates V/F. Returns false on any GPU error (caller frees + falls back).
static bool simplify_round_gpu(float3*& d_verts, int& V, int3*& d_faces, int& F,
                               float lam_len, float lam_skinny, float thresh, gpuStream_t s) {
    const int E3 = F * 3;

    // --- vertex->face adjacency (CSR) ---
    int *d_cnt = nullptr, *d_off = nullptr, *d_v2f = nullptr;
    GPU_OK(gpuMalloc(&d_cnt, sizeof(int) * (size_t)(V + 1)));
    GPU_OK(gpuMalloc(&d_off, sizeof(int) * (size_t)(V + 1)));
    GPU_OK(gpuMalloc(&d_v2f, sizeof(int) * (size_t)E3));
    GPU_OK(gpuMemsetAsync(d_cnt, 0, sizeof(int) * (size_t)(V + 1), s));
    k_count_faces<<<NB(F), BLK, 0, s>>>(d_faces, F, d_cnt);
    GPU_LAST();
    if (!cub_exclusive_sum(d_cnt, d_off, V + 1, s)) return false;
    GPU_OK(gpuMemsetAsync(d_cnt, 0, sizeof(int) * (size_t)(V + 1), s));
    k_fill_v2f<<<NB(F), BLK, 0, s>>>(d_faces, F, d_v2f, d_off, d_cnt);
    GPU_LAST();

    // --- unique undirected edges + counts ---
    uint64_t *d_edge_exp = nullptr, *d_edge_sorted = nullptr, *d_edges = nullptr;
    int *d_ecnt = nullptr, *d_num_runs = nullptr;
    GPU_OK(gpuMalloc(&d_edge_exp,    sizeof(uint64_t) * (size_t)E3));
    GPU_OK(gpuMalloc(&d_edge_sorted, sizeof(uint64_t) * (size_t)E3));
    GPU_OK(gpuMalloc(&d_edges,       sizeof(uint64_t) * (size_t)E3));
    GPU_OK(gpuMalloc(&d_ecnt,        sizeof(int) * (size_t)E3));
    GPU_OK(gpuMalloc(&d_num_runs,    sizeof(int)));
    k_expand_edges<<<NB(F), BLK, 0, s>>>(d_faces, F, d_edge_exp);
    GPU_LAST();
    if (!cub_sort_keys(d_edge_exp, d_edge_sorted, E3, s)) return false;
    if (!cub_rle(d_edge_sorted, d_edges, d_ecnt, d_num_runs, E3, s)) return false;
    int E = 0;
    GPU_OK(gpuMemcpyAsync(&E, d_num_runs, sizeof(int), gpuMemcpyDeviceToHost, s));
    GPU_OK(gpuStreamSynchronize(s));

    bool ok = true;
    if (E > 0) {
        // --- boundary vertices ---
        uint8_t* d_boundary = nullptr;
        GPU_OK(gpuMalloc(&d_boundary, sizeof(uint8_t) * (size_t)V));
        GPU_OK(gpuMemsetAsync(d_boundary, 0, sizeof(uint8_t) * (size_t)V, s));
        k_boundary<<<NB(E), BLK, 0, s>>>(d_edges, d_ecnt, E, d_boundary);
        GPU_LAST();

        // --- per-vertex QEM ---
        QEM* d_qem = nullptr;
        GPU_OK(gpuMalloc(&d_qem, sizeof(QEM) * (size_t)V));
        k_qem<<<NB(V), BLK, 0, s>>>(d_verts, d_faces, d_v2f, d_off, V, d_qem);
        GPU_LAST();

        // --- edge collapse cost ---
        float* d_cost = nullptr;
        GPU_OK(gpuMalloc(&d_cost, sizeof(float) * (size_t)E));
        k_cost<<<NB(E), BLK, 0, s>>>(d_verts, d_faces, d_v2f, d_off, d_edges, d_boundary, d_qem,
                                     E, lam_len, lam_skinny, d_cost);
        GPU_LAST();

        // --- propagate (cost<<32|id) to incident faces via atomicMin ---
        unsigned long long* d_prop = nullptr;
        GPU_OK(gpuMalloc(&d_prop, sizeof(unsigned long long) * (size_t)F));
        GPU_OK(gpuMemsetAsync(d_prop, 0xff, sizeof(unsigned long long) * (size_t)F, s)); // UINT64_MAX
        k_propagate<<<NB(E), BLK, 0, s>>>(d_edges, d_v2f, d_off, d_cost, E, d_prop);
        GPU_LAST();

        // --- collapse owners under threshold ---
        int* d_vdead = nullptr; uint8_t* d_fdead = nullptr;
        GPU_OK(gpuMalloc(&d_vdead, sizeof(int) * (size_t)V));
        GPU_OK(gpuMalloc(&d_fdead, sizeof(uint8_t) * (size_t)F));
        GPU_OK(gpuMemsetAsync(d_vdead, 0, sizeof(int) * (size_t)V, s));
        GPU_OK(gpuMemsetAsync(d_fdead, 0, sizeof(uint8_t) * (size_t)F, s));
        k_collapse<<<NB(E), BLK, 0, s>>>(d_verts, d_faces, d_edges, d_v2f, d_off, d_cost, d_prop,
                                         d_boundary, E, thresh, d_vdead, d_fdead);
        GPU_LAST();

        // --- compact vertices ---
        int *d_vkeep = nullptr, *d_vscan = nullptr;
        GPU_OK(gpuMalloc(&d_vkeep, sizeof(int) * (size_t)(V + 1)));
        GPU_OK(gpuMalloc(&d_vscan, sizeof(int) * (size_t)(V + 1)));
        k_vkeep<<<NB(V), BLK, 0, s>>>(d_vdead, V, d_vkeep);
        GPU_LAST();
        if (!cub_exclusive_sum(d_vkeep, d_vscan, V + 1, s)) ok = false;
        int newV = 0;
        if (ok) { GPU_OK(gpuMemcpyAsync(&newV, d_vscan + V, sizeof(int), gpuMemcpyDeviceToHost, s));
                  GPU_OK(gpuStreamSynchronize(s)); }
        float3* d_newV = nullptr;
        if (ok) { GPU_OK(gpuMalloc(&d_newV, sizeof(float3) * (size_t)(newV > 0 ? newV : 1)));
                  k_compact_verts<<<NB(V), BLK, 0, s>>>(d_vdead, d_vscan, d_verts, V, d_newV);
                  GPU_LAST(); }

        // --- compact faces ---
        int *d_fkeep = nullptr, *d_fscan = nullptr;
        int newF = 0; int3* d_newF = nullptr;
        if (ok) {
            GPU_OK(gpuMalloc(&d_fkeep, sizeof(int) * (size_t)(F + 1)));
            GPU_OK(gpuMalloc(&d_fscan, sizeof(int) * (size_t)(F + 1)));
            k_fkeep<<<NB(F), BLK, 0, s>>>(d_faces, d_fdead, d_vdead, d_vscan, F, d_fkeep);
            GPU_LAST();
            if (!cub_exclusive_sum(d_fkeep, d_fscan, F + 1, s)) ok = false;
            if (ok) { GPU_OK(gpuMemcpyAsync(&newF, d_fscan + F, sizeof(int), gpuMemcpyDeviceToHost, s));
                      GPU_OK(gpuStreamSynchronize(s)); }
            if (ok) { GPU_OK(gpuMalloc(&d_newF, sizeof(int3) * (size_t)(newF > 0 ? newF : 1)));
                      k_compact_faces<<<NB(F), BLK, 0, s>>>(d_faces, d_fkeep, d_fscan, d_vscan, F, d_newF);
                      GPU_LAST(); }
        }

        if (ok) GPU_OK(gpuStreamSynchronize(s));

        // swap in the compacted buffers
        if (ok) {
            gpuFree(d_verts); gpuFree(d_faces);
            d_verts = d_newV; d_faces = d_newF; V = newV; F = newF;
            d_newV = nullptr; d_newF = nullptr;
        }
        if (d_newV) gpuFree(d_newV);
        if (d_newF) gpuFree(d_newF);
        gpuFree(d_boundary); gpuFree(d_qem); gpuFree(d_cost); gpuFree(d_prop);
        gpuFree(d_vdead); gpuFree(d_fdead);
        gpuFree(d_vkeep); gpuFree(d_vscan);
        if (d_fkeep) gpuFree(d_fkeep);
        if (d_fscan) gpuFree(d_fscan);
    }

    gpuFree(d_cnt); gpuFree(d_off); gpuFree(d_v2f);
    gpuFree(d_edge_exp); gpuFree(d_edge_sorted); gpuFree(d_edges);
    gpuFree(d_ecnt); gpuFree(d_num_runs);
    return ok;
}

} // namespace

// C++-linkage entry point (declared in decimate_qem.cpp under TRELLIS_HAVE_GPU_DECIMATE).
bool decimate_qem_gpu(const std::vector<float>& in_verts, int V0,
                      const std::vector<int32_t>& in_faces, int F0,
                      int target_faces, std::vector<float>& ov, std::vector<int32_t>& of) {
    int devcount = 0;
    if (gpuGetDeviceCount(&devcount) != gpuSuccess || devcount <= 0) return false;
    if (V0 <= 0 || F0 <= 0) return false;
    if (F0 <= target_faces) { ov = in_verts; of = in_faces; return true; }

    gpuStream_t s;
    if (gpuStreamCreate(&s) != gpuSuccess) return false;

    float3* d_verts = nullptr; int3* d_faces = nullptr;
    bool ok = true;
    if (gpuMalloc(&d_verts, sizeof(float3) * (size_t)V0) != gpuSuccess) ok = false;
    if (ok && gpuMalloc(&d_faces, sizeof(int3) * (size_t)F0) != gpuSuccess) ok = false;
    if (ok && gpuMemcpyAsync(d_verts, in_verts.data(), sizeof(float3) * (size_t)V0,
                             gpuMemcpyHostToDevice, s) != gpuSuccess) ok = false;
    if (ok && gpuMemcpyAsync(d_faces, in_faces.data(), sizeof(int3) * (size_t)F0,
                             gpuMemcpyHostToDevice, s) != gpuSuccess) ok = false;
    if (ok && gpuStreamSynchronize(s) != gpuSuccess) ok = false;

    int V = V0, F = F0;
    if (ok) {
        float thresh = 1e-8f;
        const float lam_len = 1e-2f, lam_skinny = 1e-3f;
        int prevF = F, stalls = 0;
        for (int round = 0; round < 400 && F > target_faces; ++round) {
            if (!simplify_round_gpu(d_verts, V, d_faces, F, lam_len, lam_skinny, thresh, s)) { ok = false; break; }
            if (F <= target_faces) break;
            int removed = prevF - F;
            if (removed <= 0) { if (++stalls >= 2) { thresh *= 10.0f; stalls = 0; } }
            else { stalls = 0; if ((float)removed / prevF < 1e-2f) thresh *= 10.0f; }
            prevF = F;
            if (thresh > 1e12f) break;
        }
    }

    std::vector<float> hv; std::vector<int32_t> hf;
    if (ok && V > 0 && F > 0) {
        hv.resize((size_t)V * 3); hf.resize((size_t)F * 3);
        if (gpuMemcpy(hv.data(), d_verts, sizeof(float3) * (size_t)V, gpuMemcpyDeviceToHost) != gpuSuccess) ok = false;
        if (ok && gpuMemcpy(hf.data(), d_faces, sizeof(int3) * (size_t)F, gpuMemcpyDeviceToHost) != gpuSuccess) ok = false;
    } else {
        ok = false;
    }

    if (d_verts) gpuFree(d_verts);
    if (d_faces) gpuFree(d_faces);
    gpuStreamDestroy(s);
    if (!ok) return false;

    // final compaction to referenced vertices only (matches CPU decimate_qem tail)
    std::vector<int> used((size_t)V, -1); int nV = 0;
    for (int f = 0; f < F; ++f)
        for (int k = 0; k < 3; ++k) { int v = hf[3*f+k]; if (v < 0 || v >= V) return false; if (used[v] < 0) used[v] = nV++; }
    ov.assign((size_t)nV * 3, 0.f);
    for (int i = 0; i < V; ++i) if (used[i] >= 0) { ov[3*used[i]] = hv[3*i]; ov[3*used[i]+1] = hv[3*i+1]; ov[3*used[i]+2] = hv[3*i+2]; }
    of.resize((size_t)F * 3);
    for (int f = 0; f < F; ++f) for (int k = 0; k < 3; ++k) of[3*f+k] = used[hf[3*f+k]];
    printf("  decimate_qem_gpu(target=%d): V %d->%d, F %d->%d\n", target_faces, V0, nV, F0, F);
    fflush(stdout);
    return true;
}

} // namespace trellis
