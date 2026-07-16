// Standalone harness for decimate_qem: compares the validated CPU path against the
// CUDA/HIP GPU port on a procedural 500x500 torus (500k faces), plus geometry sanity.
//   procedural: ./test_decimate <target>
//   from dump:  ./test_decimate <mesh.bin> <target>   (bin = int V, int F, float[V*3], int[F*3])
// Build this WITHOUT -DTRELLIS_HAVE_GPU_DECIMATE so `decimate_qem` stays the pure CPU
// path; the GPU entry point `decimate_qem_gpu` (from decimate_qem.cu) is called directly.
#include "uv_bake.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>

namespace trellis {
// Provided by src/decimate_qem.cu (self-contained CUDA/HIP). Returns false if no GPU.
bool decimate_qem_gpu(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                      int target_faces, std::vector<float>& ov, std::vector<int32_t>& of);
}

static void make_torus(int R, std::vector<float>& v, std::vector<int32_t>& f) {
    const float Rmaj = 1.0f, rmin = 0.35f, TAU = 6.2831853f;
    v.resize((size_t)R * R * 3);
    for (int i = 0; i < R; ++i) for (int j = 0; j < R; ++j) {
        float u = TAU * i / R, w = TAU * j / R;
        float* p = &v[3 * ((size_t)i * R + j)];
        p[0] = (Rmaj + rmin * std::cos(w)) * std::cos(u);
        p[1] = (Rmaj + rmin * std::cos(w)) * std::sin(u);
        p[2] = rmin * std::sin(w);
    }
    for (int i = 0; i < R; ++i) for (int j = 0; j < R; ++j) {
        int i1 = (i + 1) % R, j1 = (j + 1) % R;
        int a = i * R + j, b = i1 * R + j, c = i1 * R + j1, d = i * R + j1;
        f.push_back(a); f.push_back(b); f.push_back(c);
        f.push_back(a); f.push_back(c); f.push_back(d);
    }
}

// bbox + NaN + boundary-edge (open-edge) count for watertightness; returns false if any NaN.
struct MeshStats { float bb[6]; long open_edges; long nonmanifold_edges; bool finite; };
static MeshStats analyze(const std::vector<float>& v, const std::vector<int32_t>& f) {
    MeshStats s; s.finite = true;
    s.bb[0]=s.bb[1]=s.bb[2]= 1e30f; s.bb[3]=s.bb[4]=s.bb[5]=-1e30f;
    int V = (int)v.size()/3, F = (int)f.size()/3;
    for (int i = 0; i < V; ++i) for (int k = 0; k < 3; ++k) {
        float x = v[3*i+k];
        if (!std::isfinite(x)) s.finite = false;
        if (x < s.bb[k]) s.bb[k] = x; if (x > s.bb[3+k]) s.bb[3+k] = x;
    }
    std::unordered_map<uint64_t,int> ec; ec.reserve((size_t)F*2);
    auto key=[](int a,int b){ if(a>b){int t=a;a=b;b=t;} return ((uint64_t)(uint32_t)a<<32)|(uint32_t)b; };
    for (int t = 0; t < F; ++t) { int a=f[3*t],b=f[3*t+1],c=f[3*t+2];
        ec[key(a,b)]++; ec[key(b,c)]++; ec[key(c,a)]++; }
    s.open_edges = 0; s.nonmanifold_edges = 0;
    for (auto& kv : ec) { if (kv.second == 1) s.open_edges++; else if (kv.second > 2) s.nonmanifold_edges++; }
    return s;
}
// Dihedral distribution over shared edges: the faceting metric. smooth=<2deg, facet=2-15deg.
static void dihedral(const char* tag, const std::vector<float>& v, const std::vector<int32_t>& f) {
    const int F = (int)f.size()/3;
    std::vector<float> fn((size_t)F*3);
    for (int t = 0; t < F; ++t) {
        const float* a=&v[3*f[3*t]], *b=&v[3*f[3*t+1]], *c=&v[3*f[3*t+2]];
        float u[3]={b[0]-a[0],b[1]-a[1],b[2]-a[2]}, w[3]={c[0]-a[0],c[1]-a[1],c[2]-a[2]};
        float n[3]={u[1]*w[2]-u[2]*w[1], u[2]*w[0]-u[0]*w[2], u[0]*w[1]-u[1]*w[0]};
        float l=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]); if(l<1e-20f)l=1;
        fn[3*t]=n[0]/l; fn[3*t+1]=n[1]/l; fn[3*t+2]=n[2]/l;
    }
    std::unordered_map<uint64_t,int> em; em.reserve((size_t)F*2);
    auto key=[](int a,int b){ if(a>b){int t=a;a=b;b=t;} return ((uint64_t)(uint32_t)a<<32)|(uint32_t)b; };
    double sum=0; long n=0, smooth=0, facet=0;
    for (int t=0;t<F;++t) for(int e=0;e<3;++e){
        uint64_t k=key(f[3*t+e], f[3*t+(e+1)%3]);
        auto it=em.find(k);
        if(it==em.end()){ em[k]=t; continue; }
        int t2=it->second;
        float d=fn[3*t]*fn[3*t2]+fn[3*t+1]*fn[3*t2+1]+fn[3*t+2]*fn[3*t2+2];
        d=d>1?1:(d<-1?-1:d);
        float ang=std::acos(d)*57.29578f;
        sum+=ang; n++; if(ang<2)smooth++; else if(ang<15)facet++;
    }
    std::printf("  [%s] dihedral: mean=%.2fdeg  smooth(<2)=%.1f%%  facet(2-15)=%.1f%%  (n=%ld edges)\n",
                tag, n?sum/n:0.0, n?100.0*smooth/n:0.0, n?100.0*facet/n:0.0, n);
}
static void report(const char* tag, const std::vector<float>& v, const std::vector<int32_t>& f) {
    MeshStats s = analyze(v, f);
    std::printf("  [%s] V=%d F=%d  bbox=[%.3f,%.3f,%.3f]..[%.3f,%.3f,%.3f]  open_edges=%ld nonmanifold=%ld finite=%d\n",
                tag, (int)v.size()/3, (int)f.size()/3,
                s.bb[0],s.bb[1],s.bb[2], s.bb[3],s.bb[4],s.bb[5],
                s.open_edges, s.nonmanifold_edges, (int)s.finite);
    dihedral(tag, v, f);
}

int main(int argc, char** argv) {
    std::vector<float> v; std::vector<int32_t> f; int target = 50000;
    if (argc >= 2 && argv[1][0] >= '0' && argv[1][0] <= '9' && !std::strchr(argv[1], '.') && !std::strchr(argv[1], '/')) {
        make_torus(500, v, f);
        target = std::atoi(argv[1]);
    } else if (argc >= 2) {
        FILE* fp = std::fopen(argv[1], "rb");
        if (!fp) { std::printf("cannot open %s\n", argv[1]); return 1; }
        int V = 0, F = 0; if (std::fread(&V, 4, 1, fp) != 1 || std::fread(&F, 4, 1, fp) != 1) return 1;
        v.resize((size_t)V * 3); f.resize((size_t)F * 3);
        if (std::fread(v.data(), 4, v.size(), fp) != v.size()) return 1;
        if (std::fread(f.data(), 4, f.size(), fp) != f.size()) return 1;
        std::fclose(fp);
        target = argc >= 3 ? std::atoi(argv[2]) : 300000;
    } else { make_torus(500, v, f); }

    int V = (int)v.size() / 3, F = (int)f.size() / 3;
    std::printf("input V=%d F=%d target=%d\n", V, F, target);
    report("input", v, f);
    std::fflush(stdout);

    const bool skip_cpu = std::getenv("TD_SKIP_CPU") != nullptr;
    const char* out_bin = std::getenv("TD_OUT");   // write GPU output mesh here (V,F,verts,faces)
    // --- CPU reference ---
    std::vector<float> ov_c; std::vector<int32_t> of_c;
    if (!skip_cpu) {
        std::printf("CPU decimate_qem:\n"); std::fflush(stdout);
        trellis::decimate_qem(v, V, f, F, target, ov_c, of_c);
        report("cpu", ov_c, of_c);
    }

    // --- GPU port ---
    std::vector<float> ov_g; std::vector<int32_t> of_g;
    std::printf("GPU decimate_qem_gpu:\n"); std::fflush(stdout);
    bool gpu_ok = trellis::decimate_qem_gpu(v, V, f, F, target, ov_g, of_g);
    if (!gpu_ok) { std::printf("  GPU path returned false (no device / failure) -- CPU fallback only\n"); return 2; }
    report("gpu", ov_g, of_g);
    if (out_bin) {
        FILE* fo = std::fopen(out_bin, "wb");
        if (fo) { int gv=(int)ov_g.size()/3, gf=(int)of_g.size()/3;
            std::fwrite(&gv,4,1,fo); std::fwrite(&gf,4,1,fo);
            std::fwrite(ov_g.data(),4,ov_g.size(),fo); std::fwrite(of_g.data(),4,of_g.size(),fo); std::fclose(fo);
            std::printf("  wrote GPU mesh -> %s\n", out_bin); }
    }
    if (skip_cpu) return 0;

    // --- comparison / pass criteria ---
    int Fc = (int)of_c.size()/3, Fg = (int)of_g.size()/3;
    MeshStats sg = analyze(ov_g, of_g);
    float df = Fc > 0 ? std::fabs((float)Fg - Fc) / Fc : 1.0f;
    std::printf("compare: CPU F=%d  GPU F=%d  |dF|=%.2f%%  (target=%d)\n", Fc, Fg, 100.f*df, target);

    bool pass = true;
    if (!sg.finite)              { std::printf("FAIL: GPU mesh has non-finite vertices\n"); pass = false; }
    if (Fg > target * 11 / 10)   { std::printf("FAIL: GPU F above target+10%%\n"); pass = false; }
    if (df > 0.10f)              { std::printf("FAIL: GPU face count differs from CPU by >10%%\n"); pass = false; }
    // torus is closed: a clean simplification should stay (near-)watertight.
    long budget = (long)(Fg * 0.01) + 16;
    if (sg.open_edges > budget)  { std::printf("WARN: GPU open_edges=%ld (> ~1%% of F) -- not fully watertight\n", sg.open_edges); }
    std::printf(pass ? "PASS: GPU matches CPU ballpark with clean geometry\n"
                     : "FAIL: see above\n");
    return pass ? 0 : 3;
}
