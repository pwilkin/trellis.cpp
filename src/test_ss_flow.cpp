// Validate the C++/GGML SS-flow DiT against PyTorch golden tensors.
//   trellis-test-ss-flow <ss_flow.gguf> <ref_dir> [gpu]
#include "trellis_model.h"
#include "dit.h"
#include "npy.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

using std::vector; using std::string;

static void compare(const char* name, const vector<float>& mine, const npy::Array& ref) {
    if ((int64_t)mine.size() != ref.numel()) {
        printf("  %-20s SIZE MISMATCH mine=%zu ref=%lld\n", name, mine.size(), (long long)ref.numel());
        return;
    }
    double maxabs = 0, sumabs = 0, refmax = 0;
    for (size_t i = 0; i < mine.size(); ++i) {
        double d = std::fabs((double)mine[i] - ref.data[i]);
        maxabs = std::max(maxabs, d); sumabs += d;
        refmax = std::max(refmax, std::fabs((double)ref.data[i]));
    }
    double rel = refmax > 0 ? maxabs / refmax : maxabs;
    printf("  %-20s max|d|=%.4e mean|d|=%.4e  rel=%.4e  %s\n",
           name, maxabs, sumabs / mine.size(), rel, rel < 2e-2 ? "OK" : "**");
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <ss_flow.gguf> <ref_dir> [gpu]\n", argv[0]); return 1; }
    const string gguf = argv[1], ref = argv[2];
    const int gpu = argc > 3 ? atoi(argv[3]) : 0;

    trellis::Model m = trellis::Model::load(gguf, gpu);
    printf("loaded %s (%zu tensors)\n", m.arch.c_str(), m.tensors.size());

    // ---- reference inputs ----
    npy::Array x  = npy::load(ref + "/input_x.npy");      // [1,8,16,16,16]
    npy::Array tA = npy::load(ref + "/input_t.npy");      // [1]
    npy::Array cA = npy::load(ref + "/input_cond.npy");   // [1,1029,1024]
    const int64_t Cin = x.shape[1], R = x.shape[2], L = R * R * R;
    const int64_t Lc = cA.shape[1], Dc = cA.shape[2];
    const float t = tA.data[0];
    printf("inputs: Cin=%lld L=%lld Lc=%lld Dc=%lld t=%.1f\n", (long long)Cin,(long long)L,(long long)Lc,(long long)Dc, t);

    trellis::DiTParams p; p.in_ch = (int)Cin; p.out_ch = (int)Cin; p.d_cond = (int)Dc;
    if (getenv("TRELLIS_F32W")) { p.cast_f32 = true; printf("(f32 weight compute)\n"); }

    // ---- host-prepared inputs ----
    // h0[c + Cin*sp] = x[c*L + sp]
    vector<float> h0(Cin * L);
    for (int64_t c = 0; c < Cin; ++c) for (int64_t sp = 0; sp < L; ++sp) h0[c + Cin * sp] = x.data[c * L + sp];
    // timestep embedding (cos|sin), 256
    vector<float> tfreq(256);
    for (int j = 0; j < 128; ++j) { float f = std::exp(-std::log(10000.f) * j / 128.f); tfreq[j] = std::cos(t * f); tfreq[128 + j] = std::sin(t * f); }
    // 3D rope cos/sin: data[token*64 + pair]
    const int half = p.head_dim / 2;       // 64
    vector<float> rcos(L * half), rsin(L * half);
    float freqs[21]; for (int j = 0; j < 21; ++j) freqs[j] = 1.0f / std::pow(10000.f, j / 21.0f);
    for (int64_t tok = 0; tok < L; ++tok) {
        int cx = tok / (R * R), cy = (tok / R) % R, cz = tok % R;
        for (int pp = 0; pp < half; ++pp) {
            float ang = 0;
            if (pp < 21) ang = cx * freqs[pp];
            else if (pp < 42) ang = cy * freqs[pp - 21];
            else if (pp < 63) ang = cz * freqs[pp - 42];
            rcos[tok * half + pp] = std::cos(ang); rsin[tok * half + pp] = std::sin(ang);
        }
    }
    { // sanity-check rope against reference
        npy::Array rc = npy::load(ref + "/rope_cos.npy");
        double md = 0; for (size_t i = 0; i < rcos.size(); ++i) md = std::max(md, std::fabs((double)rcos[i] - rc.data[i]));
        printf("rope_cos vs ref max|d|=%.3e\n", md);
    }

    // ---- build graph ----
    size_t meta = ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(32768, false) + (1 << 20);
    ggml_init_params ip{ meta, nullptr, true };
    ggml_context* c = ggml_init(ip);
    ggml_tensor* gh0  = ggml_new_tensor_2d(c, GGML_TYPE_F32, Cin, L);          ggml_set_input(gh0);
    ggml_tensor* gtf  = ggml_new_tensor_1d(c, GGML_TYPE_F32, 256);             ggml_set_input(gtf);
    ggml_tensor* gcd  = ggml_new_tensor_2d(c, GGML_TYPE_F32, Dc, Lc);          ggml_set_input(gcd);
    ggml_tensor* gcos = ggml_new_tensor_4d(c, GGML_TYPE_F32, 1, half, 1, L);   ggml_set_input(gcos);
    ggml_tensor* gsin = ggml_new_tensor_4d(c, GGML_TYPE_F32, 1, half, 1, L);   ggml_set_input(gsin);

    std::map<string, ggml_tensor*> inter;
    ggml_tensor* out = trellis::build_dit_dense(c, m, p, gh0, gtf, gcd, gcos, gsin, &inter);

    ggml_cgraph* g = ggml_new_graph_custom(c, 32768, false);
    ggml_build_forward_expand(g, out);
    for (auto& [k, v] : inter) ggml_set_output(v);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    if (!ggml_gallocr_alloc_graph(alloc, g)) { fprintf(stderr, "alloc failed\n"); return 1; }

    ggml_backend_tensor_set(gh0,  h0.data(),    0, h0.size()    * 4);
    ggml_backend_tensor_set(gtf,  tfreq.data(), 0, tfreq.size() * 4);
    ggml_backend_tensor_set(gcd,  cA.data.data(), 0, cA.numel()  * 4);
    ggml_backend_tensor_set(gcos, rcos.data(),  0, rcos.size()  * 4);
    ggml_backend_tensor_set(gsin, rsin.data(),  0, rsin.size()  * 4);

    if (ggml_backend_graph_compute(m.backend, g) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }

    printf("\ncomparisons (rel<2e-2 = OK):\n");
    const char* names[] = {"after_input_layer","t_emb_mod","after_block0","after_block1","after_block29","prefinal"};
    for (const char* n : names) { FILE* f = fopen((ref + "/" + n + ".npy").c_str(), "rb"); if (!f) continue; fclose(f); compare(n, trellis::tensor_to_f32(inter[n]), npy::load(ref + "/" + n + ".npy")); }
    // output: remap mine[c+Cin*sp] -> ref[c*L+sp]
    {
        vector<float> mine = trellis::tensor_to_f32(out);   // [Cin, L] ne0=Cin
        vector<float> remap(Cin * L);
        for (int64_t cc = 0; cc < Cin; ++cc) for (int64_t sp = 0; sp < L; ++sp) remap[cc * L + sp] = mine[cc + Cin * sp];
        compare("output", remap, npy::load(ref + "/output.npy"));
    }

    ggml_gallocr_free(alloc); ggml_free(c); m.free();
    return 0;
}
