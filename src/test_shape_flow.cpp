// Validate the C++/GGML shape-SLAT flow DiT against PyTorch golden tensors, layer by layer.
//   trellis-test-shape-flow <shape_flow_1024.gguf> <ref_dir> [gpu]
//
// Our sampled latent carries a systematic per-channel mean bias (-0.1414 in normalized
// space, where the reference emits ~0.000 -- ~120 sigma outside its seed-to-seed spread).
// The normalization constants are byte-identical to the reference's, so the defect is in
// the flow itself. This walks the DiT and reports the FIRST intermediate that diverges.
//
// Same builder as the SS flow (build_dit_dense); only the RoPE table differs, being filled
// from sparse coords rather than a dense R^3 grid -- as make_sparse_runner does.
#include "trellis_model.h"
#include "dit.h"
#include "trellis_args.h"
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

// ggml [D, L] stores (d, tok) at d + D*tok; the torch dumps are [L, D] row-major, storing
// (tok, d) at tok*D + d -- the same bytes, so these compare element-for-element.
static bool compare(const char* name, const vector<float>& mine, const npy::Array& ref) {
    if ((int64_t)mine.size() != ref.numel()) {
        printf("  %-20s SIZE MISMATCH mine=%zu ref=%lld\n", name, mine.size(), (long long)ref.numel());
        return false;
    }
    double maxabs = 0, sumabs = 0, refmax = 0, msum = 0, rsum = 0;
    for (size_t i = 0; i < mine.size(); ++i) {
        const double a = mine[i], b = ref.data[i];
        maxabs = std::max(maxabs, std::fabs(a - b)); sumabs += std::fabs(a - b);
        refmax = std::max(refmax, std::fabs(b));
        msum += a; rsum += b;
    }
    const double rel = refmax > 0 ? maxabs / refmax : maxabs;
    const bool ok = rel < 2e-2;
    // mean of each side too: a per-channel bias shows up here long before max|d| explains it
    printf("  %-20s max|d|=%.3e mean|d|=%.3e rel=%.3e | mean ours=%+.5f ref=%+.5f (d=%+.5f)  %s\n",
           name, maxabs, sumabs / mine.size(), rel,
           msum / mine.size(), rsum / mine.size(), (msum - rsum) / mine.size(), ok ? "OK" : "**");
    return ok;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <shape_flow_1024.gguf> <ref_dir> [gpu]\n", argv[0]); return 1; }
    const string gguf = argv[1], ref = argv[2];
    const int gpu = argc > 3 ? atoi(argv[3]) : 0;

    // TRELLIS_NOFA=1 swaps FlashAttention for the soft_max path: FA carries known HR
    // hazards (KV tile padding, BF16 K/V), so A/B it against the golden tensors.
    if (getenv("TRELLIS_NOFA")) { trellis::g_no_fa = true; printf("(no-fa: soft_max path)\n"); }
    trellis::Model m = trellis::Model::load(gguf, gpu);
    printf("loaded %s (%zu tensors)\n", m.arch.c_str(), m.tensors.size());

    npy::Array x  = npy::load(ref + "/input_x.npy");        // [N, 32]
    npy::Array tA = npy::load(ref + "/input_t.npy");        // [1]
    npy::Array cA = npy::load(ref + "/input_cond.npy");     // [1, Lc, 1024]
    npy::Array co = npy::load(ref + "/input_coords.npy");   // [N, 3] (int32 saved as-is)
    const int64_t L = x.shape[0], Cin = x.shape[1];
    const int64_t Lc = cA.shape[1], Dc = cA.shape[2];
    const float t = tA.data[0];
    printf("inputs: L=%lld Cin=%lld Lc=%lld Dc=%lld t=%.1f\n",
           (long long)L, (long long)Cin, (long long)Lc, (long long)Dc, t);

    trellis::DiTParams p; p.in_ch = (int)Cin; p.out_ch = (int)Cin; p.d_cond = (int)Dc;
    if (getenv("TRELLIS_F32W")) { p.cast_f32 = true; printf("(f32 weight compute)\n"); }

    // x is already [N,32] row-major = ggml [32, N] channel-major: same bytes.
    vector<float> h0(x.data.begin(), x.data.begin() + (size_t)Cin * L);

    vector<float> tfreq(256);
    for (int j = 0; j < 128; ++j) {
        const float f = std::exp(-std::log(10000.f) * j / 128.f);
        tfreq[j] = std::cos(t * f); tfreq[128 + j] = std::sin(t * f);
    }

    // 3D RoPE over the SPARSE coords (the only thing that differs from the dense SS path)
    const int half = p.head_dim / 2;   // 64
    vector<float> rcos((size_t)L * half), rsin((size_t)L * half);
    float freqs[21]; for (int j = 0; j < 21; ++j) freqs[j] = 1.0f / std::pow(10000.f, j / 21.0f);
    const float* cd = co.data.data();
    for (int64_t tok = 0; tok < L; ++tok) {
        const int cx = (int)cd[tok*3 + 0], cy = (int)cd[tok*3 + 1], cz = (int)cd[tok*3 + 2];
        for (int pp = 0; pp < half; ++pp) {
            float ang = 0;
            if (pp < 21)      ang = cx * freqs[pp];
            else if (pp < 42) ang = cy * freqs[pp - 21];
            else if (pp < 63) ang = cz * freqs[pp - 42];
            rcos[tok * half + pp] = std::cos(ang); rsin[tok * half + pp] = std::sin(ang);
        }
    }

    size_t meta = ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(32768, false) + (1 << 20);
    ggml_context* c = ggml_init({ meta, nullptr, true });
    ggml_tensor* gh0  = ggml_new_tensor_2d(c, GGML_TYPE_F32, Cin, L);         ggml_set_input(gh0);
    ggml_tensor* gtf  = ggml_new_tensor_1d(c, GGML_TYPE_F32, 256);            ggml_set_input(gtf);
    ggml_tensor* gcd  = ggml_new_tensor_2d(c, GGML_TYPE_F32, Dc, Lc);         ggml_set_input(gcd);
    ggml_tensor* gcos = ggml_new_tensor_4d(c, GGML_TYPE_F32, 1, half, 1, L);  ggml_set_input(gcos);
    ggml_tensor* gsin = ggml_new_tensor_4d(c, GGML_TYPE_F32, 1, half, 1, L);  ggml_set_input(gsin);

    std::map<string, ggml_tensor*> inter;
    ggml_tensor* out = trellis::build_dit_dense(c, m, p, gh0, gtf, gcd, gcos, gsin, &inter);

    ggml_cgraph* g = ggml_new_graph_custom(c, 32768, false);
    ggml_build_forward_expand(g, out);
    for (auto& [k, v] : inter) ggml_set_output(v);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    if (!ggml_gallocr_alloc_graph(alloc, g)) { fprintf(stderr, "alloc failed\n"); return 1; }

    ggml_backend_tensor_set(gh0,  h0.data(),      0, h0.size()   * 4);
    ggml_backend_tensor_set(gtf,  tfreq.data(),   0, tfreq.size()* 4);
    ggml_backend_tensor_set(gcd,  cA.data.data(), 0, cA.numel()  * 4);
    ggml_backend_tensor_set(gcos, rcos.data(),    0, rcos.size() * 4);
    ggml_backend_tensor_set(gsin, rsin.data(),    0, rsin.size() * 4);

    if (ggml_backend_graph_compute(m.backend, g) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }

    printf("\nwalking the DiT (rel<2e-2 = OK); the FIRST ** is where it breaks:\n");
    const char* names[] = { "after_input_layer", "t_emb_mod", "blk0_msa", "blk0_cross", "blk0_mlp",
                            "after_block0", "after_block1", "after_block29", "prefinal" };
    for (const char* n : names) {
        if (!inter.count(n)) continue;
        FILE* f = fopen((ref + "/" + n + ".npy").c_str(), "rb");
        if (!f) { printf("  %-20s (no golden dump)\n", n); continue; }
        fclose(f);
        compare(n, trellis::tensor_to_f32(inter[n]), npy::load(ref + "/" + n + ".npy"));
    }
    compare("output", trellis::tensor_to_f32(out), npy::load(ref + "/output.npy"));

    ggml_gallocr_free(alloc); ggml_free(c); m.free();
    return 0;
}
