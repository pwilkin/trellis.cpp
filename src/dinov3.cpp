#include "dinov3.h"
#include "trellis_model.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cmath>
#include <string>
#include <stdexcept>

namespace trellis {
using T = ggml_tensor;

static constexpr int D = 1024, NH = 16, HD = 64, NPREFIX = 5, NLAYERS = 24;

static T* lin(ggml_context* c, const Model& m, const std::string& p, T* x) {
    T* y = ggml_mul_mat(c, m.get(p + ".weight"), x);
    if (T* b = m.try_get(p + ".bias")) y = ggml_add(c, y, b);
    return y;
}
static T* ln(ggml_context* c, T* x, float eps, T* w = nullptr, T* b = nullptr) {
    x = ggml_norm(c, x, eps);
    if (w) x = ggml_mul(c, x, w);
    if (b) x = ggml_add(c, x, b);
    return x;
}
// half-split (NeoX) rope: x[HD,NH,N], cos/sin[HD,1,N]
static T* rope_half(ggml_context* c, T* x, T* cos, T* sin) {
    const int64_t H = x->ne[1], N = x->ne[2];
    T* x1 = ggml_cont(c, ggml_view_3d(c, x, HD/2, H, N, x->nb[1], x->nb[2], 0));
    T* x2 = ggml_cont(c, ggml_view_3d(c, x, HD/2, H, N, x->nb[1], x->nb[2], (size_t)(HD/2)*x->nb[0]));
    T* rh = ggml_concat(c, ggml_scale(c, x2, -1.0f), x1, 0);   // [-x2, x1]
    return ggml_add(c, ggml_mul(c, x, cos), ggml_mul(c, rh, sin));
}
static T* sdpa(ggml_context* c, T* q, T* k, T* v) {   // q,k,v [HD,NH,N]
    const float scale = 1.0f / std::sqrt((float)HD);
    T* q2 = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));   // [HD,N,NH]
    T* k2 = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));
    T* kq = ggml_soft_max_ext(c, ggml_mul_mat(c, k2, q2), nullptr, scale, 0.0f);
    T* v2 = ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3));   // [N,HD,NH]
    T* o = ggml_mul_mat(c, v2, kq);                         // [HD,N,NH]
    o = ggml_cont(c, ggml_permute(c, o, 0, 2, 1, 3));       // [HD,NH,N]
    return ggml_reshape_2d(c, o, D, o->ne[2]);
}
static T* attn(ggml_context* c, const Model& m, const std::string& p, T* x, T* cos, T* sin) {
    const int64_t N = x->ne[1];
    T* qkv = lin(c, m, p + ".qkv", x);                     // [3072,N], no bias
    qkv = ggml_reshape_4d(c, qkv, HD, NH, 3, N);
    auto pick = [&](int s){ return ggml_reshape_3d(c, ggml_cont(c, ggml_view_4d(c, qkv, HD, NH, 1, N, qkv->nb[1], qkv->nb[2], qkv->nb[3], (size_t)s*qkv->nb[2])), HD, NH, N); };
    T* q = pick(0); T* k = pick(1); T* v = pick(2);
    q = rope_half(c, q, cos, sin);
    k = rope_half(c, k, cos, sin);
    return lin(c, m, p + ".proj", sdpa(c, q, k, v));
}

std::vector<float> dinov3_encode(const Model& m, const std::vector<float>& chw, int S) {
    const int Hp = S / 16, NP = Hp * Hp, Ntok = NP + NPREFIX;

    // host RoPE tables [Ntok*HD]: prefix tokens identity, patches dinov3 2D half-split
    std::vector<float> rcos((size_t)Ntok * HD, 1.0f), rsin((size_t)Ntok * HD, 0.0f);
    float inv[16]; for (int j = 0; j < 16; ++j) inv[j] = 1.0f / std::pow(100.0f, (float)j / 16.0f);
    const float TWO_PI = 6.283185307179586f;
    for (int p = 0; p < NP; ++p) {
        int h = p / Hp, w = p % Hp;
        float ch = ((h + 0.5f) / Hp) * 2.0f - 1.0f, cw = ((w + 0.5f) / Hp) * 2.0f - 1.0f;
        float ang[32];
        for (int j = 0; j < 16; ++j) { ang[j] = TWO_PI * ch * inv[j]; ang[16 + j] = TWO_PI * cw * inv[j]; }
        size_t base = (size_t)(p + NPREFIX) * HD;
        for (int d = 0; d < 64; ++d) { float a = ang[d % 32]; rcos[base + d] = std::cos(a); rsin[base + d] = std::sin(a); }
    }

    size_t meta = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(16384, false) + (1 << 20);
    ggml_context* c = ggml_init({ meta, nullptr, true });
    T* img  = ggml_new_tensor_4d(c, GGML_TYPE_F32, S, S, 3, 1);       ggml_set_input(img);
    T* gcos = ggml_new_tensor_3d(c, GGML_TYPE_F32, HD, 1, Ntok);      ggml_set_input(gcos);
    T* gsin = ggml_new_tensor_3d(c, GGML_TYPE_F32, HD, 1, Ntok);      ggml_set_input(gsin);

    // patch embed: conv2d k16 s16 -> [Wp,Hp,1024,1] -> [1024, NP] (token = h*Hp + w) + bias
    T* pe = ggml_conv_2d(c, m.get("patch_embed.proj.weight"), img, 16, 16, 0, 0, 1, 1);
    pe = ggml_cont(c, ggml_permute(c, pe, 1, 2, 0, 3));               // [1024, Wp, Hp, 1]
    pe = ggml_reshape_2d(c, pe, D, NP);
    pe = ggml_add(c, pe, m.get("patch_embed.proj.bias"));
    // prefix tokens (stored f16 -> cast to f32 to match conv output)
    T* cls = ggml_cast(c, ggml_reshape_2d(c, m.get("cls_token"), D, 1), GGML_TYPE_F32);
    T* reg = ggml_cast(c, ggml_reshape_2d(c, m.get("reg_token"), D, 4), GGML_TYPE_F32);
    T* x = ggml_concat(c, ggml_concat(c, cls, reg, 1), pe, 1);        // [1024, Ntok]

    for (int i = 0; i < NLAYERS; ++i) {
        const std::string b = "blocks." + std::to_string(i);
        T* y = ln(c, x, 1e-5f, m.get(b + ".norm1.weight"), m.get(b + ".norm1.bias"));
        y = attn(c, m, b + ".attn", y, gcos, gsin);
        y = ggml_mul(c, y, m.get(b + ".gamma_1"));
        x = ggml_add(c, x, y);
        y = ln(c, x, 1e-5f, m.get(b + ".norm2.weight"), m.get(b + ".norm2.bias"));
        y = lin(c, m, b + ".mlp.fc1", y);
        y = ggml_gelu_erf(c, y);
        y = lin(c, m, b + ".mlp.fc2", y);
        y = ggml_mul(c, y, m.get(b + ".gamma_2"));
        x = ggml_add(c, x, y);
    }
    x = ln(c, x, 1e-5f);                                              // final non-affine LN
    ggml_set_output(x);

    ggml_cgraph* g = ggml_new_graph_custom(c, 16384, false);
    ggml_build_forward_expand(g, x);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    if (!ggml_gallocr_alloc_graph(alloc, g)) throw std::runtime_error("dinov3: alloc failed");
    ggml_backend_tensor_set(img,  chw.data(), 0, chw.size() * 4);
    ggml_backend_tensor_set(gcos, rcos.data(), 0, rcos.size() * 4);
    ggml_backend_tensor_set(gsin, rsin.data(), 0, rsin.size() * 4);
    if (ggml_backend_graph_compute(m.backend, g) != GGML_STATUS_SUCCESS) throw std::runtime_error("dinov3: compute failed");
    std::vector<float> out = tensor_to_f32(x);   // [D, Ntok] ggml -> flat d + D*tok
    ggml_gallocr_free(alloc); ggml_free(c);
    return out;
}

} // namespace trellis
