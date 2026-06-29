// CPU fallback for modulated deformable conv2d (matches deform_conv.cu and
// torchvision.ops.deform_conv2d v2/modulated). Used by BiRefNet's ASPPDeformable
// on builds without the CUDA kernel (Vulkan / CPU-only). Host arrays in/out, same
// NCHW C-order layout as the CUDA path; parallelized over output channels.
#include "deform_conv.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace trellis {

static inline float bilinear_cpu(const float* im, int H, int W, float h, float w) {
    if (h <= -1.f || (float)H <= h || w <= -1.f || (float)W <= w) return 0.f;
    int h_low = (int)std::floor(h), w_low = (int)std::floor(w);
    int h_high = h_low + 1, w_high = w_low + 1;
    float lh = h - h_low, lw = w - w_low, hh = 1.f - lh, hw = 1.f - lw;
    float v1 = (h_low >= 0 && w_low >= 0)             ? im[h_low * W + w_low]   : 0.f;
    float v2 = (h_low >= 0 && w_high <= W - 1)        ? im[h_low * W + w_high]  : 0.f;
    float v3 = (h_high <= H - 1 && w_low >= 0)        ? im[h_high * W + w_low]  : 0.f;
    float v4 = (h_high <= H - 1 && w_high <= W - 1)   ? im[h_high * W + w_high] : 0.f;
    return hh * hw * v1 + hh * lw * v2 + lh * hw * v3 + lh * lw * v4;
}

void deform_conv2d_cpu(const float* x, int Cin, int H, int W,
                       const float* offset, const float* mask,
                       const float* weight, const float* bias, int Cout, int K,
                       float* out, int /*gpu*/) {
    const long HW  = (long)H * W;
    const int  K2  = K * K;
    const int  pad = K / 2;
    const bool log_timing = std::getenv("TRELLIS_DBG_DEFORM") != nullptr;
    auto t0 = std::chrono::steady_clock::now();

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int oc = 0; oc < Cout; ++oc) {
        for (long rem = 0; rem < HW; ++rem) {
            const int oy = (int)(rem / W), ox = (int)(rem % W);
            float acc = bias ? bias[oc] : 0.f;
            for (int ic = 0; ic < Cin; ++ic) {
                const float* xim  = x + (long)ic * HW;
                const float* wrow = weight + ((long)oc * Cin + ic) * K2;   // weight[oc,ic,:,:]
                for (int kh = 0; kh < K; ++kh) {
                    for (int kw = 0; kw < K; ++kw) {
                        const int t = kh * K + kw;
                        const float oh = offset[((long)(2*t)   * HW) + rem];
                        const float ow = offset[((long)(2*t+1) * HW) + rem];
                        const float m  = mask[((long)t * HW) + rem];
                        const float ph = (float)(oy - pad + kh) + oh;
                        const float pw = (float)(ox - pad + kw) + ow;
                        acc += wrow[t] * m * bilinear_cpu(xim, H, W, ph, pw);
                    }
                }
            }
            out[(long)oc * HW + rem] = acc;
        }
    }

    if (log_timing) {
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[deform_cpu] Cin=%d Cout=%d H=%d W=%d K=%d -> %.1f ms\n",
                Cin, Cout, H, W, K, ms);
    }
}

#ifndef TRELLIS_DEFORM_VULKAN
// Pure-CPU build: the public entry is the CPU kernel directly. (Vulkan builds
// define deform_conv2d_run in deform_conv_vk.cpp, which falls back here.)
void deform_conv2d_run(const float* x, int Cin, int H, int W,
                       const float* offset, const float* mask,
                       const float* weight, const float* bias, int Cout, int K,
                       float* out, int gpu) {
    deform_conv2d_cpu(x, Cin, H, W, offset, mask, weight, bias, Cout, K, out, gpu);
}
#endif

} // namespace trellis
