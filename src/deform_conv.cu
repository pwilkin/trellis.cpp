// Modulated deformable conv2d (matches torchvision.ops.deform_conv2d, v2/modulated).
// Fused kernel: one thread per output element, loops over (Cin, K, K), bilinear-samples x at the
// per-location deformed positions with zero boundary. No im2col buffer.
#include "deform_conv.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

namespace trellis {

__device__ __forceinline__ float bilinear(const float* im, int H, int W, float h, float w) {
    if (h <= -1.f || (float)H <= h || w <= -1.f || (float)W <= w) return 0.f;
    int h_low = (int)floorf(h), w_low = (int)floorf(w);
    int h_high = h_low + 1, w_high = w_low + 1;
    float lh = h - h_low, lw = w - w_low, hh = 1.f - lh, hw = 1.f - lw;
    float v1 = (h_low >= 0 && w_low >= 0)        ? im[h_low * W + w_low]   : 0.f;
    float v2 = (h_low >= 0 && w_high <= W - 1)    ? im[h_low * W + w_high]  : 0.f;
    float v3 = (h_high <= H - 1 && w_low >= 0)    ? im[h_high * W + w_low]  : 0.f;
    float v4 = (h_high <= H - 1 && w_high <= W - 1)? im[h_high * W + w_high] : 0.f;
    return hh * hw * v1 + hh * lw * v2 + lh * hw * v3 + lh * lw * v4;
}

__global__ void deform_conv_kernel(const float* x, int Cin, int H, int W,
                                   const float* offset, const float* mask,
                                   const float* weight, const float* bias, int Cout, int K, int pad,
                                   float* out) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long HW = (long)H * W, total = (long)Cout * HW;
    if (idx >= total) return;
    int oc = idx / HW;
    int rem = idx % HW, oy = rem / W, ox = rem % W;
    const int K2 = K * K;
    float acc = bias ? bias[oc] : 0.f;
    for (int ic = 0; ic < Cin; ++ic) {
        const float* xim = x + (long)ic * HW;
        const float* wrow = weight + ((long)oc * Cin + ic) * K2;   // weight[oc,ic,:,:]
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int t = kh * K + kw;
                float oh = offset[((long)(2*t) * HW) + rem];
                float ow = offset[((long)(2*t+1) * HW) + rem];
                float m  = mask[((long)t * HW) + rem];
                float ph = (float)(oy - pad + kh) + oh;
                float pw = (float)(ox - pad + kw) + ow;
                acc += wrow[t] * m * bilinear(xim, H, W, ph, pw);
            }
        }
    }
    out[idx] = acc;
}

void deform_conv2d_run(const float* x, int Cin, int H, int W,
                       const float* offset, const float* mask,
                       const float* weight, const float* bias, int Cout, int K,
                       float* out, int gpu) {
    cudaSetDevice(gpu < 0 ? 0 : gpu);   // deform always runs on a real GPU even if the model is on CPU
    long HW = (long)H * W;
    size_t sx = (size_t)Cin * HW * sizeof(float);
    size_t soff = (size_t)2 * K * K * HW * sizeof(float);
    size_t smask = (size_t)K * K * HW * sizeof(float);
    size_t sw = (size_t)Cout * Cin * K * K * sizeof(float);
    size_t sout = (size_t)Cout * HW * sizeof(float);
    float *dx, *doff, *dmask, *dw, *dbias = nullptr, *dout;
    cudaMalloc(&dx, sx); cudaMalloc(&doff, soff); cudaMalloc(&dmask, smask);
    cudaMalloc(&dw, sw); cudaMalloc(&dout, sout);
    cudaMemcpy(dx, x, sx, cudaMemcpyHostToDevice);
    cudaMemcpy(doff, offset, soff, cudaMemcpyHostToDevice);
    cudaMemcpy(dmask, mask, smask, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, weight, sw, cudaMemcpyHostToDevice);
    if (bias) { cudaMalloc(&dbias, (size_t)Cout * sizeof(float)); cudaMemcpy(dbias, bias, (size_t)Cout * sizeof(float), cudaMemcpyHostToDevice); }
    long total = (long)Cout * HW;
    int threads = 256, blocks = (int)((total + threads - 1) / threads);
    deform_conv_kernel<<<blocks, threads>>>(dx, Cin, H, W, doff, dmask, dw, dbias, Cout, K, K/2, dout);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) fprintf(stderr, "deform_conv kernel: %s\n", cudaGetErrorString(e));
    cudaMemcpy(out, dout, sout, cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(doff); cudaFree(dmask); cudaFree(dw); cudaFree(dout);
    if (dbias) cudaFree(dbias);
}

} // namespace trellis
