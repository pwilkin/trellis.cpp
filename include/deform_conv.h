// Modulated deformable conv2d (torchvision deform_conv2d v2) — custom CUDA op for BiRefNet's
// ASPPDeformable. Self-contained: host arrays in/out (NCHW, C-order), device work internally.
#pragma once

namespace trellis {

// x:[Cin,H,W]  offset:[2*K*K,H,W]  mask:[K*K,H,W] (already 2*sigmoid'd)  weight:[Cout,Cin,K,K]
// bias:[Cout] or nullptr.  stride=1, dilation=1, padding=K/2 -> out:[Cout,H,W].
// gpu = CUDA device index. Falls back to a CPU implementation if built without CUDA.
void deform_conv2d_run(const float* x, int Cin, int H, int W,
                       const float* offset, const float* mask,
                       const float* weight, const float* bias, int Cout, int K,
                       float* out, int gpu);

} // namespace trellis
