// BiRefNet background removal (Swin-L backbone + ASPPDeformable decoder) in GGML.
#pragma once
#include <vector>

namespace trellis {
struct Model;

// Swin-L backbone. Input chw = [3,S,S] (S=1024 or 512), ImageNet-normalized, ggml-loadable as
// [S,S,3]. Returns the 4 stage outputs (post per-stage LayerNorm), each in torch [C,H,W] order.
struct BBOut {
    std::vector<float> f[4];           // stage features, torch [C,H,W] flat (c*H*W + h*W + w)
    int C[4] = {0}, H[4] = {0}, W[4] = {0};
};
BBOut swin_backbone(const Model& m, const std::vector<float>& chw, int S);

// Full BiRefNet forward: ImageNet-normalized 1024² input chw1024 [3,1024,1024] (ggml-loadable as
// [1024,1024,3]) -> single-channel matte LOGITS [1024,1024] (apply sigmoid for alpha). gpu = device.
std::vector<float> birefnet_matte(const Model& m, const std::vector<float>& chw1024, int gpu);

} // namespace trellis
