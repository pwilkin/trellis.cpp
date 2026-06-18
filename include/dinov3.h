// DINOv3 ViT-L/16 image conditioner -> cross-attention tokens [N, 1024].
#pragma once
#include <vector>

namespace trellis {
struct Model;

// chw: preprocessed image, torch [3,S,S] memory (== ggml [S,S,3,1]); S = 512 (or 1024).
// Returns conditioning tokens [Ntok * 1024], Ntok = (S/16)^2 + 5 (cls + 4 reg + patches),
// channel-major per token (ggml [1024, Ntok] read out as token-major [Ntok,1024]).
std::vector<float> dinov3_encode(const Model& m, const std::vector<float>& chw, int S);

} // namespace trellis
