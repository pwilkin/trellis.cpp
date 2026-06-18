// Image preprocessing for TRELLIS.2: load -> white-bg removal -> square crop ->
// premultiply on black -> resize -> ImageNet normalize -> [3,S,S] torch CHW.
#pragma once
#include <vector>
#include <string>

namespace trellis {
struct Model;
// Returns [3*S*S] float in torch CHW order (== ggml [S,S,3,1]), or empty on failure.
std::vector<float> preprocess_image(const std::string& path, int S = 512);

// BiRefNet background removal: run the matte, bbox-crop, premultiply -> square RGB uint8 cutout
// (size returned in `sz`). bm = loaded birefnet model, gpu = device for the deform kernel.
std::vector<unsigned char> birefnet_cutout(const std::string& path, const Model& bm, int gpu, int& sz);
// Resize a square RGB-uint8 cutout to SxS, ImageNet-normalize -> [3,S,S] torch CHW.
std::vector<float> normalize_cutout(const std::vector<unsigned char>& rgb, int sz, int S);
} // namespace trellis
