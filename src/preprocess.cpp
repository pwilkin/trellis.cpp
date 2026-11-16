#include "preprocess.h"
#include "birefnet.h"
#include "trellis_model.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

namespace trellis {

std::vector<float> normalize_cutout(const std::vector<unsigned char>& rgb, int sz, int S) {
    const size_t pixels = (size_t)sz * sz;
    const int channels = rgb.size() == pixels * 4 ? 4 : 3;
    if (sz <= 0 || S <= 0 || rgb.size() != pixels * channels) return {};
    std::vector<unsigned char> rs((size_t)S*S*channels);
    stbir_resize_uint8(rgb.data(), sz, sz, 0, rs.data(), S, S, 0, channels);
    const float mean[3] = {0.485f,0.456f,0.406f}, std[3] = {0.229f,0.224f,0.225f};
    std::vector<float> out((size_t)3*S*S);
    for (int c = 0; c < 3; ++c) for (int y = 0; y < S; ++y) for (int x = 0; x < S; ++x) {
        float v = rs[((size_t)y*S + x)*channels + c] / 255.0f;
        out[((size_t)c*S + y)*S + x] = (v - mean[c]) / std[c];
    }
    return out;
}

std::vector<float> cutout_to_chw01(const std::vector<unsigned char>& rgb, int sz, int S) {
    const size_t pixels = (size_t)sz * sz;
    const int channels = rgb.size() == pixels * 4 ? 4 : 3;
    if (sz <= 0 || S <= 0 || rgb.size() != pixels * channels) return {};
    std::vector<unsigned char> rs((size_t)S*S*channels);
    stbir_resize_uint8(rgb.data(), sz, sz, 0, rs.data(), S, S, 0, channels);
    std::vector<float> out((size_t)3*S*S);
    for (int c = 0; c < 3; ++c) for (int y = 0; y < S; ++y) for (int x = 0; x < S; ++x)
        out[((size_t)c*S + y)*S + x] = rs[((size_t)y*S + x)*channels + c] / 255.0f;
    return out;
}

// alpha [W*H] (>0.8 = foreground) -> bbox crop (10% margin) + premultiplied square RGBA uint8.
static std::vector<unsigned char> alpha_to_cutout(const unsigned char* rgba, int W, int H,
                                                  const std::vector<float>& alpha, int& sz) {
    int xmin = W, ymin = H, xmax = -1, ymax = -1;
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x)
        if (alpha[(size_t)y*W + x] > 0.8f) { xmin=std::min(xmin,x); ymin=std::min(ymin,y); xmax=std::max(xmax,x); ymax=std::max(ymax,y); }
    if (xmax < 0) { xmin=0; ymin=0; xmax=W-1; ymax=H-1; }
    int cx=(xmin+xmax)/2, cy=(ymin+ymax)/2;
    int half=(int)((std::max(xmax-xmin, ymax-ymin)/2 + 1) * 1.10f);
    sz = 2*half;
    std::vector<unsigned char> crop((size_t)sz*sz*4, 0);
    for (int y = 0; y < sz; ++y) for (int x = 0; x < sz; ++x) {
        int sx=cx-half+x, sy=cy-half+y;
        if (sx<0||sy<0||sx>=W||sy>=H) continue;
        float a = std::clamp(alpha[(size_t)sy*W + sx], 0.0f, 1.0f);
        const size_t dst = ((size_t)y*sz + x)*4;
        for (int c = 0; c < 3; ++c) crop[dst + c] = (unsigned char)(rgba[((size_t)sy*W + sx)*4 + c] * a);
        crop[dst + 3] = (unsigned char)std::lround(255.0f * a);
    }
    return crop;
}

bool image_has_alpha(const std::string& path) {
    int W, H, ch;
    unsigned char* img = stbi_load(path.c_str(), &W, &H, &ch, 4);
    if (!img) return false;
    bool has_alpha = false;
    for (size_t i = 0; i < (size_t)W * H; ++i) if (img[4*i+3] < 250) { has_alpha = true; break; }
    stbi_image_free(img);
    return has_alpha;
}

std::vector<unsigned char> threshold_cutout(const std::string& path, int& sz) {
    int W, H, ch;
    unsigned char* img = stbi_load(path.c_str(), &W, &H, &ch, 4);   // force RGBA
    if (!img) { fprintf(stderr, "preprocess: cannot load %s\n", path.c_str()); sz = 0; return {}; }

    // alpha: use existing alpha if the image has one and isn't all opaque; else white-bg removal.
    std::vector<float> alpha((size_t)W * H);
    bool has_alpha = false;
    for (size_t i = 0; i < (size_t)W * H; ++i) if (img[4*i+3] < 250) { has_alpha = true; break; }
    for (size_t i = 0; i < (size_t)W * H; ++i) {
        if (has_alpha) alpha[i] = img[4*i+3] / 255.0f;
        else { int mn = std::min({img[4*i], img[4*i+1], img[4*i+2]}); alpha[i] = mn < 232 ? 1.0f : 0.0f; }
    }
    std::vector<unsigned char> crop = alpha_to_cutout(img, W, H, alpha, sz);
    stbi_image_free(img);
    return crop;
}

std::vector<unsigned char> load_cutout(const std::string& path, int& sz) {
    int W, H, ch;
    sz = 0;
    unsigned char* img = stbi_load(path.c_str(), &W, &H, &ch, 4);
    if (!img) { fprintf(stderr, "preprocess: cannot load cutout %s\n", path.c_str()); return {}; }
    std::vector<unsigned char> out;
    if (W == H && W > 0) { out.assign(img, img + (size_t)W * H * 4); sz = W; }
    else fprintf(stderr, "preprocess: cutout %s is %dx%d, expected a square image\n", path.c_str(), W, H);
    stbi_image_free(img);
    return out;
}

std::vector<float> preprocess_image(const std::string& path, int S) {
    int sz;
    std::vector<unsigned char> crop = threshold_cutout(path, sz);
    if (crop.empty()) return {};
    return normalize_cutout(crop, sz, S);
}

std::vector<unsigned char> birefnet_cutout(const std::string& path, const Model& bm, int gpu, int& sz) {
    int W, H, ch;
    unsigned char* img = stbi_load(path.c_str(), &W, &H, &ch, 4);
    if (!img) { fprintf(stderr, "birefnet_cutout: cannot load %s\n", path.c_str()); sz = 0; return {}; }
    // BiRefNet is evaluated at its fixed square size, but its matte is resized
    // back to the source dimensions before cropping. Applying it to the 1024x1024
    // inference image would permanently squash non-square inputs.
    const int R = 1024;
    std::vector<unsigned char> r1024((size_t)R*R*4);
    stbir_resize_uint8(img, W, H, 0, r1024.data(), R, R, 0, 4);
    // ImageNet-normalized CHW for the matte
    const float mean[3] = {0.485f,0.456f,0.406f}, std[3] = {0.229f,0.224f,0.225f};
    std::vector<float> chw((size_t)3*R*R);
    for (int c = 0; c < 3; ++c) for (int i = 0; i < R*R; ++i)
        chw[(size_t)c*R*R + i] = (r1024[(size_t)i*4 + c] / 255.0f - mean[c]) / std[c];
    std::vector<float> logits = birefnet_matte(bm, chw, gpu);   // [R*R]
    std::vector<float> alpha1024((size_t)R*R);
    for (size_t i = 0; i < alpha1024.size(); ++i) alpha1024[i] = 1.0f / (1.0f + std::exp(-logits[i]));
    std::vector<float> alpha((size_t)W*H);
    if (!stbir_resize_float(alpha1024.data(), R, R, 0, alpha.data(), W, H, 0, 1)) {
        stbi_image_free(img);
        sz = 0;
        return {};
    }
    std::vector<unsigned char> crop = alpha_to_cutout(img, W, H, alpha, sz);
    stbi_image_free(img);
    return crop;
}

} // namespace trellis
