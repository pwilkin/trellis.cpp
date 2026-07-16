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
    std::vector<unsigned char> rs((size_t)S*S*3);
    stbir_resize_uint8(rgb.data(), sz, sz, 0, rs.data(), S, S, 0, 3);
    const float mean[3] = {0.485f,0.456f,0.406f}, std[3] = {0.229f,0.224f,0.225f};
    std::vector<float> out((size_t)3*S*S);
    for (int c = 0; c < 3; ++c) for (int y = 0; y < S; ++y) for (int x = 0; x < S; ++x) {
        float v = rs[((size_t)y*S + x)*3 + c] / 255.0f;
        out[((size_t)c*S + y)*S + x] = (v - mean[c]) / std[c];
    }
    return out;
}

// alpha [W*H] (>0.8 = foreground) -> bbox crop (10% margin) + premultiply on black -> square RGB uint8.
static std::vector<unsigned char> alpha_to_cutout(const unsigned char* rgba, int W, int H,
                                                  const std::vector<float>& alpha, int& sz) {
    int xmin = W, ymin = H, xmax = -1, ymax = -1;
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x)
        if (alpha[(size_t)y*W + x] > 0.8f) { xmin=std::min(xmin,x); ymin=std::min(ymin,y); xmax=std::max(xmax,x); ymax=std::max(ymax,y); }
    if (xmax < 0) { xmin=0; ymin=0; xmax=W-1; ymax=H-1; }
    int cx=(xmin+xmax)/2, cy=(ymin+ymax)/2;
    int half=(int)((std::max(xmax-xmin, ymax-ymin)/2 + 1) * 1.10f);
    sz = 2*half;
    std::vector<unsigned char> crop((size_t)sz*sz*3, 0);
    for (int y = 0; y < sz; ++y) for (int x = 0; x < sz; ++x) {
        int sx=cx-half+x, sy=cy-half+y;
        if (sx<0||sy<0||sx>=W||sy>=H) continue;
        float a = alpha[(size_t)sy*W + sx];
        for (int c = 0; c < 3; ++c) crop[((size_t)y*sz + x)*3 + c] = (unsigned char)(rgba[((size_t)sy*W + sx)*4 + c] * a);
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

std::vector<float> preprocess_image(const std::string& path, int S) {
    int W, H, ch;
    unsigned char* img = stbi_load(path.c_str(), &W, &H, &ch, 4);   // force RGBA
    if (!img) { fprintf(stderr, "preprocess: cannot load %s\n", path.c_str()); return {}; }

    // alpha: use existing alpha if the image has one and isn't all opaque; else white-bg removal.
    std::vector<float> alpha((size_t)W * H);
    bool has_alpha = false;
    for (size_t i = 0; i < (size_t)W * H; ++i) if (img[4*i+3] < 250) { has_alpha = true; break; }
    for (size_t i = 0; i < (size_t)W * H; ++i) {
        if (has_alpha) alpha[i] = img[4*i+3] / 255.0f;
        else { int mn = std::min({img[4*i], img[4*i+1], img[4*i+2]}); alpha[i] = mn < 232 ? 1.0f : 0.0f; }
    }
    int sz;
    std::vector<unsigned char> crop = alpha_to_cutout(img, W, H, alpha, sz);
    stbi_image_free(img);
    return normalize_cutout(crop, sz, S);
}

std::vector<unsigned char> birefnet_cutout(const std::string& path, const Model& bm, int gpu, int& sz) {
    int W, H, ch;
    unsigned char* img = stbi_load(path.c_str(), &W, &H, &ch, 4);
    if (!img) { fprintf(stderr, "birefnet_cutout: cannot load %s\n", path.c_str()); sz = 0; return {}; }
    // resize to 1024 RGBA for the matte + cutout
    const int R = 1024;
    std::vector<unsigned char> r1024((size_t)R*R*4);
    stbir_resize_uint8(img, W, H, 0, r1024.data(), R, R, 0, 4);
    stbi_image_free(img);
    // ImageNet-normalized CHW for the matte
    const float mean[3] = {0.485f,0.456f,0.406f}, std[3] = {0.229f,0.224f,0.225f};
    std::vector<float> chw((size_t)3*R*R);
    for (int c = 0; c < 3; ++c) for (int i = 0; i < R*R; ++i)
        chw[(size_t)c*R*R + i] = (r1024[(size_t)i*4 + c] / 255.0f - mean[c]) / std[c];
    std::vector<float> logits = birefnet_matte(bm, chw, gpu);   // [R*R]
    std::vector<float> alpha((size_t)R*R);
    for (size_t i = 0; i < alpha.size(); ++i) alpha[i] = 1.0f / (1.0f + std::exp(-logits[i]));
    return alpha_to_cutout(r1024.data(), R, R, alpha, sz);
}

} // namespace trellis
