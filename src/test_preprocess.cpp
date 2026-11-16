#include "preprocess.h"
#include "stb_image_write.h"

#include <cstdio>
#include <vector>

int main() {
    constexpr int W = 4;
    constexpr int H = 8;
    const char* path = "/tmp/trellis-preprocess-test.png";
    std::vector<unsigned char> rgba((size_t)W * H * 4, 0);
    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            const size_t i = ((size_t)y * W + x) * 4;
            rgba[i + 0] = 200;
            rgba[i + 1] = 100;
            rgba[i + 2] = 50;
            rgba[i + 3] = 255;
        }
    }
    if (!stbi_write_png(path, W, H, 4, rgba.data(), W * 4)) {
        std::fprintf(stderr, "failed to create test image\n");
        return 1;
    }

    int size = 0;
    const std::vector<unsigned char> cutout = trellis::threshold_cutout(path, size);
    std::remove(path);
    if (size <= 0 || cutout.size() != (size_t)size * size * 4) {
        std::fprintf(stderr, "cutout is not RGBA: size=%d bytes=%zu\n", size, cutout.size());
        return 1;
    }
    bool transparent = false;
    bool opaque = false;
    for (size_t i = 3; i < cutout.size(); i += 4) {
        transparent |= cutout[i] == 0;
        opaque |= cutout[i] == 255;
    }
    if (!transparent || !opaque) {
        std::fprintf(stderr, "cutout alpha does not preserve foreground/background\n");
        return 1;
    }

    const std::vector<float> chw = trellis::normalize_cutout(cutout, size, 16);
    if (chw.size() != 3 * 16 * 16) {
        std::fprintf(stderr, "RGBA normalization failed: values=%zu\n", chw.size());
        return 1;
    }

    // A cutout written to disk (--dump-bg) and loaded again (--cutout) must be the same bytes:
    // no second crop, no second premultiply.
    const char* cut_path = "/tmp/trellis-preprocess-test-cutout.png";
    if (!stbi_write_png(cut_path, size, size, 4, cutout.data(), size * 4)) {
        std::fprintf(stderr, "failed to write the cutout\n");
        return 1;
    }
    int loaded_size = 0;
    const std::vector<unsigned char> loaded = trellis::load_cutout(cut_path, loaded_size);
    std::remove(cut_path);
    if (loaded_size != size || loaded != cutout) {
        std::fprintf(stderr, "loaded cutout differs from the written one: size=%d\n", loaded_size);
        return 1;
    }
    // Anything that is not square cannot be a cutout.
    if (!stbi_write_png(cut_path, W, H, 4, rgba.data(), W * 4)) return 1;
    const std::vector<unsigned char> bad = trellis::load_cutout(cut_path, loaded_size);
    std::remove(cut_path);
    if (!bad.empty() || loaded_size != 0) {
        std::fprintf(stderr, "a %dx%d image was accepted as a cutout\n", W, H);
        return 1;
    }
    std::printf("PASS: %dx%d RGBA cutout, %zu normalized values\n", size, size, chw.size());
    return 0;
}
