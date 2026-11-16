#include "shape_encoder.h"
#include "sparse.h"
#include "trellis_model.h"
#include "trellis_debug.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace trellis {

ShapeLatent shape_encode_feats(const Model& m, const std::vector<float>& feats6,
                               const std::vector<std::array<int,3>>& coords_in, int grid) {
    static const int kCh[5] = { 64, 128, 256, 512, 1024 };
    static const int kBlocks[5] = { 0, 4, 8, 16, 4 };
    const int N0 = (int)coords_in.size();
    if (N0 == 0 || feats6.size() != (size_t)6 * N0) throw std::runtime_error("shape_encode: bad input");
    if (grid % 16 != 0) throw std::runtime_error("shape_encode: grid must be a multiple of 16");
    const bool dbg = std::getenv("TRELLIS_DBG_ENC") != nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    auto secs = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };

    std::vector<std::array<int,3>> coords = coords_in;
    std::vector<float> h = sparse_linear_streamed(m, "input_layer", feats6, 6, N0);   // [64, N0]
    int C = kCh[0];
    for (int i = 0; i < 5; ++i) {
        const int N = (int)coords.size();
        if (kBlocks[i] > 0) {
            std::vector<int32_t> nbr = build_neighbor_table(coords);
            for (int j = 0; j < kBlocks[i]; ++j)
                h = sparse_convnext_streamed(m, "blocks." + std::to_string(i) + "." + std::to_string(j), h, C, nbr, N);
        }
        if (i < 4) {
            S2CResult r = sparse_s2c(m, "blocks." + std::to_string(i) + "." + std::to_string(kBlocks[i]),
                                     h, C, coords, kCh[i + 1]);
            h = std::move(r.feats); coords = std::move(r.coords); C = r.C;
        }
        if (dbg || verbose())
            fprintf(stderr, "      [shape-enc] stage %d: C=%d N=%d -> %d voxels (%.1fs)\n",
                    i, C, N, (int)coords.size(), secs());
    }
    const int NL = (int)coords.size();
    h = sparse_norm_streamed(m, h, C, NL, 1e-5f);
    std::vector<float> lat64 = sparse_linear_streamed(m, "to_latent", h, C, NL);       // [64, NL]
    ShapeLatent out;
    out.coords = std::move(coords);
    out.grid = grid / 16;
    out.feats.resize((size_t)32 * NL);
    for (int n = 0; n < NL; ++n)
        for (int c = 0; c < 32; ++c) out.feats[(size_t)c + 32 * n] = lat64[(size_t)c + 64 * n];   // mean half
    return out;
}

ShapeLatent shape_encode(const Model& m, const DualGrid& dg) {
    const size_t N = dg.coords.size();
    if (dg.dual.size() != N * 3 || dg.intersected.size() != N * 3)
        throw std::runtime_error("shape_encode: dual grid has no dual vertices (voxelize with coords_only = false)");
    std::vector<float> f((size_t)6 * N);
    for (size_t i = 0; i < N; ++i)
        for (int a = 0; a < 3; ++a) {
            f[6 * i + a] = dg.dual[3 * i + a] - 0.5f;
            f[6 * i + 3 + a] = (dg.intersected[3 * i + a] ? 1.0f : 0.0f) - 0.5f;
        }
    return shape_encode_feats(m, f, dg.coords, dg.grid);
}

} // namespace trellis
