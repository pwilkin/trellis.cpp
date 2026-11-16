// FlexiDualGrid shape VAE encoder (TRELLIS.2 shape_enc_next_dc_f16c32): mesh dual grid ->
// shape SLAT latent. Used by --refine-mesh --refine-coords encoder (docs/spec/31-mesh-refine.md).
#pragma once
#include <array>
#include <vector>
#include "mesh_voxelize.h"

namespace trellis {
struct Model;

struct ShapeLatent {
    std::vector<float> feats;                    // [32*N0] channel-major: the posterior mean, raw
                                                 //   (not normalised by SHAPE_MEAN/STD)
    std::vector<std::array<int,3>> coords;       // latent voxels @ grid/16, lexicographic (x,y,z)
    int grid = 0;                                // latent grid (dual-grid resolution / 16)
};

// Input features [dual - 0.5, intersected - 0.5] (6 ch) -> input_layer 6->64 -> stages
// [64,128,256,512,1024] with [0,4,8,16,4] ConvNeXt blocks and an S2C down-block closing stages
// 0..3 -> LayerNorm (no affine, eps 1e-5) -> to_latent 1024->64 -> channels 0..31 (mean).
// `dg` must carry dual vertices (mesh_to_flexible_dual_grid with coords_only = false).
ShapeLatent shape_encode(const Model& m, const DualGrid& dg);

// Same, from explicit 6-channel input features [6*N] channel-major at `coords` @ `grid` (tests).
ShapeLatent shape_encode_feats(const Model& m, const std::vector<float>& feats6,
                               const std::vector<std::array<int,3>>& coords, int grid);

} // namespace trellis
