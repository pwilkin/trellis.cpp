// SparseStructureDecoder: SS latent [8,16,16,16] -> occupancy logits [64,64,64].
#pragma once
#include <vector>
#include <array>

namespace trellis {
struct Model;

// z: SS latent, torch [1,8,16,16,16] memory order (== ggml channels-last [16,16,16,8]).
// Returns occupancy logits in torch [1,1,64,64,64] memory (== ggml [64,64,64,1]),
// length 64*64*64. Threshold >0 for active voxels.
std::vector<float> ss_decode(const Model& m, const std::vector<float>& z);

// Active voxel coords from occupancy logits (torch [1,1,S,S,S], flat = x*S*S+y*S+z).
// out_res<=src_res downsamples by max-pool (a coarse voxel is active iff ANY of its
// (src/out)^3 sub-voxels has logit>0), matching the reference pipeline. Returns (x,y,z)
// in row-major (z fastest) order, like torch argwhere.
std::vector<std::array<int,3>> ss_coords(const std::vector<float>& logits, int src_res, int out_res);

} // namespace trellis
