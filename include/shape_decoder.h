// FlexiDualGrid shape VAE decoder: shape SLAT latent -> dual-grid head [7, M] @ res 512.
#pragma once
#include <vector>
#include <array>
#include <cstddef>
#include <cstdint>

namespace trellis {
struct Model;

struct ShapeOut {
    std::vector<float> feats7;                 // [7*M] channel-major (ne0=7): per voxel dual-grid fields
    std::vector<std::array<int,3>> coords;     // [M] voxel coords at final resolution
    int res = 512;
    std::vector<std::vector<uint8_t>> subs;    // per-C2S binarized subdiv masks (for the tex decoder)
};

// latent: shape SLAT feats [32*N0] channel-major; coords0: active voxels. resolution = final
// grid size (coords0 res * 16): 512 for res-32 input, 1024 for res-64 input (cascade HR).
ShapeOut shape_decode(const Model& m, const std::vector<float>& latent,
                      const std::vector<std::array<int,3>>& coords0, int resolution = 512);

// Cascade upsample: run from_latent + the 4 C2S up-blocks and return the grown voxel coords
// (input res * 16), discarding feats. Used by the 1024 cascade to get fine coords from the LR slat.
std::vector<std::array<int,3>> shape_upsample(const Model& m, const std::vector<float>& latent,
                                              const std::vector<std::array<int,3>>& coords0);

// [preview] shape_upsample that stops early: runs the C2S up-blocks only until the grid reaches
// `target_res` (coords0 live at `res0`), and gives up one level earlier if a level comes out
// larger than `max_level_voxels` — the estimate of a half-denoised latent can subdivide almost
// everywhere. `res_out` receives the resolution the returned coords are expressed in.
std::vector<std::array<int,3>> shape_upsample_partial(const Model& m, const std::vector<float>& latent,
                                                      const std::vector<std::array<int,3>>& coords0,
                                                      int res0, int target_res, size_t max_level_voxels,
                                                      int* res_out);

// Texture (PBR) decoder: SparseUnetVaeDecoder driven by the shape decoder's `subs` so it grows
// the IDENTICAL voxel tree. tex_latent: [32*N0]; returns 6-ch PBR [6*M] (channel-major) at the
// SAME final coords/order as shape_decode (base_color3, metallic, roughness, alpha) — pre *0.5+0.5.
std::vector<float> tex_decode(const Model& m, const std::vector<float>& tex_latent,
                              const std::vector<std::array<int,3>>& coords0,
                              const std::vector<std::vector<uint8_t>>& subs);

} // namespace trellis
