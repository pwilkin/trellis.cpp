// Mesh -> flexible dual grid: a port of o-voxel's mesh_to_flexible_dual_grid
// (TRELLIS.2 o-voxel/src/convert/flexible_dual_grid.cpp), the voxeliser behind the reference
// pipeline's encode_shape_slat. See docs/spec/31-mesh-refine.md.
#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace trellis {

struct DualGrid {
    std::vector<std::array<int,3>> coords;      // occupied voxels @grid, in o-voxel's insertion order
    std::vector<float>   dual;                  // [N*3] dual vertex as a local offset in [0,1]
                                                //   (o-voxel's dual * grid - coords); empty when coords_only
    std::vector<uint8_t> intersected;           // [N*3] the voxel's +x/+y/+z max-corner edge crosses the surface
    int grid = 0;
};

// verts: [V*3] in the TRELLIS frame, inside [-0.5, 0.5]^3 (normalize_mesh_trellis); faces [F*3].
// Every grid-line / triangle crossing registers the four voxels around the crossed edge (three
// axis sweeps), exactly as o-voxel's intersect_qef. coords_only skips the QEF accumulation and the
// dual-vertex solve. Weights are the pipeline's encode_shape_slat call values.
DualGrid mesh_to_flexible_dual_grid(const float* verts, int64_t V, const int32_t* faces, int64_t F,
                                    int grid, bool coords_only = false,
                                    float face_weight = 1.0f, float boundary_weight = 0.2f,
                                    float regularization_weight = 1e-2f);

} // namespace trellis
