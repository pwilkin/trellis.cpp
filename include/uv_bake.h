// UV unwrap (xatlas) + bake per-vertex PBR into texture images.
#pragma once
#include <vector>
#include <cstdint>

namespace trellis {

struct BakedMesh {
    std::vector<float>   verts;   // [Vo*3] atlas vertices (positions, TRELLIS space)
    std::vector<float>   uv;      // [Vo*2] normalized [0,1]
    std::vector<int32_t> faces;   // [Fo*3]
    std::vector<uint8_t> base;    // [T*T*4] RGBA base color
    std::vector<uint8_t> mr;      // [T*T*4] RGBA glTF metallic-roughness (G=rough, B=metal)
    int T = 0;
    bool ok() const { return T > 0 && !faces.empty(); }
};

// Vertex-clustering decimation: snap verts to a `grid`-cell lattice over [-0.5,0.5]^3, average
// position + pbr per cell, drop degenerate faces. Reduces a dense voxel-surface mesh enough for
// xatlas. Outputs new verts/faces/pbr6 (in place via the out vectors).
void decimate_cluster(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                      const std::vector<float>& pbr6, int grid,
                      std::vector<float>& ov, std::vector<int32_t>& of, std::vector<float>& op);

// verts [V*3], faces [F*3], pbr6 [V*6] per-vertex (base3, metallic, roughness, alpha) in [0,1].
// Unwraps with xatlas, rasterizes per-vertex PBR into a texsize×texsize atlas, dilates seams.
BakedMesh uv_bake(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                  const std::vector<float>& pbr6, int texsize);

// Voxel-native box (cube) projection atlas — no xatlas, no chart computation. Each face is
// assigned to one of 6 axis-aligned planes by its dominant signed normal; the 6 planes are
// packed into a 3×2 atlas grid and per-vertex PBR is rasterized in. O(F), trivially parallel —
// fast enough for the full (undecimated) res-1024 cascade mesh. Has mild self-overlap on folds.
BakedMesh uv_box_project(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                         const std::vector<float>& pbr6, int texsize);

} // namespace trellis
