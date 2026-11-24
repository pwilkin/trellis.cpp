// Mesh input for --refine-mesh: GLB / glTF (cgltf) and OBJ loading, and the TRELLIS.2
// normalisation into the latent grid's frame (docs/spec/31-mesh-refine.md).
#pragma once
#include <string>
#include "dual_grid.h"   // trellis::Mesh

namespace trellis {

// Loads every triangle primitive of every node (world transforms applied) of a .glb / .gltf,
// or the v/f records of an .obj (fan-triangulated polygons, negative indices), merged into one
// mesh. The format is detected by the "glTF" magic, else by extension. Returns false with
// `err` on any failure; never throws on malformed input.
bool load_mesh(const std::string& path, Mesh& out, std::string& err);

// The transform normalize_mesh_trellis applied: v_norm = R * (v - center) * scale, where R maps
// glTF (x, y, z) to the latent grid's (x, -z, y).
struct MeshNorm { float center[3] = {0, 0, 0}; float scale = 1.0f; };

// The reference pipeline's preprocess_mesh: bbox centre, scale = 0.99999 / max extent, then
// (x, y, z) -> (x, -z, y). The GLB writers rotate (x, y, z) -> (x, z, -y), the inverse, so a
// y-up input keeps its orientation on output. Returns false for an empty / degenerate mesh.
bool normalize_mesh_trellis(Mesh& m, MeshNorm& norm);

// Inverse of the scale/centre part, applied to vertices already rotated back to glTF axes by a
// writer's (x, y, z) -> (x, z, -y): v_out = v / scale + center. In-place on [V*3] floats that are
// still in grid axes, i.e. before the writer's rotation: grid (x, y, z) -> glTF (x, z, -y) means
// the glTF-frame centre is added as (cx, -cz, cy) in grid axes.
void restore_mesh_transform(std::vector<float>& verts_grid_axes, const MeshNorm& norm);

} // namespace trellis
