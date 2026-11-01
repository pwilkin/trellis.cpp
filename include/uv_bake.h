// UV unwrap (xatlas) + bake voxel-volume PBR into texture images.
#pragma once
#include <array>
#include <cstddef>
#include <vector>
#include <cstdint>

namespace trellis {

struct BakedMesh {
    std::vector<float>   verts;   // [Vo*3] atlas vertices (positions, TRELLIS space)
    std::vector<float>   uv;      // [Vo*2] normalized [0,1]
    std::vector<int32_t> faces;   // [Fo*3]
    std::vector<uint8_t> base;    // [T*T*4] RGBA base color
    std::vector<uint8_t> mr;      // [T*T*4] RGBA glTF metallic-roughness (G=rough, B=metal)
    std::vector<uint8_t> nrm;     // [T*T*4] RGBA tangent-space normal map (glTF: +Y = up the image);
                                  //   empty unless a NormalSource was given
    std::vector<float>   tangents;// [Vo*4] xyz + handedness w (glTF TANGENT); empty unless baked
    int T = 0;
    bool ok() const { return T > 0 && !faces.empty(); }
};

class TriBvh;

// Dense high-poly surface to bake a tangent-space normal map from: the narrow-band
// remeshed shell (watertight, smooth) and a BVH over it. Each shaded texel takes the
// normal of the closest shell triangle, expressed in the decimated mesh's per-vertex
// tangent frame (glTF convention: bitangent = cross(N, T) * w, which points toward
// decreasing v, i.e. up the image). Lets a 25K-face mesh shade like the dense one.
struct NormalSource {
    const TriBvh*  bvh   = nullptr;
    const float*   verts = nullptr;   // [V*3], TRELLIS space (same as the baked mesh)
    const int32_t* faces = nullptr;   // [F*3]
    float max_dist = 0.f;             // closest-point search radius (0 = unbounded)
    bool ok() const { return bvh && verts && faces; }
};

// Sparse per-voxel PBR field at grid resolution `res` (mesh space [-0.5,0.5]^3;
// voxel i covers [i/res-0.5, (i+1)/res-0.5)). feats layout [N*6] in [0,1]:
// base RGB, metallic, roughness, alpha. Matches the reference bake, which
// trilinearly samples this volume per texel (texturing.py grid_sample_3d)
// instead of interpolating decimation-averaged per-vertex colors. When `snap`
// is set (a BVH over the ORIGINAL hole-filled mesh), texels whose position
// falls off the voxel shell are first snapped to the closest surface point —
// the reference's cuBVH unsigned_distance correction.
struct VoxelPbr {
    const std::vector<std::array<int,3>>* coords = nullptr;
    const std::vector<float>* feats = nullptr;
    int res = 0;
    const TriBvh* snap = nullptr;
    bool ok() const { return coords && feats && res > 0 && !coords->empty(); }
};

// Vertex-clustering decimation: snap verts to a `grid`-cell lattice over [-0.5,0.5]^3, average
// position + pbr per cell, drop degenerate faces. Reduces a dense voxel-surface mesh enough for
// xatlas. Outputs new verts/faces/pbr6 (in place via the out vectors).
void decimate_cluster(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                      const std::vector<float>& pbr6, int grid,
                      std::vector<float>& ov, std::vector<int32_t>& of, std::vector<float>& op);

// Quadric-error simplification (meshoptimizer) to ~target_faces, compacting the vertex buffer.
// Smoothness-preserving, unlike the voxel-lattice clustering above (which stair-steps).
void decimate_simplify(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                       int target_faces, std::vector<float>& ov, std::vector<int32_t>& of);

// Faithful CPU port of CuMesh's QEM edge-collapse simplifier (refs/CuMesh/src/simplify.cu):
// Garland-Heckbert quadrics + a skinny-triangle shape penalty + flip rejection + boundary
// weighting, driven by the reference threshold ladder. Produces the reference's adaptive,
// low-sliver triangulation from a dense dual-contour mesh, unlike the meshopt/FQMS path.
void decimate_qem(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                  int target_faces, std::vector<float>& ov, std::vector<int32_t>& of);

// Drop connected components (shared-vertex face adjacency) whose face count is below
// frac*(largest component's). Removes decode floaters + spurious ground fragments (which
// shatter our mesh into 50+ pieces) while keeping legitimately large secondary shells
// (a hollow vase's inner+outer surface). In place; returns the number of components dropped.
int drop_small_components(std::vector<float>& verts, std::vector<int32_t>& faces, float frac = 0.02f);

// Fill small boundary loops (holes) with a centroid fan, port of CuMesh fill_holes
// (clean_up.cu): loops whose perimeter is below max_perimeter get one new vertex at the
// mean of their edge midpoints plus one triangle per rim edge. The reference postprocess
// runs this on the decoded mesh BEFORE the remesh (max_hole_perimeter=3e-2 in the
// [-0.5,0.5] world cube) -- open cracks make the narrow-band in/out test ambiguous and
// leak holes into the remeshed surface. In place; returns the number of holes filled.
int fill_holes(std::vector<float>& verts, std::vector<int32_t>& faces, float max_perimeter = 3e-2f);

// Taubin (lambda/mu) shrink-free Laplacian smoothing. Strips the ~1-voxel stair-step noise
// of the dual-contour surface so the quadric simplifier stays curvature-adaptive instead of
// emitting a uniform-dense sliver mesh. Boundary verts pinned. In place.
void taubin_smooth(std::vector<float>& verts, const std::vector<int32_t>& faces,
                   int iters = 5, float lambda = 0.5f, float mu = -0.53f);

// In-place cleanup of a welded surface mesh (faces only) so meshopt's guarded quadric collapse
// reaches the target without the roughening FQMS fallback: drop degenerate/duplicate faces and
// unify face orientations by BFS over manifold-edge adjacency. The narrow-band DC output is ~11%
// inconsistently wound + non-manifold, which otherwise stalls meshopt and forces FQMS.
void clean_mesh(int V, std::vector<int32_t>& faces);

// Fan-fill boundary loops of at most max_loop edges in place; returns the number of holes filled.
int fill_small_holes(std::vector<int32_t>& faces, int max_loop = 64);

// Merge vertices within `step` of each other in place (optionally remapping per-vertex RGB);
// returns the number of duplicates removed. The dual-grid decoder emits epsilon-different
// positions for corners shared across cells, so the raw mesh is full of hairline cracks that
// read as borders: they produce pinhole boundary loops and block edge-collapse simplification.
int weld_vertices(std::vector<float>& verts, std::vector<int32_t>& faces,
                  std::vector<float>* colors3 = nullptr, float step = 1.0f / 8192.0f);

// verts [V*3], faces [F*3], pbr6 [V*6] per-vertex (base3, metallic, roughness, alpha) in [0,1].
// Unwraps with xatlas, shades texels from `vox` (trilinear volume sampling) when provided,
// else from interpolated per-vertex PBR; dilates seams.
BakedMesh uv_bake(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                  const std::vector<float>& pbr6, int texsize, const VoxelPbr* vox = nullptr,
                  const NormalSource* nsrc = nullptr);

// Voxel-native box (cube) projection atlas — no xatlas, no chart computation. Faces are
// assigned to the axis-aligned plane (of 6) along which they are visible, with a second-layer
// chart set for occluded faces; texels are shaded from `vox` when provided (else per-vertex
// PBR) with a depth-tested raster. O(F) — fast even for the undecimated res-1024 cascade mesh.
BakedMesh uv_box_project(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                         const std::vector<float>& pbr6, int texsize, const VoxelPbr* vox = nullptr,
                         const NormalSource* nsrc = nullptr);

// Normal-clustered planar charts + shelf packing: deterministic unwrap with
// area-proportional texel density. Fallback when xatlas times out or is skipped.
BakedMesh uv_chart_project(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                           const std::vector<float>& pbr6, int texsize, const VoxelPbr* vox = nullptr,
                           const NormalSource* nsrc = nullptr);

// Normal-map bake on an already-unwrapped BakedMesh (verts/uv/faces/T set). `owner` is
// one int per texel: the index of the face that shaded it, or -1. Fills bm.nrm (dilated
// into the gutters) and bm.tangents. The three unwrappers above call this themselves
// when given a NormalSource; it is exposed for tests and external post-processing.
void bake_normals(BakedMesh& bm, const std::vector<int32_t>& owner, const NormalSource& src);

// Owner buffer for bake_normals from the mesh's own UV layout, no depth test (for
// unwraps with unique chart space such as xatlas, and for tests).
std::vector<int32_t> rasterize_owner(const BakedMesh& bm);

struct PresetUvBakeStats {
    size_t covered_texels = 0;
    size_t missing_voxel_texels = 0;
    size_t projected_samples = 0;
    size_t projected_samples_above_0_008 = 0;
    double maximum_source_distance = 0;
};

// Shade an already-unwrapped triangle mesh from the sparse PBR volume. The caller
// must validate UV overlap and quad-corner continuity before exporting quads.
PresetUvBakeStats bake_preset_uv(BakedMesh& mesh, const VoxelPbr& vox);

} // namespace trellis
