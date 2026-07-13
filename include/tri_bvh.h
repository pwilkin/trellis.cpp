// Median-split AABB tree over triangles with exact closest-point queries.
// Non-owning: the caller's vertex/index arrays must outlive the tree. Used by
// the texture bake (per-texel snap to the original surface, mirroring the
// reference's cuBVH unsigned_distance) and by the narrow-band remesh (UDF
// evaluation).
#pragma once
#include <cstdint>
#include <vector>

namespace trellis {

class TriBvh {
public:
    struct Hit {
        float dist2 = 1e30f;
        int32_t face = -1;
        float point[3] = {0, 0, 0};
    };

    static TriBvh build(const float* verts, int64_t V, const int32_t* faces, int64_t F);

    Hit closest(const float p[3], float max_dist = 1e30f) const;
    bool empty() const { return nodes_.empty(); }

private:
    struct Node {
        float bmin[3], bmax[3];
        int32_t left;    // internal: index of left child (right = left+1); leaf: first prim index
        int32_t count;   // 0 for internal nodes; >0 = leaf primitive count
    };
    std::vector<Node> nodes_;
    std::vector<int32_t> prim_;
    const float* verts_ = nullptr;
    const int32_t* faces_ = nullptr;
};

}  // namespace trellis
