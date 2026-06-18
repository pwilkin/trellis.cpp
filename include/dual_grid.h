// FlexiDualGrid -> triangle mesh extraction (CPU geometry).
#pragma once
#include <vector>
#include <array>
#include <cstdint>
#include "shape_decoder.h"

namespace trellis {

struct Mesh {
    std::vector<float>   verts;   // [V*3]
    std::vector<int32_t> faces;   // [F*3]
    int V() const { return (int)verts.size() / 3; }
    int F() const { return (int)faces.size() / 3; }
};

// dual-grid head [7,M] @ res -> mesh in world cube [-0.5,0.5]^3.
Mesh dual_grid_to_mesh(const ShapeOut& so);

} // namespace trellis
