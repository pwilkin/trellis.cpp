#include "dual_grid.h"
#include <cmath>
#include <unordered_map>

namespace trellis {

// edge_neighbor_voxel_offset[axis][corner][xyz] (from o_voxel flexible_dual_grid.py)
static const int OFF[3][4][3] = {
    { {0,0,0}, {0,0,1}, {0,1,1}, {0,1,0} },   // x-axis
    { {0,0,0}, {1,0,0}, {1,0,1}, {0,0,1} },   // y-axis
    { {0,0,0}, {0,1,0}, {1,1,0}, {1,0,0} },   // z-axis
};

static inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static inline float softplusf(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

Mesh dual_grid_to_mesh(const ShapeOut& so) {
    const int N = (int)so.coords.size();
    const float* F7 = so.feats7.data();   // [7, N] channel-major: F7[ch + 7*i]
    const float vs = 1.0f / so.res, org = -0.5f;

    // dual vertices: world pos = (coord + (2*sigmoid(off)-0.5)) * vs - 0.5
    Mesh mesh; mesh.verts.resize((size_t)N * 3);
    for (int i = 0; i < N; ++i)
        for (int a = 0; a < 3; ++a) {
            float dv = 2.0f * sigmoidf(F7[a + 7*i]) - 0.5f;
            mesh.verts[3*i + a] = ((float)so.coords[i][a] + dv) * vs + org;
        }

    // coord hashmap
    auto key = [](int x,int y,int z){ return ((uint64_t)(uint32_t)x<<40)|((uint64_t)(uint32_t)y<<20)|(uint32_t)z; };
    std::unordered_map<uint64_t,int> cmap; cmap.reserve(N * 2);
    for (int i = 0; i < N; ++i) cmap[key(so.coords[i][0], so.coords[i][1], so.coords[i][2])] = i;

    for (int i = 0; i < N; ++i) {
        const int x = so.coords[i][0], y = so.coords[i][1], z = so.coords[i][2];
        for (int a = 0; a < 3; ++a) {
            if (F7[3 + a + 7*i] <= 0.0f) continue;        // intersected edge?
            int q[4]; bool ok = true;
            for (int k = 0; k < 4; ++k) {
                auto it = cmap.find(key(x + OFF[a][k][0], y + OFF[a][k][1], z + OFF[a][k][2]));
                if (it == cmap.end()) { ok = false; break; }
                q[k] = it->second;
            }
            if (!ok) continue;
            float w02 = softplusf(F7[6 + 7*q[0]]) * softplusf(F7[6 + 7*q[2]]);
            float w13 = softplusf(F7[6 + 7*q[1]]) * softplusf(F7[6 + 7*q[3]]);
            if (w02 > w13) {   // quad_split_1 = [0,1,2, 0,2,3]
                mesh.faces.insert(mesh.faces.end(), { q[0],q[1],q[2], q[0],q[2],q[3] });
            } else {           // quad_split_2 = [0,1,3, 3,1,2]
                mesh.faces.insert(mesh.faces.end(), { q[0],q[1],q[3], q[3],q[1],q[2] });
            }
        }
    }
    return mesh;
}

} // namespace trellis
