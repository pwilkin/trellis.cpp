// o-voxel parity oracle for trellis-test-voxelize (B3, docs/spec/31-mesh-refine.md).
//
// Builds upstream TRELLIS.2 o-voxel/src/convert/flexible_dual_grid.cpp against Eigen and the
// torch stand-in in this directory, then runs mesh_to_flexible_dual_grid_cpu exactly as the
// Python wrapper does (vertices - aabb_min with aabb [-0.5, 0.5]^3, voxel_size = 1/grid,
// weights 1.0 / 0.2 / 1e-2 from encode_shape_slat).
//
//   ref-voxelize <in.bin> <grid> <out.bin>
//   in.bin : i64 V, F; f32 verts[V*3] (TRELLIS frame, inside [-0.5,0.5]); i32 faces[F*3]
//   out.bin: i64 N; i32 coords[N*3]; f32 dual_offset[N*3] (dual*grid - coords); u8 intersected[N*3]
#include "api.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s in.bin grid out.bin\n", argv[0]); return 1; }
    FILE* f = fopen(argv[1], "rb"); if (!f) { perror(argv[1]); return 1; }
    int64_t V, F;
    if (fread(&V, 8, 1, f) != 1 || fread(&F, 8, 1, f) != 1) return 1;
    std::vector<float> v((size_t)V * 3); std::vector<int> fc((size_t)F * 3);
    if (fread(v.data(), 4, v.size(), f) != v.size() || fread(fc.data(), 4, fc.size(), f) != fc.size()) return 1;
    fclose(f);
    const int grid = atoi(argv[2]);
    for (float& x : v) x = x - (-0.5f);
    float vs[3] = { 1.0f / grid, 1.0f / grid, 1.0f / grid };   // (aabb[1]-aabb[0]) / grid in float
    int range[6] = { 0, 0, 0, grid, grid, grid };
    auto tv = torch::from_blob(v.data(), { V, 3 }, torch::kFloat32);
    auto tf = torch::from_blob(fc.data(), { F, 3 }, torch::kInt32);
    auto ts = torch::from_blob(vs, { 3 }, torch::kFloat32);
    auto tr = torch::from_blob(range, { 2, 3 }, torch::kInt32);
    // Optional weight overrides (debugging): REF_FW / REF_BW / REF_RW.
    const float fw = getenv("REF_FW") ? (float)atof(getenv("REF_FW")) : 1.0f;
    const float bw = getenv("REF_BW") ? (float)atof(getenv("REF_BW")) : 0.2f;
    const float rw = getenv("REF_RW") ? (float)atof(getenv("REF_RW")) : 1e-2f;
    auto [coords, dual, inter] = mesh_to_flexible_dual_grid_cpu(tv, tf, ts, tr, fw, bw, rw, false);
    const int64_t N = coords.size(0);
    std::vector<float> off((size_t)N * 3);
    for (int64_t i = 0; i < N * 3; ++i) off[(size_t)i] = dual.data_ptr<float>()[i] * (float)grid - (float)coords.data_ptr<int>()[i];
    FILE* o = fopen(argv[3], "wb"); if (!o) { perror(argv[3]); return 1; }
    fwrite(&N, 8, 1, o);
    fwrite(coords.data_ptr<int>(), 4, (size_t)N * 3, o);
    fwrite(off.data(), 4, (size_t)N * 3, o);
    fwrite(inter.data_ptr<bool>(), 1, (size_t)N * 3, o);
    fclose(o);
    printf("ref-voxelize: V=%lld F=%lld grid=%d -> N=%lld\n", (long long)V, (long long)F, grid, (long long)N);
    return 0;
}
