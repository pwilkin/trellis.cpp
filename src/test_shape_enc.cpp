// Shape SLAT encoder tests (docs/spec/31-mesh-refine.md).
//
//   trellis-test-shape-enc <shape_enc.gguf> <ref_dir> [gpu]
//       Encode ref_dir/{coords0,feats0,grid}.npy (tools/ref_shape_enc.py) and compare the latent
//       with ref_dir/{latent,latent_coords}.npy: latent coords must match exactly; the latent's
//       max-based rel error must be within the decoder test's 8e-2 (the reference runs its
//       blocks in fp16).
//   trellis-test-shape-enc dump <mesh> <grid> <out.bin>
//       Voxelize a mesh (dual vertices included) for `ref_shape_enc.py --dual`.
//   trellis-test-shape-enc roundtrip <shape_enc.gguf> <shape_dec.gguf> <mesh> <grid> [gpu]
//       T8: mesh -> voxelize -> encode -> decode -> dual_grid_to_mesh, then the symmetric
//       Chamfer distance to the (normalised) input, in voxels of `grid`.
//   trellis-test-shape-enc coords <shape_enc.gguf> <shape_dec.gguf> <mesh> <grid> [gpu]
//       Token-set IoU of the encoder and voxel coordinate sources of --refine-mesh at `grid`.
#include "dual_grid.h"
#include "mesh_io.h"
#include "mesh_voxelize.h"
#include "npy.h"
#include "shape_decoder.h"
#include "shape_encoder.h"
#include "tri_bvh.h"
#include "trellis_model.h"

#include <algorithm>
#include <iterator>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace trellis;
namespace trellis { extern bool g_sparse_cast_f32; }

static int dump(const char* mesh, int grid, const char* out) {
    Mesh m; std::string err; MeshNorm nm;
    if (!load_mesh(mesh, m, err) || !normalize_mesh_trellis(m, nm)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    const DualGrid dg = mesh_to_flexible_dual_grid(m.verts.data(), m.V(), m.faces.data(), m.F(), grid, false);
    FILE* f = fopen(out, "wb");
    if (!f) { perror(out); return 1; }
    const int64_t N = (int64_t)dg.coords.size();
    fwrite(&N, 8, 1, f); fwrite(&grid, 4, 1, f);
    for (auto& c : dg.coords) fwrite(c.data(), 4, 3, f);
    fwrite(dg.dual.data(), 4, dg.dual.size(), f);
    fwrite(dg.intersected.data(), 1, dg.intersected.size(), f);
    fclose(f);
    printf("%s @%d: %lld voxels -> %s\n", mesh, grid, (long long)N, out);
    return 0;
}

// Symmetric Chamfer distance between two triangle meshes, by surface sampling + closest point.
static double chamfer_one_way(const Mesh& from, const TriBvh& to_bvh, int samples, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0, 1);
    std::vector<double> cum((size_t)from.F());
    double acc = 0;
    for (int f = 0; f < from.F(); ++f) {
        const float* a = &from.verts[(size_t)from.faces[f * 3] * 3];
        const float* b = &from.verts[(size_t)from.faces[f * 3 + 1] * 3];
        const float* c = &from.verts[(size_t)from.faces[f * 3 + 2] * 3];
        const double e1[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] }, e2[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
        const double cx = e1[1] * e2[2] - e1[2] * e2[1], cy = e1[2] * e2[0] - e1[0] * e2[2], cz = e1[0] * e2[1] - e1[1] * e2[0];
        acc += 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
        cum[(size_t)f] = acc;
    }
    double sum = 0;
    for (int s = 0; s < samples; ++s) {
        const double r = u(rng) * acc;
        const int f = (int)(std::lower_bound(cum.begin(), cum.end(), r) - cum.begin());
        float r1 = std::sqrt(u(rng)), r2 = u(rng);
        float p[3];
        for (int k = 0; k < 3; ++k)
            p[k] = (1 - r1) * from.verts[(size_t)from.faces[f * 3] * 3 + k] + r1 * (1 - r2) * from.verts[(size_t)from.faces[f * 3 + 1] * 3 + k] +
                   r1 * r2 * from.verts[(size_t)from.faces[f * 3 + 2] * 3 + k];
        sum += std::sqrt((double)to_bvh.closest(p).dist2);
    }
    return sum / samples;
}

static int roundtrip(const char* enc_path, const char* dec_path, const char* mesh, int grid, int gpu) {
    Mesh m; std::string err; MeshNorm nm;
    if (!load_mesh(mesh, m, err) || !normalize_mesh_trellis(m, nm)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    const auto t0 = std::chrono::steady_clock::now();
    auto secs = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    const DualGrid dg = mesh_to_flexible_dual_grid(m.verts.data(), m.V(), m.faces.data(), m.F(), grid, false);
    printf("voxelized @%d: %zu voxels (%.1fs)\n", grid, dg.coords.size(), secs());
    ShapeLatent lat;
    { Model em = Model::load(enc_path, gpu); lat = shape_encode(em, dg); em.free(); }
    printf("encoded: %zu latent voxels @%d (%.1fs)\n", lat.coords.size(), lat.grid, secs());
    ShapeOut so;
    { Model dm = Model::load(dec_path, gpu); so = shape_decode(dm, lat.feats, lat.coords, grid); dm.free(); }
    Mesh out = dual_grid_to_mesh(so);
    printf("decoded: %zu voxels, mesh V=%d F=%d (%.1fs)\n", so.coords.size(), out.V(), out.F(), secs());
    if (out.F() == 0) { fprintf(stderr, "empty decode\n"); return 1; }
    TriBvh bin = TriBvh::build(m.verts.data(), m.V(), m.faces.data(), m.F());
    TriBvh bout = TriBvh::build(out.verts.data(), out.V(), out.faces.data(), out.F());
    const double a = chamfer_one_way(out, bin, 200000, 1), b = chamfer_one_way(m, bout, 200000, 2);
    const double vox = 1.0 / grid;
    printf("Chamfer @%d: out->in %.3f voxels, in->out %.3f voxels, symmetric mean %.3f voxels %s\n",
           grid, a / vox, b / vox, 0.5 * (a + b) / vox, 0.5 * (a + b) / vox < 1.5 ? "OK" : "**");
    return 0.5 * (a + b) / vox < 1.5 ? 0 : 1;
}

// B4 acceptance: the token sets --refine-mesh would use from the two coordinate sources at R, with
// the same quantisation (hr_res = R, no backoff): voxel = c / 16 of the dual grid's voxels;
// encoder = c / 16 of shape_upsample(encode(dual grid)). Prints their IoU.
static int coords_iou(const char* enc_path, const char* dec_path, const char* mesh, int grid, int gpu) {
    Mesh m; std::string err; MeshNorm nm;
    if (!load_mesh(mesh, m, err) || !normalize_mesh_trellis(m, nm)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    const auto t0 = std::chrono::steady_clock::now();
    auto secs = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    const DualGrid dg = mesh_to_flexible_dual_grid(m.verts.data(), m.V(), m.faces.data(), m.F(), grid, false);
    auto quant = [&](const std::vector<std::array<int,3>>& fine) {
        std::vector<std::array<int,3>> q; q.reserve(fine.size() / 4);
        for (auto& c : fine) q.push_back({ (int)((c[0] + 0.5f) / grid * (grid / 16)), (int)((c[1] + 0.5f) / grid * (grid / 16)),
                                           (int)((c[2] + 0.5f) / grid * (grid / 16)) });
        std::sort(q.begin(), q.end()); q.erase(std::unique(q.begin(), q.end()), q.end());
        return q;
    };
    const auto qv = quant(dg.coords);
    printf("voxelized @%d: %zu voxels -> %zu voxel-mode tokens (%.1fs)\n", grid, dg.coords.size(), qv.size(), secs());
    ShapeLatent lat;
    { Model em = Model::load(enc_path, gpu); lat = shape_encode(em, dg); em.free(); }
    std::vector<std::array<int,3>> fine;
    { Model dm = Model::load(dec_path, gpu); fine = shape_upsample(dm, lat.feats, lat.coords); dm.free(); }
    const auto qe = quant(fine);
    std::vector<std::array<int,3>> inter;
    std::set_intersection(qv.begin(), qv.end(), qe.begin(), qe.end(), std::back_inserter(inter));
    const double iou = (double)inter.size() / (double)(qv.size() + qe.size() - inter.size());
    printf("encoder: %zu latent voxels, upsample -> %zu fine coords -> %zu encoder-mode tokens (%.1fs)\n",
           lat.coords.size(), fine.size(), qe.size(), secs());
    printf("token-set IoU encoder vs voxel @%d: %.4f (|v|=%zu |e|=%zu |v&e|=%zu) %s\n", grid, iou, qv.size(), qe.size(),
           inter.size(), iou >= 0.9 ? "OK" : "**");
    return iou >= 0.9 ? 0 : 1;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 5 && std::string(argv[1]) == "coords")
        return coords_iou(argv[2], argv[3], argv[4], atoi(argv[5]), argc > 6 ? atoi(argv[6]) : 0);
    if (argc > 4 && std::string(argv[1]) == "dump") return dump(argv[2], atoi(argv[3]), argv[4]);
    if (argc > 5 && std::string(argv[1]) == "roundtrip")
        return roundtrip(argv[2], argv[3], argv[4], atoi(argv[5]), argc > 6 ? atoi(argv[6]) : 0);
    if (argc < 3) {
        fprintf(stderr, "usage: %s <shape_enc.gguf> <ref_dir> [gpu]\n"
                        "       %s dump <mesh> <grid> <out.bin>\n"
                        "       %s roundtrip <shape_enc.gguf> <shape_dec.gguf> <mesh> <grid> [gpu]\n"
                        "       %s coords <shape_enc.gguf> <shape_dec.gguf> <mesh> <grid> [gpu]\n", argv[0], argv[0], argv[0], argv[0]);
        return 1;
    }
    const std::string ref = argv[2];
    const int gpu = argc > 3 ? atoi(argv[3]) : 0;
    if (getenv("TRELLIS_F32W")) g_sparse_cast_f32 = true;
    npy::Array c0 = npy::load(ref + "/coords0.npy"), f0 = npy::load(ref + "/feats0.npy"), gr = npy::load(ref + "/grid.npy");
    npy::Array lc = npy::load(ref + "/latent_coords.npy"), la = npy::load(ref + "/latent.npy");
    const int N0 = (int)c0.shape[0], grid = (int)gr.data[0];
    std::vector<std::array<int,3>> coords(N0);
    for (int i = 0; i < N0; ++i) coords[i] = { (int)c0.data[i * 3], (int)c0.data[i * 3 + 1], (int)c0.data[i * 3 + 2] };
    std::vector<float> feats((size_t)6 * N0);
    for (int i = 0; i < N0; ++i) for (int k = 0; k < 6; ++k) feats[(size_t)k + 6 * i] = f0.data[(size_t)i * 6 + k];
    Model m = Model::load(argv[1], gpu);
    const auto t0 = std::chrono::steady_clock::now();
    ShapeLatent lat = shape_encode_feats(m, feats, coords, grid);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    m.free();
    const int NL = (int)lc.shape[0];
    bool coords_ok = (int)lat.coords.size() == NL;
    for (int i = 0; coords_ok && i < NL; ++i)
        for (int a = 0; a < 3; ++a) if (lat.coords[i][a] != (int)lc.data[i * 3 + a]) { coords_ok = false; break; }
    double md = 0, mg = 0, sd = 0;
    if (coords_ok)
        for (int i = 0; i < NL; ++i) for (int c = 0; c < 32; ++c) {
            const double g = la.data[(size_t)i * 32 + c], o = lat.feats[(size_t)c + 32 * i];
            md = std::max(md, std::fabs(o - g)); mg = std::max(mg, std::fabs(g)); sd += std::fabs(o - g);
        }
    const double rel = md / std::max(mg, 1e-12);
    const bool ok = coords_ok && rel < 8e-2;
    printf("input N=%d @%d -> latent N=%zu (ref %d) in %.1fs; latent coords %s; latent rel=%.4e mean|d|=%.4e  %s\n",
           N0, grid, lat.coords.size(), NL, secs, coords_ok ? "EXACT" : "MISMATCH", rel,
           coords_ok ? sd / ((double)NL * 32) : 0.0, ok ? "OK" : "**");
    return ok ? 0 : 1;
}
