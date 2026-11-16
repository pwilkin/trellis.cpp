// Sparse submanifold 3D conv + ConvNeXt block (TRELLIS.2 shape/tex VAE).
#pragma once
#include <vector>
#include <array>
#include <cstdint>
#include <string>

struct ggml_context;
struct ggml_tensor;

namespace trellis {
struct Model;

// Neighbor table for submanifold 3x3x3 conv, TAP-MAJOR int32 [27*N]:
//   nbr[t*N + i] = row index of the voxel at coords[i] + offset(t), or N if absent.
//   tap t = kd*9 + kh*3 + kw, offset = (kd-1, kh-1, kw-1) on (x,y,z). (N = sentinel zero row.)
std::vector<int32_t> build_neighbor_table(const std::vector<std::array<int,3>>& coords);

// Graph ops (feats are channel-major ggml [C, N]; nbr is i32 ggml [N, 27]).
// Submanifold conv: weight key `prefix.weight` ggml ne [Ci,27,Co], bias `prefix.bias` [Co] -> [Co,N].
ggml_tensor* sparse_submconv(ggml_context* c, const Model& m, const std::string& prefix,
                             ggml_tensor* feats, ggml_tensor* nbr, int N);
// SparseConvNeXtBlock3d: conv -> rowLN(affine,1e-6) -> Linear(C,4C)->SiLU->Linear(4C,C) -> + input.
ggml_tensor* sparse_convnext(ggml_context* c, const Model& m, const std::string& prefix,
                             ggml_tensor* feats, ggml_tensor* nbr, int N);
// Host-streamed ConvNeXt block for very large sparse levels. The output is identical in
// layout to sparse_convnext, but each voxel chunk is a separate short-lived GPU graph and
// only referenced neighbour feature columns are uploaded. This avoids the growing concat
// chain / full sparse-table residency that can make stage-3 shape decode request 20-30+ GB.
std::vector<float> sparse_convnext_streamed(const Model& m, const std::string& prefix,
                                             const std::vector<float>& feats_in, int C,
                                             const std::vector<int32_t>& nbr, int N);

// SparseResBlockC2S3d up-block (channel->spatial ×2). Host-orchestrated (subdiv readback).
// feats_in: [Cin*N] channel-major; returns new feats [Cout*M] + new coords (res ×2).
// ext_subdiv: if non-null, use this [8*N] binarized mask (tex decoder guide_subs); else predict
// via to_subdiv (shape decoder). The returned `subdiv` is the [8*N] mask actually used.
struct C2SResult { std::vector<float> feats; std::vector<std::array<int,3>> coords; int C = 0; std::vector<uint8_t> subdiv; };
C2SResult sparse_c2s(const Model& m, const std::string& prefix,
                     const std::vector<float>& feats_in, int Cin,
                     const std::vector<std::array<int,3>>& coords, int Cout,
                     const std::vector<uint8_t>* ext_subdiv = nullptr);

// SparseResBlockS2C3d down-block (spatial->channel /2), the encoder's mirror of sparse_c2s:
//   h = conv1(SiLU(LN_affine(x, 1e-6)))     Cin -> Cout/8 at the fine resolution
//   h = S2C(h); xs = S2C(x)                 2x2x2 children packed into channels (empty slots 0)
//   h = conv2(SiLU(LN(h, 1e-6)))            Cout -> Cout at the coarse resolution
//   out = h + mean over groups of 8*Cin/Cout consecutive channels of xs
// Parents are coord/2 in lexicographic (x,y,z) order (the reference's unique() of the linear
// code); child slot s = lx + 2*ly + 4*lz holds fine channel c at s*C + c. Host-orchestrated and
// streamed in bounded chunks like sparse_c2s (the fine level can be tens of millions of voxels).
struct S2CResult { std::vector<float> feats; std::vector<std::array<int,3>> coords; int C = 0; };
S2CResult sparse_s2c(const Model& m, const std::string& prefix,
                     const std::vector<float>& feats_in, int Cin,
                     const std::vector<std::array<int,3>>& coords, int Cout);

// Voxel-local streamed ops on host tables [C*N] (bounded GPU working set):
//   linear: out = W x + b (weight key prefix.weight [Cout, Cin] torch layout, prefix.bias)
//   norm:   LayerNorm over channels, no affine, eps
std::vector<float> sparse_linear_streamed(const Model& m, const std::string& prefix,
                                          const std::vector<float>& feats_in, int Cin, int N);
std::vector<float> sparse_norm_streamed(const Model& m, const std::vector<float>& feats_in, int C, int N, float eps);

} // namespace trellis
