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

// SparseResBlockC2S3d up-block (channel->spatial ×2). Host-orchestrated (subdiv readback).
// feats_in: [Cin*N] channel-major; returns new feats [Cout*M] + new coords (res ×2).
// ext_subdiv: if non-null, use this [8*N] binarized mask (tex decoder guide_subs); else predict
// via to_subdiv (shape decoder). The returned `subdiv` is the [8*N] mask actually used.
struct C2SResult { std::vector<float> feats; std::vector<std::array<int,3>> coords; int C = 0; std::vector<uint8_t> subdiv; };
C2SResult sparse_c2s(const Model& m, const std::string& prefix,
                     const std::vector<float>& feats_in, int Cin,
                     const std::vector<std::array<int,3>>& coords, int Cout,
                     const std::vector<uint8_t>* ext_subdiv = nullptr);

} // namespace trellis
