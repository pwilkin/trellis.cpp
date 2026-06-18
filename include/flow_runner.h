// Flow-DiT runner (dense grid or sparse voxel) + FlowEuler guidance-interval sampler.
#pragma once
#include <vector>
#include <array>
#include <functional>
#include <map>
#include <string>
#include "dit.h"

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;
typedef struct ggml_gallocr* ggml_gallocr_t;

namespace trellis {
struct Model;

struct SamplerParams {
    int   steps             = 12;
    float guidance_strength = 7.5f;
    float guidance_rescale  = 0.0f;
    float gi0               = 0.6f;
    float gi1               = 1.0f;
    float rescale_t         = 1.0f;
    float sigma_min         = 1e-5f;
};

// One DiT graph (built once for a fixed token count N), re-run per sampler step.
// Token axis N = R^3 (dense) or number of active voxels (sparse); RoPE tables are
// supplied by the factory (grid index math vs real voxel coords).
class DitRunner {
public:
    DitRunner(const Model& m, const DiTParams& p, int N, int n_cond,
              const std::vector<float>& rope_cos, const std::vector<float>& rope_sin);
    ~DitRunner();
    // xt: [in_ch*N] channel-major. cond: [d_cond*n_cond]. Returns velocity [out_ch*N].
    std::vector<float> forward(const std::vector<float>& xt, float t_scaled, const float* cond);
    int N() const { return N_; }
private:
    const Model& m_; DiTParams p_; int N_, Lc_;
    ggml_context* ctx_ = nullptr; ggml_cgraph* g_ = nullptr; ggml_gallocr_t alloc_ = nullptr;
    ggml_tensor *gh0_, *gtf_, *gcond_, *gcos_, *gsin_, *gout_;
    std::vector<float> rcos_, rsin_;   // re-uploaded each forward (gallocr may reuse input buffers)
    std::map<std::string, ggml_tensor*> inter_;   // [dbg] named intermediates for NaN localization
    bool dbg_nan_ = false, dbg_done_ = false;
};

// Dense factory: RoPE from R^3 grid (ij meshgrid, z fastest). N = R^3.
DitRunner* make_dense_runner(const Model& m, const DiTParams& p, int R, int n_cond);
// Sparse factory: RoPE from real voxel coords [N][3]. N = coords.size().
DitRunner* make_sparse_runner(const Model& m, const DiTParams& p,
                              const std::vector<std::array<int,3>>& coords, int n_cond);

// FlowEuler guidance-interval sampler over an arbitrary forward functor.
using FlowFwd = std::function<std::vector<float>(const std::vector<float>&, float, const float*)>;
std::vector<float> sample_flow(const FlowFwd& fwd, std::vector<float> sample,
                               const float* cond, const float* neg_cond,
                               const SamplerParams& sp,
                               std::vector<std::vector<float>>* trace = nullptr);

} // namespace trellis
