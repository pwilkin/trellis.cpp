// Flow-DiT runner (dense grid or sparse voxel) + FlowEuler guidance-interval sampler.
#pragma once
#include <memory>
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
class GraphExec;

struct SamplerParams {
    int   steps             = 12;
    float guidance_strength = 7.5f;
    float guidance_rescale  = 0.0f;
    float gi0               = 0.6f;
    float gi1               = 1.0f;
    float rescale_t         = 1.0f;
    float sigma_min         = 1e-5f;
    // SpaceControl (SDEdit-style) start: when x0 is set and start_u > 0, the sampler starts
    // at t0 = rescale(1 - start_u) from t0*noise + (1-t0)*x0. start_u is CONTINUOUS: the
    // first step runs from t0 down to the next grid time, so every strength is distinct
    // (rounding to whole steps made 0.05..0.35 all start from the same place).
    float        start_u    = 0.0f;
    const float* x0         = nullptr;   // same layout/size as the noise sample
};

// One branch of the classifier-free guidance pair. `cond` is the cross-attention context that
// every TRELLIS.2 stage uses; `proj` carries the extra per-token view-aligned features and is
// only read in Pixal3D's proj mode, where a null `proj` means the all-zero (negative) branch —
// the runner zeroes the device tensor rather than making the caller materialize the buffer.
// The implicit constructor keeps every cross-mode call site (and the reference tests) writing
// plain `cond.data()`.
struct FlowCond {
    const float* cond = nullptr;   // [d_cond * n_cond]
    const float* proj = nullptr;   // [proj_ch * N]
    FlowCond() = default;
    FlowCond(const float* c) : cond(c) {}
    FlowCond(const float* c, const float* p) : cond(c), proj(p) {}
};

// One DiT graph (built once for a fixed token count N), re-run per sampler step.
// Token axis N = R^3 (dense) or number of active voxels (sparse); RoPE tables are
// supplied by the factory (grid index math vs real voxel coords).
class DitRunner {
public:
    DitRunner(const Model& m, const DiTParams& p, int N, int n_cond,
              const std::vector<float>& rope_cos, const std::vector<float>& rope_sin);
    ~DitRunner();
    // xt: [in_ch*N] channel-major. Returns velocity [out_ch*N].
    std::vector<float> forward(const std::vector<float>& xt, float t_scaled, const FlowCond& c);
    int N() const { return N_; }
private:
    const Model& m_; DiTParams p_; int N_, Lc_;
    ggml_context* ctx_ = nullptr; ggml_cgraph* g_ = nullptr;
    // Owns graph allocation and execution for both direct and scheduled backends.
    std::unique_ptr<GraphExec> exec_;
    ggml_tensor *gh0_, *gtf_, *gcond_, *gproj_ = nullptr, *gcos_, *gsin_, *gout_;
    std::vector<float> rcos_, rsin_;   // re-uploaded each forward (gallocr may reuse input buffers)
    std::map<std::string, ggml_tensor*> inter_;   // [dbg] named intermediates for NaN localization
    bool dbg_nan_ = false, dbg_done_ = false;
    int  n_fwd_ = 0;
};

// Dense factory: RoPE from R^3 grid (ij meshgrid, z fastest). N = R^3.
DitRunner* make_dense_runner(const Model& m, const DiTParams& p, int R, int n_cond);
// Sparse factory: RoPE from real voxel coords [N][3]. N = coords.size().
DitRunner* make_sparse_runner(const Model& m, const DiTParams& p,
                              const std::vector<std::array<int,3>>& coords, int n_cond);

// FlowEuler guidance-interval sampler over an arbitrary forward functor.
using FlowFwd = std::function<std::vector<float>(const std::vector<float>&, float, const FlowCond&)>;
std::vector<float> sample_flow(const FlowFwd& fwd, std::vector<float> sample,
                               const FlowCond& cond, const FlowCond& neg_cond,
                               const SamplerParams& sp,
                               std::vector<std::vector<float>>* trace = nullptr);

} // namespace trellis
