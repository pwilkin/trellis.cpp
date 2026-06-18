// Shared TRELLIS.2 flow-DiT graph builder (dense path, B=1).
#pragma once
#include <map>
#include <string>

struct ggml_context;
struct ggml_tensor;

namespace trellis {
struct Model;

struct DiTParams {
    int n_blocks   = 30;
    int n_heads    = 12;
    int head_dim   = 128;
    int d_model    = 1536;
    int d_mlp      = 8192;     // int(1536 * 5.3334)
    int d_cond     = 1024;
    int in_ch      = 8;
    int out_ch     = 8;
    float ln_eps       = 1e-6f;
    float final_ln_eps = 1e-5f;
    float rms_eps      = 1e-12f;
    bool  cast_f32     = false;   // cast f16 weights to f32 before matmul (precision test)
};

// Build the dense SS-flow forward graph (B=1). All input tensors live in `gctx`
// and must be flagged ggml_set_input by the caller; weights come from `m`.
//   h0   : [in_ch, L]          patchified input (channel-major)
//   tfreq: [256]               sinusoidal timestep embedding (host-computed)
//   cond : [d_cond, Lc]        conditioning tokens
//   cos/sin: [1, head_dim/2, 1, L]  precomputed 3D-RoPE tables
// Returns the [out_ch, L] velocity; `inter` (optional) collects named intermediates.
ggml_tensor* build_dit_dense(ggml_context* gctx, const Model& m, const DiTParams& p,
                             ggml_tensor* h0, ggml_tensor* tfreq, ggml_tensor* cond,
                             ggml_tensor* cos, ggml_tensor* sin,
                             std::map<std::string, ggml_tensor*>* inter = nullptr);

} // namespace trellis
