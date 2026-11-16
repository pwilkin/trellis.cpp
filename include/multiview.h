// Multi-view conditioning: view specs, view directions and per-token blend weights.
//
// TRELLIS.2 is camera-unaware: every flow cross-attends to global DINOv3 tokens with no pose
// input. Multi-view conditioning (the ComfyUI-Trellis2 `FlowEulerMultiViewSampler` approach)
// runs the flow once per view and blends the per-view velocities token by token, weighting each
// token by how much it faces that view's camera:
//
//     d_i      = (cos el * sin az, sin el, cos el * cos az)   viewer frame: X right, Y up,
//                                                             Z toward the front camera
//     score_i  = p . d_i^grid        p = token position in [-1, 1]^3 on the latent grid
//     w[i]     = softmax_i(tau * score_i)
//
// The viewer frame equals the output glTF frame. The latent grid is z-up with the object's
// front on -y (the GLB writers rotate (x,y,z) -> (x,z,-y)), so the default `-y` preset maps
// viewer (X,Y,Z) to grid (X,-Z,Y). The `ref-z` / `ref-x` presets reproduce the reference
// sampler exactly, including its normalisation (see docs/spec/30-multiview.md).
#pragma once
#include <array>
#include <string>
#include <vector>

namespace trellis {

struct ViewSpec {
    std::string image;
    float az_deg = 0.0f;   // 0 front, 90 camera on viewer +X ("right"), 180 back, 270 left
    float el_deg = 0.0f;   // +90 top, -90 bottom
};

enum class FrontAxis { NegY, PosY, PosX, NegX, RefZ, RefX };

struct BlendParams {
    float     temperature = 2.0f;
    FrontAxis front       = FrontAxis::NegY;
    float     yaw_offset  = 0.0f;   // degrees, added to every view's azimuth
    bool ref_norm() const { return front == FrontAxis::RefZ || front == FrontAxis::RefX; }
};

constexpr int kMaxViews = 12;

// "path[:az[:el]]". Parsed from the RIGHT: at most two trailing ":<token>" pieces are stripped,
// and only when they parse as an angle, so "C:\dir\a.png" and "C:\dir\a.png:180" both work.
// az may be a name: front|back|left|right|top|bottom = (0,0) (180,0) (270,0) (90,0) (0,90)
// (0,-90); an explicit elevation after a name overrides the name's own.
bool parse_view_spec(const std::string& s, ViewSpec& out, std::string& error);
// -y (default) | +y | +x | -x | ref-z | ref-x
bool parse_front_axis(const std::string& s, FrontAxis& out);
const char* front_axis_name(FrontAxis f);

// Unit vector toward the camera in viewer coordinates (yaw offset applied).
std::array<double,3> view_dir_viewer(float az_deg, float el_deg, float yaw_offset);
// The same direction expressed on the latent grid axes (x, y, z) by the front-axis preset.
std::array<float,3> view_dir_grid(float az_deg, float el_deg, const BlendParams& bp);

// View-major weights: w[v*N + n]; for each n the sum over v is 1. A single view gives all 1.
// `grid` is the resolution the coords live on (geometric normalisation (c+0.5)/grid*2-1);
// `ref_model_res` is the flow model's own `resolution` and is used only by the ref-* presets,
// which normalise as c/ref_model_res*2-1 like the reference (0 -> fall back to `grid`).
std::vector<float> view_weights_sparse(const std::vector<std::array<int,3>>& coords, int grid,
                                       const std::vector<ViewSpec>& views, const BlendParams& bp,
                                       int ref_model_res = 0);
// Dense R^3 grid, token n = x*R*R + y*R + z (matches make_dense_runner). ref-* presets use the
// reference's endpoint-inclusive linspace(-1, 1, R).
std::vector<float> view_weights_dense(int R, const std::vector<ViewSpec>& views, const BlendParams& bp);

// Mean weight per view over all tokens (a cheap sanity signal for logs).
std::vector<double> mean_view_weights(const std::vector<float>& w, int V);

} // namespace trellis
