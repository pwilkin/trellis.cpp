#pragma once
#include <cstdint>
#include <string>

// Run the full TRELLIS.2 image->3D pipeline: reconstruct `img` into a textured
// GLB at `outglb`. `gpu` selects the device (>=0 GPU, <0 CPU), `M` is the model
// directory, `seed` the RNG seed. Returns 0 on success. Behavior is further
// tuned by the TRELLIS_* environment variables (see README).
int trellis_run(const std::string& img, const std::string& outglb, int gpu, const std::string& M, uint32_t seed);
