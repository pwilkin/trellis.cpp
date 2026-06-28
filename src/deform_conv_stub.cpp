// Stub for non-CUDA builds: the modulated deformable conv lives in deform_conv.cu
// (CUDA only) and is used solely by BiRefNet's ASPPDeformable. Vulkan/CPU builds
// don't compile the kernel, so this stub satisfies the link; it throws if the
// BiRefNet background-removal path is actually exercised (use threshold bg-removal).
#include "deform_conv.h"
#include <stdexcept>

namespace trellis {

void deform_conv2d_run(const float*, int, int, int,
                       const float*, const float*,
                       const float*, const float*, int, int,
                       float*, int) {
    throw std::runtime_error(
        "deform_conv2d_run: this build has no CUDA kernel; BiRefNet background "
        "removal (TRELLIS_BIREFNET) is unavailable — use the default threshold bg-removal.");
}

} // namespace trellis
