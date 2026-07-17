// Minimal glTF 2.0 binary (.glb) writer — geometry only (POSITION + indices).
#pragma once
#include <cstdint>

namespace trellis {

// verts: [V*3] float32 in TRELLIS world space ([-0.5,0.5]^3); faces: [F*3] int32.
// Applies the TRELLIS->glTF rotation (x,y,z)->(x,z,-y) and writes a single-mesh GLB.
// colors: optional [V*3] float RGB in [0,1] -> emitted as COLOR_0 (vertex colors).
bool write_glb(const char* path, const float* verts, int64_t V, const int32_t* faces, int64_t F,
               const float* colors = nullptr, int64_t seed = -1, const char* copyright = nullptr);

// Textured GLB: POSITION + NORMAL + TEXCOORD_0 + indices, PBR material with embedded
// base-color and metallic-roughness textures (lossy WebP via EXT_texture_webp when built
// with TRELLIS_WEBP, else PNG). verts/faces/uv from uv_bake; base/mr are [T*T*4] RGBA.
// double_sided=false matches the reference's remeshed output (consistent orientation);
// pass true for un-remeshed fallback paths where winding is not guaranteed.
bool write_glb_textured(const char* path, const float* verts, int64_t V, const float* uv,
                        const int32_t* faces, int64_t F,
                        const unsigned char* base_rgba, const unsigned char* mr_rgba, int T,
                        bool double_sided = false, int64_t seed = -1, const char* copyright = nullptr);

// Debug helper: binary little-endian PLY in raw TRELLIS space (no axis swap).
// colors: optional [V*3] float RGB in [0,1] -> written as uchar red/green/blue per vertex.
bool write_ply(const char* path, const float* verts, int64_t V, const int32_t* faces, int64_t F,
               const float* colors = nullptr);

} // namespace trellis
