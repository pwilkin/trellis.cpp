# trellis.cpp

A standalone, **GGML-based C++** implementation of Microsoft's
[TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) image-to-3D pipeline,
driven **end-to-end from a text prompt**: the text→image step is produced by
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (the local
Z-Image model), and everything downstream — background removal, image conditioning,
the three flow transformers, the VAE decoders, mesh extraction and GLB export — runs
in native C++/GGML with no Python at runtime.

Target acceptance test: generate a 3D asset from the prompt
**"green goblin with metal helmet and spiked club"**.

> Status: **complete end-to-end** — bg removal, conditioning, all three flows, VAE decoders, mesh +
> UV-textured PBR export, 1024 cascade, and BiRefNet background removal all run in native C++/GGML.
> `trellis-cli <image.png> <out.glb>` produces a **UV-textured GLB with a PBR material** — a green
> goblin with a silver metal helmet and spiked club. Default is the **1024 cascade** (LR `flow_512`
> → upsample → HR `flow_1024` → res-1024 decode, sharper geometry); `TRELLIS_512=1` selects the
> lighter res-512 path. The cascade runs on a 16 GB card thanks to **FlashAttention with padded K/V**
> (`src/dit.cpp::sdpa`): the manual softmax needed a single ~18 GB score-matrix alloc at the
> sparse-structure stage, and at the HR token count (≈53k) ggml's tiled FA NaN'd on the unpadded last
> key-tile — zero-padding K/V to a 256 multiple + BF16 fixes both. Every neural component is validated
> against PyTorch (BiRefNet matches to rel 4e-4). See [ROADMAP.md](ROADMAP.md) and `docs/spec/`.
>
> ```
> ./build/trellis-cli assets/goblin.png out/goblin.glb      # image -> UV-textured GLB (atlas + PBR)
> python tools/render_glb.py out/goblin.glb out/view.png    # quick multi-view render
> ```
> The mesh is cluster-decimated (vertex clustering, to the tri budget) then UV-unwrapped. **The
> default unwrap is a voxel-native 6-way box projection** (O(faces), seconds): xatlas chart-compute
> is ~superlinear in faces and only spreads across ~2 cores even though it's multithreaded, so it
> used to dominate the bake (minutes). Per-vertex PBR (base color + metallic/roughness from the tex
> decoder) is baked into the atlas. Env knobs: `TRELLIS_512=1` (res-512 path), `TRELLIS_NOTEX=1`
> (skip texturing), `TRELLIS_DECIM=<grid>` (`0` = no decimation, keep the full res-1024 mesh),
> `TRELLIS_TEX=<atlas>`, `TRELLIS_XATLAS=1` (decimate→xatlas unwrap instead — tighter UV packing,
> much slower), `TRELLIS_BIREFNET=1` (BiRefNet background removal instead of the white-bg threshold —
> for images with real backgrounds; full Swin-L + deformable-conv decoder on GPU, ~13s).
> Every neural component is validated against PyTorch (the `trellis-test-*` binaries + `tools/ref_*.py`):
> SS sampler matches torch to rel 4.3e-3 (exact voxel match), DiT 2.8e-3, DINOv3 1.8e-2, sparse conv 1e-3,
> C2S exact. f16 is the default and matches torch fine (the earlier "needs f32" was a graph-input-reuse
> bug, not precision); `TRELLIS_F32=1` forces f32 compute.

## Pipeline

```
text prompt
  │  stable-diffusion.cpp (Z-Image)              [external binary]
  ▼
RGB image
  │  BiRefNet / RMBG  (background removal)        [GGML]   → RGBA cutout
  ▼
DINOv3 ViT-L/16 feature extractor                [GGML]   → patch tokens [N,1024]
  │  (image conditioning, + null cond for CFG)
  ▼
① Sparse-Structure flow DiT (1.3B, dense 16³)    [GGML]
  │  → 8-ch 16³ latent → SS conv3d decoder → 64³ occupancy → active voxels
  ▼
② Shape-SLAT flow DiT (1.3B, sparse)             [GGML]   → 32-ch latent / active voxel
  │  → FlexiDualGrid shape decoder (sparse ConvNeXt) → dual grid → FlexiCubes mesh
  ▼
③ Texture-SLAT flow DiT (1.3B, sparse)           [GGML]   → 32-ch latent / active voxel
  │  → Sparse U-Net tex decoder (6-ch PBR per voxel)
  ▼
textured mesh → GLB
```

All three flow stages use a `FlowEulerGuidanceIntervalSampler` (rectified-flow Euler,
12 steps, classifier-free guidance with a guidance interval + rescale). Optional
512→1024 cascade for higher resolution.

## Models

Downloaded to `/media/ilintar/D_SSD/models/trellis2/` (the repo never stores weights):

| role | source | notes |
|------|--------|-------|
| SS flow DiT | `microsoft/TRELLIS.2-4B` `ss_flow_img_dit_1_3B_64` | 1.3B, bf16 |
| Shape SLAT flow | `…/slat_flow_img2shape_dit_1_3B_{512,1024}` | 1.3B, bf16; `_1024` drives the cascade's HR pass |
| Tex SLAT flow | `…/slat_flow_imgshape2tex_dit_1_3B_{512,1024}` | 1.3B, bf16; `_1024` drives the cascade's HR pass |
| Shape decoder | `…/shape_dec_next_dc_f16c32` | FlexiDualGrid VAE |
| Tex decoder | `…/tex_dec_next_dc_f16c32` | Sparse U-Net VAE, 6-ch |
| SS decoder | `microsoft/TRELLIS-image-large` `ss_dec_conv3d_16l8` | reused from v1 |
| Image cond | `timm/vit_large_patch16_dinov3.lvd1689m` | ungated mirror of DINOv3 ViT-L (same weights) |
| BG removal | `ZhengPeng7/BiRefNet` | ungated BiRefNet (RMBG-2.0 substitute) |

The two helper models are HF-gated upstream; the ungated equivalents above avoid needing
a token. See `docs/spec/` and `tools/` for safetensors→GGUF conversion.

## Building

GGML is vendored in `thirdparty/ggml` (CUDA build). See [ROADMAP.md](ROADMAP.md).

```
cmake -B build -DGGML_CUDA=ON
cmake --build build -j
```

## Layout

```
src/            C++ implementation (models, ops, pipeline, drivers)
src/test_*.cpp  parity / unit tests vs reference tensors (built as the trellis-test-* binaries)
include/        public headers
tools/          python conversion scripts (safetensors → GGUF) + reference-dump checks, via the uv venv
docs/spec/      reverse-engineered per-component architecture spec
thirdparty/     vendored ggml (gitignored), plus stb + xatlas
```
