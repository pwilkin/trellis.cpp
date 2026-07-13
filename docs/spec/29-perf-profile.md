# 29 — E2E performance profile and optimization findings (res 1024, Strix Halo)

Measured on the goblin benchmark (`goblin_ref.png`, seed 42, res 1024,
Vulkan backend, 2026-07-12). E2E wall: **6:09–6:40** (goblin-class),
**12:39** (turret-class). `perf record -g --call-graph dwarf` over the full
run: 33K samples, ~140 core-seconds of CPU against ~370 s wall — the run is
GPU-bound; a core spinning in `ggml_vk_wait_for_fence` (21 % of CPU cycles)
is just the fence wait while the GPU works.

## Wall-time breakdown (goblin, 399 s instrumented run)

| stage | wall | notes |
|---|---|---|
| preprocess + DINOv3 (512+1024) | ~25 s | |
| sparse-structure flow | 47 s | 12 steps, 22 forwards (guidance-doubled) |
| shape SLAT flow LR 512 | 28 s | 12 steps, 20 forwards |
| shape SLAT flow HR cascade (17.2k tok) | 166 s | 12 steps, 20 forwards, 8.3 s/fwd |
| texture SLAT flow | 91 s | 12 steps, 12 forwards (gs=1 → single pass) |
| decodes (shape, PBR) | ~20 s | |
| postprocess (weld→…→WebP GLB) | ~24 s | weld 2.3, fill 3.7, bvh 0.9, remesh 4.3, simplify 7.6, bake 5.6 |

`sample_flow` now prints `[flow] N steps, M forwards, X s` per flow.

## CPU hotspots (share of all cycles)

- 29.5 % `TriBvh::closest` — remesh candidate-cell UDF queries + bake texel
  snap. All queries bounded and pruned; the cost is query volume.
- 21.1 % `ggml_vk_wait_for_fence` — GPU wait, not real work.
- 9.3 % xatlas charting (`UniformGrid2` + `addFaceToChart`), 5.3 %
  `meshopt_simplifyEdge`, ~6 % allocator/memmove, rest small.

## What was tried

| change | outcome |
|---|---|
| Tighter remesh BVH bounds (`dil = band+1` with proof; center pass radius `eps+keep`; vertex pass `eps+2·cell`) | **kept** — bit-identical output, remesh 4.7→4.3 s |
| CFG batch-2 forward (cond+uncond in one graph) | **rejected before implementation** — at 4k–17k tokens the GEMMs are compute-bound; batching saves sync overhead only. The 2× heuristic applies to memory-bound (small-batch) regimes. Uncond forwards cost ~99 s/run; the only way to cut that is fewer guided steps (quality/parity tradeoff). |
| `TRELLIS_FA_FAST=1` (F16 K/V + F16-accumulate FA instead of BF16+F32) | **rejected** — RADV/gfx1151 crashes or hangs in the sparse shape flows (dense SS flow alone improved 47→31–40 s). Flag retained, default-off, for future driver revisits. BF16 K/V remains mandatory for range (HR activations exceed F16 ±65504). |
| HIP/ROCm backend (`build-hip`, gfx1151) | **rejected as default** — with ggml's default HIP-graph capture the FIRST DiT forward never completes (>10 min; the 32k-node graph appears to break capture). With `GGML_CUDA_DISABLE_GRAPHS=1` it runs correctly but trails Vulkan on every flow: SS 52.0 s (vs 47.1), shape LR 40.0 (vs 28.2), shape HR 193.6 (vs 166.5) — ~10–40 % slower, est. goblin ~7:40 vs 6:09. Numerically fine (cond stats match Vulkan). |
| flow-loop buffer reuse | **skipped** — reallocation shows as ~4 % CPU but the run is GPU-bound; no wall effect. |

## Reference pipeline timings (same inputs, seed 42, 1024_cascade)

Python reference on the old box's RTX 5060 Ti 16 GB (CUDA 13.3, torch 2.12,
native flash_attn 2.8.3 sparse attention, timm-mirror DINOv3, rembg nulled —
inputs pre-matted). Model load 85 s excluded from per-model totals.

| input | generate | postprocess (to_glb) | total | ours (Vulkan iGPU) | ratio |
|---|---|---|---|---|---|
| goblin | 115.6 s | 16.2 s | **2:12** | 6:09 | 2.8× |
| turret | 208.2 s | 26.2 s | **3:54** | 12:39 | 3.2× |

The ~3× gap tracks the hardware gap (discrete Blackwell w/ CUDA flash-attn vs
RDNA3.5 iGPU w/ Vulkan FA); our CPU postprocess (~24 s) is at parity with
their CUDA postprocess (16–26 s). Caveat: running both models in one process
OOMs the 31 GB box (turret residue + goblin peak > RAM) — run per-process.

## Same code on discrete CUDA (RTX 5060 Ti 16 GB, box) — hardware vs implementation

Our port, unmodified, built with GGML_CUDA (g++-14 host, arch 86;120), same
inputs/seed. E2E includes GGUF load; reference "cold" = generate+post+85 s load.

| | ours CUDA 5060 Ti | ours Vulkan iGPU | reference cold, same GPU |
|---|---|---|---|
| goblin flows (SS/LR/HR/tex) | 19.1 / 7.5 / 56.1 / 33.5 = 116 s | 47.1 / 28.2 / 166.5 / 90.6 = 332 s | — |
| goblin E2E | **3:16** | 6:09 | 3:37 |
| turret flows | 19.1 / 15.5 / 149.5 / 89.0 = 273 s | — | — |
| turret E2E | **7:23** | 12:39 | 5:20 |

- Flows scale 2.9× from iGPU-Vulkan to the mid-range Blackwell card — the
  local runtime is a hardware ceiling, not an implementation gap.
- Same-GPU vs reference: goblin-class at parity (faster cold-start);
  turret-class ~1.4× slower — the gap opens with token count (~45k-token HR
  attention), pointing at ggml-CUDA FA / sparse-conv scaling vs
  flash-attn-varlen + flex_gemm as the remaining implementation headroom.

## Standing conclusions

- Vulkan is the right backend on this hardware; the flows are within ~2× of
  what the GPU can do on these graphs and further gains need either fewer
  forwards (steps / guidance interval — quality knobs, not parity-safe
  defaults) or upstream ggml-vulkan FA/matmul improvements.
- The postprocess (~24 s) is no longer worth optimizing relative to the flows.
