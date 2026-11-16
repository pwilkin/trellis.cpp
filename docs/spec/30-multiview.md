# 30 — Multi-view conditioning (`--view`)

Variable-N multi-view conditioning for the TRELLIS.2 family. The method follows
ComfyUI-Trellis2's `FlowEulerMultiViewSampler`. Each flow runs once per view and the
per-view velocities are blended token by token. This port generalises the reference from
its four named views to arbitrary azimuth/elevation and N ≤ 12. It also fixes the
reference's axis convention.

Code: `include/multiview.h` / `src/multiview.cpp` (views, directions, weights),
`sample_flow_mv` in `src/flow_runner.cpp`, the per-view preprocessing / DINO / `CondSet`
plumbing in `src/trellis_cli.cpp`, the CLI flags in `src/trellis_args.cpp`, multi-image
`/generate` in `src/trellis-server.cpp`.

## Method

For each view *i* (az θ, el φ in degrees, `--yaw-offset` added to θ):

```
d_i      = (cos φ sin θ, sin φ, cos φ cos θ)    viewer frame = glTF frame: X right, Y up, Z → front camera
score_i  = p · d_i^grid                          p = token position in [-1, 1]^3
w_i      = softmax_i(τ · score_i)                τ = --blend-temp (default 2.0)
pred_v   = Σ_i w_i · guided_velocity(x_t, t, cond_i, neg)   then the ordinary Euler step
```

`guided_velocity` is the single-view guidance interval + CFG + CFG-rescale, applied **per view**
as in the reference mixin chain. All views share one negative (the zero tensor at the same
`x_t`, `t`), so it is evaluated once per guided step. This is exact, and a guided step costs
V+1 forwards instead of 2V. A non-finite per-view velocity is zeroed before blending, so one bad
view cannot poison the others. With V = 1, `sample_flow_mv` *is* `sample_flow`: same calls, same
arithmetic, same log line.

Blended stages (`--mv-stages`, default all):

| stage | grid of the weights | conds |
|---|---|---|
| sparse structure (dense 16³) | 16, token `x*256 + y*16 + z` | DINOv3@512 |
| shape SLAT, res 512 / LR cascade | 32 | DINOv3@512 |
| shape SLAT, HR cascade | `hr_res/16` (64 at 1024, 96 at 1536) | DINOv3@1024 |
| texture SLAT | 32 (res 512, or the mixed `--tex-res 512` path) or `hr_res/16` | as the shape stage it follows |

Each view is preprocessed independently: auto/threshold/BiRefNet matte with bbox crop, then
DINOv3 at 512 (and 1024 for the cascade). BiRefNet and DINOv3 are each loaded once and looped
over the views. Peak VRAM is unchanged from single-view: one `DitRunner` graph serves every view
and only the cond tensor is re-uploaded. Each view adds 4.2 MB (512) + 16.8 MB (1024) of host
memory.

## Axis convention

trellis.cpp sparse coords are `{x, y, z}` = torch `coords[:, 1:4]`. The grid is z-up with the
object's front on −y; the GLB writers rotate `(x, y, z) → (x, z, −y)`. The default preset
maps viewer `(X, Y, Z)` to grid `(X, −Z, Y)`:

| `--front-axis` | grid x | grid y | grid z | meaning |
|---|---|---|---|---|
| `-y` (default) | X | −Z | Y | TRELLIS canonical: front on grid −y, z up |
| `+y` | −X | Z | Y | object generated facing away |
| `+x` | Z | X | Y | front on grid +x |
| `-x` | −Z | −X | Y | front on grid −x |
| `ref-z` | Z | −Y | X | reference `front_axis='z'` (front = +c[0], right = +c[2]) |
| `ref-x` | X | Y | Z | reference `front_axis='x'` (front = +c[2], right = +c[0]) |

The reference sampler reads `z = coords[:,1]`, `x = coords[:,3]`, assuming a `[b, z, y, x]`
layout. In TRELLIS axes, its `front_axis='z'` splits front/back along grid **x** (lateral) and
right/left along grid **z** (vertical). Neither of its presets uses grid y, the actual depth
axis. `ref-z` / `ref-x` keep that behaviour, bit for bit, for parity and for ComfyUI-identical
output.

Token position: the geometric presets use the cell centre, `p = (c + 0.5)/G·2 − 1`, on the
grid G the tokens live on. The `ref-*` presets copy the reference: sparse
`p = c/R_model·2 − 1`, with `R_model` the flow GGUF's own `resolution`, and dense
`linspace(−1, 1, D)`.

Embedded flow resolutions (`trellis.config_json`): ss_flow 16, shape_flow_512 32,
shape_flow_1024 64, tex_flow_512 32, **tex_flow_1024 64**.

### Reference quirks kept only in `ref-*` mode

- Sparse normalisation by the model resolution: in the 1536 cascade, grid 96 normalised by
  64 gives p up to +2.
- Cascade LR bug: the reference builds the LR sampler with the HR model's resolution (64),
  so LR coords span p ∈ [−1, 0). Reproduce it with `TRELLIS_MV_REF_LRBUG=1`.
- Unknown view names score −10. Not reachable here: every view has a direction.

## Tests

`trellis-test-multiview [weights_dir] [step_dir]` (CPU only, no weights):

- **T1**: weights vs `tools/ref_multiview_weights.py`. 960 cases: grids 32/64/96 sparse and
  16 dense; view sets {1}, {0,180}, {0,90,180,270}, six with top/bottom, and
  {(30,10),(150,−20),(260,45)}; τ ∈ {0.5,1,2,5}; all presets; yaw 0 and 37. Cardinal `ref-*`
  cases are checked against a literal transcription of the reference functions; all others
  against the generalised formula. The script self-checks the two formulas against each other
  on the cardinal cases (2.5e-8). Result: **max |Δw| = 6.0e-8, rows sum to 1 ± 5e-8**.
- **T2**: fake-forward sampler tests.
  - T2a: V = 1 equals `sample_flow` by **memcmp**, with CFG on/off and rescale on/off.
  - T2b: V = 3 with identical conds matches single view (1.6e-7).
  - T2c: one-hot weights give exactly the selected view.
  - T2d: forward count is 12·V + guided (34 / 58 at V = 2 / 4) and 12·V for gs = 1.
- **T3**: full multi-view trajectories vs `tools/ref_multiview_step.py`, which transcribes the
  reference sampler maths (guidance interval → CFG → rescale → blend → Euler) over the same fake
  model. Cases: dense with CFG + rescale and with CFG; sparse with CFG, no CFG, gs = 0. Result:
  **max rel ≤ 4.0e-7** at every step.

Caveat on T1/T3: the oracle is a transcription of the reference code, not the reference
itself. `MV_REF=<ComfyUI-Trellis2 checkout>` makes both scripts import and run the real
sampler under a stub package (torch, CPU) and check the transcription against it. That path
was not run in the session that produced these numbers.

### End-to-end (NVIDIA Thor iGPU, Vulkan, seed 42)

**A0 noise floor.** Two runs of the unmodified binary give bit-identical `--dump-post` files
(`tools/compare_post.py`), both at res 512 and at res 1024 + `--tex-res 512`. Vulkan on this
device is deterministic run to run.

**T4 (N = 1 regression).** The new binary's `.post` is **bit-identical** to the baseline
(goblin) in every form:

- res 512: positional image; `--view goblin.png:0`; `--view goblin.png:123:10` (N = 1 ignores
  angles);
- res 1024 + `--tex-res 512`: positional image and `--view goblin.png:123:10`.

**A4 smoke.** The same image twice at az 0/180, res 512, completes. Forward counts:

| stage | forwards | formula |
|---|---|---|
| SS | 34 | 12·2 + 10 guided |
| shape | 32 | 12·2 + 8 |
| texture | 24 | 12·2 |

Flow time is 202 s, against 124 s for N = 1.

**T5 (round trip on rendered views).**

Method:

1. `tools/render_views.py render` renders the showcase GLB orthographically at az
   0/90/180/270, el 0, as RGBA with an alpha matte. These renders are the input views.
2. The output GLB is rendered at the same four angles, plus four **held-out** oblique views
   (az 45/135/225/315, el 20) that no view conditions on.
3. `tools/render_views.py compare` crops each silhouette to its alpha bbox and reports IoU and
   the mean RGB error (0–255) over the pixels both renders cover.

Two caveats on reading these numbers:

- With orthographic cameras, the back silhouette is exactly the mirror of the front one, and
  270 of 90. Front/back IoUs are therefore identical by construction and cannot tell the two
  apart. Colour error and the held-out views can.
- The colour error has a large constant floor of about 18–20 even for an almost perfect shape,
  as in the refine runs below. It comes from texture tint and bake differences, so compare it
  between runs only.

Knight (`assets/showcase/knight`, res 512):

| run | IoU 0/90/180/270 | colour 0/90/180/270 | mean colour | held-out IoU | held-out colour |
|---|---|---|---|---|---|
| single view (az 0) | .94 / .72 / .94 / .72 | 36.2 / 41.4 / 36.4 / 37.9 | 38.0 | 0.834 | 38.0 |
| N=2 (0, 180) | .85 / .66 / .85 / .66 | 41.1 / 42.6 / **25.7** / 38.7 | 37.0 | 0.761 | 37.9 |
| **N=4, `-y`, τ 2** | .86 / .82 / .86 / .82 | 29.7 / 27.8 / **19.8** / 26.8 | **26.0** | 0.800 | **27.0** |
| N=4, `ref-z` | .78 / .86 / .78 / .86 | 40.3 / 38.3 / 13.9 / 33.1 | 31.4 | 0.634 | 33.8 |
| N=4, `ref-x` | .87 / .77 / .87 / .77 | 37.6 / 43.6 / 30.0 / 42.6 | 38.4 | 0.727 | 40.0 |
| N=4, `-y`, yaw +180 (control) | .89 / .84 / .89 / .84 | 31.9 / 37.3 / 21.6 / 30.9 | 30.4 | 0.864 | 29.8 |
| N=4, τ 1 | — | back 29.9 | 31.1 | 0.788 | 33.0 |
| N=4, τ 4 | — | back 12.8 | 26.6 | 0.816 | 26.6 |

What these show:

- **Multi-view helps where the views add information.** At N = 4, the colour error drops by
  about 30% on every view, held-out ones included. Side-view IoU rises from 0.72 to 0.82. The
  back view's colour error goes from 36 to 20, so the cape comes from the back photo instead of
  being hallucinated.
- **The blended outline is slimmer.** Blended outputs come out somewhat narrower than the
  single-view one: front IoU 0.94 → 0.85, held-out IoU 0.83 → 0.80. The plan's
  "back IoU ≥ single + 0.05" criterion is ill-posed with orthographic renders; see the first
  caveat above. On silhouette alone it is *not* met. On the back view's appearance, the
  measurable part, multi-view wins clearly (36.4 → 19.8). Blending velocities from views that
  disagree erodes the outline (risk R3). Mitigations from the plan, still to evaluate: unit
  direction scores, a per-view multiplier, annealing τ.
- **Front axis.** `-y` beats both reference presets on every aggregate:
  - against `ref-z`: held-out IoU 0.80 vs 0.63, colour 27.0 vs 33.8;
  - against `ref-x`: 0.80 vs 0.73, colour 27.0 vs 40.0.

  `ref-z` paints the front with the wrong views (front colour error 40). The single-view
  output also keeps the input's front at glTF +Z, i.e. grid −y, which independently confirms
  the canonical-front assumption of §0.2.
- **The yaw-180 control does not separate cleanly.** It is worse on colour (front 31.9 vs
  29.7, held-out 29.8 vs 27.0) but better on IoU (0.86 vs 0.80). TRELLIS predicts a whole
  canonical object from any single view (R2), so the knight, whose front and back silhouettes
  match, survives swapped views. **R1 is therefore settled only against `ref-z`/`ref-x`**, not
  against an arbitrary rotation. An object with a strongly asymmetric silhouette is needed to
  close it.
- **τ.** τ = 1 (the ComfyUI node default) is clearly worse; τ = 2 and τ = 4 are close, with
  τ = 4 slightly better held-out. **Open question 1 is resolved by keeping the sampler default
  τ = 2.0.**

Chest (`assets/showcase/chest`, res 512): not usable as an orientation probe. Its az 0 render
shows the chest's *side*. TRELLIS imposes its canonical pose regardless, rotating the chest 90°
so the lock face sits at az 0 in single view and at N = 2. Front-face IoU is 0.84 at N = 4,
but the side views stay at 0.60. **Multi-view conditioning cannot override the model's pose
prior** (R2). Views should show the object in its canonical orientation, front at az 0.

Mean IoU / colour (conditioned views; held-out):

| chest run | conditioned | held-out |
|---|---|---|
| single | 0.714 / 29.6 | 0.720 / 34.0 |
| N=2 | 0.705 / 29.1 | 0.721 / 33.2 |
| N=4 | 0.722 / 30.8 | 0.797 / 35.7 |

**Timings, res 512, Thor Vulkan** (the knight's flows; the chest is larger):

| | SS | shape | texture | flows total |
|---|---|---|---|---|
| N=1 (goblin) | 61.7 s (22 fwd) | 38.8 s (20) | 23.4 s (12) | 124 s |
| N=2 | 93.0 s (34) | 62.0 s (32) | 46.6 s (24) | 202 s |

At N = 4 the full knight run took about 240 s, against 112 s for N = 1, including
preprocessing and the GLB postprocess. This is in line with §4's estimate of
`12·N + guided` forwards.

**1024 acceptance (knight, res 1024 + `--tex-res 512`).** No NaN/Inf in any `[stats]` line.

| views | SS | shape LR | shape HR | texture | wall |
|---|---|---|---|---|---|
| N=2 | 93 s (34 fwd) | 16 s (32) | 148 s (32) | 12 s (24) | 377 s |
| N=4 | 159 s (58) | 30 s (56) | 269 s (56) | 26 s (48) | 596 s |

**T6 (server).** All checks pass:

- Two `image` parts with `azimuths=front,180` return a GLB; a single image unchanged also
  returns a GLB.
- 400s, each with a message: mismatched azimuth count, missing azimuths with two images,
  elevation 95, `model=pixal3d` with two images, `/refine` without mesh or without image.
- No staged request files are left behind.

**Final binary.** A full rebuild, then `t4_512_pos` and the res-1024 run (the latter during the
memory measurement) re-checked against the A0 baselines: **bit-identical**.
