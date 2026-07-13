# 28 — Divergence matrix: PyTorch reference vs trellis.cpp

Stage-by-stage comparison of the reference pipeline (space `app.py` `extract_glb`
path — the ground truth users compare against, see 27 §2.3) and this port.
Reference cites are `file:line` into the read-only clones listed in 27 §1
(TRELLIS.2 `75fbf01`, CuMesh `12289e1`, FlexGEMM `6dd94a8`, space `ebf60b2`);
"ours" cites are into `src/`. Post-process semantics are per 27; this file adds
the "ours" side and ranks the deltas.

---

## Executive summary — where the divergences start

The neural front half (DINOv3 → SS flow → shape cascade → decoders) is a
transcription with parameter parity; its only divergences are (a) a different
RNG stream (bit-exact output comparison is impossible by construction — parity
is checked distributionally via `slat_stats`) and (b) the opt-out NOFIX
robustness guards in the sampler. **The divergences that change what a user
sees start at the first post-neural stage and compound from there.** Ranked:

| # | Divergence | Class | Impact/effort | Status |
|---|---|---|---|---|
| 1 | **Narrow-band DC remesh not wired into `trellis_cli`** — port exists (`remesh_dc.cpp`, per 27 §4) and runs in `post-replay`, but the main CLI still ships the raw dual-grid mesh through weld/fill/simplify. Reference geometry is *always* the ~1-voxel offset shell (27 §2.3, §4.6). | missing-stage (main path) | high / low — call site is `trellis_cli.cpp:276-296` | port **closed this session**; CLI wiring **open** |
| 2 | **No `NORMAL` accessor in the GLB** — reference exports area-weighted vertex normals (postprocess.py:211-216, 27 §3.8); `write_glb_textured` writes only POSITION+TEXCOORD_0 (`mesh_glb.cpp:115`). glTF mandates *flat* normals when unspecified → faceted shading. | missing-stage | high / low | open |
| 3 | **Cleanup-ladder gaps**: no `remove_small_connected_components(1e-5)` (floating debris survives — also bloats chart count and bake cost), no `repair_non_manifold_edges`, no `remove_duplicate_faces` as a mesh op (only chart-level flagging), no `unify_face_orientations`; `weld_vertices` is ours-extra. | missing-stage + ours-extra | med / med | open |
| 4 | **doubleSided=true vs reference false** (`mesh_glb.cpp:116` vs postprocess.py:303) — masks winding bugs, costs fill-rate; blocked on #1/#3 (consistent winding needed first). | param-diff | med / low (after #1) | open |
| 5 | **Texel snap order**: reference snaps *every* texel to the closest point on the hole-filled original mesh before sampling (postprocess.py:254-256); ours samples at the raster point and snaps only on trilinear miss (`uv_bake.cpp:61-69`). | algo-diff | med / low | BVH snap **closed this session**; unconditional order open |
| 6 | **Hole fill semantics**: perimeter < 3e-2 world units + centroid apex vertex (clean_up.cu:450-712) vs ≤64 boundary edges + fan from a loop vertex (`uv_bake.cpp:279-367`). | algo-diff | low-med / med | open |
| 7 | **TELEA inpaint vs BFS dilation** (postprocess.py:288-292 vs `uv_bake.cpp:95-113`): smooth diffusion vs nearest-texel copy in the unmasked atlas area. | algo-diff | low / med | open |
| 8 | **Simplifier family**: CuMesh parallel independent-set QEM (27 §3.2) vs meshopt error-ladder + FQMS tail (`uv_bake.cpp:189-277`); same 300k face target. | algo-diff | low / high | open (accepted) |
| 9 | **Unwrap**: pre-clustering + stock xatlas per cluster now mirrors the reference structure (27 §6); residual param-diffs in the pack step (padding 2 vs 0, blockAlign, explicit resolution cap loop) and ours-extra L1/L2 triage. Chart *layout* is inherently non-comparable (xatlas packer RNG). | param-diff + ours-extra | low / n.a. | pre-clustering + pack cap **closed this session** |
| 10 | **Preprocess**: default matte is a white-background threshold, not BiRefNet (opt-in `--birefnet`; space uses remote BRIA RMBG-2.0); crop margin 1.10 vs reference exactly 1.0 (trellis2_image_to_3d.py:156 vs `preprocess.cpp:36`). | algo-diff (default) + param-diff | med (bad inputs) / low | open |
| 11 | **Export format**: PNG textures vs WebP (`extension_webp=True`, space app.py:549); MR texture R=255 vs 0; baked alpha channel dropped (A=255). | param-diff (cosmetic) | nil-low / low | open (accepted) |
| 12 | **Sampler NOFIX guards** (`flow_runner.cpp:160-167`): CFG-rescale ratio clamp [0.2,5] + non-finite velocity zeroing. No-ops in-distribution; alters trajectories only on pathological inputs. `TRELLIS_NOFIX=1` restores raw behavior. | ours-extra | nil (in-dist) / n.a. | open by design |

**Closed this session:** narrow-band DC remesh port (`remesh_dc.cpp`), TriBvh
closest-point snap in `VoxSampler` (`tri_bvh.cpp`, `uv_bake.cpp:61-69`),
atlas pack-resolution cap loop (`uv_bake.cpp:800-815`), normal-cone
pre-clustering with per-cluster xatlas meshes (`uv_bake.cpp:656-771`).

**Remaining open:** rows 1-2 above (wiring + normals), cleanup-ladder ops,
doubleSided, unconditional texel snap, hole-fill metric, TELEA vs BFS, WebP,
offset-shell implications (below), NOFIX guards (intentional).

**Offset-shell geometry implications** (once #1 is wired): with
`project_back=0` the output surface sits ~1 voxel *outside* the true surface
(27 §4.0, §4.6); gaps thinner than 2·eps fuse (intentional smoothing);
sharp features round to voxel scale (simple DC, no QEF). These are *reference
behaviors we must reproduce*, not defects — texel snap (#5) is what pulls the
colors back onto the true surface.

---

## The matrix

Column key — **Class**: match / param-diff / algo-diff / missing-stage /
ours-extra. **Verify**: cheapest local A/B. **Rank**: priority (impact/effort).

| Stage | Reference | Ours | Class | Expected visual impact | Verify | Rank |
|---|---|---|---|---|---|---|
| **1. Preprocess / bg removal** | Max-1024 LANCZOS downscale; alpha kept if image has one, else RMBG-2.0 matte (BiRefNet locally, remote BRIA in the space); bbox over `alpha > 0.8·255`; square crop, margin exactly 1.0 (`size * 1`); premultiply on black (trellis2_image_to_3d.py:127-162; space app.py:326-356). ImageNet-normalized at 512 and 1024. | `preprocess.cpp:48-65`: alpha used if any pixel <250, else **white-threshold matte** (min channel <232, line 59); bbox `alpha>0.8`; crop margin **1.10** (line 36); premultiply (line 43); stb resize (not LANCZOS). `--birefnet` runs a full BiRefNet port (`birefnet.cpp`) at fixed 1024². ImageNet normalize `preprocess.cpp:16-26`. | algo-diff (default matte) + param-diff (margin, resampler) | Threshold matte fails on non-white BGs and halos → garbage conditioning. 10% margin shrinks the object in frame → slightly different DINOv3 features, mild reconstruction framing shift even on clean inputs. | Run same PNG with/without `--birefnet`; compare `slat_stats cond_512` and final renders. Alpha-bearing inputs bypass the matte entirely. | P3 (med/low): default to BiRefNet, drop margin to 1.0 |
| **2. DINOv3 conditioning** | `get_cond` (trellis2_image_to_3d.py:164-186): DINOv3 ViT-L features at the pipeline resolutions; `neg_cond = zeros_like(cond)` (line 183). | `dinov3.cpp:56-115`: full transcription (patch embed, 2D half-split RoPE with prefix-identity, 24 blocks, gamma scales, final non-affine LN); cond at 512 (+1024 for cascade) `trellis_cli.cpp:83-95`; zero neg cond (line 90). | match (transcription) | None beyond float noise (verified against reference activations in IMPL_NOTES_dinov3). | `test_dinov3` harness; `slat_stats` cond lines in the CLI log. | — |
| **3. SS flow + decode** | Dense flow at 16³×8ch, `FlowEulerGuidanceIntervalSampler`, space defaults steps 12 / cfg 7.5 / rescale 0.7 / rescale_t 5.0 (space app.py sliders, 27 §2.3); decode `decoder(z_s) > 0` then `max_pool3d > 0.5` to res 32 (trellis2_image_to_3d.py:227-233). | `trellis_cli.cpp:97-116`: same defaults (line 104: steps 12, gr 0.7, rt 5.0, gi [0.6,1.0] from checkpoint pipeline.json); sampler math `flow_runner.cpp:118-174` (t-rescale line 126, interval gate 136, CFG-rescale 146-163 matching classifier_free_guidance_mixin.py:9-26); decode `ss_decoder.cpp:83-129`, `logits > 0` + any-in-2³-block ≡ maxpool>0.5. | match (modulo RNG + NOFIX guards) | Different noise stream ⇒ a *different valid sample*, not an error. NOFIX ratio clamp only bites on OOD inputs. | `TRELLIS_DBG_STEP=1` per-step stats; `TRELLIS_NOFIX=1` A/B; `--voxply` dumps the res-32 occupancy for viewing. | — |
| **4. Shape SLAT cascade** | `sample_shape_slat_cascade` (trellis2_image_to_3d.py:277-364): LR flow @res32 → denorm → decoder upsample → quantize `((c+0.5)/512·(hr//16)).int()` (line 331) with −128 backoff, floor 1024, cap 49152 tokens (335-339) → HR flow with cond_1024; shape norm constants from checkpoint json. Space defaults cfg 7.5 / rescale 0.5 / steps 12 / rescale_t 3.0. | `trellis_cli.cpp:140-196`: identical backoff loop (168-183), identical quantization (line 173), same sampler params (line 128), SHAPE_MEAN/STD hardcoded from checkpoint (lines 42-43); upsample via `shape_upsample` (`shape_decoder.cpp:98-103`). | match (modulo RNG) | None structural. | `--dump-slat` writes `/tmp/hr_slat.bin` for the reference-decoder diff; token/backoff lines in the log. | — |
| **5. Shape decode + dual grid** | `decode_shape_slat` → SparseUnetVaeDecoder → FlexiDualGrid (o-voxel `flexible_dual_grid.py`): dual vert `(coord + 2σ(off) − 0.5)/res − 0.5`, quad per intersected edge, diagonal by softplus-weight product, fixed winding tables. | `shape_decoder.cpp:56-96` (ConvNeXt stages + C2S, chunked linears) + `dual_grid.cpp:17-56`: same offset tables (lines 8-12), same sigmoid/softplus vertex + diagonal rule (46-52). | match (transcription) | None. | `test_shape_dec` / `test_slat_shape`; `--decim 0 --no-weld --no-fill` in post-replay isolates raw decode output. | — |
| **6. Tex flow + decode** | `sample_tex_slat` (trellis2_image_to_3d.py:391-432): noise width `in_ch−32`, concat normalized shape SLAT; space defaults cfg 1.0 / rescale 0.0 / steps 12 / rescale_t 3.0; decode `decoder(slat, guide_subs=subs)·0.5+0.5` (line 451). | `trellis_cli.cpp:219-251`: concat forward (227-234), params line 235 (gs 1.0, gi [0.6,0.9], rt 3.0), TEX_MEAN/STD denorm (238); `tex_decode` with guide_subs (`shape_decoder.cpp:105-110`); `·0.5+0.5` + clamp (245-247; ref clamps at uint8 quantize — same result). | match (modulo RNG) | None structural (gs=1.0 ⇒ CFG/rescale path never runs at defaults). | `slat_stats`; PBR voxel count log line; `TRELLIS_DUMP_POST` dumps the 6-ch field for replay. | — |
| **7. Hole fill** | CuMesh `fill_holes(3e-2)`, run in `decode_latent` (trellis2_image_to_3d.py:474) and again in the to_glb prologue (postprocess.py:110): boundary loops with world perimeter <3e-2 get **one new centroid vertex** + one fan triangle per boundary edge, winding from canonical edge packing (arbitrary) (clean_up.cu:450-712, 27 §3.1). | `fill_small_holes` (`uv_bake.cpp:279-367`): loops of **≤64 edges** (topological bound, `uv_bake.h:50`), fan from `loop[0]` (no new vertex); directed pass keeps neighbor-consistent winding, second undirected pass catches flipped tears. Called `trellis_cli.cpp:278` and post-decimation (296). | algo-diff | Both close small holes; thresholds cross over (3e-2 ≈ 15-30 edges at res 512-1024, ours up to 64). Fan-from-vertex gives skinnier triangles than centroid fan. Ours' winding handling is *stronger* than the reference's (which relies on the later remesh to discard orientation). Matters mainly as remesh-UDF input hygiene. | `post-replay --no-fill` A/B; `glb_metrics.py` boundary-edge count. | P6 (low-med/med) |
| **8. Remesh** | Production path (postprocess.py:165-187): `remesh_narrow_band_dc` — octree band `\|UDF−eps\|<0.87·cell`, pseudo-sign `UDF−eps`, simple DC (mean of crossings, cell-center fallback), far-edge ownership, fixed quad tables, effectively-always split-1, `project_back=0` ⇒ ~1-voxel offset shell, watertight (27 §4); then only `simplify(target)`. | `remesh_dc.cpp:34-259`: faithful port — same domain inflation/eps (42-44), same predicate (103-137, candidates via triangle-AABB dilation instead of octree: provably a superset, same surviving set), corner mapping v/res (163-165), DC kernel (180-207), quad tables + split-1-always with the upstream diagonal bug reproduced as documented (213-239, 27 §4.5), intended (non-inert) missing-voxel quad drop (228-233). **Wired only into `post_replay.cpp:70-74` — `trellis_cli.cpp` does not call it.** | match (port) / **missing-stage (CLI wiring)** | Without it the CLI ships voxel-native dual-grid geometry: different silhouette (no 1-voxel inflation), thin-gap fusion absent, not watertight, orientation mixed. With it: reference-shaped closed shell. Largest single geometry delta on the main path. | `post-replay dump.bin out.glb` vs `--no-remesh`; `--band 2` sensitivity; `glb_metrics.py` watertight flag + component count; render compare. | **P1 (high/low)** |
| **9. Simplify** | `CuMesh.simplify(300000)` (space default; face target): parallel independent-set QEM edge collapse, midpoint/boundary-endpoint placement (no optimal solve), λ_edge 1e-2, λ_skinny 1e-3, flip veto, ×10 threshold escalation (cumesh.py:320-359, simplify.cu, 27 §3.2). Space also pre-simplifies at 16.7M faces (no-op) (space app.py:529). | `decimate_simplify` (`uv_bake.cpp:189-277`): meshopt error ladder 0.01→1.0 (+SimplifyPrune on rung 1) then unguarded FQMS quadric tail to target; target 300k cascade / 150k non-cascade (`trellis_cli.cpp:288-289`); post-pass weld+fill (293-296). Legacy `--decim GRID` clustering; `--decim 0` off. | algo-diff (same family, same target) | Both quadric-driven to the same count; triangle-size distribution and boundary handling differ. FQMS can emit non-manifold configurations the reference's manifold-only collapses would not (and vice versa: ref's independent-set passes overshoot differently). Rarely visible after texturing. | `post-replay --faces N`; `glb_metrics.py` F, non-manifold-edge count, texel-density CoV. | P8 (low/high — accepted) |
| **10. Unwrap (charting + packing)** | `uv_unwrap` (cumesh.py:408-480, 27 §6): `remove_degenerate_faces` → GPU normal-cone clustering (merge cost = merged-cone-angle + 0.1·area + 1e-4·perim²/area, threshold π/2, refine off) → **stock xatlas per cluster**, all-default ChartOptions/PackOptions (padding **0**, resolution 0 ⇒ ~1024 estimate, blockAlign false, rotateCharts true, bruteForce false ⇒ packer RNG), one shared atlas, UVs atlas-normalized. | `uv_bake` (`uv_bake.cpp:551-998`): L1 sliver/dup `faceIgnoreData` (563-589), L2 tiny-component strip with reserved band (594-653, 857-947), normal-cone BFS clusters (seed-hemisphere gate, **12k-face cap**, 656-708), per-cluster xatlas meshes (754-771), 300 s chart timeout (779-792), pack `resolution=TX, padding=2, blockAlign=true` + 4-attempt cap loop (800-815), squeeze normalize (819, 847); fallbacks `uv_chart_project` / `--box-uv`. | param-diff (pack: padding/blockAlign/explicit cap) + ours-extra (L1/L2 triage, cluster cap, timeout) | Chart layouts are non-comparable by design (packer RNG) — what matters is density uniformity and seam behavior. Our padding 2 at bake-T *reduces* bleed vs reference padding 0 at ~1024 packer units. L2 strip trades tiny-component fidelity for tractability. Cluster cap can split what the reference keeps whole → more seams on large smooth regions. | `glb_metrics.py` chart count, texel-density mean/CoV, zero-UV-area % vs an HF-space GLB; `post-replay --atlas 1024/2048`, `--box-uv` A/B. | P9 (low/n.a.) |
| **11. Texture raster** | nvdiffrast on UV-as-clip-space at T (2048 default), 100k-face chunks, later chunk wins overlap (postprocess.py:229-243); per-texel 3D position interpolated on the decimated mesh (249). | CPU barycentric raster, 0.5-texel centers, −0.001 edge tolerance, later face wins (uv_bake.cpp:956-990); identical position interpolation (974-977). | match (equivalent) | Sub-texel edge-rule differences only; same surface either way. | Diff two bakes of the same dump (`post-replay` twice) — deterministic; inspect `_base.png`. | — |
| **12. Attribute sampling** | **Every** masked texel: BVH `unsigned_distance` → closest point on the hole-filled *original* mesh via barycentric reconstruction (postprocess.py:254-256, 27 §5) → `grid_sample_3d` trilinear with **renormalization over present voxels** (grid_sample.cu:170-193, 27 §7), convention `g=(p+0.5)·res`. | `VoxSampler` (`uv_bake.cpp:40-88`): identical trilinear convention + renormalization (40-60; equivalence proven 27 §7); **snap is conditional** — closest-point resample only when the direct trilinear misses (61-69); ring-average shell crawl as no-BVH fallback (70-87, ours-extra). Snap BVH = pre-decimation mesh (`trellis_cli.cpp:301-303`). | algo-diff (conditional vs unconditional snap) | Texels whose raster point is inside a populated voxel neighborhood sample at the *decimated* surface point, not the closest original-surface point → slight color smearing where simplification moved the surface (creases, thin parts). Miss-path behavior now matches. | `post-replay --no-snap` vs default vs (edit) unconditional-snap build; look at crease sharpness. | P5 (med/low) |
| **13. Inpaint / dilation** | `cv2.INPAINT_TELEA` over `~mask`, radius 3 (base) / 1 (MR/alpha) (postprocess.py:288-292) — smooth diffusion filling the whole empty atlas. | Multi-source BFS `dilate_full` (`uv_bake.cpp:95-113`) — nearest-written-texel copy, unbounded. | algo-diff | Affects only never-rasterized texels (chart gutters): visible as blocky vs smooth color under bilinear/mip sampling at chart edges; padding 2 keeps it mostly hidden. | Zoom `_base.png` chart margins; render with mipmapping vs without. | P7 (low/med) |
| **14. Material / axis / export** | `PBRMaterial` (postprocess.py:296-304): RGBA base (alpha from attrs), MR=(R=0,G=rough,B=metal), factors 1.0, alphaMode OPAQUE, **doubleSided=False**; axis `(x,y,z)→(x,z,-y)` + `uv.y=1−uv.y` (313-315); **vertex normals exported** (211-216, area-weighted per 27 §3.8); trimesh GLB, **WebP** textures (space app.py:549). | `write_glb_textured` (`mesh_glb.cpp:86-145`): rotation match (92); factors 1.0 match (116); **doubleSided=true** (116); **PNG** (97-99,118); MR **R=255** (`uv_bake.cpp:986`; glTF ignores R — cosmetic); base **A=255** (alpha attr dropped, `uv_bake.cpp:985`; ref is OPAQUE anyway); **no NORMAL accessor** (115) ⇒ spec-mandated flat shading; no UV flip needed (our raster is y-down by construction — match). | param-diff (doubleSided, WebP) + missing-stage (normals; alpha channel) | **Normals: high** — faceted vs smooth shading on every curved surface. doubleSided: hides winding errors, minor perf; safe to flip only after remesh wiring guarantees orientation. WebP: file size only. | Load in a strict viewer (Babylon sandbox / `gltf-validator`); `glb_metrics.py` reports doubleSided/MIME/MR-channel stats; render compare after adding normals. | **P2 (high/low)** normals; P4 doubleSided; P11 rest |
| **15. Non-remesh cleanup ladder** | remesh=False branch (postprocess.py:134-162, unused in production): simplify(3×target) → `remove_duplicate_faces` → `repair_non_manifold_edges` → `remove_small_connected_components(1e-5)` → `fill_holes(3e-2)` → simplify(target) → repeat cleanup → `unify_face_orientations`; doubleSided=True. Semantics 27 §3.3-3.6. | Main CLI path today (`trellis_cli.cpp:276-296`): `weld_vertices` (ours-extra, eps=1/(8·res)) → `fill_small_holes` → decimate ladder → weld+fill again. **Missing:** duplicate-face removal (only chart-level flagging, `uv_bake.cpp:563-589`), non-manifold edge repair, small-component removal, orientation unification. | missing-stage (4 ops) + ours-extra (weld) | Floating sub-voxel debris survives → speckle in silhouette + thousands of extra components feeding the unwrap (the L2 strip exists largely to absorb this). Mixed winding is invisible only while doubleSided=true. Note this row describes what our main output *actually* goes through until row-8 wiring lands; the reference never ships this branch. | `glb_metrics.py` component count/top-5 + winding-consistency % on our GLB vs an HF-space GLB; `post-replay --no-weld` isolates weld. | P3 (med/med) — `remove_small_connected_components` first: cheapest, most visible |

---

## Verification toolbox (referenced by the matrix)

- **`post-replay <dump.bin> <out.glb>`** (`post_replay.cpp`) — re-runs
  everything after the neural stages from a `TRELLIS_DUMP_POST=<path>` dump
  (`trellis_cli.cpp:263-275`). Flags: `--no-remesh`, `--band N`, `--no-snap`,
  `--box-uv`, `--faces N`, `--atlas T`, `--decim GRID|0`, `--no-weld`,
  `--no-fill`, `--no-bake`.
- **Env hooks:** `TRELLIS_DUMP_POST` (dump post-stage inputs),
  `TRELLIS_DBG_STEP` (per-flow-step pred/sample stats, `flow_runner.cpp:129`),
  `TRELLIS_DBG_NAN` (per-layer DiT NaN breakdown, `flow_runner.cpp:34,63-79`),
  `TRELLIS_NOFIX=1` (disable sampler guards, `flow_runner.cpp:130`).
- **CLI debug flags:** `--voxply` (res-32 occupancy PLY), `--dump-slat`
  (`/tmp/hr_slat.bin` for the reference-decoder diff).
- **`tools/glb_metrics.py`** — CPU-only geometry/UV/material metrics
  (components, boundary/non-manifold edges, winding consistency, watertight,
  chart count, texel-density stats, MIME/doubleSided/MR channels) with a
  side-by-side table for ours-vs-space GLB comparison.
- **`tools/render_glb_fast.py`** — quick render compare of two GLBs.

## Addendum — closed after the matrix was written (same session)

- **xatlas-ignored degenerate faces**: xatlas silently ignores sub-epsilon faces; they remain in its output with twin/(0,0) UVs and rasterize as atlas-spanning wedges. Now detected via chart-coverage (faces in no chart) and dropped — they are zero-area in 3D. Failed (Invalid) charts are rerouted to the planar strip.
- **Remesh wired into `trellis_cli`** (was post-replay-only when the matrix was written); the meshopt ladder now reaches the 300k target on both benchmarks with no FQMS stage, `[fqms]` no longer appears.
- **NORMAL accessor added to `write_glb_textured`** — area-weighted vertex normals in the rotated frame; viewers now smooth-shade.
- Duplicate faces are dropped outright (not ignored); sliver charting is left to xatlas under the 12k-face normal-cone cluster cap.

## Addendum 2 — oracle validation (reference postprocess on our dumps)

The reference postprocess (`o_voxel.postprocess.to_glb`, exact HF-space args:
`remesh=True, remesh_band=1, remesh_project=0`, 300k faces, 2048² WebP) was run
on the CUDA box against **our own post-stage dumps** (`goblin_post.bin`,
`turret_post.bin`), isolating postprocess parity from the neural stages. Oracle
GLBs live in `refs/oracle/`. Build recipe quirks (torch 2.12 + GCC15/nvcc):
host compiler g++-14 **and** `NVCC_APPEND_FLAGS=-std=c++20` (nvcc's cudafe
strips `typename` from `List_inl.h:202`; C++20 makes it optional), `MAX_JOBS=4`
(31 GB RAM), git submodules required (`CuMesh/third_party/cubvh` + Eigen,
`o-voxel/third_party/eigen`).

`glb_metrics.py`, oracle vs ours (goblin / turret):

| metric | oracle | ours |
|---|---|---|
| faces | 283k / 300k | 295k / 298k |
| bbox | match to 4 decimals | match |
| winding consistency | 100% / 100% | 100% / 100% |
| welded boundary edges | **27 / 81** | 15,318 / 3,503 |
| welded non-manifold edges | 47 / 1,076 | 218 / 3,214 |
| UV charts | **1,564 / 16,304** | 20,106 / 44,390 |
| uv bbox coverage | full [0,1]² | 0.66×0.99 / 0.73×0.94 |
| texel density mean (T=1) | **0.167 / 0.075** | 0.056 / 0.020 |
| texel density CV | **0.24 / 0.44** | 1.69 / 1.90 |
| doubleSided | False | True |

Visual renders (model-viewer, `tools/mv_preview/`) are at parity on both
models. Remaining evidence-backed deltas, all atlas-side (P4 backlog):

1. **Chart fragmentation** — 2.7–13× more charts than the oracle; our
   normal-cone clustering fragments far more than cumesh's (90° cone,
   `global_iterations=1`, smooth_strength=1). Drives seam count, vertex
   duplication (+30%), and padding waste.
2. **Atlas occupancy / texel density** — oracle fills the full atlas at ~3×
   our texel density with 4–7× tighter density spread; our pack-retry loop
   over-shrinks (`0.96^attempt` × page ratio) and leaves 27–34% of atlas width
   unused.
3. **Residual boundary cracks** — welded-boundary edges 15.3k vs 27 (goblin):
   dropped degenerate/uncovered faces and the tiny-component strip leave real
   holes; the oracle charts everything.
4. **doubleSided=false** now that remesh guarantees orientation (winding is
   100% on both); WebP vs PNG is cosmetic (file size).

## Addendum 3 — P4 parity pass (closes Addendum 2's deltas)

All four Addendum-2 items were implemented, verified per dump with
`post-replay` + `glb_metrics.py` + model-viewer renders vs `refs/oracle/`:

1. **Clustering** (`uv_bake.cpp`): the greedy seed-flood normal-cone clustering
   was replaced with a faithful CPU port of cumesh `compute_charts`
   (atlas.cu:1071-1210) — bottom-up chart merging with the exact reference
   cost (`merged_cone_half_angle + 0.1·area + 1e-4·perim²/area`, collapse iff
   argmin of both charts and cost ≤ π/2, cones recomputed per round). Goblin:
   1,677 clusters vs oracle's 1,564 charts (was 17k+). Cluster meshes now go
   to xatlas the way the reference sends them: positions only, no normals, no
   custom epsilon.
2. **Packing**: replaced the shrink-only 2048-target retry loop with the
   reference scheme — stock PackOptions (padding 0, resolution 0 → xatlas
   grows a single ~1024–1300² atlas at its own density estimate), then UVs
   normalized by the final atlas width/height so charts always fill the full
   [0,1]². Texel density now *exceeds* the oracle (goblin 0.181 vs 0.167,
   turret 0.088 vs 0.075 at T=1) with p5–p95 spread at parity; the pack-cap
   machinery is gone.
3. **Cracks**: xatlas-ignored (degenerate) and Invalid-chart faces are kept,
   emitted point-collapsed onto one charted twin UV (they are zero-area in 3D).
   The tiny-component strip triage now engages only on soup-scale inputs
   (>2000 components, i.e. un-remeshed fallback paths). Goblin welded-boundary
   edges: 15,318 → 307 (oracle 27).
4. **TELEA inpaint** replaces BFS dilation in `uv_bake` (base r=3, MR r=1),
   an OpenCV-faithful FMM port. Bug learned the hard way: the initial narrow
   band must lie on the KNOWN side (as in OpenCV) — seeding it with unknown
   pixels leaves the entire first ring unpainted (black), which reads as dark
   veins along every chart seam at padding 0.
5. **Material/export**: doubleSided=false when remeshed (callers pass
   `rm.F()==0`), MR red channel 0, base alpha from the decoded alpha attr
   (alphaMode stays OPAQUE), lossy WebP textures (q80) via `EXT_texture_webp`
   (libwebp v1.5.0 FetchContent, `TRELLIS_WEBP=ON` default, PNG fallback).
   `glb_metrics.py` follows the extension's source indirection.

Remaining known deltas (accepted): chart count still ~2× oracle on the turret
(32k vs 16k; density median matches, mean above oracle — xatlas subdivides our
clusters more than theirs); turret welded-boundary 3.1k vs 81 (pre-existing in
the decimated geometry, not opened by the bake); our WebP files are larger
(noisier inpaint far-field + alpha channel content); zero-UV-area faces 1.8%
(the kept point-collapsed faces the reference deletes).
