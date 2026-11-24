# 31 — Mesh refinement (`--refine-mesh`)

Re-generates an existing mesh from one or more views, the way ComfyUI's
`Trellis2MeshRefinerMultiView` does (cuzelac/ComfyUI-Trellis2-MultiViewRefiner). The input
mesh fixes **which latent cells exist**. The shape SLAT on those cells is then re-sampled from
noise, conditioned on the view(s) with the multi-view blend of spec 30, and the texture flow, PBR
decode, postprocess and GLB writer run as usual. No sparse-structure stage and no LR cascade
stage run.

```
views → cutout → DINOv3 (512 at --res 512, else 1024)
mesh  → load (GLB/glTF/OBJ) → normalise (preprocess_mesh) → flexible dual grid @R (R = --res)
      → fine coords @R:
          encoder: shape_encode → latent @R/16 → shape_upsample (shape decoder) → coords @R
          voxel:   the occupied voxels of the dual grid directly
      → quantise to hr_res/16 with the token backoff (quantize_with_backoff)
      → shape SLAT flow (shape_flow_512 at R = 512, else shape_flow_1024), blended, from noise
      → shape decode @hr_res → texture SLAT flow (blended) → PBR decode → postprocess → GLB
```

`--refine-coords auto` (default) uses the encoder when `shape_enc.gguf` is in the model
directory, and voxel coords otherwise. `--refine-restore-transform on` (default) maps the result
back to the input's centre and scale. The axis swap needs no extra step: the writers' rotation
undoes it, so a y-up input keeps its orientation.

Code: `src/mesh_io.cpp` (cgltf v1.14 vendored in `thirdparty/cgltf/`; hand-written OBJ
parser), `src/mesh_voxelize.cpp`, `sparse_s2c` / `sparse_linear_streamed` /
`sparse_norm_streamed` in `src/sparse.cpp`, `src/shape_encoder.cpp`, and the
`refine` branches in `src/trellis_cli.cpp`. `POST /refine` is in `src/trellis-server.cpp`.

## Normalisation (`normalize_mesh_trellis`)

This is the reference `preprocess_mesh`:

1. Take the bbox centre.
2. Scale by `0.99999 / max_extent`.
3. Map `(x, y, z) → (x, −z, y)`.

Our version clamps to [−0.5, 0.5] where the reference asserts. `restore_mesh_transform`
applies the inverse scale and centre in grid axes, just before the writers.

## Voxeliser (`mesh_to_flexible_dual_grid`)

This is a port of o-voxel's `flexible_dual_grid.cpp`, with the reference call values:

- aabb [−0.5, 0.5]³, `voxel_size = 1/grid`
- face weight 1.0, boundary weight 0.2, regularisation 1e-2

The passes:

- **intersect**: three axis scan-line sweeps in double. Each grid-edge crossing registers the
  four voxels around the edge and flags `intersected[axis]` on the (0,0) one. It accumulates
  the plane QEF and the crossing points. Voxels are kept in first-touch order.
- **face**: a triangle/voxel-box overlap test (plane and three 2D edge projections). It adds
  the plane QEF **unweighted**: o-voxel's `face_weight` only enables the pass.
- **boundary**: edges used by exactly one face, visited in sorted (v0, v1) order. A DDA through
  the voxels adds `boundary_weight · Q_line`.
- **solve**: add the regulariser `1e-2 · cnt · Q(mean point)`, then try an unconstrained 3×3
  solve. If the result leaves the voxel, the best of 6 face, 12 edge and 8 corner candidates is
  taken. Output is the local offset `dual · grid − coords`, as `encode_shape_slat` feeds it.

The reference uses Eigen `colPivHouseholderQr`. The port has small float solvers that mirror
Eigen 3.4's algorithm: pivoting, norm downdates, nonzero-pivot cut-off, Householder
convention, and the order of `Qᵀb` and of the back substitution. It also mirrors the rounding
of the two Eigen kernels that matter on aarch64/NEON:

- the 3×3 matrix-vector product: rows 0–1 are a fused multiply-add chain, row 2 is
  `p0 + (p1 + p2)`. It is reproduced with `std::fma`, which is exactly rounded everywhere.
- the `pᵀQp` reduction: `(l0 + l2) + (l1 + l3)`.

`mesh_voxelize.cpp` is compiled with `-ffp-contract=off`. Without it, GCC fuses borderline
triangle/box tests differently.

### Parity (B3)

Oracle: `tools/ref_voxelize/`. It compiles upstream `flexible_dual_grid.cpp` against Eigen
3.4.0 and a ~40-line torch stand-in (CMake option `TRELLIS_BUILD_REF_ORACLES`; the upstream
source stays in its own checkout). Use `trellis-test-voxelize parity <mesh|box|sphere> <grid>
<dir>` with `trellis-ref-voxelize`. Both sides are built with `-O2 -ffp-contract=off` on
aarch64.

| case | voxels | coords / order / intersected | dual offsets |
|---|---|---|---|
| box @64 | 6,074 | identical | bit-identical |
| UV sphere @64 | 10,561 | identical | bit-identical |
| UV sphere @256 | 168,976 | identical | bit-identical |
| chest.glb @256 | 939,846 | identical | bit-identical |
| knight.glb @512 | 792,931 | identical | bit-identical |

How the gap was closed: the first port matched coords and flags exactly, but only 84% of the
chest's duals were within 1e-3 voxel. Dumping both sides' per-voxel QEFs showed them
identical except in the boundary `b = −A·v0` column (last ulp). Re-solving our QEFs with real
Eigen reproduced the oracle exactly. Each remaining mismatch was then a summation order:

- `Qᵀb` adds the top entry last;
- `pᵀQp` pairs its lanes;
- the NEON matrix-vector product, as above.

On other CPUs, Eigen's own vectorisation differs, and the oracle would move by ulps there too.

Analytic checks (`trellis-test-voxelize`, no oracle needed):

- the box's flagged edges equal the analytic set (6,072/6,072);
- every voxel lies within one voxel of the surface;
- every surface sample is covered;
- single-face voxels put their dual vertex on the face plane (< 4e-6 voxel);
- corner voxels land within 1e-2 voxel of the corner, which is the regulariser's pull.

## Encoder (`shape_encode`)

Configuration (`shape_enc_next_dc_f16c32_fp16.json`, upstream `microsoft/TRELLIS.2-4B`, 284
tensors, 0.71 GB as GGUF):

1. `input_layer` 6→64 on `[dual − 0.5, intersected − 0.5]`.
2. Stages `[64,128,256,512,1024]` with `[0,4,8,16,4]` ConvNeXt blocks. Stages 0–3 each close
   with a `SparseResBlockS2C3d`, at `blocks.{i}.{num_blocks[i]}`.
3. LayerNorm, no affine, eps 1e-5.
4. `to_latent` 1024→64, keeping channels 0..31 (`sample_posterior=False`).

`sparse_s2c` mirrors `sparse_c2s`:

- conv1 runs at the fine level (Cin → Cout/8) after an affine LN + SiLU.
- Pack into parents `coord/2`, in lexicographic order: child slot `lx + 2·ly + 4·lz`, channel
  `slot·C + c`.
- Non-affine LN + SiLU, then conv2 at the coarse level.
- Skip connection: the mean over groups of `8·Cin/Cout` packed channels of x.

Every step is a host-orchestrated, bounded GraphRun (≤ 65,536 voxels, compacted neighbour
columns), like the decoder's streamed paths. The fine level of a dense res-1024 mesh (20M voxels
for the chest) therefore never needs its full feature table on the device.

### Parity (B4)

`tools/ref_shape_enc.py` is a vectorised, gather-based, pure-torch f32 implementation on CPU.
`trellis-test-shape-enc <shape_enc.gguf> <ref_dir> [gpu]`, threshold as the decoder test
(max-based rel < 8e-2):

| input | N @grid | latent N | latent coords | latent rel | mean abs Δ |
|---|---|---|---|---|---|
| synthetic sphere shell, random features | 161,454 @256 | 532 | exact | 1.7e-4 | 7.5e-4 |
| chest.glb dual grid (B3 voxeliser) | 939,846 @256 | 1,451 | exact | 2.5e-4 | 8.9e-4 |

Both rows were measured on the CPU backend. GPU-backend parity is in the end-to-end section.

**T8 round trip** (`trellis-test-shape-enc roundtrip`, Vulkan): voxelise → encode → decode
(`shape_decode`) → `dual_grid_to_mesh`, then a symmetric Chamfer distance against the
normalised input (200k surface samples each way).

| mesh @512 | voxels | latent | decoded voxels | out→in | in→out | symmetric mean (bar 1.5) |
|---|---|---|---|---|---|---|
| chest | 4,786,858 | 6,240 | 4,832,377 | 0.131 | 0.189 | **0.160 voxels** |
| knight | 792,931 | 1,280 | 803,471 | 0.191 | 0.270 | **0.231 voxels** |

Encoding the chest's 4.8M voxels takes 31 s on Thor/Vulkan.

## Coordinates: encoder vs voxel

S2C makes the encoder's latent cells exactly the parents `c // 16` of the fine voxels. The
refiner's quantisation `(c + 0.5)/R · (hr_res/16)` is also `c // 16` when there is no backoff.
The two coordinate sources therefore differ only where the decoder's upsampler prunes a
cell's children, or where the backoff lowers `hr_res`. Voxel mode (B2) is a close stand-in,
and the encoder mode follows the reference exactly.

## End-to-end (NVIDIA Thor iGPU, Vulkan, seed 42)

Views are the orthographic renders of spec 30 (T5) at az 0 and 180. Evaluation uses the same
renderer and metric as spec 30, including the four held-out oblique views (az 45/135/225/315,
el 20). Bboxes are measured in the glTF frame, with restore on.

| run | tokens | shape / tex flow | wall | IoU conditioned / held-out | colour error | bbox vs input |
|---|---|---|---|---|---|---|
| chest, voxel coords, res 512 | 6,240 @32 | 169 s / 127 s | 429 s | **0.976 / 0.988** | 20.5 / 20.4 | size within 0.3 % |
| chest, encoder coords, res 512 | 6,240 @32 (identical set) | 169 s / 127 s | 476 s | 0.976 / 0.988 | 20.5 / 20.4 | identical to voxel run |
| knight, encoder coords, res 1024 | 6,009 @64 | 207 s / 155 s | 562 s | **0.960 / 0.957** | 19.2 / 19.9 | size within 1.2 %, centre within 1.3e-3 |

For comparison, a from-scratch N=2 generation of the chest scores IoU 0.70 / 0.72. Refinement
keeps the input geometry, and its colour error is the lowest of any run.

At res 512 the encoder and voxel coordinate sources give the **same** token set, so the outputs
match exactly; the difference is ~45 s of encode + upsample.

**Token-set IoU, encoder vs voxel, at R = 1024** (`trellis-test-shape-enc coords`, Vulkan):

| mesh | voxel tokens | encoder tokens | IoU |
|---|---|---|---|
| chest | 32,579 | 32,577 | **0.9999** (the upsampler pruned 2 cells) |
| knight | 6,009 | 6,009 | **1.0000** |

Open question 3: the two sources are equivalent in practice. `auto` keeps the plan's rule
(encoder when `shape_enc.gguf` is present), which follows the reference exactly. Voxel mode is
a free stand-in that skips the ~0.7 GB weight load and 30–60 s of encode + upsample.

**Encoder parity on the GPU backend** (Vulkan, f16 weights; the CPU rows are above):

| input | latent coords | rel | mean abs Δ |
|---|---|---|---|
| sphere shell @256 | exact | 2.0e-3 | 5.7e-3 |
| chest @256 | exact | 2.8e-3 | 6.7e-3 |

**Memory.** Thor has unified memory, so there is no separate VRAM figure. These are the peak
process VmHWM and the peak drop in system MemAvailable, which also catches GPU allocations:

| run | peak VmHWM | peak MemAvailable drop |
|---|---|---|
| normal res 1024 + `--tex-res 512` (goblin) | 8.9 GiB | 7.9 GiB |
| refine, encoder coords, res 1024 (knight, 2 views) | 6.5 GiB | 5.7 GiB |

The plan's "1024 VRAM + 1 GB" criterion is not directly measurable on a UMA device. By these
host-side measures the encoder refine stays below a normal 1024 run, since SS and LR are
skipped. The encoder's weights (0.71 GB) are loaded only for the encode step.

**`/refine`** (T6): a multipart chest GLB plus two images at az 0/180 returns a 5.3 MB GLB. A
missing mesh or image returns 400. Staged files are removed.

## Limits

- Texture uses the mesh's own resolution; the res-512 "mixed" texture path needs an LR SLAT,
  which refine does not have. `--tex-res` is ignored with a notice. Dense 1024 refines may show
  the outer-skin speckle (spec 22).
- Interior faces and thin double walls voxelise as shells and are refined as such.
- `--dump-post` writes grid-frame vertices (no restore transform), like a normal run.
- `--retopo` writes its quad GLB through the external driver, from the POST dump, so it cannot
  restore the input transform. The combination is rejected unless
  `--refine-restore-transform off` is given.
- `--model pixal3d` is rejected (spec 30).
