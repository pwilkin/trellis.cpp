# 27 — Reference post-process analysis (to_glb / CuMesh / FlexGEMM grid_sample)

Line-by-line analysis of the reference GLB post-processing stack, written as the
authoritative source for (a) the CPU port of the narrow-band DC remesh and
(b) the divergence matrix. All citations are `file:line` into the read-only
clones listed in §1. Where a kernel detail is load-bearing, the code is quoted.

---

## 1. Provenance

| Repo | Path | Commit | Contents used here |
|---|---|---|---|
| microsoft/TRELLIS.2 | `/devel/alt/refs/TRELLIS.2` | `75fbf01` | `o-voxel/o_voxel/postprocess.py`, `app.py`, `example.py`, `trellis2/pipelines/trellis2_image_to_3d.py`, `trellis2/pipelines/trellis2_texturing.py`, `trellis2/representations/mesh/base.py` |
| CuMesh | `/devel/alt/refs/CuMesh` | `12289e1` | `cumesh/{cumesh,remeshing,bvh,xatlas}.py`, `src/remesh/{simple_dual_contour,svox2vert}.cu`, `src/{clean_up,simplify,atlas,geometry,connectivity}.cu`, `src/dtypes.cuh`, `src/hash/*`, `third_party/xatlas/*` |
| FlexGEMM | `/devel/alt/refs/FlexGEMM` | `6dd94a8` | `flex_gemm/ops/grid_sample/{grid_sample,grid_sample_torch}.py`, `flex_gemm/kernels/cuda/grid_sample/grid_sample.cu`, `flex_gemm/ops/utils.py` |
| HF space (ground truth for user-visible output) | `/devel/alt/refs/trellis2-space` | `ebf60b2` | `app.py`, vendored `trellis2/` tree |
| cubvh (BVH impl) | **not vendored** — git submodule `third_party/cubvh` → `https://github.com/JeffreyXiang/cubvh.git`, branch `trellis.2` (CuMesh/.gitmodules) | n/a | Python-side contract read from `CuMesh/cumesh/bvh.py` |

The vendored xatlas (`CuMesh/third_party/xatlas/xatlas.{cpp,h}`) was diffed
against stock `jpcy/xatlas` master: **byte-identical except the namespace rename
`xatlas` → `cumesh_xatlas`** (and a missing trailing newline). No algorithmic
modification whatsoever. See §6.

The two `trellis2/` trees (GitHub vs space) differ only in packaging details
(pipeline base-class loading, lazy-import table, an `h.coords` vs `v.coords`
variable rename in `fdg_vae.py` where both tensors share coords). Neither
`postprocess.py` nor any file analyzed below differs between them except
`app.py` (§2.3).

---

## 2. `to_glb` — verified op order and parameters

Function: `o-voxel/o_voxel/postprocess.py:14-331`.
Signature defaults (postprocess.py:23-31): `decimation_target=1000000`,
`texture_size=2048`, `remesh=False`, `remesh_band=1`, `remesh_project=0.9`,
`mesh_cluster_threshold_cone_half_angle_rad=radians(90)`,
`mesh_cluster_refine_iterations=0`, `mesh_cluster_global_iterations=1`,
`mesh_cluster_smooth_strength=1`.
**No production caller uses `remesh=False` or `remesh_project=0.9`** — see §2.3.

### 2.0 Input normalization (postprocess.py:60-93)

- `aabb` → float32 `(2,3)` tensor.
- If `voxel_size` given: `grid_size = ((aabb[1]-aabb[0]) / voxel_size).round().int()` (line 78).
- Else `grid_size` given: `voxel_size = (aabb[1]-aabb[0]) / grid_size` (line 87).
- With the universal caller values `aabb=[[-0.5]*3,[0.5]*3]` and `grid_size=res`
  (or `voxel_size=1/res`): `voxel_size = 1/res`, `grid_size = (res,res,res)`.

Note the mesh arriving at `to_glb` from `decode_latent` has **already** been
hole-filled once: `trellis2_image_to_3d.py:474` calls `m.fill_holes()` →
`Mesh.fill_holes` (`trellis2/representations/mesh/base.py:35-57`), which is the
same CuMesh `fill_holes(max_hole_perimeter=3e-2)` as below.

### 2.1 Common prologue (both branches)

1. `mesh.init(vertices, faces)` (postprocess.py:105-106).
2. `mesh.fill_holes(max_hole_perimeter=3e-2)` (line 110). Semantics: §3.1.
3. `vertices, faces = mesh.read()` (line 113) — **the "original mesh" used for
   all later BVH lookups is the hole-filled mesh, not the raw input.**
4. `bvh = cumesh.cuBVH(vertices, faces)` (line 122) — built once, reused for
   (a) the remesh UDF and (b) the texel snap in step 2.4.

### 2.2 The two branches (postprocess.py:133-187)

**Branch `remesh=False`** (lines 134-162) — *not used by any production caller*:
1. `mesh.simplify(decimation_target * 3)` (line 136)
2. `mesh.remove_duplicate_faces()` (141)
3. `mesh.repair_non_manifold_edges()` (142)
4. `mesh.remove_small_connected_components(1e-5)` (143)
5. `mesh.fill_holes(max_hole_perimeter=3e-2)` (144)
6. `mesh.simplify(decimation_target)` (149)
7. repeat cleanup: `remove_duplicate_faces()` (154), `repair_non_manifold_edges()` (155),
   `remove_small_connected_components(1e-5)` (156), `fill_holes(3e-2)` (157)
8. `mesh.unify_face_orientations()` (162)
→ material later gets `doubleSided=True` (line 303).

**Branch `remesh=True`** (lines 165-187) — *the production path*:
```python
center = aabb.mean(dim=0)                       # (0,0,0)
scale  = (aabb[1] - aabb[0]).max().item()       # 1.0
resolution = grid_size.max().item()             # res (512/896/1024/…)
mesh.init(*cumesh.remeshing.remesh_narrow_band_dc(
    vertices, faces,
    center = center,
    scale  = (resolution + 3 * remesh_band) / resolution * scale,   # line 174
    resolution = resolution,
    band = remesh_band,            # 1
    project_back = remesh_project, # 0 in ALL production callers
    bvh = bvh))
mesh.simplify(decimation_target)                # line 185 — the ONLY cleanup here
```
No duplicate-face / non-manifold / component / hole / orientation pass runs after
remesh. Material later gets `doubleSided=False` (line 303). Full remesh
algorithm: §4.

### 2.3 What the callers actually pass (the divergence-matrix row)

| Caller | pre-`to_glb` | decimation_target | texture_size | remesh | band | project | attr grid |
|---|---|---|---|---|---|---|---|
| `example.py:26-47` | `mesh.simplify(16777216)` (line 27, "nvdiffrast limit" — for the preview render, but it mutates the mesh handed to `to_glb`) | 1,000,000 | 4096 | True | 1 | 0 | `voxel_size=mesh.voxel_size` |
| GitHub `app.py:472-514` `extract_glb` | `decode_latent` only (line 492); **no** extra simplify | UI slider 100k–1M, **default 500,000** (app.py:531) | slider 1024–4096, default 2048 (532) | True (503) | 1 (504) | 0 (505) | `grid_size=res` (499) |
| **space `app.py:507-551` `extract_glb`** (ground truth) | `decode_latent` (528) **then `mesh.simplify(16777216)` (529)** | UI slider 100k–500k, **default 300,000** (space app.py:570) | slider 1024–4096, default 2048 (571) | True (540) | 1 (541) | 0 (542) | `grid_size=res` (536) |

Both apps: `glb.export(glb_path, extension_webp=True)` (GitHub 512, space 549).
Space vs GitHub `app.py` further differs in:
- Space `preprocess_image` (space app.py:326-356) reimplements
  `pipeline.preprocess_image` inline and gets its alpha matte from a **remote**
  `briaai/BRIA-RMBG-2.0` gradio client (`remove_background`, space app.py:317-323;
  client at 667; `pipeline.rembg_model = None` at 669). Algorithm is otherwise
  identical to `Trellis2ImageTo3DPipeline.preprocess_image`
  (trellis2_image_to_3d.py:127-162): max-1024 LANCZOS downscale, alpha>0.8·255
  bbox, square crop, premultiply. GitHub app calls `pipeline.preprocess_image`
  (app.py:321) with the local BiRefNet RMBG-2.0.
- `image_to_3d` bodies are identical (GitHub app.py:351-401 ↔ space 386-436):
  `pipeline.run(image, seed, preprocess_image=False, …sampler params…,
  pipeline_type={'512':'512','1024':'1024_cascade','1536':'1536_cascade'}[resolution],
  return_latent=True)`, then `mesh.simplify(16777216)` for the preview.
  All sampler-slider defaults identical (ss: 7.5/0.7/12/5.0; shape: 7.5/0.5/12/3.0;
  tex: 1.0/0.0/12/3.0; resolution radio default "1024").
- Note `mesh.simplify(16777216)` is a **face**-count cap (`CuMesh.simplify(target_num_faces)`,
  cumesh.py:320-333 — returns immediately if `num_faces <= target`), not a
  vertex cap.

### 2.4 UV unwrap + normals (postprocess.py:201-216)

```python
out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
    compute_charts_kwargs={
        "threshold_cone_half_angle_rad": radians(90.0),
        "refine_iterations": 0,
        "global_iterations": 1,
        "smooth_strength": 1},
    return_vmaps=True)
mesh.compute_vertex_normals()
out_normals = mesh.read_vertex_normals()[out_vmaps]
```
`uv_unwrap` internally first calls `remove_degenerate_faces()` (cumesh.py:436)
— this mutates the mesh, so the normals computed afterwards are on the
degenerate-free mesh. Full unwrap semantics: §6. Vertex-normal formula: §3.8.
`out_vmaps` maps unwrapped (chart-split, per-chart duplicated) vertices back to
CuMesh vertex indices; `out_vertices = new_vertices[vmaps]` (cumesh.py:471-472).

### 2.5 Texture bake (postprocess.py:229-266)

1. `uvs_rast = cat([out_uvs*2-1, zeros, ones], -1).unsqueeze(0)` (line 232) —
   UV in [0,1] mapped to clip space xy, z=0, w=1.
2. Rasterize in **chunks of 100,000 faces** into a `(1,T,T,4)` buffer
   (T=texture_size), lines 236-243: nvdiffrast stores `face_id_local+1` in
   channel 3 (0 = empty); the loop adds the chunk offset `i` and merges via
   `torch.where(mask_chunk, rast_chunk, rast)` — later chunks overwrite earlier
   ones on overlap.
3. `mask = rast[0,...,3] > 0` (246); `pos = dr.interpolate(out_vertices, rast, out_faces)[0][0]`
   (249) — per-texel barycentric 3D position **on the remeshed/decimated mesh**.
4. **Texel snap to original surface** (lines 254-256):
   ```python
   _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
   orig_tri_verts = vertices[faces[face_id.long()]]         # (N,3,3)
   valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
   ```
   i.e. each texel's 3D point is replaced by the **closest point on the
   hole-filled original mesh** (barycentric reconstruction — see §5).
5. **Sparse trilinear sample** (lines 259-266):
   ```python
   attrs[mask] = grid_sample_3d(
       attr_volume, cat([zeros, coords], -1),
       shape=[1, C, *grid_size], 
       grid=((valid_pos - aabb[0]) / voxel_size).reshape(1,-1,3),
       mode='trilinear')
   ```
   Convention resolved in §7.

### 2.6 Texture post-processing, material, axes (postprocess.py:278-331)

- Channel split by `attr_layout` and 8-bit quantize `clip(x*255, 0, 255).astype(uint8)`
  (281-284): base_color 0:3, metallic 3:4, roughness 4:5, alpha 5:6.
- Inpaint with `cv2.INPAINT_TELEA` on `mask_inv = ~mask`: radius 3 for
  base_color, radius 1 for metallic/roughness/alpha (288-292).
- `PBRMaterial` (296-304): `baseColorTexture = RGBA(base_color, alpha)`,
  `baseColorFactor=[255,255,255,255]`,
  `metallicRoughnessTexture = (R=0, G=roughness, B=metallic)`,
  `metallicFactor=1.0`, `roughnessFactor=1.0`, `alphaMode='OPAQUE'`,
  **`doubleSided = (not remesh)`** → `False` on the production path.
- Axis conversion (313-315): `v[:,1], v[:,2] = v[:,2], -v[:,1]` for vertices and
  normals (numpy tuple assignment ⇒ `(x,y,z) → (x, z, -y)`); `uv[:,1] = 1-uv[:,1]`.
- `trimesh.Trimesh(..., process=False, visual=TextureVisuals(uv, material))` (317-323).

---

## 3. Per-op semantics (CuMesh)

All Python methods in `cumesh/cumesh.py` are thin wrappers over `_C` (CUDA);
semantics below are from the CUDA sources.

Shared connectivity definitions (`src/connectivity.cu`):
- **Edge**: packed `uint64 = (min(v0,v1) << 32) | max(v0,v1)` per face edge
  (connectivity.cu:117-119), radix-sorted and uniqued with counts →
  `edges[E]`, `edge2face_cnt[E]`, CSR `edge2face` (123-278).
- **Boundary edge**: `edge2face_cnt == 1` (connectivity.cu:396, 410-412).
  **Boundary vertex**: endpoint of a boundary edge (383-405).
- **Manifold edge / `manifold_face_adj`**: edges with **exactly 2** adjacent
  faces yield a face pair `(f0,f1)` (connectivity.cu:659: `if (end-start != 2) return`).
- **Connected components**: parallel union-find (hook/compress) over
  `manifold_face_adj` (818-860) — i.e. faces joined only across *manifold* edges.
- **Boundary loop** (895-1125): connected components of boundary edges (linked
  via shared vertices); a component is a "loop" iff every boundary edge in it
  has ≥1 other same-component boundary edge at *both* endpoints
  (`is_bound_conn_comp_loop_kernel`, 913-955). Open chains are not loops.

### 3.1 `fill_holes(max_hole_perimeter=3e-2)` — clean_up.cu:450-712

1. Ensure boundary loops computed.
2. Per-loop perimeter = Σ boundary-edge lengths (segmented sum, 466-498).
3. Keep loops with `perimeter < max_hole_perimeter` **strictly less-than**
   (`LessThanOp`, 443-447 applied at 503-509). Units are world units — with the
   mesh in [-0.5,0.5]³, 3e-2 = 3% of the cube edge.
4. For each kept loop: create **one new vertex** = mean of the loop's edge
   midpoints (`compute_loop_boundary_midpoints` 405-420 + segmented sum + div,
   646-682) — equal to the loop vertex centroid on a simple loop.
5. Emit **one fan triangle per boundary edge**: `{e0, e1, new_vertex}` where
   `e0 = int(edge & 0xFFFFFFFF)` (= max index) and `e1 = int(edge >> 32)`
   (= min index) (`connect_new_vertices_kernel`, 423-440). ⚠ Winding comes from
   the *canonical* (min,max) edge packing, **not** from loop traversal order →
   fan triangles have arbitrary, mutually inconsistent orientation. In the
   non-remesh branch this is later fixed by `unify_face_orientations`; on the
   production path holes are filled only *before* remeshing, so it never matters
   for output orientation.
6. Vertices/faces appended in place; caches cleared.

### 3.2 `simplify(target_num_faces, options={})` — cumesh.py:320-359 + simplify.cu

Python driver (cumesh.py:338-356): `thresh=1e-8`, `lambda_edge_length=1e-2`,
`lambda_skinny=1e-3`; loop `simplify_step(...)` until `faces <= target`; if a
pass removes `< 1%` of faces, `thresh *= 10`. `simplify_step`
(simplify.cu:531-588) per pass:

1. **Per-vertex QEM** (`get_qem_kernel`, 35-62): sum of plane quadrics `p·pᵀ`
   over incident faces, `p=(n̂, -n̂·v0)` with **normalized** normal, **no area
   weighting** (dtypes.cuh:213-236; `evaluate` = `vᵀQv`, v=(x,y,z,1),
   dtypes.cuh:240-268).
2. **Per-edge collapse cost** (`get_edge_collapse_cost_kernel`, 158-225):
   - target position `v = w0·v0 + (1-w0)·v1` with **boundary rule**
     (184-189): both-or-neither boundary → midpoint (w0=0.5); exactly one
     boundary endpoint → keep it (w0=1 if v0 boundary else 0). Same rule reused
     at collapse time (379-384). There is **no optimal-position QEM solve** in
     the path actually taken (a `solve_optimal` exists in dtypes.cuh:54 but is
     unused here).
   - `cost = (Q0+Q1)(v) + λ_edge·|v1-v0|² + λ_skinny·mean_skinny·|v1-v0|²`
     (194-222), where per surviving incident triangle
     `skinny_term = 1 - clamp(4√3·area_new / Σ|e_new|², 0, 1)`
     (`process_incident_tri`, 87-140), averaged over the triangles incident to
     either endpoint (excluding the ≤2 collapsed ones).
   - **Flip guard**: if any incident triangle's new normal has
     `old_normal·new_normal < 0`, cost = ∞ (127-128, 207-216).
3. **Independent-set selection** (`propagate_cost_kernel` 269-296 +
   `collapse_edges_kernel` 341-374): each edge packs `(cost_float_bits<<32|edge_id)`
   and atomicMin's it into every face incident to either endpoint; an edge
   collapses only if `cost <= thresh` **and** it owns the min-pack on *all*
   faces around both endpoints. This makes each pass a parallel independent-set
   collapse (comparable to a coarse "multiple-choice" QEM), not a strict
   greedy priority queue — **ordering differs from classic QSlim**.
4. **Collapse** (377-407): `vertices[e0] = v_new`, drop `e1`, drop faces
   containing both endpoints, rewire `e1→e0`; compact vertices/faces (411-528).
   Note `e0 = int(e >> 32)` = min index, `e1` = max index (simplify.cu:178-179):
   the surviving vertex slot is the smaller index (unless the boundary rule
   moved the position to `v1`, in which case slot `e0` receives `v1`'s position).

Stopping: face count ≤ target (the pass may overshoot below target).

### 3.3 `remove_duplicate_faces()` — clean_up.cu:249-328

Duplicate = same vertex *set* (each face's indices bubble-sorted into a
temporary copy, 200-216; original order/winding preserved). Faces radix-sorted
by sorted triple with original index as payload (stable) → keep the first of
each equal-key group = **lowest original face index** (219-237); mask scattered
back to original order; `_remove_faces` + `remove_unreferenced_vertices`.
Orientation-agnostic: two mutually reversed faces count as duplicates.

### 3.4 `repair_non_manifold_edges()` — clean_up.cu:787-866

Vertex-split repair, always recomputes `manifold_face_adj` first (791):
1. For every *manifold* edge (exactly-2 face pair), pair up the two faces'
   corner slots holding the shared vertices (`construct_vertex_adj_pairs_kernel`
   715-768; corner slot id = `3*face + local_index`).
2. Parallel union-find over all `3F` corner slots with those pairs
   (hook/compress loop 812-836).
3. Each union-find class becomes one output vertex (position read through any
   member slot, `index_vertice_kernel` 771-784); faces rewritten to class ids
   (857).

Effect: vertices are duplicated so that faces remain connected **only through
manifold edges**; corner fans meeting at a non-manifold edge (≥3 faces) or a
bowtie vertex are separated into distinct vertices *with identical coordinates*.
Face count unchanged; no face is deleted.

### 3.5 `remove_small_connected_components(min_area)` — clean_up.cu:945-1072

Components = union-find over `manifold_face_adj` (§3 header; parts touching
only via non-manifold edges are separate components). Per-component area =
Σ face areas (`compute_face_areas`, geometry.cu:10-37: `0.5·|(v1-v0)×(v2-v0)|`).
Keep component iff `area >= min_area` (`GreaterThanOrEqualToOp`, 938-942).
`to_glb` uses `min_area=1e-5` (world units²; the whole object spans ≤ 1 unit).

### 3.6 `unify_face_orientations()` — clean_up.cu:1191-1251

1. Per manifold edge, `flipped` flag (`get_flip_flags_kernel`, 1131-1172): find
   the two shared vertices' local indices in each face; `direction = (i1-i0+3)%3`
   per face; `flipped = (direction1 == direction2)` — equal cyclic direction
   means inconsistent winding.
2. Union-find with **parity** (1-bit flip carried in the LSB of the parent
   pointer; `hook_edges_with_orientation_kernel` 1075-1111,
   `compress_components_with_orientation_kernel` 1114-1128).
3. Faces whose accumulated parity is 1 get `(x,y,z) → (x,z,y)` (1175-1188).

Only *relative* consistency per component; there is **no global inside/outside
(outward-normal) decision** — the component root's original winding wins.
Only used on the (non-production) `remesh=False` branch, which also sets
`doubleSided=True`.

### 3.7 `remove_degenerate_faces(abs_thresh=1e-24, rel_thresh=1e-12)` — clean_up.cu:331-384

Remove face if (a) any repeated vertex index, or (b)
`area < min(rel_thresh · max_edge_len², abs_thresh)` (line 358 — note `fminf`:
*both* thresholds must exceed the area). Called with defaults from
`uv_unwrap` (cumesh.py:436).

### 3.8 `compute_vertex_normals()` — geometry.cu:74-130

Per vertex: sum of **unnormalized** cross products `(v1-v0)×(v2-v0)` over
incident faces (⇒ area-weighted), then normalize; if the normalized result is
NaN (zero sum), fall back to the **first** incident face's (unnormalized) cross
product (95-109). Face order in `vert2face` comes from the CSR construction
(atomic fill — nondeterministic order on GPU; a CPU port should note the NaN
fallback is order-sensitive but occurs only on degenerate fans).

---

## 4. `remesh_narrow_band_dc` — CPU-port blueprint

Source: `CuMesh/cumesh/remeshing.py:24-252` +
`CuMesh/src/remesh/svox2vert.cu` + `CuMesh/src/remesh/simple_dual_contour.cu`.
Everything is float32; integer voxel coords are int32.

### 4.0 Inputs and derived constants

From `to_glb` (postprocess.py:166-180) with `aabb=±0.5`, original scale 1:
- `center = (0,0,0)`; `scale = (res + 3·band)/res` (domain inflated by 3 band
  voxels so the offset surface never touches the domain boundary);
- `resolution = res = grid_size.max()`; `band = 1`; `project_back = 0`;
- `bvh` = BVH over the **hole-filled original mesh** (§2.1).
- `eps = band * scale / resolution` (remeshing.py:100) — in inflated-domain
  units; ≈ `band·(res+3·band)/res²` in world units (≈ 1.003 voxels for
  res=1024, band=1).

**Key design decision (the UDF pseudo-sign):** the implicit function is
`f(x) = UDF(x) − eps`, where UDF is the *unsigned* distance to the original
mesh (remeshing.py:98-99 comment; 123-124, 163-164). The extracted isosurface
is therefore the **offset surface at distance `eps` (≈ 1 voxel) around the
original mesh** — a closed inflated shell; "inside" (f<0) is the band
|UDF|<eps around the surface. There is no normal-based pseudo-sign and no
half-voxel threshold; sign is simply `UDF < eps`. With `project_back=0`
(all production callers), **the output geometry stays on that offset surface —
it is NOT projected back onto the original mesh**.

### 4.1 Sparse narrow-band grid construction (octree-style, remeshing.py:102-141)

```python
base_resolution = resolution
while base_resolution > 32:
    assert base_resolution % 2 == 0
    base_resolution //= 2                     # 1024→512→…→32; 896→448→224→112→56→28
coords = dense base_resolution³ grid (meshgrid, int32)
while True:
    cell_size = scale / base_resolution
    pts = ((coords.float() + 0.5) / base_resolution - 0.5) * scale + center   # voxel CENTERS
    distances = bvh.unsigned_distance(pts)[0] - eps
    subdiv_mask = |distances| < 0.87 * cell_size
    coords = coords[subdiv_mask]
    if base_resolution >= resolution: break
    base_resolution *= 2; coords *= 2
    coords = (coords[:,None,:] + OFFSETS[None]).reshape(-1,3)  # 8 children each
```
(remeshing.py:103-141; `OFFSETS` = the 8 binary corner offsets, 85-88.)

- Start level: halve `resolution` until ≤ 32 (must stay even at every halving;
  for res=1024 the levels are 32,64,128,256,512,1024 — 6 UDF evaluation rounds;
  for res=896: 28,56,112,224,448,896).
- Refinement predicate: keep a voxel iff `|UDF(center) − eps| < 0.87·cell` at
  the *current* level. 0.87 ≳ √3/2: since UDF (hence f) is 1-Lipschitz and any
  point of a voxel (closed cube) is within √3/2·cell of its center, **every
  voxel whose closed cube intersects the isosurface is provably kept, at every
  level** (an ancestor cube containing a zero-crossing point has its center
  within √3/2·ancestor_cell ≤ 0.87·ancestor_cell of it). The band is therefore
  a superset of all cells touching the offset surface.
- Final iteration prunes at full resolution *before* breaking, so the surviving
  voxel set is exactly `{v : |UDF(center_v) − eps| < 0.87·scale/res}`.

### 4.2 Active-vertex extraction and dedup (remeshing.py:146-158, svox2vert.cu)

Voxels are inserted into a linear-probing hashmap
(key = flat index `x·H·D + y·D + z` with `W=H=D=resolution`, value = row index;
capacity `2·Nvox`; uint32 keys iff `res³ < 2³²` else uint64 — remeshing.py:8-21,
149-152; Murmur3 finalizer hash, hash.cuh:1-20).

`get_sparse_voxel_grid_active_vertices` (svox2vert.cu:135-231) emits, per active
voxel `(x,y,z)`, corner `(x+i,y+j,z+k)` (i,j,k∈{0,1}) iff `(i,j,k)==(0,0,0)`
**or** the voxel whose coordinate equals that corner is *not* active (or out of
bounds) (svox2vert.cu:37-58, 92-119). This guarantees **coverage** (every corner
of every active voxel appears: if voxel `(corner)` is active it emits it as its
own (0,0,0) corner) but only *partial* dedup — a corner whose same-coordinate
voxel is inactive is emitted once per active voxel that owns it in the
−direction (up to 7 duplicates). Duplicates are harmless: all copies get the
same UDF value, and the vertex hashmap insert (below) keeps one arbitrary
winner (`linear_probing_insert` overwrites on equal key, hash.cuh:24-41).

### 4.3 UDF at grid vertices (remeshing.py:160-165)

```python
pts_vert = (grid_verts.float() / resolution - 0.5) * scale + center   # voxel CORNERS
distances_vert = bvh.unsigned_distance(pts_vert)[0] - eps
```
Note the two sampling formulas: voxel **centers** use `(c+0.5)/res`, grid
**vertices** use `v/res` — both mapped through the *inflated* domain
`[center - scale/2, center + scale/2]`.

Grid vertices go into a second hashmap with per-dim size `resolution+1`
(capacity `2·Nvert`; remeshing.py:177-179), mapping vertex coord → row into
`distances_vert`.

### 4.4 Dual-contouring kernel (`simple_dual_contour`, simple_dual_contour.cu:28-156)

One thread per active voxel. For each of the voxel's 12 edges (4 per axis),
fetch corner values via the vertex hashmap (`get_vertex_val`, 12-25 — no
missing-key handling; correctness relies on §4.2 coverage) and test the sign
crossing:

```c
// axis X, u,v ∈ {0,1}:
float val1 = f(vx,   vy+u, vz+v);
float val2 = f(vx+1, vy+u, vz+v);
if ((val1 < 0 && val2 >= 0) || (val1 >= 0 && val2 < 0)) {
    float t = -val1 / (val2 - val1);          // linear zero crossing
    intersection_sum += (vx + t, vy+u, vz+v); // grid-vertex coordinate units
    intersection_count++;
}
```
(lines 54-83; Y and Z analogous, 85-143.) Sign convention: `f < 0` is inside
(within the band), `f >= 0` outside — a vertex exactly on the isosurface counts
as outside.

**Dual vertex placement** (146-155): the **plain average of the edge-crossing
points** — *no QEF, no normals, no clamping to the cell*:
```c
out_vertices[voxel] = intersection_sum / intersection_count;   // if any crossing
else                = (vx+0.5, vy+0.5, vz+0.5);                // fallback: cell center
```

**Edge ownership for topology** (71-81, 101-111, 131-141): each voxel reports
crossing direction only for its 3 "far" edges — the `u==1,v==1` edge per axis,
i.e. the X-edge `(vx,vy+1,vz+1)→(vx+1,vy+1,vz+1)`, the Y-edge
`(vx+1,vy,vz+1)→(vx+1,vy+1,vz+1)`, and the Z-edge
`(vx+1,vy+1,vz)→(vx+1,vy+1,vz+1)` (all incident to the voxel's max corner):
`out_intersected[voxel][axis] = +1` if `val1<0 && val2>=0` (inside→outside along
+axis), `-1` if `val1>=0 && val2<0`, else `0`.

### 4.5 Quad assembly and triangulation (remeshing.py:189-233)

Per owned crossing edge, the 4 voxels sharing it are `voxel + offset` with
(remeshing.py:71-76):
```python
edge_neighbor_voxel_offset = [
  [[0,0,0],[0,0,1],[0,1,1],[0,1,0]],   # x-axis edge (cyclic order around +x)
  [[0,0,0],[1,0,0],[1,0,1],[0,0,1]],   # y-axis edge
  [[0,0,0],[0,1,0],[1,1,0],[1,0,0]]]   # z-axis edge
```
Rows with `intersected != 0` are gathered (191-193); the 4 voxel coords are
looked up in the *voxel* hashmap (195-199) and the quad kept only if all 4 are
found (200-201). ⚠ Two implementation notes:
1. As proven in §4.1, away from the domain boundary all 4 voxels of a genuinely
   crossing edge are always active, so this filter should never fire; the
   inflated domain (`+3·band` voxels) keeps the band off the boundary.
2. The check **as written is inert**: the lookup result is cast `.int()`
   (int32; the miss sentinel `0xFFFFFFFF` wraps to −1) and then compared with
   the Python literal `0xffffffff` (line 199-200) — `-1 != 4294967295` promotes
   to int64 and is always True. A CPU port should implement the intended check
   (drop quads referencing a missing voxel); given note 1 it is defensive only.

Unreferenced dual vertices are compacted (206-210), then
`mesh_vertices = (dual_verts / resolution - 0.5) * scale + center` (213) —
same corner mapping as §4.3 (dual verts live in grid-vertex coordinate units).

**Winding + diagonal choice** (79-83, 215-233). Split tables:
```python
quad_split_1_n = [0,1,2, 0,2,3];  quad_split_1_p = [0,2,1, 0,3,2]   # diagonal q0–q2
quad_split_2_n = [0,1,3, 3,1,2];  quad_split_2_p = [0,3,1, 3,2,1]   # diagonal q1–q3
```
`_p` (reversed winding) is selected where `intersected_dir == +1`, `_n` where
`-1` (215-219, 224-228) — this orients quads consistently with the crossing
direction (outward = away from the band).

The diagonal is then chosen by comparing `align0` vs `align1` (220-233), each
computed as `|n(t[0],t[1],t[2]) · n'(t[1],t[2],t[3])|` over the **first four
entries of the flattened 6-index split list** — ⚠ **this is a latent upstream
bug that a faithful port must reproduce**: for split 1 the indices
`(t[0..3]) = (q0,q1,q2,q0)` make `n` and `n'` normals of the *same* triangle
(`align0 = |n(q0,q1,q2)|²`), and for split 2 `(q0,q1,q3,q3)` makes
`n' = (q3−q1)×(q3−q1) = 0` (`align1 = 0` exactly). Consequently
`align0 > align1` **always selects split 1 (diagonal q0–q2), except when
triangle (q0,q1,q2) has an exactly-zero cross product, in which case split 2 is
used**. The "pick the more planar split" heuristic is effectively dead code.
Output `mesh_triangles = chosen.reshape(-1,3)` interleaves each quad's two
triangles.

### 4.6 `band` and `project_back` numerically

- `band` (=1) enters in exactly two places: `eps = band·scale/resolution`
  (the offset distance ≈ band voxels) and the domain inflation
  `scale ·= (res + 3·band)/res` from the caller (postprocess.py:174).
- `project_back` (remeshing.py:238-250): if > 0,
  ```python
  _, face_id, uvw = bvh.unsigned_distance(mesh_vertices, return_uvw=True)
  projected = (vertices[faces[face_id]] * uvw[...,None]).sum(1)   # closest point on original mesh
  mesh_vertices -= project_back * (mesh_vertices - projected)
  ```
  — a plain lerp by factor `project_back` toward the closest original-surface
  point (1.0 = full snap). **All production callers pass 0**, so the exported
  geometry is the ~1-voxel offset shell (the signature default 0.9 is unused).

### 4.7 Output topology guarantees

- Exactly one dual vertex per active voxel (referenced ones kept), one quad
  (two triangles) per sign-crossing grid edge — standard dual contouring on a
  sparse grid.
- Because `f = UDF − eps` is continuous, negative only inside a bounded band,
  and every band-touching cell is present (§4.1), the crossing-edge set forms
  the complete boundary of the voxelized `f<0` region ⇒ the quad surface is
  **closed (watertight) and consistently oriented** (outward from the band),
  provided the band does not touch the domain boundary — which the `+3·band`
  inflation ensures.
- Standard DC caveats apply: quad meshes from DC can contain **non-manifold
  edges/vertices** where the offset surface passes through a cell in multiple
  sheets (thin gaps < 2·eps between surface parts merge, since the offset
  shell of nearby geometry fuses — this is *intentional smoothing* of the
  band construction), and the *simple* variant (mean of crossings, no QEF)
  rounds sharp features to voxel scale. Self-intersections cannot occur at
  `project_back=0` (vertices stay in their cells' neighborhoods); at
  `project_back>0` they can.
- The subsequent `mesh.simplify(decimation_target)` (§3.2) has an infinite-cost
  flip guard but no manifold guard; its input here is the DC mesh unchanged.

### 4.8 CPU-port notes

- Replace both hashmaps with any exact int3→index map; capacity/probing details
  are irrelevant to results. The only numeric requirements are float32
  arithmetic in: point mapping formulas (§4.1/§4.3), `t = -val1/(val2-val1)`,
  the crossing average, and the two comparisons (`< 0.87·cell`, `val < 0`).
- UDF must be the exact closest-point distance to the hole-filled mesh
  (triangle-exact, not point-sampled) — see §5.
- Parallelism is embarrassing per level / per voxel; determinism on CPU is
  straightforward (the only GPU nondeterminism — duplicate-vertex hash winner —
  doesn't affect values).
- Vectorizable cost centers: BVH UDF queries (≈ Σ levels |coords| + |grid_verts|
  queries; ~ a few million at res 1024) and the per-voxel 12-edge loop.

---

## 5. cuBVH contract (`cumesh/bvh.py`)

Implementation lives in the (not-vendored) `cubvh` submodule
(JeffreyXiang/cubvh, branch `trellis.2`; API identical to ashawkey/cubvh).
Python contract:

- `cuBVH(vertices, triangles)` (bvh.py:12-23): builds from CPU numpy copies;
  asserts `triangles.shape[0] > 8` (line 20).
- `unsigned_distance(positions, return_uvw=False)` (bvh.py:54-82) returns
  `(distances, face_id, uvw)`:
  - `distances`: float32 `[N]`, **unsigned** (≥0) Euclidean distance to the
    closest point on the mesh surface (triangle-exact);
  - `face_id`: **int64** `[N]`, index of the closest triangle into the `faces`
    array the BVH was built from;
  - `uvw`: float32 `[N,3]` **barycentric coordinates of the closest point on
    that triangle**, ordered to match the face's vertex order and summing to 1;
    only computed when `return_uvw=True`.
- Closest-point reconstruction used by both consumers is
  `closest = Σ_k vertices[faces[face_id]][k] * uvw[k]`
  (postprocess.py:254-256; remeshing.py:242-249) — the CPU port needs exactly
  this contract: `{distance, face_id, uvw}` per query, nothing else.
  (`ray_trace` and `signed_distance` exist, bvh.py:25-113, but are unused by
  the post-process path.)

Uses on the GLB path: (1) UDF at voxel centers/corners during remesh (distances
only), (2) texel snap (face_id + uvw; distance discarded), (3) `project_back`
lerp (unused at 0).

---

## 6. UV unwrap: CuMesh clustering + stock xatlas

### 6.1 Structure (`uv_unwrap`, cumesh.py:408-480)

1. `remove_degenerate_faces()` (436; §3.7).
2. **GPU coarse clustering** `compute_charts(...)` (439; §6.2) segments faces
   into `num_charts` clusters and builds per-cluster vertex/face lists
   (`read_atlas_charts`, atlas.cu:1213-1252 — per-cluster vertices are already
   split/duplicated, `construct_chart_mesh` atlas.cu:954-1068).
3. Each cluster is added to **stock xatlas as a separate mesh**
   (`Atlas.add_mesh`, cumesh.py:453-458; no normals, no input UVs) —
   *not* as a precomputed chart; xatlas re-runs its own chart segmentation
   *within* each cluster mesh.
4. `xatlas.compute_charts(**defaults)` then `xatlas.pack_charts(**defaults)`
   (459-460) — `to_glb` passes **no** xatlas kwargs, so the Python defaults in
   `cumesh/xatlas.py` apply (§6.3).
5. Results gathered per cluster (`get_mesh`, 465-474): xatlas `xref` maps new
   vertices to cluster-local indices → composed with the cluster vmap to global
   CuMesh vertex ids; faces offset-concatenated; UVs concatenated. Final
   `vertices = new_vertices[vmaps]` (chart-split vertices duplicate positions).

The binding (`third_party/xatlas/binding.cpp:105-144`) returns UVs **divided by
the final atlas width/height** (126-137), so `out_uvs ∈ [0,1]` regardless of the
atlas's internal pixel resolution.

### 6.2 GPU clustering `compute_charts` — atlas.cu:1071-1210

Parameters from `to_glb`: `threshold_cone_half_angle_rad = π/2`,
`refine_iterations = 0`, `global_iterations = 1`, `smooth_strength = 1`, plus
Python-side defaults `area_penalty_weight = 0.1`,
`perimeter_area_ratio_weight = 0.0001` (cumesh.py:361-391 — note the *method*
defaults `refine_iterations=100, global_iterations=3` are overridden by
`to_glb`).

Algorithm (one global iteration, refine disabled):
- Init: every face is its own chart (1092-1099).
- Repeat until no merge happens (1105-1179):
  - Chart adjacency graph over manifold face pairs, with shared-boundary
    lengths aggregated per chart pair (`init_chart_adj_kernel` 49-107 +
    sort/reduce 306-378).
  - Per chart: **normal cone** (axis, half-angle). Axis = normalized sum of
    (unit) face normals; half-angle = max angular deviation of member faces
    from the axis (`compute_chart_normal_cones`, 486-678). Chart area =
    Σ face areas; perimeter = Σ boundary edge lengths.
  - Merge cost per adjacent pair (155-195):
    `cost = merged_cone_half_angle + 0.1·(areaA+areaB)
          + 0.0001·(perimA+perimB−2·shared)²/(areaA+areaB)`,
    where the merged cone is the exact 1-D interval hull of the two cones
    placed `axis_angle` apart (176-183).
  - Parallel independent merges: per chart, min-cost incident pair
    (`propagate_cost_kernel` 198-222, ties → lower edge id); an edge collapses
    iff it is the min for *both* charts **and** `cost ≤ π/2`
    (`collapse_edges_kernel` 225-277, which also updates the winner's cone by
    the same interval-hull rule, 253-273).
  - Compact chart ids and loop.
- `refine_iterations = 0` ⇒ the boundary-smoothing reassignment pass
  (`refine_charts_kernel`, 681-803, weight `smooth_strength`) **never runs**
  on the production path; `smooth_strength` is thus inert.
- `reassign_chart_ids` (843-887): charts are re-split into face-connected
  components (union-find over manifold adjacency restricted to same chart id).

Net effect: clusters are normal-cone-limited (≤ 90° half-angle), size- and
compactness-penalized, connected groups of faces — deliberately coarser than
final charts; xatlas then does the real charting inside each.

### 6.3 xatlas options actually in effect (all stock defaults)

From `cumesh/xatlas.py` defaults (55-96, 104-139), identical to stock
`xatlas::ChartOptions` / `PackOptions` defaults (third_party/xatlas/xatlas.h:173-232):

| ChartOptions | value | | PackOptions | value |
|---|---|---|---|---|
| maxChartArea | 0 (∞) | | maxChartSize | 0 |
| maxBoundaryLength | 0 (∞) | | **padding** | **0 px** |
| normalDeviationWeight | 2.0 | | texelsPerUnit | 0 (auto) |
| roundnessWeight | 0.01 | | **resolution** | **0 → ≈1024² estimate** |
| straightnessWeight | 6.0 | | bilinear | true |
| normalSeamWeight | 4.0 | | blockAlign | false |
| textureSeamWeight | 0.5 | | bruteForce | false |
| maxCost | 2.0 | | rotateCharts | true |
| **maxIterations** | **1** | | rotateChartsToAxis | true |
| useInputMeshUvs | false | | | |
| fixWinding | false | | | |

Implications for the port:
- The reference relies on **stock xatlas behavior** for chart growth (the usual
  normal-deviation/roundness/straightness metric), LSCM parameterization, and
  the random-placement packer (bruteForce=false ⇒ xatlas's internal
  rand — atlas-level determinism caveat), all at `maxIterations=1`.
- Charting is **not** CUDA-parallelized; the CUDA part is only the coarse
  pre-clustering (§6.2), which (a) bounds per-xatlas-mesh size and (b) forces
  chart boundaries at cluster boundaries (each cluster is an isolated mesh, so
  xatlas can never merge across them; cluster-boundary vertices are duplicated).
- Packing is a single shared atlas across all cluster meshes; final UVs are
  atlas-normalized ([0,1]) and consumed at `texture_size` (2048) independent of
  the packer's internal ≈1024 resolution estimate. `padding=0` + `bilinear=true`
  means charts are separated by only the bilinear guard band in *packer* texels
  (≈1024 units) — at a 2048 bake this halves the effective guard width; the
  Telea inpaint of unmasked texels (§2.6) is what prevents visible bleed.

### 6.4 What was changed vs stock jpcy/xatlas

Nothing, verbatim (§1 diff): the `xatlas.cpp`/`xatlas.h` sources are unmodified
apart from the `cumesh_xatlas` namespace rename. The binding
(`binding.cpp:151-186`) exposes exactly the stock option fields listed above
(plus createImage omitted). A CPU port can therefore link stock xatlas
directly and reproduce reference charting/packing behavior, provided it feeds
the same cluster meshes in the same order and uses the §6.3 defaults.

---

## 7. `grid_sample_3d` — coordinate convention, resolved

Op: `flex_gemm/ops/grid_sample/grid_sample.py:192-212` → trilinear path
`_trilinear_fwd` (77-123) → CUDA kernel
`hashmap_lookup_grid_sample_3d_trilinear_neighbor_map_weight_kernel`
(`flex_gemm/kernels/cuda/grid_sample/grid_sample.cu:149-201`).

Exact kernel math (grid_sample.cu:170-193):
```c
base_x = floor(q.x - 0.5f);  // etc.
for i in 0..7:
    x = base_x + (i&1); y = base_y + ((i>>1)&1); z = base_z + ((i>>2)&1);
    if (in bounds && hashmap hit):
        n[i] = row index;
        w[i] = (1-|q.x-x-0.5|)·(1-|q.y-y-0.5|)·(1-|q.z-z-0.5|);
        w_sum += w[i];
for i in 0..7: w[i] /= max(w_sum, 1e-12f);
```
then `out = Σ w[i]·feats[n[i]]` (triton `indice_weighed_sum`, grid_sample.py:113-117).
The torch fallback (`grid_sample_torch.py:60-123`) implements the identical
convention (offsets ±0.5 then truncation; weight `1-|neigh+0.5-q|`; division by
clamped valid-weight sum, line 114-121).

**Convention:** the voxel with integer coordinate `c` is centered at continuous
grid coordinate `c + 0.5`. Query grids passed by the two callers:
- `to_glb`: `grid = (pos − aabb[0]) / voxel_size = (pos + 0.5)·res` (postprocess.py:264)
- `postprocess_mesh`: `grid = (pos + 0.5)·resolution` (trellis2_texturing.py:330)
- `MeshWithVoxel.query_attrs`: `(xyz − origin)/voxel_size` (mesh/base.py:222-231)

⇒ all three are the **same** mapping `g = (p + 0.5)·res` for `p ∈ [-0.5,0.5]³`,
`g ∈ [0,res]`. A voxel is sampled exactly at its center when
`p = (c + 0.5)/res − 0.5`. In `align_corners` language this is
`align_corners=False` half-texel-center sampling, expressed in **unnormalized**
grid units.

**Resolution of the 12-ovoxel_mesh.md open question:** our VoxSampler's
`gf = (p+0.5)·res − 0.5; c0 = floor(gf); w = 1−|gf−c|` is algebraically
identical: `gf = g − 0.5`, so `floor(gf) = floor(g−0.5) = base` and
`1−|gf−c| = 1−|g−c−0.5|` — **there is no 0.5-voxel offset; the conventions
match exactly.**

**The one real behavioral subtlety:** missing/out-of-range corner voxels do
**not** contribute zero — the surviving weights are **renormalized by the sum
of valid weights** (`w[i] /= max(w_sum,1e-12)`, grid_sample.cu:191-193; same in
torch fallback). Only when *all 8* corners are absent does the output collapse
to 0 (w_sum clamped to 1e-12). A zero-fill implementation darkens texels near
the sparse volume's surface (which is *most* bake texels, since the mesh runs
along the voxel shell) — this must be renormalization, not zero-padding.
Bounds: corners outside `[0,W)×[0,H)×[0,D)` are treated as absent (179).
`mode='nearest'` (unused on the bake path) is **truncation** `int(q)` of the
grid coordinate, not rounding (grid_sample.cu:42-44).

---

## 8. SPEC-DELTA vs existing transcriptions

### 8.1 `docs/spec/01-pipeline.md`

Checked against `trellis2_image_to_3d.py` (`run` 488-595, `preprocess_image`
127-162, `get_cond` 164-186, `decode_latent` 455-486): the transcription is
accurate (pipeline_type table, ss_res map, cascade quantization loop, tex noise
width `in_channels−32`, `decode_tex_slat(tex_slat, subs)`, denorm order,
premultiplied preprocess). Deltas:
- 01:112 "`MeshWithVoxel.query_attrs(xyz)` … (used by GLB texture bake)" —
  the GLB bake does **not** call `query_attrs`; `to_glb` invokes
  `grid_sample_3d` directly (postprocess.py:260-266). `query_attrs`
  (mesh/base.py:222-231) is the same math but is used by the preview renderer
  path only. Cosmetic mislabel.
- 01:13-16 describes `example.py` only. The **space app** (ground truth) adds
  `mesh.simplify(16777216)` *inside* `extract_glb` before `to_glb`
  (space app.py:529) and uses decimation_target default 300,000 /
  texture_size 2048 — not 1M/4096. See §2.3 matrix.

### 8.2 `docs/spec/11-tex_dec.md`

- 11:49 / 11:60 "guide_subs=None at inference" and the CRITICAL open question
  §11:175 — **incorrect for the image-to-3D pipeline** (the trellis.cpp
  target): `decode_tex_slat` there passes `guide_subs=subs`
  (trellis2_image_to_3d.py:450: `self.models['tex_slat_decoder'](slat,
  guide_subs=subs)`), with `subs` returned by the shape decoder
  (`decode_shape_slat(..., return_subs=True)`, 385). The upsampling target is
  therefore externally supplied at inference; the "decoder cannot invent
  sub-voxels" concern is resolved by `guide_subs`, not by spatial caches or
  `pred_subdiv`. The `(slat)`-only call described in 11 is the *texturing*
  pipeline (trellis2_texturing.py:282).
- 11:99 bake description matches `postprocess_mesh` (trellis2_texturing.py:287-371).
  Its open question 4 (grid alignment, "off-by-scale risk") is resolved in §7:
  `(pos+0.5)*resolution` samples voxel centers at `c+0.5`; conventions agree.
- 11:171 "UV bake texture is 2048×2048×6" — correct for both pipelines'
  defaults; space UI default confirms 2048.

### 8.3 `docs/spec/12-ovoxel_mesh.md`

- **12:92, 12:115, 12:129 "missing neighbor contributes 0" — WRONG.** Missing
  corners are excluded and the remaining trilinear weights are renormalized
  (grid_sample.cu:185-193; §7). This is the single most consequential
  inaccuracy for texture parity.
- 12:133 open question 1 — resolved (§7): voxel centers at integer+0.5 grid
  units; the two caller formulas are equivalent; our VoxSampler convention is
  exactly right; only the renormalization differs.
- 12:53 / 12:100: `uv_unwrap` description omits (a) the internal
  `remove_degenerate_faces()` (cumesh.py:436) which mutates the mesh before
  normal computation, and (b) the actual two-stage structure. The
  characterization "CUDA in cumesh; CPU fallback: use the xatlas library …
  UVs will differ" (12:100, 12:135) understates fidelity: **stock, unmodified
  xatlas already is the reference parameterizer/packer** (§6.4); a CPU port
  linking stock xatlas and reproducing the §6.2 clustering reproduces reference
  behavior, not an approximation.
- 12:51-52 op order is correct; add that on the production branch the *only*
  post-remesh op is `simplify(decimation_target)` — no dedupe/repair/
  components/holes/orientation pass (postprocess.py:165-187), and that
  `decode_latent` has already hole-filled the mesh before `to_glb`
  (trellis2_image_to_3d.py:474).
- 12:99 fallback advice "skip remesh … set doubleSided=True (matches the
  not-remesh branch)" — valid as an MVP fallback, but note the reference
  outputs users compare against are **always** remeshed (space passes
  remesh=True; §2.3), with vertices on the ~1-voxel offset shell
  (project_back=0, §4.6). Geometry cannot match without the remesh.
- 12:38 says callers pass `remesh_project=0` — correct; worth stating
  explicitly that the signature default 0.9 is dead (§4.6).
- to_glb's own docstring (postprocess.py:48) "decimation_target: target number
  of vertices" is wrong upstream — it is a **face** target
  (`CuMesh.simplify(target_num_faces)`, cumesh.py:320-333; both app UIs label
  it "target face count").

### 8.4 `docs/spec/25-mesh_glb.md`

- 25:24-62 rotation analysis `(x,y,z)→(x,z,-y)` — verified correct, including
  the numpy tuple-assignment reasoning (postprocess.py:313-315).
- 25:59-62 "(`doubleSided=True` is set on the Python material anyway, so
  winding is not even visually critical)" — **wrong for the production path**:
  `doubleSided = True if not remesh else False` (postprocess.py:303) and all
  callers remesh ⇒ `doubleSided=False`, winding is visually load-bearing. The
  DC winding rule is §4.5.
- 25:210 "`mesh.simplify(16777216)` (nvdiffrast vertex-count cap) — only needed
  if downstream rasterization; not for export" — it is a *face*-count cap, and
  in the **space** app it also runs inside `extract_glb` before `to_glb`
  (space app.py:529), i.e. it *is* on the ground-truth export path (a no-op
  unless the decoded mesh exceeds 16.7M faces).
- 25:66-81 aabb reasoning fine for the geometry MVP, but "NONE feed into vertex
  positions for the geometry path" holds only with remesh skipped; on the
  reference path `aabb`/`grid_size` set the remesh domain (§4.0) and thereby
  every output vertex.

---

## 9. Resolved open questions

1. **grid_sample alignment (12-ovoxel Q1, 11-tex_dec Q4):** voxel `c` centered
   at grid coordinate `c+0.5`; callers' `(pos+0.5)·res` ≡ `(pos−aabb₀)/voxel`;
   our VoxSampler `gf=(p+0.5)·res−0.5` + floor + trilinear is the identical
   convention. Remaining fix: renormalize weights over present voxels instead
   of zero-filling (§7).
2. **Remesh algorithm (12-ovoxel Q2, first half):** fully specified in §4 —
   octree-refined narrow band on `|UDF−eps|<0.87·cell`, pseudo-sign
   `UDF−eps`, simple DC (mean of edge crossings, cell-center fallback),
   far-edge ownership, fixed quad tables, effectively-always split-1
   triangulation, `project_back=0` ⇒ output is the ≈1-voxel offset shell,
   watertight away from domain boundary.
3. **uv_unwrap internals (12-ovoxel Q2, second half):** GPU normal-cone
   clustering (§6.2, cost `cone_half_angle + 0.1·area + 1e-4·perim²/area`,
   merge threshold π/2, refine disabled) + **unmodified stock xatlas** per
   cluster with all-default Chart/Pack options (§6.3-6.4).
4. **doubleSided (12-ovoxel Q6):** production path (remesh=True) ⇒ `False`;
   texturing pipeline ⇒ `True` (trellis2_texturing.py:355).
5. **Ground-truth caller parameters:** space `extract_glb` =
   `simplify(16777216)` → `to_glb(grid_size=res, aabb=±0.5,
   decimation_target=UI (default 300k), texture_size=UI (default 2048),
   remesh=True, remesh_band=1, remesh_project=0)`; export
   `extension_webp=True`. GitHub app differs only in defaults (500k) and the
   missing pre-simplify (§2.3).
6. **cuBVH contract (needed by texel snap and remesh):** `unsigned_distance` →
   (distance ≥ 0 float32, closest int64 `face_id`, float32 barycentric `uvw`
   of the closest point in face-vertex order); closest point reconstructed as
   `Σ tri_vert·uvw` (§5).
7. **xatlas provenance:** vendored copy is byte-identical to stock jpcy/xatlas
   (namespace rename only) — the port may depend on stock xatlas semantics
   (§1, §6.4).
8. **Vertex normals in the GLB:** area-weighted (unnormalized cross-product
   sum) per-vertex normals with first-face NaN fallback, computed after
   `uv_unwrap`'s degenerate-face removal, gathered through `out_vmaps`
   (§2.4, §3.8).
9. **Simplify semantics:** parallel independent-set QEM edge collapse with
   midpoint/boundary-endpoint placement (no optimal-point solve), edge-length
   and skinny penalties (λ=1e-2/1e-3), normal-flip veto, threshold escalation
   ×10 when a pass removes <1% of faces; target counts **faces** (§3.2).
