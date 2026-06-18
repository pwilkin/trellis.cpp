# ovoxel_mesh

## ovoxel_mesh — O-Voxel extension + final textured GLB export

### 0. Role overview & inference vs training

The `o_voxel` package (built as the `o_voxel._C` PyTorch CUDA/C++ extension, see `o-voxel/setup.py`) provides 5 functional groups. For the C++ inference port, only the GLB-export and grid-sampling paths matter. Per-file role:

| File | Function | Needed at inference? |
|---|---|---|
| `o_voxel/postprocess.py` `to_glb()` | Mesh cleanup + (optional) remesh + decimation + UV unwrap + PBR texture baking → `trimesh.Trimesh` GLB. **This is the final export used by `example.py` and `app.py`** | YES (this is THE export step) |
| `o_voxel/rasterize.py` `VoxelRenderer` | Splats voxels to a camera image (preview/video render). Calls `_C.rasterize_voxels_cuda`. Used only by `trellis2/renderers/voxel_renderer.py` for visualization. | NO (preview only; not on the GLB path) |
| `o_voxel/serialize.py` `encode_seq/decode_seq` | Z-order (Morton) / Hilbert 30-bit encode/decode of integer voxel coords. Used (a) by the `.vxz` codec for compressed voxel storage and (b) potentially as ordering for windowed sparse attention. | NO for GLB export. (Trellis2's *own* sparse-attention serialize lives in `trellis2/modules/sparse/serialize.py`, a different module — covered by another component. The o_voxel one is only needed if you read `.vxz` data files.) |
| `o_voxel/io/*` (`vxz.py`, `ply.py`, `npz.py`) | Voxel file I/O. `.vxz` = SVO (sparse-voxel-octree) + per-attr-channel compressed (lzma/zstd/deflate) chunked format, used for training data. | NO (training/data-prep) |
| `o_voxel/convert/*` (`flexible_dual_grid.py`, `volumetic_attr.py`) | mesh↔dual-grid (used in shape encode), and textured-mesh→volumetric attr (data prep). | The `mesh_to_flexible_dual_grid` half is used by the texturing pipeline's encode (separate component); the `_to_mesh` half by the VAE decode. NOT part of GLB export. |
| `trellis2/utils/mesh_utils.py` | Pure-numpy PLY read/write (`read_ply`, `write_ply`, `write_pbr_ply`). Not on the GLB path; uses `plyfile`/`struct`. | NO (utility/debug) |
| `trellis2/representations/__init__.py` | Lazy-import registry mapping `Mesh`,`Voxel`,`MeshWithVoxel`,`MeshWithPbrMaterial` → submodules. | structural only |

Answering the focus questions directly:
- **serialize** = Morton/Hilbert curve ordering of voxel integer coords. NOT used by the GLB export. It IS the curve used inside the `.vxz` storage codec; whether Trellis2 sparse-attention uses *this* exact function is no — sparse attention imports `encode_seq` from `trellis2/modules/sparse/serialize.py` (different file, same algorithm class). Skip for the export port.
- **rasterize** = a voxel *splatting renderer to a camera image* (color/depth/alpha), NOT texture baking. It is for preview video. The actual PBR-onto-UV baking is done in `to_glb` via nvdiffrast + `grid_sample_3d`, not via this op.
- **postprocess** = the whole export: hole-fill, simplify/decimate (or dual-contouring remesh), UV unwrap, bake PBR texture by trilinearly sampling the attribute volume, build `trimesh` PBRMaterial, axis-swap, return `Trimesh`. Then caller does `.export("*.glb", extension_webp=True)`.

---

### 1. Data flowing into `to_glb` (from `decode_latent`)

`Trellis2ImageTo3DPipeline.decode_latent` (`trellis2/pipelines/trellis2_image_to_3d.py:456`) builds a `MeshWithVoxel` (`trellis2/representations/mesh/base.py:190`):
- `vertices`: (N,3) float, in [-0.5,0.5] object space
- `faces`: (M,3) int
- `origin = [-0.5,-0.5,-0.5]`
- `voxel_size = 1/resolution` (resolution = 512 or 1024)
- `coords = v.coords[:, 1:]` — (L,3) int voxel coords (column 0, the batch index, dropped)
- `attrs = v.feats` — (L,C) float voxel attributes, C=6
- `voxel_shape = torch.Size([B, C, X, Y, Z])` = `[*v.shape, *v.spatial_shape]`
- `layout = pbr_attr_layout = {'base_color':slice(0,3),'metallic':slice(3,4),'roughness':slice(4,5),'alpha':slice(6... actually slice(5,6)}` → channels: 0-2 base_color RGB, 3 metallic, 4 roughness, 5 alpha. C=6 total. Values are in [0,1] (decoder output is `*0.5+0.5`).

`example.py`/`app.py` then call `o_voxel.postprocess.to_glb(vertices, faces, attr_volume=mesh.attrs, coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size (or grid_size=res), aabb=[[-0.5]*3,[0.5]*3], decimation_target, texture_size, remesh=True, remesh_band=1, remesh_project=0)`.

Note the **two near-identical bake routines**: `o_voxel/postprocess.py:to_glb` (the production path; does remesh/decimate) and `Trellis2TexturingPipeline.postprocess_mesh` (`trellis2_texturing.py:287`; for the *texturing-only* pipeline where the mesh is given). The port should implement the `to_glb` behavior; `postprocess_mesh` is the same baking math without the remesh/decimate (it can reuse the same texel-baking code).

---

### 2. `to_glb` exact op order

Inputs normalized: `aabb`→(2,3) float tensor; `grid_size = round((aabb[1]-aabb[0])/voxel_size).int()` if voxel_size given, else `voxel_size=(aabb[1]-aabb[0])/grid_size`. With aabb=±0.5 and voxel_size=1/res → grid_size=res in each axis.

Steps (all CUDA via `cumesh`/`nvdiffrast`/`flex_gemm` in the reference):
1. `mesh = cumesh.CuMesh(); mesh.init(vertices,faces)`; `mesh.fill_holes(max_hole_perimeter=3e-2)`; read back.
2. `bvh = cumesh.cuBVH(vertices,faces)` (over the *original* hole-filled mesh; kept for later attribute correction).
3. Branch A (`remesh=False`): `simplify(decimation_target*3)` → `remove_duplicate_faces()`,`repair_non_manifold_edges()`,`remove_small_connected_components(1e-5)`,`fill_holes(3e-2)` → `simplify(decimation_target)` → same cleanup again → `unify_face_orientations()`.
   Branch B (`remesh=True`, the production default): `cumesh.remeshing.remesh_narrow_band_dc(vertices,faces, center=aabb.mean(0), scale=(resolution+3*remesh_band)/resolution*scale, resolution=grid_size.max(), band=remesh_band, project_back=remesh_project, bvh=bvh)` (Dual Contouring narrow-band remesh) → `simplify(decimation_target)`.
4. UV unwrap: `out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(compute_charts_kwargs={threshold_cone_half_angle_rad=radians(90), refine_iterations=0, global_iterations=1, smooth_strength=1}, return_vmaps=True)`. `mesh.compute_vertex_normals(); out_normals = mesh.read_vertex_normals()[out_vmaps]`.
5. **Texture baking** (texel→3D→attr):
   - `ctx = dr.RasterizeCudaContext()`
   - clip-space UVs: `uvs_rast = cat([out_uvs*2-1, zeros, ones], -1).unsqueeze(0)` → (1,V,4)
   - rasterize the UV triangles into a `texture_size×texture_size` buffer, **in chunks of 100000 faces**, accumulating; alpha channel stores `face_id+1+chunk_offset`. `rast = where(rast_chunk[...,3:4]>0, rast_chunk, rast)`.
   - `mask = rast[0,...,3] > 0` (valid texels)
   - `pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]` → per-texel 3D position on the *new* (decimated/remeshed) mesh.
   - **Attribute correction**: snap each valid texel position to the *original* mesh surface: `_, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)`; `orig_tri = vertices[faces[face_id]]`; `valid_pos = (orig_tri * uvw[...,None]).sum(1)`.
   - **Trilinear sample the attribute volume**: `attrs[mask] = grid_sample_3d(attr_volume, coords4=cat([zeros, coords],-1), shape=[1,C,X,Y,Z], grid=((valid_pos-aabb[0])/voxel_size).reshape(1,-1,3), mode='trilinear')`. (`flex_gemm.ops.grid_sample.grid_sample_3d` = trilinear interpolation of a *sparse* volume given by `coords`+`feats`; grid coords are in **voxel index units**, i.e. continuous voxel coordinate = (world - aabb_min)/voxel_size.)
6. **Channel split & 8-bit quantize**: `base_color = clip(attrs[...,0:3]*255,0,255).uint8` (H,W,3); `metallic=...[3:4]` ; `roughness=...[4:5]`; `alpha=...[5:6]`.
7. **Inpaint UV seams**: `mask_inv=(~mask).uint8`; `base_color=cv2.inpaint(base_color,mask_inv,3,INPAINT_TELEA)`; metallic/roughness/alpha inpainted with radius 1 (then `[...,None]`).
8. **Build PBR material** (trimesh): `PBRMaterial(baseColorTexture=Image(concat([base_color,alpha],-1)) (RGBA), baseColorFactor=[255,255,255,255], metallicRoughnessTexture=Image(concat([zeros_like(metallic), roughness, metallic],-1))  (R=0,G=roughness,B=metallic per glTF spec), metallicFactor=1, roughnessFactor=1, alphaMode='OPAQUE', doubleSided=(not remesh))`.
9. **Axis conversion (Y-up → glTF)**: for vertices & normals: `v[:,1],v[:,2] = v[:,2], -v[:,1]` (swap Y/Z, negate new Z). For UVs: `uv[:,1] = 1-uv[:,1]` (flip V).
10. Return `trimesh.Trimesh(vertices, faces, vertex_normals, process=False, visual=TextureVisuals(uv, material))`. Caller: `glb.export(path, extension_webp=True)`.

---

### 3. serialize.py (Morton / Hilbert) — exact algorithm

`encode_seq(coords[N,3] int, permute=[0,1,2], mode='z_order'|'hilbert') -> codes[N] int32` (30-bit). `decode_seq` is the inverse. permute reorders which coord maps to x/y/z; decode uses `permute.index(0/1/2)`.

Z-order (`z_order.cu`): each 10-bit coord expanded by `expandBits` then `code = xx*4 + yy*2 + zz` (x is highest). `expandBits(v)`: `v=(v*0x00010001)&0xFF0000FF; v=(v*0x00000101)&0x0F00F00F; v=(v*0x00000011)&0xC30C30C3; v=(v*0x00000005)&0x49249249`. Decode: `x=extractBits(code>>2); y=extractBits(code>>1); z=extractBits(code)`. `extractBits` is the inverse mask chain (see ggml notes). Max 10 bits/axis → res ≤ 1024.

Hilbert (`hilbert.cu`): standard Skilling transform on `point[3]={x,y,z}`, `m=1<<9`. Encode: inverse-undo-excess loop (q from m down to 2), gray-encode (`point[i]^=point[i-1]`), then XOR by `t` accumulated from `point[2]` bits; finally same `expandBits`+`xx*4+yy*2+zz` packing. Decode mirror with `m=2<<9`. Code fully quoted in hilbert.cu lines 33-129.

For the GLB port these are **not required** unless you load `.vxz` files at runtime (the pipeline does not for image-to-3D; it reads from model weights). Flag: training data only.

---

### 4. rasterize.py — voxel splat renderer (preview only)

`VoxelRenderer.render(position[N,3], attrs[N,C], voxel_size, extrinsics[4,4], intrinsics[3,3])` → `{attr:(C,H,W), depth:(H,W), alpha:(H,W)}`. Builds OpenGL projection from OpenCV intrinsics (`intrinsics_to_projection`: ret[0,0]=2fx, [1,1]=2fy, [0,2]=2cx-1, [1,2]=-2cy+1, [2,2]=far/(far-near), [2,3]=near*far/(near-far), [3,2]=1), passes `view.T`, `(persp@view).T`, campos, `0.5/fx`,`0.5/fy`, res*ssaa to `_C.rasterize_voxels_cuda` (signature in `rasterize/api.h`: positions(N,3) in [0,1]^3, attrs(N,1)..., voxel_size, viewmatrix(4,4), projmatrix(4,4), campos(3), tan_fovx, tan_fovy, H, W → color(C,H,W),depth,alpha). SSAA downsample via bilinear `F.interpolate(antialias=True)`. **Not on the GLB export path; skip for the export port** (only needed if you implement preview rendering).

---

### 5. Minimal pure-CPU port for GLB output

What you MUST port (CPU is sufficient; no learned weights involved):
- A **GLB writer** (binary glTF 2.0): one mesh primitive with POSITION (vec3 f32), NORMAL (vec3 f32), TEXCOORD_0 (vec2 f32), indices (u32), a PBR `metallicRoughness` material referencing two images (baseColor RGBA + metallicRoughness RGB), images embedded (WebP per `extension_webp=True`, or PNG fallback). Apply the **axis swap** (Y/Z swap + negate Z) and **V-flip** exactly as in §2.9. trimesh is the reference lib; for C++ use tinygltf or cgltf, or hand-roll the JSON+BIN chunks.
- **Trilinear sparse-volume sampling** (`grid_sample_3d` equivalent): given sparse `coords[L,3]` + `feats[L,C]` defining an XYZ grid of `voxel_shape`, sample at continuous grid index `(world-aabb_min)/voxel_size`; gather the 8 neighbor voxels (hash lookup on integer coords; missing neighbor contributes 0), trilinear weights. This is the per-texel attribute bake.
- **UV-space rasterization** (replaces nvdiffrast): for each triangle, its UV coords map to a `texture_size×texture_size` image; rasterize the triangle in UV pixel space, and per covered texel barycentric-interpolate the triangle's 3D vertex positions → texel 3D pos, then sample the volume. This is straightforward CPU scanline rasterization (no GPU needed).
- **Seam inpaint**: replicate `cv2.inpaint(...,INPAINT_TELEA)` — or a cheaper dilation/push-pull fill of invalid texels (a few-pixel border dilation is the load-bearing behavior; Telea quality is not critical for correctness).
- Channel quantization (`*255` clip uint8) and the metallicRoughness packing **R=0, G=roughness, B=metallic** (glTF convention).

CUDA-only steps in the reference and their CPU fallbacks:
- `cumesh` hole-fill / simplify / non-manifold repair / `remove_small_connected_components` / `unify_face_orientations`: CUDA. **CPU fallback**: skip decimation entirely (export the raw mesh) or use a CPU decimator (e.g. quadric edge-collapse). Decimation only affects file size/poly-count, not correctness of texturing — the BVH attribute-correction step re-samples from the original mesh regardless.
- `cumesh.remeshing.remesh_narrow_band_dc` (Dual Contouring): CUDA. **CPU fallback**: skip remesh (`remesh=False` path) and bake directly onto the decoded marching-cubes/DC mesh; set `doubleSided=True` (matches the `not remesh` branch).
- `mesh.uv_unwrap` (xatlas-style chart packing): CUDA in cumesh. **CPU fallback**: use the `xatlas` library (CPU, MIT) for chart computation + packing — produces `out_vertices/out_faces/out_uvs/out_vmaps` with identical semantics (vmaps index original vertices for normals).
- `bvh.unsigned_distance(...,return_uvw=True)`: CUDA. **CPU fallback**: a CPU BVH / closest-point-on-triangle returning (face_id, barycentric uvw). Needed only if you decimate/remesh; if you bake on the original mesh you can skip the correction (set `valid_pos=pos`, `attr_volume` sampled directly).
- `nvdiffrast` rasterize/interpolate: replace with the CPU UV rasterizer above.
- `dr.RasterizeCudaContext`, `flex_gemm.grid_sample_3d`: CPU reimplementations as described.

**Simplest correct CPU pipeline**: (1) take decoded mesh as-is (no remesh/decimate), (2) xatlas UV unwrap, (3) CPU UV rasterize → per-texel 3D pos, (4) trilinear sparse-volume sample for the 6 PBR channels, (5) quantize + dilate-fill seams, (6) axis-swap + V-flip, (7) write GLB with baseColor(RGBA) and metallicRoughness(0,R,M) textures, doubleSided=true.

## Weight key patterns

NONE — ovoxel_mesh has no learned parameters. `o_voxel/postprocess.py`, `rasterize.py`, `serialize.py`, `io/*`, and `trellis2/utils/mesh_utils.py` are all pure algorithmic CUDA/C++/numpy code with NO nn.Module state_dict entries; nothing appears in safetensors. The `MeshWithVoxel`/`MeshWithPbrMaterial`/`Voxel` classes (`trellis2/representations/mesh/base.py`, `voxel/voxel_model.py`) are plain data containers (attributes: `vertices`,`faces`,`origin`,`voxel_size`,`coords`,`attrs`,`voxel_shape`,`layout`; PbrMaterial fields `base_color_texture/_factor`,`metallic_*`,`roughness_*`,`alpha_*`,`alpha_mode`,`alpha_cutoff`), not registered modules. The inputs (vertices/faces/coords/attrs) are produced at runtime by the VAE decoder (a different, weighted component); this component only consumes them. The `o_voxel._C` extension exports only functions (ext.cpp): hashmap_*, mesh_to_flexible_dual_grid_cpu, textured_mesh_to_volumetric_attr_cpu, {z_order,hilbert}_{encode,decode}_{cpu,cuda}, {en,de}code_sparse_voxel_octree[_attr_{parent,neighbor}]_cpu, rasterize_voxels_cuda — no parameters.

## GGML notes

No GGML graph ops are strictly required for ovoxel_mesh — it is post-network CPU geometry/IO. Implement as plain C++ host code, not as a ggml compute graph. Per-op mapping:

- Trilinear sparse-volume sample (grid_sample_3d): CUSTOM CPU. Build an integer-coord hash map (xyz -> row index into feats[L,6]); for each query, floor the continuous voxel coord, gather 8 corner voxels (0 if absent), trilinear blend. Not a ggml primitive (it's a gather over a sparse hashmap). Memory: hashmap over L voxels (L up to a few hundred k); feats L×6 f32.

- UV-space triangle rasterization: CUSTOM CPU scanline rasterizer; per-texel barycentric interpolation of 3 vertex positions. No ggml op. Output buffer texture_size² × 3 f32 + mask (texture_size up to 4096 → 4096²×4 ≈ 268 MB f32; bake channel-by-channel or in tiles to bound memory; reference does face-chunking of 100k).

- Morton/Hilbert encode/decode (serialize): CUSTOM CPU bit-twiddling (expandBits/extractBits chains quoted in spec). Trivial scalar loops. Only needed if reading .vxz; not for GLB.

- Mesh decimation / hole-fill / non-manifold repair / DC remesh / uv_unwrap / closest-point-BVH: NOT ggml; these are CUDA-lib (`cumesh`) ops. CPU fallbacks: xatlas (UV), optional quadric decimator, optional CPU BVH; or skip decimation/remesh entirely and bake on the raw mesh.

- Seam inpaint (cv2.inpaint TELEA): CUSTOM CPU; a few-iteration dilation/push-pull of invalid texels is an acceptable substitute.

- voxel splat renderer (rasterize_voxels_cuda): preview-only, OUT OF SCOPE for export.

GLB write: use tinygltf/cgltf or hand-rolled glTF2 JSON+BIN; embed two textures (WebP via libwebp to match extension_webp=True, else PNG). metallicRoughness texture packs G=roughness, B=metallic, R=0. Apply vertex/normal axis swap (y,z)->(z,-y) and uv V-flip before writing. All f32 buffers, indices u32.

Numerical exactness to match reference: attribute quantize = clip(x*255,0,255) round-to-uint8; trilinear weights standard; coords for sampling are in voxel-index units = (world - aabb_min)/voxel_size with aabb=±0.5, voxel_size=1/res, so = (world+0.5)*res.

## Open questions

1. `flex_gemm.ops.grid_sample.grid_sample_3d` boundary/padding & align_corners convention is not visible in the read files (background job to locate flex_gemm source was launched but not yet confirmed). I inferred standard trilinear with grid in voxel-index units and zero outside the sparse set. CONFIRM the exact corner alignment (does index i correspond to voxel-center or voxel-corner?) by reading flex_gemm/ops/grid_sample.py + its CUDA kernel — this shifts samples by 0.5 voxel and matters for exact texture match. The two callers differ: postprocess uses grid=((pos-aabb0)/voxel_size); texturing pipeline uses grid=((pos+0.5)*resolution) — confirm these are equivalent (they are, given aabb=±0.5, voxel_size=1/res).

2. `cumesh.remeshing.remesh_narrow_band_dc` and `mesh.uv_unwrap` algorithm details (chart threshold/packing) are opaque (compiled lib). For a faithful port, decide whether to (a) reproduce via xatlas (UVs will differ but be valid) or (b) skip remesh. Output GLB geometry will differ from reference unless cumesh is reproduced; texture content should still be correct since baking re-samples original surface via BVH. CONFIRM acceptable.

3. `cv2.inpaint INPAINT_TELEA` exact pixels are not reproducible without porting Telea; a dilation fallback changes seam pixels slightly. CONFIRM this tolerance is acceptable (it only affects pixels outside the UV charts).

4. Whether the C++ runtime must read `.vxz` at all (it appears training-only; image-to-3D loads weights, decoder emits coords/feats directly). If never, the entire serialize + io + SVO codec can be dropped. CONFIRM no .vxz asset is loaded at inference.

5. `extension_webp=True` — confirm target consumers accept WebP-in-GLB (KHR_texture_basisu? no — trimesh uses image/webp). If not, emit PNG. Verify desired image format.

6. `doubleSided` is `True` when not remeshing and `False` when remeshing (postprocess) but always `True` in texturing pipeline — pick per chosen mesh path.
