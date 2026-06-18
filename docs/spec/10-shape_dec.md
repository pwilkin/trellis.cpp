# shape_dec

## TRELLIS.2 shape_dec — FlexiDualGridVaeDecoder spec

### 0. Role in pipeline (inference only)
`shape_slat_decoder` = `FlexiDualGridVaeDecoder`. Called from `decode_shape_slat` (pipelines/trellis2_image_to_3d.py:381-389) as:
```
self.models['shape_slat_decoder'].set_resolution(resolution)   # resolution default 256
ret = self.models['shape_slat_decoder'](slat, return_subs=True)
# returns: (meshes: List[Mesh], subs: List[SparseTensor])
```
Input `slat` is the shape SLAT: a `SparseTensor` with **32 channels** (`latent_channels=32`) defined on **active voxels at resolution 32** (coords `[N,4]` = `[batch, z, y, x]`, batch col first). The decoder upsamples geometry 32→256 (3 actual C2S upsamples used out of 4 up_blocks — see §3) and emits a `FlexiDualGrid` (per-voxel dual vertex offset + 3 edge-intersection flags + 1 split weight), then runs `flexible_dual_grid_to_mesh` (pure CPU/CUDA geometry) to produce a triangle mesh. `subs` (the per-level subdivision predictions) are returned to guide the **texture** decoder (`tex_slat_decoder(slat, guide_subs=subs)`).

IMPORTANT: this class is `pred_subdiv=True` (the base default; FlexiDualGridVaeDecoder does not pass pred_subdiv so it is True). So each up-block predicts its own subdivision from `to_subdiv`.

`self.training` is False at inference — only the `else` branch of `FlexiDualGridVaeDecoder.forward` matters.

### 1. SparseTensor conventions
- `x.feats`: `[N, C]` float. `x.coords`: `[N, 4]` int = `[batch_idx, X, Y, Z]` (col 0 batch). `coords[:,1:]` are spatial.
- `x.replace(feats)`: returns new SparseTensor, same coords, new feats.
- `spatial_shape`: dense grid dims; `_scale`: Fraction scale tuple (tracks current resolution multiplier, starts (1,1,1) at res 32, ×2 per C2S).
- Spatial cache holds neighbor maps and the `channel2spatial_2` / `subdivision` entries used by upsampling.

### 2. Decoder config (configs/scvae/shape_vae_next_dc_f16c32_fp16.json, decoder block)
```
resolution      = 256
model_channels  = [1024, 512, 256, 128, 64]   # per level
latent_channels = 32
num_blocks      = [4, 16, 8, 4, 0]            # ConvNeXt blocks per level
block_type      = ["SparseConvNeXtBlock3d"] * 5
up_block_type   = ["SparseResBlockC2S3d"] * 4 # 4 up blocks (between the 5 levels)
out_channels    = 7   (passed by FlexiDualGridVaeDecoder.__init__ as first arg to base)
use_fp16        = true   (blocks run in fp16; norms via LayerNorm32 internally upcast to fp32)
voxel_margin    = 0.5    (default)
pred_subdiv     = True
```
block_args (use_checkpoint, training-only — ignore for inference): per level [F,F,F,T,T].

### 3. Module construction (base `SparseUnetVaeDecoder.__init__`, out_channels=7)
```
self.from_latent   = SparseLinear(latent_channels=32, model_channels[0]=1024)   # Linear 32->1024 + bias
self.output_layer  = SparseLinear(model_channels[-1]=64, out_channels=7)        # Linear 64->7 + bias
self.blocks = ModuleList of 5 sublists (one per level i):
  level i: num_blocks[i] x SparseConvNeXtBlock3d(model_channels[i])
  then, if i < 4: one SparseResBlockC2S3d(model_channels[i], model_channels[i+1], pred_subdiv=True)
```
Concretely:
- `blocks[0]`: 4x ConvNeXt(1024); then C2S 1024->512  (res 32->64)
- `blocks[1]`: 16x ConvNeXt(512); then C2S 512->256   (res 64->128)
- `blocks[2]`: 8x ConvNeXt(256); then C2S 256->128    (res 128->256)
- `blocks[3]`: 4x ConvNeXt(128); then C2S 128->64     (res 256->512)  **WARNING/RECONCILE**
- `blocks[4]`: 0 ConvNeXt(64); no up-block (last level)

**Reconcile 32→256 (8x=3 upsamples) vs 4 up_blocks:** There are 4 C2S up-blocks, giving 32→64→128→256→512. The geometry/output grid uses `self.resolution=256`. The C2S in block[3] takes the spatial scale to 512. The dual-grid extraction divides coords by `2*N` for the geometric position regardless (see §6: `mesh_vertices=(coords+dual_vertices)/(2N)-0.5` in train, but inference uses `(coords+dual_vertices)*voxel_size + aabb[0]` with `voxel_size=(aabb[1]-aabb[0])/grid_size`, `grid_size=resolution=256`). So the **final feature grid coords are at resolution 512**, while voxel_size is computed for grid_size=256 — i.e. the decoder produces a 512-res sparse field but the dual-grid mapping normalizes using 256. Implementer note: trust `set_resolution(256)` → `grid_size=256` in `flexible_dual_grid_to_mesh`, and the output_layer coords are the level-4 (512-scaled) coords. The number of *active* upsamples is data-dependent via predicted subdivisions; the nominal spatial resolution after all 4 C2S is 512. **OPEN: confirm whether the released checkpoint uses resolution=256 with 4 up-blocks giving 512 coords then /512, vs an intended 256 grid — see open_questions.**

### 4. Forward (inference path, base.forward with return_subs=True)
```
h = from_latent(x)                       # SparseLinear 32->1024
h = h.type(fp16)
subs = []
for i, res_level in enumerate(self.blocks):
    for j, block in enumerate(res_level):
        if i < 4 and j == len(res_level)-1:     # the up-block (C2S)
            # inference, pred_subdiv=True -> block returns (h, sub)
            h, sub = block(h)
            subs.append(sub)
        else:
            h = block(h)                         # ConvNeXt
h = h.type(x.dtype)                              # back to fp32
h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))   # final LayerNorm, NO affine (weight/bias=None), eps default 1e-5, normalized over last dim (64)
h = output_layer(h)                              # SparseLinear 64->7
return h, subs                                   # (return_subs=True branch)
```
Then FlexiDualGridVaeDecoder.forward (else branch) post-processes `h` (the 7-ch SparseTensor), see §5.

### 5. Output head decode (FlexiDualGridVaeDecoder.forward, fdg_vae.py:97-110), inference
`h.feats` is `[M, 7]`. Channels:
- `[..., 0:3]` = dual-vertex offset logits → `vertices = (1 + 2*voxel_margin) * sigmoid(feats[...,0:3]) - voxel_margin`. With voxel_margin=0.5: `vertices = 2*sigmoid(x) - 0.5`, range (-0.5, 1.5). Stored as SparseTensor `vertices` (same coords as h).
- `[..., 3:6]` = 3 edge-intersection logits → `intersected = feats[...,3:6] > 0` (bool, per x/y/z edge of the voxel). SparseTensor `intersected`.
- `[..., 6:7]` = split weight logit → `quad_lerp = softplus(feats[...,6:7])` (positive scalar). SparseTensor `quad_lerp`.

Then for each batch item (zip over batched SparseTensors `vertices`,`intersected`,`quad_lerp` — iterating a SparseTensor yields per-batch sub-tensors):
```
Mesh(*flexible_dual_grid_to_mesh(
    v.coords[:, 1:],   # spatial coords [Nv,3] int (the 512-scale grid coords)
    v.feats,           # dual vertex offsets [Nv,3] float
    i.feats,           # intersected flags  [Nv,3] bool
    q.feats,           # split weight       [Nv,1] float
    aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],
    grid_size=self.resolution,   # 256
    train=False))
```
out_list[0] = list of Mesh; returns (meshes, subs).

### 6. flexible_dual_grid_to_mesh — INFERENCE (train=False), CPU/GPU geometry to port
Inputs: `coords [N,3]` int (voxel index per active dual-grid cell), `dual_vertices [N,3]` float (offset within cell), `intersected_flag [N,3]` bool, `split_weight [N,1]` float (the softplus output), aabb 2x3, grid_size=256.

Constants (static, build once):
```
edge_neighbor_voxel_offset [1,3,4,3] int =
  x-axis: [[0,0,0],[0,0,1],[0,1,1],[0,1,0]]
  y-axis: [[0,0,0],[1,0,0],[1,0,1],[0,0,1]]
  z-axis: [[0,0,0],[0,1,0],[1,1,0],[1,0,0]]
quad_split_1 = [0,1,2, 0,2,3]   # two triangles, diagonal 0-2
quad_split_2 = [0,1,3, 3,1,2]   # two triangles, diagonal 1-3
```
Steps:
1. `voxel_size = (aabb[1]-aabb[0]) / grid_size` (= 1/256 per axis). 
2. Build a 3D spatial hashmap of active voxels: key = coords (with batch col 0 prepended), value = row index 0..N-1. (Custom op: `hashmap_insert_3d_idx_as_val` + `hashmap_lookup_3d`. Port as a std::unordered_map<uint64 packed (z,y,x), uint32 idx> or a dense grid lookup.)
3. For each active cell n and each of the 3 axes a with `intersected_flag[n,a]==True`, gather the 4 neighbor voxels = `coords[n] + edge_neighbor_voxel_offset[a]` (a quad of 4 voxels around that edge). Stack → `connected_voxel [M,4,3]` where M = number of true flags.
4. Look up each of the 4 neighbor voxels in the hashmap → `connected_voxel_indices [M,4]`. Mark `valid` = all 4 found (lookup != 0xffffffff). Keep valid → `quad_indices [L,4]` (indices into the dual-vertex array). L = number of emitted quads.
5. Compute world-space dual vertices for the *whole* set: `mesh_vertices = (coords.float() + dual_vertices) * voxel_size + aabb[0]`  → `[N,3]` (these are the mesh's output vertices; ALL N dual vertices become mesh vertices, indices preserved).
6. Choose diagonal per quad. `split_weight` is NOT None here (it is quad_lerp), so:
```
sw = split_weight[quad_indices]          # [L,4,1]
sw02 = sw[:,0]*sw[:,2]; sw13 = sw[:,1]*sw[:,3]
mesh_triangles = where(sw02 > sw13,
                       quad_indices[:, quad_split_1],     # [L,6]
                       quad_indices[:, quad_split_2]).reshape(-1,3)   # [2L,3]
```
   (The min-angle fallback `split_weight is None` branch using cross-product normals is NOT taken at inference.)
7. Return `(mesh_vertices [N,3] float, mesh_triangles [2L,3] int)`.
Wrapped into `Mesh(vertices, faces)` → `Mesh.vertices=float`, `Mesh.faces=int`, `vertex_attrs=None`.

NOTE the geometric mapping uses grid_size=256 (`voxel_size=1/256`) but `coords` come from the level-4 grid (512-scale) — see §3 reconcile / open_questions. The position formula `(coords + dual_vertices)*voxel_size + aabb[0]` is the load-bearing one.

(Training-only branch, ignore for port: adds a center vertex per quad with weighted blend `quad_split_train=[0,1,4,1,2,4,2,3,4,3,0,4]`, and a different normalization `(coords+dual_vertices)/(2N)-0.5`.)

### 7. Block internals (from sparse_unet_vae.py) — exact op order

**SparseConvNeXtBlock3d(channels)** (the bulk of params):
```
h = conv(x)                              # SparseConv3d(C, C, k=3) submanifold, +bias
h = h.replace(norm(h.feats))             # LayerNorm32(C, affine=True, eps=1e-6) over channel dim
h = h.replace(mlp(h.feats))              # Linear(C, 4C) -> SiLU -> Linear(4C, C)
return h + x                             # residual
```
mlp = `nn.Sequential(Linear(C,4C), SiLU(), Linear(4C,C))`. mlp_ratio=4.0 default. Second Linear was zero-init (zero_module) at init but loaded from checkpoint.

**SparseResBlockC2S3d(channels=Cin, out_channels=Cout, pred_subdiv=True)** (the up-block, channel→spatial 2x upsample):
```
subdiv = to_subdiv(x)                     # SparseLinear(Cin, 8): predicts 8 octant occupancy logits
h = x.replace(norm1(x.feats))            # LayerNorm32(Cin, affine=True, eps=1e-6)
h = h.replace(silu(h.feats))
h = conv1(h)                             # SparseConv3d(Cin, Cout*8, k=3)
subdiv_bin = subdiv.replace(subdiv.feats > 0)        # bool [N,8]
h = updown(h, subdiv_bin)                # SparseChannel2Spatial(2): reshape Cout*8 -> spatial 2x2x2, scatter to children selected by subdiv_bin
x = updown(x, subdiv_bin)                # same upsample applied to x (uses cached map)
h = h.replace(norm2(h.feats))            # LayerNorm32(Cout, affine=False, eps=1e-6) -> normalize only, no params
h = h.replace(silu(h.feats))
h = conv2(h)                             # SparseConv3d(Cout, Cout, k=3) (zero-init at init)
h = h + skip_connection(x)               # skip = repeat_interleave: x.feats[:, j] tiled to Cout, factor = Cout//(Cin//8)
return h, subdiv
```
- `skip_connection` (lambda, NO params): `x.feats.repeat_interleave(Cout // (Cin // 8), dim=1)`. After C2S, x has `Cin//8` channels per child voxel; repeat_interleave expands to Cout. For the 4 up-blocks: (Cin//8, Cout) = (128,512),(64,256),(32,128),(16,64); factor = 4 each.
- `SparseChannel2Spatial(2)`: each parent voxel's `Cout*8` features are reshaped to 8 children × Cout; only children with subdiv_bin True are emitted. New coords = `parent*2 + octant_offset`, octant index decoded as `subidx//(2**i) % 2` for axis i (DIM=3). This is the geometry growth step (sparse scatter — custom GGML op).

`to_subdiv` output (subdiv SparseTensor, the returned `sub`) is what populates `subs` and guides the texture decoder. At inference subdiv is binarized by `>0`.

### 8. Activation / norm formulas
- SiLU(x) = x * sigmoid(x).
- Softplus(x) = ln(1+exp(x)) (PyTorch default beta=1, threshold=20: for x>20 returns x).
- Sigmoid standard.
- LayerNorm32: cast feats to fp32, LayerNorm over last dim (channels), eps as given (1e-6 in blocks; final F.layer_norm uses default eps=1e-5), cast back. affine=True → has weight+bias; affine=False (norm2 in up-block) → no params, pure normalize.
- Final `F.layer_norm(h.feats, h.feats.shape[-1:])` before output_layer: normalized_shape = [64], no weight/bias, eps=1e-5.

### 9. Dtype flow
from_latent runs in input dtype (fp32) → cast to fp16 for all blocks → cast back to fp32 before final LayerNorm + output_layer. LayerNorm32 always computes in fp32 regardless. For the C++ port, computing the whole decoder in fp32 is acceptable (fp16 is a runtime optimization); replicate fp16 only if matching bit-exact intermediate isn't required.

### 10. Custom geometry ops summary (CPU port)
- 3D spatial hashmap insert/lookup (or dense grid index) for active-voxel → row-index.
- Edge-neighbor quad gather (3 axis offset tables of 4 voxels).
- Quad validity (all 4 neighbors active), diagonal selection via split_weight product comparison.
- Vertex world position: `(coord + offset) * (1/256) + (-0.5)`.
- All N dual vertices are mesh vertices (no compaction); faces index into them; downstream `fill_holes`/`simplify` (cumesh) are separate post-processing not part of decoder.

## Weight key patterns

Top-level module is the decoder root (likely stored under a checkpoint key like `models.shape_slat_decoder` or `decoder`; the relative parameter paths below are what matters). All SparseConv3d weights are stored PERMUTED as `(Co, Kd, Kh, Kw, Ci)` = `(out, 3, 3, 3, in)` (see conv_flex_gemm.py:34), bias `(Co,)`. SparseLinear / nn.Linear weights are standard `(out, in)`, bias `(out,)`.

Input/output projections:
- `from_latent.weight`  [1024, 32]
- `from_latent.bias`    [1024]
- `output_layer.weight` [7, 64]
- `output_layer.bias`   [7]

ConvNeXt blocks: `blocks.{i}.{j}.*` for level i, block j (j in 0..num_blocks[i]-1):
- `blocks.{i}.{j}.norm.weight`  [Ci]   (LayerNorm affine)
- `blocks.{i}.{j}.norm.bias`    [Ci]
- `blocks.{i}.{j}.conv.weight`  [Ci, 3, 3, 3, Ci]   (permuted spconv)
- `blocks.{i}.{j}.conv.bias`    [Ci]
- `blocks.{i}.{j}.mlp.0.weight` [4*Ci, Ci]   (mlp = Sequential: index 0 = Linear, 1 = SiLU(no params), 2 = Linear)
- `blocks.{i}.{j}.mlp.0.bias`   [4*Ci]
- `blocks.{i}.{j}.mlp.2.weight` [Ci, 4*Ci]
- `blocks.{i}.{j}.mlp.2.bias`   [Ci]
Per-level Ci: level0=1024 (j=0..3), level1=512 (j=0..15), level2=256 (j=0..7), level3=128 (j=0..3), level4 has 0 ConvNeXt blocks.

Up-blocks (SparseResBlockC2S3d) live at the LAST index of levels 0..3, i.e. `blocks.{i}.{num_blocks[i]}` :
- level0: `blocks.0.4.*`  (Cin=1024, Cout=512)
- level1: `blocks.1.16.*` (Cin=512,  Cout=256)
- level2: `blocks.2.8.*`  (Cin=256,  Cout=128)
- level3: `blocks.3.4.*`  (Cin=128,  Cout=64)
Each up-block UP = `blocks.{i}.{num_blocks[i]}`:
- `{UP}.norm1.weight` [Cin]        (LayerNorm32 affine=True)
- `{UP}.norm1.bias`   [Cin]
- `{UP}.norm2`        NO params    (affine=False — emits nothing in state_dict)
- `{UP}.conv1.weight` [Cout*8, 3, 3, 3, Cin]   (permuted spconv; out=Cout*8)
- `{UP}.conv1.bias`   [Cout*8]
- `{UP}.conv2.weight` [Cout, 3, 3, 3, Cout]    (permuted spconv)
- `{UP}.conv2.bias`   [Cout]
- `{UP}.to_subdiv.weight` [8, Cin]   (SparseLinear/Linear)
- `{UP}.to_subdiv.bias`   [8]
- skip_connection: lambda, NO params (repeat_interleave).
Concrete shapes:
  blocks.0.4: norm1[1024], conv1[4096,3,3,3,1024], conv2[512,3,3,3,512], to_subdiv[8,1024]
  blocks.1.16: norm1[512],  conv1[2048,3,3,3,512],  conv2[256,3,3,3,256], to_subdiv[8,512]
  blocks.2.8:  norm1[256],  conv1[1024,3,3,3,256],  conv2[128,3,3,3,128], to_subdiv[8,256]
  blocks.3.4:  norm1[128],  conv1[512,3,3,3,128],   conv2[64,3,3,3,64],   to_subdiv[8,128]

No params for: final F.layer_norm (functional), SparseChannel2Spatial, SparseSpatial2Channel, SparseUpsample/Downsample, skip_connection lambdas, SiLU.

NOTE: if checkpoint predates the flex_gemm permute, conv weights might be stored as `(Co, Ci, 3, 3, 3)` and permuted at load — verify by inspecting actual tensor shapes (Co,Ci both present so order detection is by which 3-tuple is the kernel). Given conv_flex_gemm.py permutes in __init__ before training, the saved weights are `(Co,3,3,3,Ci)`.

## GGML notes

Per-op GGML mapping:

1. SparseConv3d (submanifold, stride=1, k=3, same coords): CUSTOM op. This is the dominant cost. flex_gemm does masked implicit gemm over a precomputed neighbor map (27 taps). For GGML: precompute, per voxel, the 27 neighbor row-indices (a [N,27] int gather map; absent neighbors = -1). Then conv = sum over 27 taps of (gathered feats [N,Ci]) @ W_tap[Ci,Co]. Implement as: a custom gather to build [N,27,Ci], then a batched matmul against weight reshaped [27,Ci,Co] and reduce-sum over the 27 axis, + bias. Weight stored [Co,3,3,3,Ci]; tap t maps to (kd,kh,kw)=(t//9,(t//9)%3,t%3) offset (-1,0,1). Neighbor map is shared across all convs at the same resolution/dilation (cache by resolution) — build once per level. This is the single hardest op; no native ggml sparse conv exists.

2. SparseLinear / nn.Linear: ggml_mul_mat + ggml_add (bias). Trivial. (from_latent, output_layer, to_subdiv, ConvNeXt mlp.)

3. LayerNorm over channel dim: ggml_norm (normalizes last dim) then mul by weight + add bias for affine; for affine=False (norm2) just ggml_norm. eps must be set (1e-6 blocks / 1e-5 final). Note feats are [N,C] row-major so last-dim norm matches.

4. SiLU: ggml_silu. Softplus: not a single ggml op — compute log(1+exp(x)) via ggml_exp/ggml_log/add, or a custom map op (only 1 channel, N elements — cheap). Sigmoid: ggml_sigmoid.

5. SparseChannel2Spatial (upsample): CUSTOM gather/scatter. Input feats [N, Cout*8]; reshape to [N*8, Cout]; for each parent, select children octants where subdiv>0; output [M, Cout] where M=sum of selected. Produces new coords [M,3] = parent*2 + octant. Needs: (a) to_subdiv logits -> bool mask [N,8]; (b) prefix-sum to allocate; (c) gather feats[parent*8 + octant]. Implement as CPU-side index computation producing a gather index array, then ggml_get_rows. The x-branch (skip) is repeat_interleave then same upsample — also custom but reduces to a gather. Octant decode: child offset along axis i = (octant >> i) & 1.

6. Residual add: ggml_add. ConvNeXt residual h+x; up-block h + skip(x).

7. Subdivision prediction (to_subdiv): Linear 8 outputs -> threshold >0 -> drives #5. The bool [N,8] masks accumulate into `subs` returned for the texture decoder (must be exported/kept).

8. Final mesh extraction (flexible_dual_grid_to_mesh): pure CPU C++ (NOT ggml). Implement: dense or hash voxel index, edge-neighbor quad gather (3 fixed offset tables), validity test, world-position `(coord+offset)/256 - 0.5`, diagonal split by `sw[0]*sw[2] vs sw[1]*sw[3]`, emit triangles. O(N) and O(M). No GPU needed.

Memory: dominant tensors are conv1 of blocks.0.4 [4096,3,3,3,1024] = ~452M params just that layer's weight (4096*27*1024 ~ 113M floats = 226MB fp16). Total decoder is large at the 1024-channel level. Active voxel count N grows ~8x per up-level (bounded by predicted subdivisions). Pre-build and cache the 27-neighbor map per resolution to avoid recompute across the many ConvNeXt blocks at each level (level1 has 16 ConvNeXt = 16 convs sharing one map).

fp16: blocks run fp16 in reference; safe to run fp32 in port. Norms always fp32 (LayerNorm32). Keep conv accumulation in fp32.

Inference-only: ignore use_checkpoint (gradient checkpointing), the training branch of forward (subs_gt, the Mesh built with train=True center-vertex split), sample_posterior, KL. Only `forward(..., return_subs=True)` else-branch + train=False mesh extraction are needed.

## Open questions

1. RESOLUTION/UPSAMPLE RECONCILE (critical): config has 4 up_blocks (C2S) → coords reach 512-scale after blocks.3.4, but `self.resolution=256` is used as grid_size in flexible_dual_grid_to_mesh, and voxel_size=1/256. Need to confirm against a real run whether: (a) the final output coords are at 512 and the /256 normalization is intentional (giving vertices that can lie at half-integer 512 positions mapped through 256 voxel_size — yields offsets in (-0.5,1.5) range matching voxel_margin), or (b) only 3 of the 4 up-blocks actually fire / the checkpoint differs. The vertices formula `2*sigmoid-0.5` range (-0.5,1.5) suggests dual vertices can extend one voxel beyond the cell, consistent with the dual-grid-on-512 + 256 grid interpretation. STRONGLY recommend running the reference decoder once and printing `h.coords.max()` and `_scale` before output_layer to nail the actual final resolution. (Note prompt states 32->256=8x=3 upsamples; code has 4 C2S blocks → 16x to 512. This discrepancy MUST be resolved by reference execution.)

2. SparseConv neighbor semantics: confirm submanifold conv keeps output coords == input coords (it does for flex_gemm submanifold). The ConvNeXt conv does not change active set; only C2S changes it. Verify no padding/dilation other than 1.

3. to_subdiv binarization threshold is exactly `>0` (logit), confirmed in code. Confirm the texture decoder consumes `subs` as `guide_subs` with the same >0 semantics (it calls block(h, subdiv=guide_subs[i]); SparseResBlockC2S3d with pred_subdiv=False uses subdiv directly then binarizes `subdiv.feats>0`). So `subs` passed are the raw logits SparseTensors, not pre-binarized — verify when porting the texture decoder.

4. Coordinate axis order: coords are [batch, X, Y, Z]; the edge_neighbor_voxel_offset and octant decode assume a specific axis ordering. flexible_dual_grid uses coords[:,1:] directly and offsets index [.,.,3] in same order. Confirm X/Y/Z vs Z/Y/X matches between sparse-conv coord storage and the dual-grid offset tables (both use coords[:,1:] consistently, so internally consistent, but the final mesh orientation depends on it — verify against rendered output).

5. Whether `from_latent` input slat needs the shape_slat normalization (mean/std) applied before decode: in decode_latent the shape_slat is the raw decoder output of an upstream flow; normalization (shape_slat_normalization) is applied in sample_tex_slat for the TEX path, not shown applied before shape decode here — confirm the shape decoder expects raw (un-normalized) slat. (decode_shape_slat passes slat straight in.)

6. Final F.layer_norm eps: uses torch default (1e-5) since not specified; confirm.

7. cumesh post-processing (fill_holes, simplify) is applied in decode_latent (m.fill_holes()) — these are separate library ops (CuMesh). Out of scope for shape_dec core but required for output parity; flagged as a downstream component to port or stub.
