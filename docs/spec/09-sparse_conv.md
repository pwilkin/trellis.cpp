# sparse_conv

## TRELLIS.2 Sparse 3D Convolution — Implementation Spec

### 0. Backend selection (decisive)
`trellis2/modules/sparse/config.py`:
- `CONV = 'flex_gemm'` is the **default and intended** backend (env `SPARSE_CONV_BACKEND` may override to `'none' | 'spconv' | 'torchsparse' | 'flex_gemm'`). The released TRELLIS.2 weights are produced/used with `flex_gemm`.
- `DEBUG = False`, `ATTN = 'flash_attn'`.

`SparseConv3d`/`SparseInverseConv3d` (`conv/conv.py`) are thin `nn.Module` dispatchers that lazily import `conv_{CONV}.py` and delegate to `sparse_conv3d_init` / `sparse_conv3d_forward`.

### 1. The conv is SUBMANIFOLD ONLY (flex_gemm backend) — `conv/conv_flex_gemm.py`
Hard constraint enforced in `sparse_conv3d_init`:
```python
assert stride == 1 and (padding is None), 'Currently flex_gemm implementation only support submanifold sparse convolution (stride=1, padding=None)'
```
So **every** `sp.SparseConv3d` in the VAE is a submanifold conv. Key semantics for the C++ port:
- **Submanifold**: the output has the *exact same set of active coordinates* as the input (same `coords`, same `layout`, same active count N). The conv never creates or deletes voxels. Confirmed by `out = x.replace(out)` in the forward (replace keeps `coords`/`layout`/`scale`/`spatial_cache`, only swaps `feats`).
- A 3x3x3 kernel at output voxel p gathers, for each of the 27 kernel offsets `d ∈ {-1,0,1}^3`, the feature of the input voxel at `p + dilation*d` **only if that neighbor is itself active**; missing neighbors contribute **zero** (i.e. they are skipped, not zero-padded with a learned bias term — bias added once at the end). This is the standard Minkowski/spconv `SubMConv3d` rule.
- No spatial-shape change, no stride, no transpose conv. `SparseInverseConv3d` raises `NotImplementedError` for flex_gemm — it is never used with this backend.

Resolution/spatial reduction is therefore done **outside** the conv, by separate gather/scatter resample modules (Section 4), never by strided conv.

#### Parameters and weight layout (critical for key prediction)
```python
self.in_channels, self.out_channels
self.kernel_size = (k,k,k)            # always (3,3,3) in the VAE
self.stride = (1,1,1); self.dilation = (1,1,1)
self.weight = nn.Parameter(torch.empty((out_channels, in_channels, kd, kh, kw)))   # init Co,Ci,Kd,Kh,Kw
if bias: self.bias = nn.Parameter(torch.empty(out_channels)) else register_parameter("bias", None)
# THEN permuted in __init__ and re-wrapped as Parameter:
self.weight = nn.Parameter(self.weight.permute(0, 2, 3, 4, 1).contiguous())   # -> (Co, Kd, Kh, Kw, Ci)
```
So the **stored/checkpoint weight shape is `(Co, Kd, Kh, Kw, Ci) = (out, 3,3,3, in)`**, NOT PyTorch's `(out,in,3,3,3)`. State-dict keys are `...conv.weight` and `...conv.bias` (the Parameter lives directly on the SparseConv3d module as attributes `weight`/`bias`, there is no nested `self.conv` for flex_gemm — unlike spconv backend which nests `self.conv`).

#### Forward — `sparse_conv3d_forward`
```python
flex_gemm.ops.spconv.set_algorithm('masked_implicit_gemm_splitk')   # config.FLEX_GEMM_ALGO
flex_gemm.ops.spconv.set_hashmap_ratio(2.0)                          # config.FLEX_GEMM_HASHMAP_RATIO
Co, Kd, Kh, Kw, Ci = self.weight.shape
neighbor_cache_key = f'SubMConv3d_neighbor_cache_{Kw}x{Kh}x{Kd}_dilation{self.dilation}'
neighbor_cache = x.get_spatial_cache(neighbor_cache_key)
out, neighbor_cache_ = sparse_submanifold_conv3d(
    x.feats, x.coords,
    torch.Size([*x.shape, *x.spatial_shape]),
    self.weight, self.bias, neighbor_cache, self.dilation)
if neighbor_cache is None: x.register_spatial_cache(neighbor_cache_key, neighbor_cache_)
out = x.replace(out)
```
- `x.feats`: `(N, Ci)`, `x.coords`: `(N, 4)` int32 `= [batch, z, y, x]` (coords in `[0,1023]`; channel-first batch index at column 0). Coords for one batch are contiguous.
- The **neighbor map** (the gather index, one per (output voxel, kernel offset)) is computed once per scale and cached on the tensor under `SubMConv3d_neighbor_cache_3x3x3_dilation(1, 1, 1)`. All convs at the same scale (same `_scale` key) reuse it. This is the central data structure the C++ port must build.
- Output `feats`: `(N, Co)`, same `coords`.

The math per output voxel p:  `out[p] = bias + Σ_{d active neighbor} weight[:, d, :] · feats[p+d]` where `weight[:, d, :]` is the `(Co, Ci)` slice for kernel offset d.

### 2. SparseConvNeXtBlock3d (the workhorse block; `models/sc_vaes/sparse_unet_vae.py`)
Attributes:
```python
self.norm = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)     # affine (weight+bias)
self.conv = sp.SparseConv3d(channels, channels, 3)                       # 3x3x3 submanifold, in=out=channels
self.mlp = nn.Sequential(
    nn.Linear(channels, int(channels*mlp_ratio)),    # mlp.0   (mlp_ratio default 4.0)
    nn.SiLU(),                                        # mlp.1
    zero_module(nn.Linear(int(channels*mlp_ratio), channels)),   # mlp.2 (init zero)
)
```
Forward (`_forward`):
```python
h = self.conv(x)                       # submanifold 3x3x3 conv FIRST (spatial mixing)
h = h.replace(self.norm(h.feats))      # LayerNorm over channel dim
h = h.replace(self.mlp(h.feats))       # pointwise MLP (per-voxel, ignores coords)
return h + x                           # residual (elementwise add of feats; same coords)
```
Note this is NOT classic ConvNeXt (no depthwise pointwise split; conv is full Ci→Co=channels). Order is: **conv → LayerNorm → Linear → SiLU → Linear(zero) → +residual**. The mlp does not use the conv's spatial info beyond what `h` already carries. `mlp.2` is zero-initialized so the block starts as identity.

### 3. ResBlocks
All ResBlocks use: `norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)` (affine) and `norm2 = LayerNorm32(out_channels, elementwise_affine=False, eps=1e-6)` (NO affine — pure normalization, no params). `conv2 = zero_module(SparseConv3d(out,out,3))` (zero init). Activation is `F.silu`.

#### SparseResBlockS2C3d (spatial→channel **down**, used as `down_block_type` in encoder)
```python
self.conv1 = sp.SparseConv3d(channels, out_channels // 8, 3)
self.conv2 = zero_module(sp.SparseConv3d(out_channels, out_channels, 3))
self.skip_connection = lambda x: x.replace(
    x.feats.reshape(x.feats.shape[0], out_channels, channels*8//out_channels).mean(dim=-1))
self.updown = sp.SparseSpatial2Channel(2)
```
Forward:
```python
h = norm1(x); h = silu(h)
h = conv1(h)              # channels -> out_channels//8  (at fine resolution)
h = updown(h)            # SparseSpatial2Channel(2): packs 2^3=8 children -> *8 channels => out_channels
x = updown(x)            # x packed too: channels -> channels*8
h = norm2(h); h = silu(h)
h = conv2(h)             # out_channels -> out_channels (at coarse resolution), zero-init
h = h + skip(x)          # skip: reshape (N, out_channels, channels*8//out_channels).mean(-1): channels*8 -> out_channels
```
So downsampling is done by `SparseSpatial2Channel` (gather 8 children into channel dim), NOT by strided conv. conv1 runs at fine resolution producing `out/8` channels, then S2C multiplies channels by 8 → `out`.

#### SparseResBlockC2S3d (channel→spatial **up**, used as `up_block_type` in decoder)
```python
self.conv1 = sp.SparseConv3d(channels, out_channels * 8, 3)
self.conv2 = zero_module(sp.SparseConv3d(out_channels, out_channels, 3))
self.skip_connection = lambda x: x.replace(
    x.feats.repeat_interleave(out_channels // (channels // 8), dim=1))
if pred_subdiv: self.to_subdiv = sp.SparseLinear(channels, 8)   # predicts which of 8 children are active
self.updown = sp.SparseChannel2Spatial(2)
```
Forward:
```python
if pred_subdiv: subdiv = self.to_subdiv(x)     # (N,8) logits
h = norm1(x); h = silu(h)
h = conv1(h)                                    # channels -> out_channels*8  (coarse res)
subdiv_binarized = subdiv.replace(subdiv.feats > 0)
h = updown(h, subdiv_binarized)               # Channel2Spatial(2): split *8 channels into up to 8 children -> out_channels (fine res, expands voxel count)
x = updown(x, subdiv_binarized)
h = norm2(h); h = silu(h)
h = conv2(h)                                   # out_channels -> out_channels (zero-init)
h = h + skip(x)
return (h, subdiv) if pred_subdiv else h
```
Upsampling = `SparseChannel2Spatial` which **creates new voxels** (children) gated by the predicted `subdiv` mask. `to_subdiv` (a Linear `channels→8`) predicts per-parent which of the 8 octant children exist; children with logit>0 are materialized. This is the only place new coords are created in the decoder.

#### Other ResBlock variants (present in code, not used in shape_vae config but may appear in tex_vae configs):
- `SparseResBlock3d`: generic, `resample_mode ∈ {'nearest','spatial2channel'}`, `downsample`/`upsample` flags; uses `SparseDownsample`/`SparseUpsample` (avg-pool / nearest-interp) for 'nearest' mode. Has `conv1`, `conv2`, `skip_connection` (`SparseLinear` if `channels!=out` else `Identity`), and `to_subdiv` only when `upsample`.
- `SparseResBlockDownsample3d` / `SparseResBlockUpsample3d`: 'nearest'-style resample via `SparseDownsample(2)`/`SparseUpsample(2)`, `to_subdiv` only in upsample variant.

### 4. Resample primitives (custom gather/scatter — `modules/sparse/spatial/`)
These contain the real "hard" sparse logic (the conv is comparatively simple given a neighbor map):

**SparseSpatial2Channel (factor=2)** `spatial2channel.py`:
- DIM=3, factor=2 → 8 children. New (coarse) coords = floor(coord/2), deduped via Morton-like code `code = Σ c·OFFSET`, `code.unique(return_inverse=True)`.
- `subidx = Σ (coord%2)·2^i` ∈[0,8): octant index of each input voxel within its parent.
- `new_feats = zeros(N_parent*8, C); new_feats[idx*8 + subidx] = x.feats; reshape(N_parent, 8*C)`. So output channels = `8*C`, output voxel count = N_parent (≤ N/1). Missing children → zeros.
- Caches `(new_coords, idx, subidx)` under `spatial2channel_2`, and registers inverse `channel2spatial_2` cache on output.

**SparseChannel2Spatial (factor=2)** — inverse:
- Needs a `subdivision` `(N_parent,8)` bool (from `to_subdiv`>0) when no cache. `N_leaf = sub.sum(-1)`, `subidx = sub.nonzero()[:,-1]`. New (fine) coords = `parent*2 + octant_offset(subidx)`, repeat_interleaved by N_leaf. `idx` maps each child back to its parent row.
- `x_feats = x.feats.reshape(N*8, C_out)` (splits the 8·C_out channels into 8 groups); `new_feats = x_feats[idx*8 + subidx]`. Output channels = `C_in/8`, voxel count = Σ active children.

**SparseDownsample (factor)** `basic.py`: mean/max pool of children into parent (coord//factor, scatter_reduce). Output channels unchanged. Caches `downsample_{factor}` and inverse `upsample_{factor}`.

**SparseUpsample (factor)** `basic.py`: nearest-neighbor expansion gated by `subdivision`; `new_feats = x.feats[idx]` (replicate parent feature to each child). Channels unchanged.

`SparseLinear` (`linear.py`) = `nn.Linear` applied to `.feats` only: `input.replace(super().forward(input.feats))`. Keys `...weight`, `...bias`.

### 5. Encoder/Decoder assembly (`SparseUnetVaeEncoder` / `Decoder`, shape config)
- `model_channels=[64,128,256,512,1024]`, `latent_channels=32`, encoder `num_blocks=[0,4,8,16,4]`, all `block_type=SparseConvNeXtBlock3d`, `down_block_type=SparseResBlockS2C3d`.
- Encoder: `input_layer = SparseLinear(in_channels, 64)` (in=6 for FDG: vertices-0.5 cat intersected-0.5). Then `blocks[i] = [ConvNeXt × num_blocks[i]] + (S2C downblock to next channels, except last stage)`. Tail: `F.layer_norm(feats, feats.shape[-1:])` (no affine) → `to_latent = SparseLinear(1024, 2*32=64)` → chunk into mean,logvar.
- Decoder mirrored: `model_channels=[1024,512,256,128,64]`, `num_blocks=[4,16,8,4,0]`, `up_block_type=SparseResBlockC2S3d`, `from_latent=SparseLinear(32,1024)`, tail `F.layer_norm` → `output_layer=SparseLinear(64, out=7)`. The C2S up-blocks emit `subdiv` predictions consumed to grow voxels.
- fp16: blocks run in fp16 (`h.type(self.dtype)`), input/output layers and final layer_norm in fp32; `LayerNorm32` always upcasts to fp32 internally then casts back.

### 6. Constants
- LayerNorm eps = 1e-6 (norm1/norm2/the ConvNeXt norm); tail `F.layer_norm` uses default eps 1e-5.
- mlp_ratio default 4.0; SiLU activations; conv2 and mlp.2 zero-initialized.
- factor=2 everywhere; 8 = 2^3 children per parent.
- coords dtype int32, layout `[batch, z, y, x]`, values in [0,1023].

## Weight key patterns

Backend = flex_gemm. SparseConv3d stores weight/bias DIRECTLY on the module (no nested `.conv`), with PERMUTED shape (Co, Kd, Kh, Kw, Ci) = (out, 3,3,3, in). SparseLinear/nn.Linear are standard.

Encoder (`encoder.` prefix when wrapped in VAE, otherwise as below):
- input_layer.weight  (64, 6),  input_layer.bias  (64)
- to_latent.weight     (64, 1024), to_latent.bias  (64)   # 64 = 2*latent(32)
- blocks.{i}.{j}.* where i=stage 0..4, j=block index within stage.

ConvNeXtBlock3d (block j in a stage with C channels, mlp_ratio 4):
- blocks.{i}.{j}.norm.weight (C), blocks.{i}.{j}.norm.bias (C)
- blocks.{i}.{j}.conv.weight (C, 3,3,3, C), blocks.{i}.{j}.conv.bias (C)
- blocks.{i}.{j}.mlp.0.weight (4C, C), blocks.{i}.{j}.mlp.0.bias (4C)
- blocks.{i}.{j}.mlp.2.weight (C, 4C), blocks.{i}.{j}.mlp.2.bias (C)
  (mlp.1 is SiLU, no params)

Encoder down block = SparseResBlockS2C3d (last j of stages 0..3; channels=C_in, out=C_out=next):
- blocks.{i}.{j}.norm1.weight (C_in), .norm1.bias (C_in)        # affine
- blocks.{i}.{j}.norm2  -> NO weight/bias (elementwise_affine=False)
- blocks.{i}.{j}.conv1.weight (C_out//8, 3,3,3, C_in), .conv1.bias (C_out//8)
- blocks.{i}.{j}.conv2.weight (C_out, 3,3,3, C_out), .conv2.bias (C_out)
- skip_connection is a lambda -> NO params
- updown (SparseSpatial2Channel) -> NO params

Decoder up block = SparseResBlockC2S3d (channels=C_in, out=C_out=next, pred_subdiv=True):
- blocks.{i}.{j}.norm1.weight (C_in), .norm1.bias (C_in)
- blocks.{i}.{j}.norm2 -> NO params
- blocks.{i}.{j}.conv1.weight (C_out*8, 3,3,3, C_in), .conv1.bias (C_out*8)
- blocks.{i}.{j}.conv2.weight (C_out, 3,3,3, C_out), .conv2.bias (C_out)
- blocks.{i}.{j}.to_subdiv.weight (8, C_in), .to_subdiv.bias (8)
- skip_connection lambda, updown (SparseChannel2Spatial) -> NO params

Decoder:
- from_latent.weight (1024, 32), from_latent.bias (1024)
- output_layer.weight (7, 64), output_layer.bias (7)

Generic blocks (if a config uses them):
SparseResBlock3d / Downsample3d / Upsample3d:
- norm1.weight/bias (affine), norm2 (no params)
- conv1.weight (Co,3,3,3,Ci), conv1.bias; conv2.weight (Co,3,3,3,Co), conv2.bias
- skip_connection.weight/bias ONLY when channels!=out_channels (SparseLinear); else Identity (no params)
- to_subdiv.weight (8,Ci)/.bias (8) ONLY in upsample variants
- updown (SparseDownsample/SparseUpsample/S2C/C2S) -> NO params

IMPORTANT for the C++ loader: if released checkpoints were saved under the spconv backend instead, conv params are nested as `...conv.conv.weight` with UNPERMUTED shape (out,in,3,3,3) and torchsparse uses `...conv.kernel` of shape (27, in, out). Detect which by inspecting tensor rank/shape and permute to the flex_gemm (Co,Kd,Kh,Kw,Ci) layout at load time. Given config defaults flex_gemm, expect the direct `conv.weight` (Co,3,3,3,Ci) form.

## GGML notes

Hardest custom GGML ops (no native equivalents):

1. SUBMANIFOLD 3x3x3 CONV (core). Two phases:
   a) Build neighbor map: for each active voxel p (key = packed (b,z,y,x)) and each of 27 offsets d (dilation=1 → {-1,0,1}^3), look up whether p+d is active; produce gather index `nmap[p, off] = input_row or -1`. Needs a hash map over coords (flex_gemm uses hashmap_ratio 2.0). In GGML: implement as a custom CUDA/CPU op that builds a coord->row hash table once per scale and emits an int32 tensor `nmap (N, 27)`. Cache it on the tensor (reused by all convs at that scale).
   b) Gemm: out(N,Co) = bias + Σ_{off=0..26} gather(feats, nmap[:,off]) @ W[off] where W reshaped to (27, Ci, Co) from stored (Co,27,Ci) i.e. (Co,Kd,Kh,Kw,Ci). Implement as 27 masked gather+matmul accumulations, or one fused custom kernel. Each off: rows with nmap==-1 contribute 0. Can be done with ggml_get_rows (gather) + ggml_mul_mat per offset + accumulate, but the -1/invalid masking needs a custom gather (or append a zero row at index N and map -1 -> N). RECOMMENDED: pad feats with a zero row, remap -1 to that index, then 27× (ggml_get_rows + ggml_mul_mat) + sum + bias. This avoids a fully custom matmul. Weight slice per offset is (Ci,Co) contiguous if stored (Co,Kd,Kh,Kw,Ci) is reordered to (27,Ci,Co) at load.

2. SparseSpatial2Channel / Channel2Spatial (down/up by octant packing). Custom gather/scatter:
   - S2C: compute parent coords = coord>>1, dedup (sort+unique or hashmap) → idx (N->N_parent) and subidx (octant 0..7). Scatter feats into (N_parent, 8, C) at [idx, subidx], zero elsewhere; reshape to (N_parent, 8C). Needs unique/sort (custom) + scatter (ggml has no scatter → custom op or build index then get_rows from a zero-init buffer written by a kernel).
   - C2S: requires subdiv mask (N_parent,8) from to_subdiv>0; expand parents into children, gather feats[idx*8+subidx]. Pure gather (ggml_get_rows) once the child index list is built (custom prefix-sum to materialize child coords).
   Both create/destroy voxel rows → dynamic output size; not expressible as static ggml graph. Must run as custom op that also outputs new coords.

3. to_subdiv: just an nn.Linear on feats (ggml_mul_mat + bias) then >0 threshold (custom compare → bool/int mask).

4. SparseDownsample(mean/max) / SparseUpsample(nearest): scatter_reduce / gather — custom (only if a config uses 'nearest' resample; shape_vae uses S2C/C2S instead).

Memory considerations:
- Neighbor map (N,27) int32 cached per scale; sizeable at fine resolution (N up to ~1e6 voxels → 27e6 ints = ~108MB). Build once, reuse across all convs of that stage.
- Dynamic voxel counts between stages mean buffers must be (re)allocated at runtime; cannot pre-size the graph. Plan a resizable arena / per-stage graph rebuild.
- fp16 feats in blocks, fp32 for norms (LayerNorm32 upcasts) and input/output linears + tail layer_norm.
- coords int32 [b,z,y,x], values 0..1023 → fits packed into a single int64/uint32 Morton-ish key for hashing; use 11 bits per axis (1024) + batch.

Standard ggml ops cover: SparseLinear / nn.Linear (mul_mat+add), SiLU, residual add (ggml_add), LayerNorm with/without affine (ggml_norm + optional mul/add; eps 1e-6 for norm1/norm2/ConvNeXt-norm, 1e-5 for tail F.layer_norm). chunk for mean/logvar = ggml_view. sigmoid/softplus for decoder head are elementwise.

## Open questions

1. flex_gemm is a compiled CUDA extension not present locally (`flex_gemm.ops.spconv.sparse_submanifold_conv3d`); I inferred submanifold gather semantics (zero for missing neighbors, identical output coords) from the explicit assert, `out = x.replace(out)` (coords unchanged), the neighbor_cache naming `SubMConv3d_...`, and the spconv backend which uses `spconv.SubMConv3d` for stride=1/pad=None. Confirm the exact accumulation (whether the center offset is always present — it is, since p is active) and weight offset ordering (the d→linear-index mapping) by reading flex_gemm source or running the reference; the (Co,Kd,Kh,Kw,Ci) permute means offset enumeration order is z(d=Kd) outer, then h(Kh), then w(Kw) — verify against flex_gemm's internal offset iteration to map W[off] correctly.

2. Checkpoint backend: need to confirm the released safetensors were saved with flex_gemm (direct `conv.weight` of shape (Co,3,3,3,Ci)) vs spconv (`conv.conv.weight` (Co,Ci,3,3,3)) vs torchsparse (`conv.conv.kernel` (27,Ci,Co)). Inspect an actual checkpoint's keys/shapes to pick the right loader/permutation. The config files default to fp16 + flex_gemm, strongly suggesting flex_gemm layout.

3. The tex_vae configs (tex_vae_next_dc_*.json) were not read; they may use SparseResBlock3d with resample_mode='nearest' (needing SparseDownsample/Upsample) or different in/out channels (in=? for texture). Read them before porting the texture VAE.

4. `o_voxel.convert.flexible_dual_grid_to_mesh` (decoder head, FDG) is an external compiled module — out of scope for sparse_conv but needed for the full decoder; the 7 output channels are [3 vertex offsets (sigmoid scaled), 3 intersected logits, 1 quad_lerp (softplus)].

5. Whether `spatial_shape` (max coord+1 per axis) is needed at inference for the neighbor hashmap bounds — flex_gemm receives `torch.Size([*x.shape, *x.spatial_shape])`; confirm it's only used for hashmap sizing, not for behavior.
