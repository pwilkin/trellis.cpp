# shape_dec


# FlexiDualGridVaeDecoder ("shape_dec_next_dc_f16c32_fp16") — inference spec

Source of truth: `/tmp/TRELLIS.2/trellis2/models/sc_vaes/fdg_vae.py` (FlexiDualGridVaeDecoder),
`/tmp/TRELLIS.2/trellis2/models/sc_vaes/sparse_unet_vae.py` (SparseUnetVaeDecoder + blocks),
`/tmp/TRELLIS.2/trellis2/modules/sparse/spatial/spatial2channel.py` (SparseChannel2Spatial),
`/tmp/TRELLIS.2/o-voxel/o_voxel/convert/flexible_dual_grid.py` (flexible_dual_grid_to_mesh),
config `/tmp/t2cfg/shape_dec_next_dc_f16c32_fp16.json`.

## 0. What it is / where it sits in the pipeline
This is the **shape SLat decoder** (`self.models['shape_slat_decoder']`). Stage ① produced active voxel
coords + a sampled structured latent (SLat) `slat: SparseTensor` with `feats [N, 32]` (latent_channels=32)
and `coords [N, 4]` = (batch, x, y, z) int. trellis2_image_to_3d.py line 385 calls it as
`ret = self.models['shape_slat_decoder'](slat, return_subs=True)` after `set_resolution(resolution)`
(resolution ∈ {256,384,512,...}, default 256). In our runtime, batch is always a single object (b=0).

Output: a triangle `Mesh` (vertices [V,3] float, faces [F,3] int) in world space (aabb [-0.5,0.5]^3),
plus `subs` (the predicted subdivision logits per up-block; needed downstream by the texture decoder
`decode_tex_slat(tex_slat, subs)` — KEEP these around).

This forward runs in EVAL mode (`self.training == False`) and `pred_subdiv == True`. All `subs_gt`,
`subdivision` spatial-cache, `train=True` branches in flexible_dual_grid_to_mesh, the FlexiDualGridVaeEncoder,
and `Mesh.fill_holes/simplify/remove_faces` (cumesh CUDA) are TRAINING/POSTPROCESS-ONLY — DO NOT PORT.

## 1. Config (exact)
```
resolution      = 256            # mesh grid_size for world-coord mapping (overridable: 256/384/512)
model_channels  = [1024, 512, 256, 128, 64]
latent_channels = 32
num_blocks      = [4, 16, 8, 4, 0]      # ConvNeXt blocks per stage
block_type      = SparseConvNeXtBlock3d (all 5 stages)
up_block_type   = SparseResBlockC2S3d   (4 up-blocks, between stages)
out_channels    = 7
voxel_margin    = 0.5            # default in FlexiDualGridVaeDecoder.__init__
pred_subdiv     = True
use_fp16        = True           # blocks compute in fp16; norms/layernorm cast to fp32 internally
```

The decoder's `self.blocks` is built (SparseUnetVaeDecoder.__init__, lines 426-444) as a ModuleList of
5 stages. Stage i has `num_blocks[i]` ConvNeXt blocks; for i<4 an up-block (SparseResBlockC2S3d) is
**appended as the last element** of that stage's list. So `blocks.{i}.{j}`:
- stage 0: j=0..3 ConvNeXt (1024ch), j=4 = C2S 1024->512   (to_subdiv on 1024)
- stage 1: j=0..15 ConvNeXt (512ch),  j=16 = C2S 512->256   (to_subdiv on 512)
- stage 2: j=0..7 ConvNeXt (256ch),   j=8  = C2S 256->128   (to_subdiv on 256)
- stage 3: j=0..3 ConvNeXt (128ch),   j=4  = C2S 128->64    (to_subdiv on 128)
- stage 4: j=(none, num_blocks=0) — empty, no ConvNeXt, no up-block. Confirmed by keys: no `blocks.4.*`.

So the actual graph is: from_latent → [stage0 4×ConvNeXt → C2S0] → [stage1 16×ConvNeXt → C2S1]
→ [stage2 8×ConvNeXt → C2S2] → [stage3 4×ConvNeXt → C2S3] → (stage4 empty) → final F.layer_norm →
output_layer → 7ch.

NOTE on voxel scale: input voxel grid is the coarse SLat grid; each C2S multiplies coords by 2 and
materializes children. With 4 up-blocks the final grid is 16× the input grid scale. (The README phrasing
"32->64->128->256->512" is the *abstract* scale-doubling; the literal coords after each C2S = coords*2 +
child offset. The absolute integer grid the mesh extractor uses is `grid_size = resolution`.)

## 2. SparseTensor conventions (must replicate)
- `coords`: int tensor `[N, 4]` = `(b, x, y, z)`. Column 0 is batch (=0 here).
- `feats`: float tensor `[N, C]`.
- `replace(new_feats)`: same coords, new feats. `.type(dtype)` casts feats.
- `spatial_shape`: per-axis max(coord)+1 (only used by C2S cache MAX computation; for forward we don't
  actually need it because in eval we go through the `subdivision is not None` branch, see §5).

## 3. from_latent  (SparseLinear 32 -> 1024)
```
h.feats = slat.feats @ from_latent.weight.T + from_latent.bias     # [N, 1024]
h.coords = slat.coords                                              # unchanged
h = h.type(fp16)
```
weight `from_latent.weight [1024,32]`, bias `from_latent.bias [1024]`.

## 4. SparseConvNeXtBlock3d  (the workhorse; 4+16+8+4 = 32 blocks total)
Per `SparseConvNeXtBlock3d._forward` (lines 284-288), residual order is:
```
h = conv(x)                              # SparseConv3d k=3, submanifold, in=out=C
h = norm(h.feats)                        # LayerNorm32 over last dim C, elementwise_affine=True, eps=1e-6
h = mlp(h.feats)                         # Linear(C, 4C) -> SiLU -> Linear(4C, C)   (zero_module init on 2nd, irrelevant at inference)
return h + x                             # residual
```
mlp_ratio defaults to 4.0 (so hidden = 4*C). Note `mlp.0`=Linear(C,4C), `mlp.1`=SiLU(no params), `mlp.2`=Linear(4C,C).
LayerNorm32 = standard LayerNorm but computed in fp32 then cast back (norm.py). eps=1e-6.

IMPORTANT: the conv is applied to x DIRECTLY (no norm/activation before conv, unlike ResBlock).
The norm comes AFTER conv, then the channelwise MLP. This is ConvNeXt-style (depthwise-ish 3x3x3 conv +
pointwise MLP), but here `conv` is a full SparseConv3d (not depthwise), in=out=C.

## 5. SparseResBlockC2S3d  (channel-to-spatial up-block; 4 of them) — eval/pred_subdiv path
Per `SparseResBlockC2S3d._forward` (lines 240-256) with `pred_subdiv=True`:
```
# (a) predict subdivision from the PRE-norm input x  (channels = stage-in C = model_channels[i])
subdiv_logits = to_subdiv(x.feats)            # SparseLinear C -> 8   => [N, 8]
subdiv_binarized = (subdiv_logits > 0)        # bool [N, 8]   (eval: thresholded; train uses GT cache)

# (b) main branch
h = norm1(x.feats)                            # LayerNorm32 affine=True eps=1e-6 over C
h = SiLU(h)
h = conv1(h)                                  # SparseConv3d C -> out_C*8 , k=3 submanifold
h = SparseChannel2Spatial(2)(h, subdiv_binarized)   # spatial upsample: each parent voxel emits one child
                                              #   voxel per set bit of its 8-bit mask. feats reshaped
                                              #   [N, out_C*8] -> per child [out_C].  See §5.1.
# (c) skip branch (x is ALSO upsampled with the SAME subdiv so coords align)
x_up = SparseChannel2Spatial(2)(x, subdiv_binarized)        # x has C channels -> reshape [N,C*8]->child [C... ]
skip = x_up.feats.repeat_interleave(out_C // (C // 8), dim=1)   # channel match: see lambda line 235
# Actually skip_connection runs on x_up: x.replace(x_up.feats.repeat_interleave(out_C//(C//8),1))
h = norm2(h.feats)                            # LayerNorm32 affine=False eps=1e-6 (no weight/bias)
h = SiLU(h)
h = conv2(h)                                  # SparseConv3d out_C -> out_C, k=3 submanifold (zero-init; nonzero at inference from ckpt)
h = h + skip
return h, subdiv_logits                       # subdiv_logits collected into `subs`
```

### 5.1 SparseChannel2Spatial(factor=2) on a sparse tensor WITHOUT cache (eval path)
(spatial2channel.py lines 67-93; cache is None because in the decoder we never ran the paired
SparseSpatial2Channel — encoder side. So the `subdivision is not None` branch executes.)
DIM=3, factor=2, factor**DIM = 8.
```
sub      = subdivision.feats                 # bool [N, 8], child-occupancy mask per parent voxel
N_leaf   = sub.sum(dim=-1)                    # [N] children kept per parent (0..8)
subidx   = sub.nonzero()[:, -1]              # [K] for each kept child, its 0..7 local index, in
                                             #     row-major(N then bit) order. K = sum(N_leaf)
new_coords = x.coords.clone(); new_coords[:,1:] *= 2          # parent*2
new_coords = repeat_interleave(new_coords, N_leaf, dim=0)     # [K,4], parent coord repeated per kept child
for i in 0..2:  new_coords[:, i+1] += (subidx // 2**i) % 2    # add child offset:
                                             #   i=0 -> x += bit0, i=1 -> y += bit1, i=2 -> z += bit2
idx = repeat_interleave(arange(N), N_leaf)   # [K] parent index for each child

x_feats = x.feats.reshape(N*8, -1)           # split the C*8 (or C) channel block into 8 sub-blocks of size feats_C/8
new_feats = x_feats[idx*8 + subidx]          # [K, feats_C/8]  gather child block
```
KEY index math: child local index `s = subidx ∈ [0,7]`, decoded as `s = x_off + 2*y_off + 4*z_off`
(bit0=x, bit1=y, bit2=z). The flattened feature for parent p has 8 contiguous sub-blocks; child s reads
sub-block s. So for the **main branch h** (feats_C = out_C*8): each child gets out_C feats =
`h.feats[p, s*out_C : (s+1)*out_C]`. For the **skip x_up** (feats_C = C): the reshape is `N*8` rows of
size C/8, child gets `x.feats[p, s*(C/8):(s+1)*(C/8)]` then repeat_interleaved by `out_C//(C//8)` to out_C.
  - stage0: C=1024, out_C=512 -> child skip block C/8=128, repeat factor 512//128=4 -> 512. ✓
  - stage1: C=512,  out_C=256 -> 64, factor 256//64=4 -> 256. ✓
  - stage2: C=256,  out_C=128 -> 32, factor 128//32=4 -> 128. ✓
  - stage3: C=128,  out_C=64  -> 16, factor 64//16=4 -> 64.  ✓

The new sparse set after a C2S has K voxels (K = number of set subdivision bits across all parents).
This is how the active voxel set GROWS from coarse to fine while staying sparse.

## 6. Final head (SparseUnetVaeDecoder.forward lines 498-500)
```
h = h.type(fp32)                              # back to input dtype
h = F.layer_norm(h.feats, h.feats.shape[-1:])  # LayerNorm over last dim, NO affine (no weight/bias), eps default 1e-5
h = output_layer(h.feats)                     # SparseLinear 64 -> 7
```
output_layer.weight [7,64], output_layer.bias [7]. Result feats `[Nf, 7]` at the finest sparse grid,
coords `[Nf, 4]`.

## 7. Split the 7 channels (fdg_vae.py eval branch, lines 99-108)
```
vertices_feats   = (1 + 2*voxel_margin) * sigmoid(feats[:, 0:3]) - voxel_margin     # voxel_margin=0.5 -> 2*sigmoid(x)-0.5, range (-0.5,1.5)
intersected_bool = feats[:, 3:6] > 0          # bool [Nf,3]  (x/y/z-edge intersected flags)
quad_lerp        = softplus(feats[:, 6:7])    # [Nf,1] split weight, >0
```
`coords` passed to mesh extractor is `h.coords[:, 1:]` i.e. drop batch col -> `[Nf, 3]` int xyz.

## 8. flexible_dual_grid_to_mesh — eval (train=False) — PURE CPU GEOMETRY
Inputs: `coords [N,3] int`, `dual_vertices = vertices_feats [N,3] float`, `intersected_flag [N,3] bool`,
`split_weight = quad_lerp [N,1] float`, `aabb=[[-.5,-.5,-.5],[.5,.5,.5]]`, `grid_size = resolution` (256).
(flexible_dual_grid.py lines 142-283; here N = Nf, the finest voxel count.)

Constants (lines 173-183):
```
edge_neighbor_voxel_offset[3][4][3] =       # for each of 3 edge axes, the 4 voxels sharing that edge
  axis x: [[0,0,0],[0,0,1],[0,1,1],[0,1,0]]
  axis y: [[0,0,0],[1,0,0],[1,0,1],[0,0,1]]
  axis z: [[0,0,0],[0,1,0],[1,1,0],[1,0,0]]
quad_split_1 = [0,1,2, 0,2,3]               # diagonal 0-2
quad_split_2 = [0,1,3, 3,1,2]               # diagonal 1-3
```
voxel_size = (aabb[1]-aabb[0]) / grid_size = 1/256 per axis (line 212).

Algorithm:
```
# vertex world positions  (line 268; the line-222 formula is DEAD — overwritten)
mesh_vertices = (coords.float() + dual_vertices) * voxel_size + aabb[0]    # [N,3], aabb[0]=-0.5

# build hashmap: key = encoded (b=0,x,y,z) -> value = voxel row index  (lines 225-226)
#   replicate with a std::unordered_map<uint64,int> keyed on (x,y,z); value = i.

# For each voxel i and each axis a in {0,1,2} where intersected_flag[i,a] is True:
#   the 4 neighbor voxels = coords[i] + edge_neighbor_voxel_offset[a]      # (4,3)
#   look up all 4 in hashmap. If ALL 4 exist (valid), they form a quad:
#       quad_indices = [idx0, idx1, idx2, idx3]    (the 4 voxel row indices)
# (lines 229-239: connected_voxel_valid = all 4 found; collect L quads.)

# Triangulate each quad (split_weight provided branch, lines 258-265):
#   w = split_weight[quad_indices]           # the scalar quad_lerp at each of the 4 corners
#   d02 = w[0]*w[2];  d13 = w[1]*w[3]
#   if d02 > d13:  tris = quad[quad_split_1]   # corners (0,1,2),(0,2,3)
#   else:          tris = quad[quad_split_2]   # corners (0,1,3),(3,1,2)
#   emit 2 triangles (6 indices) into face list.

return mesh_vertices [N,3], mesh_triangles [2L, 3]
```
NOTE: vertices are ALL N voxels' dual vertices (even unreferenced ones stay in the array; faces index
into them). That matches reference (it does not compact). Faces are int32.

This entire section (§7 split + §8 extraction) is **pure CPU, portable C++** — no GGML, no GPU needed.
The `_C.hashmap_*_cuda` calls are just a GPU hashmap; replace with a host hashmap.

## 9. Returned `subs`
`return_subs=True` -> decoder returns `(h, subs)` where `subs` is the list of 4 `subdiv_logits`
SparseTensors (one per up-block, the PRE-binarization logits feats `[N_i,8]` with their coords).
These are consumed by the texture decoder. Persist coords+logits per up-block.


## Weight key map


All keys from /devel/alt/trellis.cpp/docs/spec/keys/ckpts__shape_dec_next_dc_f16c32_fp16.keys.txt (f16).
Conv weights are stored ALREADY PERMUTED to (Co, Kd, Kh, Kw, Ci) — see conv_flex_gemm.py line 34
`weight.permute(0,2,3,4,1).contiguous()`. So shape [Co,3,3,3,Ci]. Linear weights are torch [out,in].

== Stem / head ==
from_latent.weight      [1024,32]   SparseLinear 32->1024     (h = x@W.T + b)
from_latent.bias        [1024]
output_layer.weight     [7,64]      SparseLinear 64->7
output_layer.bias       [7]
(final F.layer_norm BEFORE output_layer has NO weights — affine=False.)

== Stage 0: blocks.0.{0..3} ConvNeXt (C=1024, hidden=4096) ==
blocks.0.{j}.conv.weight  [1024,3,3,3,1024]   SparseConv3d k3 (Co=1024,Ci=1024)
blocks.0.{j}.conv.bias    [1024]
blocks.0.{j}.norm.weight  [1024]              LayerNorm32 affine eps1e-6
blocks.0.{j}.norm.bias    [1024]
blocks.0.{j}.mlp.0.weight [4096,1024]         Linear 1024->4096
blocks.0.{j}.mlp.0.bias   [4096]
blocks.0.{j}.mlp.2.weight [1024,4096]         Linear 4096->1024
blocks.0.{j}.mlp.2.bias   [1024]
  (j=0..3; mlp.1 = SiLU, no params)

== Stage 0 up-block: blocks.0.4 = SparseResBlockC2S3d (C=1024 -> out=512) ==
blocks.0.4.to_subdiv.weight [8,1024]   SparseLinear 1024->8
blocks.0.4.to_subdiv.bias   [8]
blocks.0.4.norm1.weight     [1024]     LayerNorm32 affine eps1e-6 (on C=1024)
blocks.0.4.norm1.bias       [1024]
blocks.0.4.conv1.weight     [4096,3,3,3,1024]  SparseConv3d 1024 -> 512*8=4096
blocks.0.4.conv1.bias       [4096]
blocks.0.4.conv2.weight     [512,3,3,3,512]    SparseConv3d 512 -> 512
blocks.0.4.conv2.bias       [512]
  (norm2 = LayerNorm32 affine=False -> NO weight/bias in ckpt, confirmed absent. skip_connection is a
   lambda repeat_interleave -> NO params.)

== Stage 1: blocks.1.{0..15} ConvNeXt (C=512, hidden=2048) ==
blocks.1.{j}.conv.weight  [512,3,3,3,512]
blocks.1.{j}.conv.bias    [512]
blocks.1.{j}.norm.weight  [512]   ; .norm.bias [512]
blocks.1.{j}.mlp.0.weight [2048,512] ; .mlp.0.bias [2048]
blocks.1.{j}.mlp.2.weight [512,2048] ; .mlp.2.bias [512]
  (j=0..15)

== Stage 1 up-block: blocks.1.16 = SparseResBlockC2S3d (C=512 -> out=256) ==
blocks.1.16.to_subdiv.weight [8,512]  ; .to_subdiv.bias [8]
blocks.1.16.norm1.weight     [512]    ; .norm1.bias [512]
blocks.1.16.conv1.weight     [2048,3,3,3,512]   (512 -> 256*8=2048)
blocks.1.16.conv1.bias       [2048]
blocks.1.16.conv2.weight     [256,3,3,3,256]    (256 -> 256)
blocks.1.16.conv2.bias       [256]

== Stage 2: blocks.2.{0..7} ConvNeXt (C=256, hidden=1024) ==
blocks.2.{j}.conv.weight  [256,3,3,3,256] ; .conv.bias [256]
blocks.2.{j}.norm.weight  [256] ; .norm.bias [256]
blocks.2.{j}.mlp.0.weight [1024,256] ; .mlp.0.bias [1024]
blocks.2.{j}.mlp.2.weight [256,1024] ; .mlp.2.bias [256]
  (j=0..7)

== Stage 2 up-block: blocks.2.8 = SparseResBlockC2S3d (C=256 -> out=128) ==
blocks.2.8.to_subdiv.weight [8,256] ; .to_subdiv.bias [8]
blocks.2.8.norm1.weight     [256]   ; .norm1.bias [256]
blocks.2.8.conv1.weight     [1024,3,3,3,256]   (256 -> 128*8=1024)
blocks.2.8.conv1.bias       [1024]
blocks.2.8.conv2.weight     [128,3,3,3,128]    (128 -> 128)
blocks.2.8.conv2.bias       [128]

== Stage 3: blocks.3.{0..3} ConvNeXt (C=128, hidden=512) ==
blocks.3.{j}.conv.weight  [128,3,3,3,128] ; .conv.bias [128]
blocks.3.{j}.norm.weight  [128] ; .norm.bias [128]
blocks.3.{j}.mlp.0.weight [512,128] ; .mlp.0.bias [512]
blocks.3.{j}.mlp.2.weight [128,512] ; .mlp.2.bias [128]
  (j=0..3)

== Stage 3 up-block: blocks.3.4 = SparseResBlockC2S3d (C=128 -> out=64) ==
blocks.3.4.to_subdiv.weight [8,128] ; .to_subdiv.bias [8]
blocks.3.4.norm1.weight     [128]   ; .norm1.bias [128]
blocks.3.4.conv1.weight     [512,3,3,3,128]    (128 -> 64*8=512)
blocks.3.4.conv1.bias       [512]
blocks.3.4.conv2.weight     [64,3,3,3,64]      (64 -> 64)
blocks.3.4.conv2.bias       [64]

== Stage 4: NONE (num_blocks[4]=0, no up-block since last stage). No blocks.4.* keys. ==

Total keys accounted for: every line in the .keys.txt dump maps above. No unmapped keys remain.
norm2 (all up-blocks) and the final pre-output layer_norm are affine=False => no params (correctly absent).


## GGML plan


This component is fundamentally a SUBMANIFOLD SPARSE network and CANNOT reuse ss_decoder.cpp's dense
ggml_conv_3d (that path builds a full DxWxHxC grid; here grids reach 256^3-512^3 with only ~10^5-10^6
active voxels). The existing helpers reusable: Linear (matmul + bias), LayerNorm pattern, SiLU, sigmoid,
softplus — but the conv must be a NEW submanifold sparse conv. Recommend a hybrid CPU/GGML design.

== Core new primitive: submanifold SparseConv3d (k=3, stride=1) ==
Submanifold = output voxel set == input voxel set; for output voxel p, sum over the 27 kernel taps t
(offset in [-1,0,1]^3) of input voxel (p+offset_t) IF that voxel exists in the active set, times
weight[:, t, :] (the (Kd,Kh,Kw) slice). Plan:
  1. Build a host hashmap (x,y,z)->row index ONCE per sparse level (coords change after each C2S).
  2. Precompute a neighbor map: for each output row p and each of 27 taps, the input row index or -1.
     (Mirrors flex_gemm's `neighbor_cache` keyed by kernel size — cache per spatial level.)
  3. GGML execution of the gather-GEMM. Two viable implementations:
     (A) "gather then matmul" per tap: for tap t with offset, gather input feats rows where neighbor
         exists into a dense [Nout, Ci] buffer (zeros where missing), matmul by weight slice
         W_t [Co, Ci] (ggml_mul_mat), accumulate (ggml_add) across 27 taps; add bias once at end.
         This is 27 ggml_mul_mat of [Nout,Ci]x[Ci,Co]. Reshape ckpt weight [Co,3,3,3,Ci] -> 27 slices
         [Co,Ci]; in ggml store as a single [Ci, Co, 27] tensor and index. Gather is the only custom op.
     (B) Implement gather as a precomputed index tensor + ggml_get_rows. ggml_get_rows(feats[Ci,N],
         idx[Nout]) gives [Ci,Nout]; rows with missing neighbor map to a zero row (append a zero row at
         index N and point missing taps there). Then 27× get_rows + mul_mat + add. This keeps everything
         on-graph (GPU-friendly) and only needs CPU to build the 27 index vectors per level. PREFERRED.
   Layout: keep feats as ggml [C, N] (ne0=C contiguous per voxel) so mul_mat (W[Ci,Co] x feats[Ci,N])
   works directly; or [N, C]. Match whatever dit.cpp Linear helper expects and reuse it for to_subdiv,
   mlp, from_latent, output_layer.

== ConvNeXt block (×32) ==
  conv (sparse, above) -> LayerNorm (reuse ggml_norm + affine; cast fp32) -> Linear C->4C -> SiLU
  -> Linear 4C->C -> add residual (ggml_add with the pre-conv feats). All per-voxel ops are plain
  ggml on the [N,C] feat matrix; only conv needs the neighbor map. Build one ggml graph PER STAGE
  (coords constant within a stage) like ss_decoder's run_seg pattern, feeding feats in/out as host
  buffers between stages. Neighbor map computed on CPU before building the stage graph and uploaded as
  index tensors (ggml_set_input I32).

== C2S up-block (×4) — split CPU/GGML ==
  on-graph (GGML): to_subdiv Linear C->8 ; norm1+SiLU ; conv1 (sparse C->out*8) ; conv2 (sparse out->out).
  CPU between: (1) read subdiv logits [N,8], threshold >0 -> mask; (2) compute child coords + scatter
  indices (the SparseChannel2Spatial gather: for each parent, for each set bit s, child reads feature
  sub-block s). Produce: new_coords [K,3]; a gather index list mapping child k -> (parent p, sub-block s)
  for BOTH the main branch (block size out_C, source = conv1 output [N, out_C*8]) and the skip branch
  (block size C/8 from x [N,C], then repeat_interleave to out_C). Implement the channel-block gather as
  ggml_get_rows on a RESHAPED feat tensor: reshape conv1_out [N, out_C*8] -> [out_C, 8*N] is NOT directly
  a row gather; simpler to do the [K,out_C] gather on CPU OR build a [8*N, out_C] view and get_rows with
  idx = p*8+s. Recommended: reshape conv1_out to [out_C, 8, N] (ne0=out_C), permute to [out_C, N, 8] then
  flatten last two -> rows indexed by (s*N+p) or (p*8+s); use ggml_get_rows with the CPU-built index
  [K]. Skip branch identical with block size C/8 then a repeat (ggml_repeat / reshape+concat, factor 4).
  Then norm2 (affine=False: just ggml_norm, no mul/add) + SiLU + conv2 (sparse, on NEW coords -> build
  new neighbor map for K voxels) + add skip. Output coords = new_coords; feats [K, out_C].

== Head ==
  fp32 layer_norm (no affine) + output_layer Linear 64->7. Then DOWNLOAD feats [Nf,7] + coords [Nf,3]
  to host.

== Pure-CPU geometry (no GGML): §7 + §8 ==
  Implement entirely on host in C++:
   - channel split: 2*sigmoid(f[0:3])-0.5 ; f[3:6]>0 ; softplus(f[6]).
   - mesh extraction: host hashmap (x,y,z)->idx; edge_neighbor_voxel_offset table; for each voxel & axis
     with intersected bit set, look up 4 neighbors, if all present emit quad; pick diagonal by
     split_weight product (w0*w2 vs w1*w3); emit 2 triangles. vertex pos = (coord+dual)*(1/res)-0.5.
   - emit vertices [N,3] float, faces [2L,3] int32. This is the final mesh (OBJ/GLB export downstream).

== Memory / perf notes ==
  - Neighbor map + child-gather indices are the only CPU-side combinatorics; everything else is GEMM.
  - Cache neighbor maps per spatial level (keyed like flex_gemm `SubMConv3d_neighbor_cache_3x3x3`).
  - fp16 weights: load as f16, but LayerNorm/F.layer_norm compute in fp32 (cast feats up, norm, cast
    down) exactly as LayerNorm32/ss_decoder's chln does.
  - Single batch (b=0) — ignore the batch column except to carry it through coords.
  - Reuse ss_decoder.cpp's run_seg(staged graph + gallocr) scaffolding for the per-stage GGML graphs.


## Reuse


REUSE from existing code:
- src/ss_decoder.cpp `run_seg` pattern (per-segment ggml_context + ggml_gallocr + upload/compute/download)
  — directly applicable to run one GGML graph per decoder stage. Copy the scaffold.
- src/ss_decoder.cpp `chln` LayerNorm-with-cast pattern is the template for LayerNorm32 (but here norm is
  over the channel dim of a [N,C] feature matrix, NOT a spatial grid — simpler: ggml_norm over ne0=C after
  laying feats as [C,N], then mul weight + add bias; for affine=False skip mul/add).
- src/dit.cpp Linear helper (matmul weight.T + bias) for from_latent, to_subdiv, mlp.0/mlp.2, output_layer.
- ggml_silu, ggml_sigmoid, ggml_add, ggml_mul, ggml_get_rows, ggml_mul_mat — all stock.
- Model weight loader (m.get(key)) and the f16 ckpt loading already used by ss_decoder/dit.

GENUINELY NEW (no existing equivalent):
- Submanifold SparseConv3d (k=3, stride=1): ss_decoder uses DENSE ggml_conv_3d which is unusable at
  256-512 resolution. Need host-side neighbor-map construction (hashmap over coords, 27-tap gather
  indices) + 27× get_rows/mul_mat accumulation. This is the central new kernel.
- SparseChannel2Spatial (subdivision-driven upsample): new CPU index logic (child coords + per-child
  channel-sub-block gather) feeding ggml_get_rows. No analogue exists (ss_decoder's pixel_shuffle is a
  DENSE 2x reshuffle of a full grid with NO subdivision mask — fundamentally different; the sparse C2S
  only materializes children whose to_subdiv logit > 0).
- to_subdiv thresholding + collecting `subs` for the texture decoder.
- flexible_dual_grid_to_mesh: brand-new pure-CPU geometry (hashmap neighbor lookup, quad assembly,
  diagonal split, triangle emission). Portable, no deps.

DO NOT PORT (training/postprocess only):
- FlexiDualGridVaeEncoder, SparseUnetVaeEncoder, to_latent, posterior sampling.
- SparseResBlock3d/Downsample/Upsample/S2C3d, SparseSpatial2Channel, SparseDownsample,
  SparseUpsample — none are used by this decoder config (only ConvNeXt + C2S are).
- train=True branch of flexible_dual_grid_to_mesh (quad_split_train, mid_vertices) and the no-split_weight
  cross-product diagonal branch (split_weight is always provided here = quad_lerp).
- subdivision/subs_gt spatial-cache (only built when self.training).
- Mesh.fill_holes / simplify / remove_faces (cumesh CUDA), MeshWithVoxel/grid_sample, renderers, VXZ IO,
  dual_grid.py (dataset preprocessing -> mesh_to_flexible_dual_grid, the INVERSE direction).


## Open questions


1. INPUT COORD SCALE: confirm the integer coords of the SLat handed to from_latent. trellis2_image_to_3d.py
   builds slat coords via `((hr_coords+0.5)/lr_resolution*(hr_resolution//16)).int()` then `.unique`.
   So the decoder's input grid is `resolution//16` per axis (e.g. 256//16=16). After 4× C2S (×2 each) the
   finest grid = (resolution//16)*16 = resolution. This makes voxel_size=1/resolution consistent with the
   mesh extractor. VERIFY by dumping `slat.coords.max()` and `h.coords.max()` (finest) from a reference
   run — finest max should be < resolution.

2. SUBMANIFOLD vs PADDED conv: confirm flex_gemm `sparse_submanifold_conv3d` truly keeps coords fixed
   (output set == input set) and zero-pads missing neighbors (no implicit grid growth). conv_flex_gemm.py
   asserts stride==1, padding==None => submanifold. Confirm there's no separate "generative"/inverse conv
   that adds voxels (there isn't in this decoder — growth only via C2S). Recommend dumping in/out coords
   of one conv from a reference run to confirm identical.

3. Exact final F.layer_norm eps: `F.layer_norm(feats, feats.shape[-1:])` uses default eps=1e-5 (not 1e-6
   like the affine LayerNorm32 blocks). Confirm against torch default and use 1e-5 for the pre-output norm,
   1e-6 for the in-block norms.

4. to_subdiv threshold semantics in eval: `subdiv_binarized = subdiv.feats > 0` (strictly >0). Confirm no
   sigmoid/topk gating. (Code path confirms plain >0; flag only because a wrong threshold collapses the
   mesh.) Also confirm whether ANY parent ever yields 0 children (N_leaf=0) — handle K possibly < 8*N.

5. Child-index orientation in C2S: subidx decode is `coord[i+1] += (subidx//2**i)%2` => bit0->x, bit1->y,
   bit2->z. The conv1 output channel block order must match (sub-block s read by child s). Both are the
   same `subidx`, so they're consistent BY CONSTRUCTION; but verify the channel reshape order of
   conv1_out [N, out_C*8] is (s outer, out_C inner) i.e. block s = feats[:, s*out_C:(s+1)*out_C].
   x.feats.reshape(N*8, -1) implies row-major split of the LAST dim into 8 groups of size feats_C/8 with
   s being the OUTER group index — confirm by a tiny reference dump (feed a known feat, check child feats).

6. quad corner ORDER vs winding: edge_neighbor_voxel_offset gives the 4 quad corners in a fixed cyclic
   order per axis; triangle winding (front-face normals) depends on it. If exported mesh looks inside-out,
   the offset tables / split index lists may need the documented order preserved exactly (they're quoted
   verbatim in the spec). Verify normals against a reference mesh.

