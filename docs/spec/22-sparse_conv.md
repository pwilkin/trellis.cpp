# sparse_conv

## sparse_conv: Submanifold SparseConv3d + ConvNeXt/C2S/S2C blocks (TRELLIS.2 shape & tex VAE)

This component is the sparse-3D convolution machinery used by the **FlexiDualGrid shape VAE** (and the analogous tex VAE) decoder/encoder. The backend in the checkpoint is `flex_gemm` (config.CONV='flex_gemm', see `conv/config.py`). All math below is **inference-only**; training-only branches are flagged.

### 0. Data model (SparseTensor)
A `SparseTensor` (`modules/sparse/basic.py`) is two parallel arrays:
- `coords`: int32 `[N, 4]` = `(batch, x, y, z)`. Coords in `[0,1023]`. Data for one batch is contiguous (sorted by batch). For our pipeline batch=1.
- `feats`: `[N, C]` float (fp16 in torso, fp32 at norm boundaries).
- `_scale`: a per-axis Fraction tracking voxel size; used only as cache key (we can key caches by integer resolution instead).
- `_spatial_cache`: dict keyed by `str(_scale)`; memoizes neighbor maps, downsample/upsample index maps, layout. **In a single forward this caching is just an optimization** — semantically every op recomputes from coords.

`.replace(feats)` keeps coords, swaps feats. This is the ubiquitous functional update.

### 1. SparseConv3d == submanifold 3×3×3 conv (the core kernel)
From `conv/conv.py` + `conv/conv_flex_gemm.py`: every `sp.SparseConv3d(Ci, Co, 3)` constructed in the VAE has `stride=1, padding=None` ⇒ asserts to **submanifold** conv (`flex_gemm.ops.spconv.sparse_submanifold_conv3d`). kernel_size always 3, dilation 1, bias True.

**Submanifold semantics (exact):**
- **Output coords == input coords** (same N, same `coords`). No new active voxels are created; `.replace(out)` is used (see `sparse_conv3d_forward`).
- For each output voxel `o` at coord `p=(x,y,z)`:
  `out[o] = bias + Σ_{k=0..26} W[k] · feats[ neighbor(o,k) ]`
  where the 27 kernel taps `k` enumerate offsets `(dx,dy,dz) ∈ {-1,0,1}³` (dilation=1). `neighbor(o,k)` is the index of the input voxel whose coord is `p + offset(k)` **if it is active**, else that tap contributes 0 (zero padding for missing neighbors).
- `W[k]` is a `[Co, Ci]` matrix (one per kernel tap). `bias` is `[Co]`.

**Kernel tap ↔ offset ordering.** Weight stored as `(Co, Kd, Kh, Kw, Ci)` (see below). The flat tap index over `(Kd,Kh,Kw)` with each dim ∈{0,1,2} maps to offset `(kd-1, kh-1, kw-1)`. We must match flex_gemm's axis convention: coords are `(x,y,z)` = `(coord[1],coord[2],coord[3])`; flex_gemm treats them as `(D,H,W)`-ordering consistent with the permute `(Co,Ci,Kd,Kh,Kw)->(Co,Kd,Kh,Kw,Ci)`, i.e. **Kd indexes the first spatial coord (x), Kh the second (y), Kw the third (z)**. Concretely:
```
tap t = kd*9 + kh*3 + kw,  kd,kh,kw ∈ {0,1,2}
offset = (kd-1, kh-1, kw-1) added to (x,y,z)
```
The center tap is t=13 (offset (0,0,0)) → always a self-match.

### 2. Weight tensor layout (checkpoint)
`conv_flex_gemm.sparse_conv3d_init` builds `weight` as `(Co, Ci, 3,3,3)` then **permutes to `(Co, 3,3,3, Ci)` contiguous** (line 34). So every checkpoint key `*.conv.weight` / `*.conv1.weight` / `*.conv2.weight` has shape **`[Co, Kd=3, Kh=3, Kw=3, Ci]`** (confirmed by the keys dump, e.g. `blocks.0.0.conv.weight f16 [1024,3,3,3,1024]`). Bias `[Co]`.

NOTE: this is the **same memory order** the existing dense `conv3d()` helper in `src/ss_decoder.cpp` expects after its own reshaping, but the dense helper calls `ggml_conv_3d` on a packed `[k,k,k,IC*OC]` weight — that path is NOT reusable for sparse (it does a dense spatial conv). For sparse we treat the weight as 27 separate `[Co,Ci]` GEMM matrices indexed by tap.

### 3. SparseConvNeXtBlock3d (the dominant block; `block_type`)
`sparse_unet_vae.py` L265. Composition (depthwise-free — it's a full 3×3×3 conv, NOT depthwise despite the ConvNeXt name):
```
def _forward(x):
    h = self.conv(x)                       # SparseConv3d(C, C, 3)  submanifold
    h = h.replace(self.norm(h.feats))      # LayerNorm32(C, affine=True, eps=1e-6) over channel dim
    h = h.replace(self.mlp(h.feats))       # Linear(C,4C) -> SiLU -> Linear(4C,C) [zero-init]
    return h + x                           # residual on feats (coords identical)
```
Weights per block prefix `P`:
- `P.conv.weight [C,3,3,3,C]`, `P.conv.bias [C]`
- `P.norm.weight [C]`, `P.norm.bias [C]`  (LayerNorm over last dim of feats)
- `P.mlp.0.weight [4C,C]`, `P.mlp.0.bias [4C]`
- `P.mlp.2.weight [C,4C]`, `P.mlp.2.bias [C]`
(mlp.1 = SiLU, no params. mlp_ratio=4.0.)

`LayerNorm32` = standard LayerNorm computed in fp32 over the channel dimension of `feats` (`[N,C]`, normalize over C), eps=1e-6, affine. Reuse the existing LayerNorm helper but apply it per-row over C (here feats is already `[N,C]`, so it's a plain row-wise LayerNorm — simpler than the channels-last permute dance in ss_decoder).

### 4. SparseResBlockC2S3d (decoder up-block; channel→spatial, ×2 upsample)
`sparse_unet_vae.py` L217. Used as `up_block_type` between decoder stages. Constructed with `(channels=C_in, out_channels=C_out, pred_subdiv=True)`. Internals:
- `norm1 = LayerNorm32(C_in, affine=True, eps=1e-6)`
- `norm2 = LayerNorm32(C_out, affine=False, eps=1e-6)`  ← **NO weight/bias** (elementwise_affine=False). Confirmed: keys have `norm1.{weight,bias}` but NO `norm2.*`.
- `conv1 = SparseConv3d(C_in, C_out*8, 3)` → weight `[C_out*8,3,3,3,C_in]`
- `conv2 = zero_module(SparseConv3d(C_out, C_out, 3))` → `[C_out,3,3,3,C_out]`
- `to_subdiv = SparseLinear(C_in, 8)` (pred_subdiv=True) → `[8, C_in]`, bias `[8]`
- `updown = SparseChannel2Spatial(2)`
- `skip_connection = lambda x: x.replace(x.feats.repeat_interleave(C_out // (C_in//8), dim=1))` — **no params**.

Inference `_forward` (training=False, pred_subdiv=True):
```
subdiv = to_subdiv(x)                       # SparseLinear -> feats [N,8]  (logits)
h = norm1(x.feats); h = silu(h)
h = conv1(h)                                # submanifold conv, feats now [N, C_out*8]
subdiv_bin = (subdiv.feats > 0)             # [N,8] bool
h = updown(h, subdiv_bin)                   # SparseChannel2Spatial -> new coords (×2 res), feats [M, C_out]
x = updown(x, subdiv_bin)                   # SAME index map (cache) -> feats [M, C_in]
h = norm2(h.feats); h = silu(h)             # norm2 affine=False
h = conv2(h)                                # submanifold conv on the NEW (upsampled) coords
h = h + skip_connection(x)                  # x feats repeat_interleaved to C_out
return h, subdiv                            # subdiv returned for guide; inference path discards unless return_subs
```
Key detail: **`conv1` runs on the OLD coords**, then C2S upsamples, then **`conv2` runs on the NEW coords**. Two distinct neighbor maps.

### 5. SparseChannel2Spatial(2) — the up-resampler (decoder)
`spatial/spatial2channel.py` L58. DIM=3, factor=2, so `factor**DIM = 8`.

Inference (no cache, subdivision provided as bool `[N,8]`):
```
sub = subdivision.feats                     # [N,8] bool: which of 8 octants to spawn
N_leaf = sub.sum(-1)                         # [N] number of active children per parent
subidx = sub.nonzero()[:, -1]               # [M] flat child-octant index in 0..7, M=sum(N_leaf)
new_coords = x.coords.clone(); new_coords[:,1:] *= 2
new_coords = repeat_interleave(new_coords, N_leaf)        # [M,4], parent coord ×2, repeated per child
for i in 0..2: new_coords[:, i+1] += (subidx // 2**i) % 2 # decode octant -> +{0,1} per axis
idx = repeat_interleave(arange(N), N_leaf)               # [M] parent index per child
# feature redistribution: parent feats viewed as 8 channel-blocks
x_feats = x.feats.reshape(N*8, C_out)        # x.feats was [N, C_out*8]
new_feats = x_feats[idx*8 + subidx]          # [M, C_out]  pick the octant's channel block
```
**Octant index math**: child `(x,y,z)` offset from `subidx` via `axis i gets bit (subidx>>i)&1`. i=0→x, i=1→y, i=2→z. This MUST match the channel-block ordering of conv1's `C_out*8` output (octant o occupies channels `[o*C_out : (o+1)*C_out]`).

### 6. SparseResBlockS2C3d (encoder down-block; spatial→channel, ×2 downsample)
`sparse_unet_vae.py` L179. Symmetric to C2S. `(channels=C_in, out_channels=C_out)`:
- `norm1 = LayerNorm32(C_in, affine=True)`, `norm2 = LayerNorm32(C_out, affine=False)` (no params)
- `conv1 = SparseConv3d(C_in, C_out//8, 3)` → `[C_out//8,3,3,3,C_in]`
- `conv2 = zero_module(SparseConv3d(C_out, C_out, 3))` → `[C_out,3,3,3,C_out]`
- `updown = SparseSpatial2Channel(2)`
- `skip = lambda x: x.replace(x.feats.reshape(N, C_out, C_in*8//C_out).mean(-1))` — no params.
```
h = norm1(x.feats); h = silu(h)
h = conv1(h)                                 # feats [N, C_out//8] on OLD coords
h = updown(h)                                # S2C: parent coords (÷2), feats packed to [M, C_out]
x = updown(x)
h = norm2(h.feats); h = silu(h)
h = conv2(h)                                 # on NEW (downsampled) coords
h = h + skip(x)
```
(Encoder is needed only for the offline shape-latent encoding path; at i2-3D inference time we only run the **decoder**. Flag: encoder = preprocessing/offline, decoder = runtime.)

### 7. SparseSpatial2Channel(2) — down-resampler (encoder)
`spatial2channel.py` L7. Inverse of C2S:
```
coord_parent = coords; coord_parent[1:] //= 2
subidx = sum_i (coords[1:,i] % 2) * 2**i            # octant 0..7
group voxels by parent code -> unique -> new_coords [M,4], idx [N] (parent index per child), subidx [N]
new_feats = zeros(M*8, C); new_feats[idx*8 + subidx] = x.feats; reshape -> [M, C*8]
```
Children that don't exist leave their 8-block slot = 0. Output feats `[M, C_in*8]` then conv2 maps to C_out. (For tex/shape *decoder* runtime we don't need this, but it's the exact inverse and shares the same hashmap+octant kernels.)

### 8. Shape VAE decoder full structure (the runtime target)
From `configs/scvae/shape_vae_next_dc_f16c32_fp16_ft_512.json` + keys dump:
- `model_channels = [1024, 512, 256, 128, 64]`, `latent_channels=32`, `num_blocks=[4,16,8,4,0]`, `resolution=512`, `voxel_margin=0.5`, fp16, pred_subdiv=True, out_channels=7.
- `from_latent = SparseLinear(32, 1024)`  → `from_latent.weight [1024,32]`, bias `[1024]`
- `output_layer = SparseLinear(64, 7)` → `[7,64]`, bias `[7]`
- Stage 0 (C=1024): `blocks.0.0..0.3` ConvNeXt(1024); `blocks.0.4` = C2S(1024→512): conv1 `[4096,3,3,3,1024]`, conv2 `[512,3,3,3,512]`, norm1 `[1024]`, to_subdiv `[8,1024]`.
- Stage 1 (C=512): `blocks.1.0..1.15` ConvNeXt(512); `blocks.1.16` = C2S(512→256): conv1 `[2048,3,3,3,512]`, conv2 `[256,..]`, to_subdiv `[8,512]`.
- Stage 2 (C=256): `blocks.2.0..2.7` ConvNeXt(256); `blocks.2.8` = C2S(256→128): conv1 `[1024,..,256]`, conv2 `[128,..]`, to_subdiv `[8,256]`.
- Stage 3 (C=128): `blocks.3.0..3.3` ConvNeXt(128); `blocks.3.4` = C2S(128→64): conv1 `[512,..,128]`, conv2 `[64,..]`, to_subdiv `[8,128]`.
- Stage 4 (C=64): num_blocks[4]=0 → **no blocks** (the last entry has no up-block since i<len-1 fails).

Decoder forward (`SparseUnetVaeDecoder.forward`, inference):
```
h = from_latent(x)                          # [N,32]->[N,1024]
h = h.type(fp16)
for i,stage in enumerate(blocks):
  for j,block in enumerate(stage):
    if i<4 and j==last:                      # the C2S up-block
       # pred_subdiv True, not training, guide_subs None:
       h = block(h, subdiv=None)            # pred_subdiv internally => to_subdiv(h)
    else:
       h = block(h)                          # ConvNeXt
h = h.type(fp32)
h = layer_norm(h.feats, over last dim)       # F.layer_norm, NO affine (eps default 1e-5)
h = output_layer(h)                          # [N,64]->[N,7]
```
Wait — in inference with pred_subdiv=True and guide_subs=None, the C2S block is called and returns `(h, sub)`; `forward` does `h = block(h, subdiv=guide_subs[i] if guide_subs else None)` ONLY in the `not pred_subdiv` branch. With pred_subdiv=True it takes `h, sub = block(h); subs.append(sub)`. So each up-block **predicts its own subdivision** from to_subdiv(h)>0. Final `return_subs` controls whether subs are returned.

**Final head (FlexiDualGridVaeDecoder.forward, inference, training=False):** out feats `[N,7]`:
- vertices  = `(1 + 2*voxel_margin)*sigmoid(feats[:,0:3]) - voxel_margin`  (voxel_margin=0.5 ⇒ `2*sigmoid(x)-0.5`)
- intersected = `feats[:,3:6] > 0`  (bool, 3 edges)
- quad_lerp = `softplus(feats[:,6:7])`
Then `flexible_dual_grid_to_mesh(coords, vertices, intersected, quad_lerp, aabb=[-0.5..0.5]^3, grid_size=resolution=512)` (in `o_voxel.convert`) → final mesh. The dual-grid→mesh conversion is a separate non-NN component (out of scope here, but consumes these three feats fields per active voxel at the final resolution).

### 9. Resolution / N growth
Decoder input coords = active voxels from Stage ① (e.g. at 64³, ~few-thousand voxels). Each C2S up-block ×2 the spatial resolution and grows N by `sum(to_subdiv>0)` children. Final resolution 512³ over the 4 up-blocks (64→128→256→512). N can reach up to ~1e6 (max_active_voxels). All ops are per-voxel; the only neighbor coupling is the 3×3×3 conv.

### 10. Constants summary
- kernel 3, dilation 1, pad-equivalent = drop missing neighbors (submanifold).
- factor=2, octant count 8, octant bit order axis i ← (subidx>>i)&1 (i=0→x).
- LayerNorm eps 1e-6 (block norms), 1e-5 (final F.layer_norm, default).
- mlp_ratio 4. SiLU activations. conv2 zero-initialized (just affects training; weights loaded as-is).
- subdiv threshold: `> 0` on raw logits.

## Weight key map

All keys verified against /devel/alt/trellis.cpp/docs/spec/keys/ckpts__shape_dec_next_dc_f16c32_fp16.keys.txt (292 keys) and ckpts__shape_enc_next_dc_f16c32_fp16.keys.txt (284 keys). All f16.

=== SHAPE DECODER (runtime) ===
Top-level:
  from_latent.weight [1024,32], from_latent.bias [1024]      (SparseLinear 32->1024)
  output_layer.weight [7,64], output_layer.bias [7]          (SparseLinear 64->7)
  (final F.layer_norm before output_layer has NO params)

ConvNeXtBlock3d (block prefix P, channels C): keys =
  P.conv.weight [C,3,3,3,C], P.conv.bias [C]
  P.norm.weight [C], P.norm.bias [C]
  P.mlp.0.weight [4C,C], P.mlp.0.bias [4C]
  P.mlp.2.weight [C,4C], P.mlp.2.bias [C]
  Instances:
    Stage0 C=1024: blocks.0.0 .. blocks.0.3
    Stage1 C=512 : blocks.1.0 .. blocks.1.15
    Stage2 C=256 : blocks.2.0 .. blocks.2.7
    Stage3 C=128 : blocks.3.0 .. blocks.3.3

C2S up-block (prefix P, C_in->C_out): keys =
  P.conv1.weight [C_out*8,3,3,3,C_in], P.conv1.bias [C_out*8]
  P.conv2.weight [C_out,3,3,3,C_out], P.conv2.bias [C_out]
  P.norm1.weight [C_in], P.norm1.bias [C_in]     (affine)
  P.to_subdiv.weight [8,C_in], P.to_subdiv.bias [8]
  (NO norm2.* — affine=False; NO skip_connection params — lambda repeat_interleave)
  Instances (verified):
    blocks.0.4  : conv1 [4096,3,3,3,1024], conv2 [512,3,3,3,512], norm1 [1024], to_subdiv [8,1024]   (1024->512)
    blocks.1.16 : conv1 [2048,3,3,3,512],  conv2 [256,3,3,3,256], norm1 [512],  to_subdiv [8,512]    (512->256)
    blocks.2.8  : conv1 [1024,3,3,3,256],  conv2 [128,3,3,3,128], norm1 [256],  to_subdiv [8,256]    (256->128)
    blocks.3.4  : conv1 [512,3,3,3,128],   conv2 [64,3,3,3,64],   norm1 [128],  to_subdiv [8,128]    (128->64)
  Stage4 (blocks.4.*): NONE (num_blocks=0, no up-block).
  grep confirms 0 keys containing "skip" in decoder.

=== SHAPE ENCODER (offline/preprocessing) ===
  input_layer.weight [64,6], input_layer.bias [64]    (SparseLinear 6->64; 6 = vertices(3)+intersected(3) per FlexiDualGridVaeEncoder)
  to_latent.weight [64,1024]? -> dump shows to_latent.weight [64,1024], to_latent.bias [64]
     NOTE: encoder model_channels=[64,128,256,512,1024]; to_latent = SparseLinear(model_channels[-1]=1024, 2*latent=64). So weight is [64,1024] (out=64=2*32). Confirmed.
  (final F.layer_norm before to_latent: no params)
  ConvNeXt blocks at C per stage: blocks.0.0 (C=64, but num_blocks[0]=0 so blocks.0.0 is actually the S2C down-block, see below);
  S2C down-block (prefix P, C_in->C_out): keys =
    P.conv1.weight [C_out//8,3,3,3,C_in], P.conv1.bias [C_out//8]
    P.conv2.weight [C_out,3,3,3,C_out], P.conv2.bias [C_out]
    P.norm1.weight [C_in], P.norm1.bias [C_in]   (NO norm2, NO skip params, NO to_subdiv)
    Instances (verified):
      blocks.0.0 : conv1 [16,3,3,3,64], conv2 [128,3,3,3,128], norm1 [64]     (64->128, C_out//8=16)
      blocks.1.4 : conv1 [32,3,3,3,128], conv2 [256,3,3,3,256], norm1 [128]   (128->256, C_out//8=32)
      (further stages 256->512, 512->1024 analogously at blocks.2.8, blocks.3.16)
  Encoder num_blocks=[0,4,8,16,4]: stage0 has 0 ConvNeXt + the down-block at blocks.0.0; etc.

=== conv weight layout (ALL conv*.weight) ===
  shape [Co, 3, 3, 3, Ci] = (Co, Kd, Kh, Kw, Ci), contiguous, from permute(0,2,3,4,1) of nn (Co,Ci,3,3,3).
  In GGML/ggml ne order (reversed): ne = [Ci, 3, 3, 3, Co]? — ggml stores reversed, so a torch [Co,3,3,3,Ci] tensor loaded by the existing loader has ne0=Ci (fastest), ne1=Kw, ne2=Kh, ne3=Kd, and Co as the 5th dim. Since ggml is 4D max, load as ne=[Ci, 27, Co] or [Ci, 3*3*3, Co] (flatten the 3 kernel dims) — see ggml plan.

The tex VAE (tex_dec_next_dc_f16c32_fp16.keys.txt) has the IDENTICAL structure (SparseUnetVaeDecoder, same block_type/up_block_type lists); only out_channels and in/latent dims differ — same kernels apply.

## GGML plan

This component CANNOT reuse the dense `ggml_conv_3d` path in src/ss_decoder.cpp — that does a dense spatial conv over a full grid. Sparse submanifold conv must be implemented as: coord hashmap → [N,27] neighbor index table → 27 indexed GEMMs accumulated. The norm/linear/mlp parts CAN reuse existing helpers.

### Custom CPU/host kernels (precompute per conv resolution, done on host in C++, not in the ggml graph)
1. **Coord hashmap build**: hash `coords[i] = (x,y,z)` (batch fixed=1) into an open-addressing or std::unordered_map<uint64_t,int32> keyed by `(x<<40)|(y<<20)|z` → row index. O(N).
2. **Neighbor table `nbr [N,27]` (int32)**: for each voxel i, for each tap t=0..26 with offset `(kd-1,kh-1,kw-1)`, look up `(x+dx,y+dy,z+dz)`; store row index or -1 if absent. Center tap t=13 == i. Cache per resolution (matches flex_gemm neighbor_cache). This is the "[N,27] neighbor gather" kernel.
3. **Octant index maps** for C2S/S2C: given subdiv-bool `[N,8]` (computed from `to_subdiv` logits>0 in the graph, read back to host), build `new_coords [M,4]`, parent-index `idx [M]`, octant `subidx [M]` per Section 5/7. Pure host integer code.

### In-graph ggml ops
**Submanifold conv (per conv layer):**
- Load weight as ggml tensor `W` with ne = `[Ci, 27, Co]` (reinterpret the contiguous `[Co,3,3,3,Ci]` torch buffer: torch-contiguous flat index = ((((co*3+kd)*3+kh)*3+kw)*Ci+ci); flatten (kd,kh,kw)->t gives [Co,27,Ci] torch ⇒ ggml ne=[Ci,27,Co]). bias ne=[Co].
- For each tap t in 0..26:
  - `nbr_t` = column t of nbr table = int32 `[N]` ggml tensor (input). Replace -1 with a sentinel row (append a zero row to feats so -1 maps to it; or mask).
  - `gathered_t = ggml_get_rows(feats, nbr_t)` → `[Ci, N]` (ggml: feats stored ne=[Ci,N], get_rows picks rows). For missing neighbors point at an appended zero row → contributes 0.
  - `Wt = ggml_view` of W at tap t → `[Ci, Co]`; `acc += ggml_mul_mat(Wt, gathered_t)` → `[Co, N]`.
  - This is the "get_rows + matmul accumulate" pattern, 27 times. Sum via ggml_add chain.
- Add bias (reshape `[Co,1]`, broadcast). Output feats `[Co,N]`.
  - Optimization: build one big `[Ci*27, N]` gathered matrix and a single `[Ci*27, Co]` matmul (im2col-style). Preferred: gather all 27 into a contiguous `[27*Ci, N]` then one ggml_mul_mat with W reshaped `[27*Ci, Co]`. This matches flex_gemm "implicit_gemm" and is far faster. Missing neighbors = zero rows.

**ConvNeXt block:** conv (above) → row-wise LayerNorm over Ci (feats ne=[C,N]: ggml_norm normalizes ne0=C — exactly what's needed; affine mul/add with `[C]`) → Linear(C,4C) (ggml_mul_mat) → ggml_silu → Linear(4C,C) → ggml_add residual. Reuse existing Linear + LayerNorm helpers from src/dit.cpp (feats already channel-major ne0=C, so NO permute needed — simpler than ss_decoder's chln).

**to_subdiv:** Linear(C,8) → feats `[8,N]` → read back to host → bool = (>0). (Linear via ggml_mul_mat.)

**C2S/S2C feature redistribution:** after octant maps built on host, do gather with ggml_get_rows on a reshaped feats view (`[C_out, N*8]` viewed from `[C_out*8, N]`) using index `idx*8+subidx`. skip_connection: C2S repeat_interleave (host index gather or ggml_get_rows with a repeat index), S2C reshape+mean (ggml_reshape + ggml_mean over the redundancy axis).

### Graph segmentation
Like ss_decoder.cpp's run_seg pattern: run each conv as its own subgraph OR (better) one subgraph per stage, reading feats back to host only at C2S boundaries where new coords/N must be computed (subdiv depends on to_subdiv output). Pipeline: [from_latent] → stage0 convnext×4 (one graph) → [to_subdiv readback + host C2S map build] → conv1 graph → C2S gather → conv2 graph → ... repeat. Keep neighbor tables on host, upload as int32 ggml inputs.

### Memory notes
- feats fp16 in torso, fp32 for norms (cast around ggml_norm; or just run fp32 — N is large but C modest). Final layer_norm + output_layer in fp32.
- N up to ~1e6 at 512³, C down to 64 there → feats ~256MB fp32; intermediate stages smaller N. nbr table [N,27] int32 ~108MB at peak. Manageable; build per-resolution and free.
- The im2col gather `[27*Ci, N]` at stage0 (Ci=1024, N small ~thousands) is fine; at stage3 (Ci=128, N~1e6) it's 27*128*N*2B fp16 ≈ 7GB — TOO big. Use the per-tap accumulate (27 get_rows+mul_mat, no materialized im2col) at high-N stages, or tile N into chunks. Recommend per-tap accumulate as the default, im2col only for low-N stage0/1.

## Reuse

REUSE from existing code:
- src/dit.cpp Linear helper (ggml_mul_mat + bias): directly usable for from_latent, output_layer, to_subdiv, mlp.0/mlp.2, and as the GEMM primitive inside the submanifold conv.
- src/dit.cpp LayerNorm helper: usable for the ConvNeXt P.norm (affine) and the C2S/S2C norm1 (affine) and norm2 (affine=False → skip the mul/add). feats are ne0=C row-major so ggml_norm applies over C directly — SIMPLER than ss_decoder.cpp's chln() which needs permutes because it holds data channels-LAST. Do NOT reuse chln(); write a trivial row-norm.
- ggml_silu: builtin, used in ConvNeXt mlp and C2S/S2C activations.
- The run_seg/gallocr subgraph harness pattern in src/ss_decoder.cpp is a good template for chunking the decode into per-stage graphs with host-side readback at C2S boundaries.
- npy.h / model loader (Model::get) for weight access by key.

GENUINELY NEW (no existing analog):
- Coord hashmap (uint64 pack of x,y,z -> row idx). Nothing in the repo builds a sparse coord index.
- [N,27] neighbor-gather table builder. New.
- Submanifold conv as get_rows+mul_mat accumulate over 27 taps. The existing conv3d() in ss_decoder.cpp is DENSE (ggml_conv_3d) and is the wrong primitive — must NOT be reused for sparse.
- SparseChannel2Spatial / SparseSpatial2Channel octant index math (new_coords, idx, subidx) on host. New.
- to_subdiv-driven dynamic N growth (variable output size per up-block). New control flow; ss_decoder's fixed dense pixel_shuffle (pixel_shuffle()) is the dense analog but assumes ALL 8 children exist — sparse keeps only subdiv>0 children, so pixel_shuffle is NOT reusable.
- Final FlexiDualGrid head (sigmoid/threshold/softplus split of 7 channels) + flexible_dual_grid_to_mesh — the mesh extraction is a separate component (o_voxel.convert), out of scope here but consumes this component's [N,7] output.

## Open questions

1. flex_gemm tap-axis convention: I inferred tap t=kd*9+kh*3+kw with kd→x(coord[1]), kh→y, kw→z and offset=(kd-1,kh-1,kw-1), based on the permute (Co,Ci,Kd,Kh,Kw)->(Co,Kd,Kh,Kw,Ci) and coords=(b,x,y,z). flex_gemm itself is not installed (find found no flex_gemm package), so the exact (D,H,W)↔(x,y,z) mapping inside sparse_submanifold_conv3d is not source-verified. Because submanifold conv is symmetric in offsets and the SAME convention is used for build+load, a consistent self-chosen ordering will be correct ONLY IF it matches how the checkpoint weights were trained. RECOMMEND: dump one reference conv (e.g. blocks.3.0.conv on a tiny known sparse input) from a working Python install of TRELLIS.2 to validate tap↔offset and octant-bit↔axis ordering before trusting the C++ port. This is the single highest-risk ambiguity.

2. C2S octant channel-block layout: I assumed conv1's C_out*8 output is laid out as 8 contiguous C_out-blocks with octant o = bit-decoded (o>>i)&1 for axis i, matching SparseChannel2Spatial's `x_feats.reshape(N*8, C_out)` + `idx*8+subidx`. The reshape implies octant index = the SECOND-fastest grouping (feats[:, o*C_out:(o+1)*C_out]); confirm subidx bit→axis order (i=0→x) matches the conv weight's spatial-tap training. Tie to the same reference dump as (1).

3. Final F.layer_norm eps: code uses `F.layer_norm(h.feats, h.feats.shape[-1:])` with default eps=1e-5. Block LayerNorm32 uses eps=1e-6. Confirmed from source; flagged so they're not unified by mistake.

4. Whether runtime needs the ENCODER at all: at i2-3D inference the decoder consumes a sampled SLat latent (from the SLat flow DiT), so the sparse ENCODER (S2C blocks) is offline/preprocessing only. Confirm against the inference entrypoint (example.py pipeline) — but the keys are present so I documented both; encoder kernels are the exact inverse and share the hashmap/octant code.

5. guide_subs path: decoder.forward supports guide_subs only when pred_subdiv=False; the shipped shape decoder has pred_subdiv=True, so each up-block self-predicts subdivision. Confirm the inference pipeline does NOT pass guide_subs (it cannot, given pred_subdiv=True asserts). I assumed self-prediction.
