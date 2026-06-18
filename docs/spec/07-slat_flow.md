# slat_flow

## SLatFlowModel (Structured Latent Flow / sparse DiT denoiser)

Source: `/tmp/TRELLIS.2/trellis2/models/structured_latent_flow.py`. Two checkpoints both use class `ElasticSLatFlowModel` (subclass of `SLatFlowModel` via `SparseTransformerElasticMixin`; the mixin is training-only memory management — at inference it behaves identically to `SLatFlowModel`, so port `SLatFlowModel.forward` exactly).

### 0. Config values (from `configs/gen/slat_flow_*_512_bf16.json`)
Common to BOTH variants:
- `resolution=32`, `model_channels=1536`, `cond_channels=1024`, `out_channels=32`
- `num_blocks=30`, `num_heads=12` (so head_dim = 1536/12 = **128**), `mlp_ratio=5.3334` (MLP hidden = `int(1536*5.3334)` = **8192**)
- `pe_mode="rope"`, `share_mod=true`, `initialization="scaled"`, `qk_rms_norm=true`, `qk_rms_norm_cross=true`
- `dtype` runtime = bfloat16 (`convert_to` casts only `self.blocks`; t_embedder/input/out layers stay fp32 unless cast). `num_head_channels` default 64 is overridden by explicit `num_heads=12`.

Variant differences:
- **img2shape**: `in_channels=32`. No concat cond. `out=32` (shape SLat).
- **imgshape2tex**: `in_channels=64`. The model receives `x` (noise, 32ch) and `concat_cond` (shape latent, 32ch); they are concatenated channel-wise -> 64. `out=32` (texture/PBR SLat).

### 1. Data model: SparseTensor
A `SparseTensor` (subclass of `VarLenTensor`) holds:
- `coords`: int tensor `[N, 4]` = `[batch_idx, x, y, z]` (xyz are voxel integer coords in [0, resolution)). Active voxels only (N = total active voxels across the batch).
- `feats`: `[N, C]` float tensor, row-aligned with coords.
- `layout`: list of `slice` objects, one per batch element, computed from `coords[:,0]` via `bincount`+`cumsum`. Rows of each batch are contiguous (asserted). This defines per-batch token ranges for block-diagonal attention.
- `_spatial_cache`: dict keyed by scale; caches `layout`, `seqlen`, and the RoPE phases (key `rope_phase_3d_freq1.0-10000.0_hd128`).
- `shape` = `[B, *feats.shape[1:]]` where B = `coords[:,0].max()+1`.

`x.replace(new_feats)` returns a new SparseTensor with same coords/layout/cache but new feats. All elementwise/Linear/Norm ops operate on `feats` (the `[N, C]` matrix) and keep coords fixed.

`VarLenTensor` (cond) is the same minus coords: just `feats [T, ...]` + `layout`. `VarLenTensor.from_tensor_list([t0,t1,...])` concatenates along dim 0 and records per-element slices.

### 2. Submodules and forward order

#### 2.1 Constructor (`__init__`)
```
self.t_embedder = TimestepEmbedder(1536)              # frequency_embedding_size=256
if share_mod:
    self.adaLN_modulation = nn.Sequential(SiLU(), Linear(1536, 6*1536=9216, bias=True))
# pe_mode=="rope" -> NO self.pos_embedder created (ape branch only)
self.input_layer = SparseLinear(in_channels, 1536)    # 32->1536 or 64->1536
self.blocks = ModuleList([ModulatedSparseTransformerCrossBlock(...) x 30])
self.out_layer = SparseLinear(1536, 32)
```

#### 2.2 TimestepEmbedder (`models/sparse_structure_flow.py`)
- `timestep_embedding(t, dim=256, max_period=10000)`: sinusoidal. `half=128`; `freqs = exp(-ln(10000)*arange(128)/128)`; `args = t[:,None]*freqs[None]`; `emb = cat([cos(args), sin(args)], -1)` -> `[B, 256]`.
- `mlp = Sequential(Linear(256,1536), SiLU(), Linear(1536,1536))` -> `t_emb [B,1536]`.

#### 2.3 forward(x, t, cond, concat_cond=None)
```
1. if concat_cond is not None: x = sp.sparse_cat([x, concat_cond], dim=-1)   # feats cat on channel dim -> [N,64]
2. if cond is list: cond = VarLenTensor.from_tensor_list(cond)                # DINOv3 tokens, var len per batch
3. h = self.input_layer(x)                                                    # SparseLinear -> feats [N,1536]
4. h = manual_cast(h, dtype)                                                  # -> bf16
5. t_emb = self.t_embedder(t)                                                 # [B,1536]
6. if share_mod: t_emb = self.adaLN_modulation(t_emb)                         # [B,9216]
7. t_emb = manual_cast(t_emb, dtype); cond = manual_cast(cond, dtype)
8. # pe_mode=="rope": NO ape addition here (the "if pe_mode=='ape'" block skipped)
9. for block in self.blocks: h = block(h, t_emb, cond)
10. h = manual_cast(h, x.dtype)                                               # back to fp32
11. h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))                  # FINAL LayerNorm, NO affine, eps default 1e-5
12. h = self.out_layer(h)                                                     # SparseLinear 1536->32
return h   # SparseTensor, feats [N,32], same coords as input x
```
Note the final non-affine `F.layer_norm` over last dim (1536) before out_layer. eps is PyTorch default 1e-5 (not 1e-6).

### 3. ModulatedSparseTransformerCrossBlock (the 30 blocks)
File `modules/sparse/transformer/modulated.py`. With `share_mod=True`:
- `self.modulation = nn.Parameter(randn(6*1536=9216)/1536**0.5)`  (NO per-block adaLN_modulation Sequential when share_mod)
- Norms: `norm1=LayerNorm32(1536, affine=False, eps=1e-6)`, `norm2=LayerNorm32(1536, affine=TRUE, eps=1e-6)`, `norm3=LayerNorm32(1536, affine=False, eps=1e-6)`. **norm2 has learnable weight+bias; norm1/norm3 do not.**
- `self_attn = SparseMultiHeadAttention(1536, num_heads=12, type="self", attn_mode="full", qkv_bias=True, use_rope=True, rope_freq=(1.0,10000.0), qk_rms_norm=True)`
- `cross_attn = SparseMultiHeadAttention(1536, ctx_channels=1024, num_heads=12, type="cross", attn_mode="full", qkv_bias=True, qk_rms_norm_cross=True)`  (no rope)
- `mlp = SparseFeedForwardNet(1536, mlp_ratio=5.3334)`

forward `_forward(x, mod, context)` with `mod = t_emb [B,9216]`:
```
shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp = (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
   # self.modulation [9216] broadcasts over batch -> [B,9216], chunk -> six [B,1536]
h = x.replace(norm1(x.feats))                 # LN over 1536, no affine
h = h * (1 + scale_msa) + shift_msa           # per-BATCH modulation broadcast to per-voxel (see 3.1)
h = self.self_attn(h)                         # full self-attn over voxels of same batch + 3D RoPE
h = h * gate_msa
x = x + h
h = x.replace(norm2(x.feats))                 # LN over 1536, WITH affine
h = self.cross_attn(h, context)              # cross-attn to DINOv3 tokens (context)
x = x + h
h = x.replace(norm3(x.feats))                 # LN over 1536, no affine
h = h * (1 + scale_mlp) + shift_mlp
h = self.mlp(h)
h = h * gate_mlp
x = x + h
return x
```

#### 3.1 Modulation broadcast (CRITICAL for the port)
`h * (1+scale) + shift` and `h * gate`: `h` is a SparseTensor (feats `[N,1536]`), the modulation operand is a dense tensor `[B,1536]`. In `SparseTensor.__elemwise__` (basic.py:717): `other = broadcast_to(other, self.shape=[B,1536]); other = other[self.batch_boardcast_map]`. `batch_boardcast_map` (basic.py:562) = `repeat_interleave(arange(B), seqlen)` -> `[N]` mapping each voxel to its batch row. So scale/shift/gate are gathered per-voxel by batch index. In the C++ port: for each voxel i, use modulation params of its batch. With B=1 (typical inference), this is just a single broadcast vector applied to all N voxels.

### 4. SparseMultiHeadAttention (`modules/sparse/attention/modules.py`)
head_dim = 1536/12 = 128.

#### 4.1 Self-attention (type="self", full, rope, qk_rms_norm)
- `self.to_qkv = Linear(1536, 4608, bias=True)`  (3*1536)
- `qk_rms_norm`: `self.q_rms_norm = SparseMultiHeadRMSNorm(128, 12)`, `self.k_rms_norm = SparseMultiHeadRMSNorm(128, 12)`
- `self.rope = SparseRotaryPositionEmbedder(head_dim=128, dim=3, rope_freq=(1.0,10000.0))`
- `self.to_out = Linear(1536, 1536)` (bias=True, default)

forward:
```
qkv = to_qkv(x.feats)                                  # [N,4608]
qkv = _fused_pre(qkv, num_fused=3)                     # reshape feats [N,4608]->[N,3,12,128] (VarLenTensor)
# qk_rms_norm OR use_rope branch taken:
q,k,v = qkv.unbind(dim=-3)                             # three [N,12,128]
q = q_rms_norm(q); k = k_rms_norm(k)                   # RMSNorm per head
q,k = rope(q,k)                                        # 3D RoPE using x.coords xyz
qkv = qkv.replace(stack([q.feats,k.feats,v.feats], dim=1))   # [N,3,12,128]
h = sparse_scaled_dot_product_attention(qkv)          # block-diagonal over batches
h = _reshape_chs(h, (-1,))                             # [N,12,128]->[N,1536]
h = to_out(h)
```
Attention is FULL but block-diagonal per batch: `q_seqlen[i] = layout[i].stop-start`; attention computed independently within each batch's voxel set (xformers `BlockDiagonalMask`/flash_attn varlen with `cu_seqlens`). For the C++ port with B=1 this is plain dense attention over all N voxels. Scale = 1/sqrt(128) (SDPA default). No causal mask.

#### 4.2 SparseMultiHeadRMSNorm
```
scale = dim**0.5 = sqrt(128)
gamma = Parameter(ones(heads=12, dim=128))            # shape [12,128]
forward(x): x=x.float(); x = normalize(x, dim=-1) * gamma * scale; return x.to(orig_dtype)
```
i.e. per-head L2-normalize the 128-dim vector, then multiply by per-(head,channel) gamma and by sqrt(128). Operates on `[N,12,128]`.

#### 4.3 SparseRotaryPositionEmbedder — 3D RoPE (`attention/rope.py`)
Constructor (head_dim=128, dim=3, rope_freq=(1.0,10000.0)):
```
freq_dim = head_dim//2//dim = 128//2//3 = 21
freqs = arange(21)/21                                  # [21], floats 0..20/21
freqs = rope_freq[0] / (rope_freq[1] ** freqs) = 1.0 / 10000**freqs   # [21]
```
forward(q,k) — q,k are VarLenTensor feats `[N,12,128]`, coords from `q.coords` `[N,4]`:
```
coords = q.coords[...,1:]                              # xyz [N,3]
phases = _get_phases(coords.reshape(-1)).reshape(N,3,-1)   # outer(coords_flat, freqs) -> [N*3,21] -> polar(1,phase) complex -> [N,3,21]
# concat axes: phases is [N, 3*21=63] complex after reshape(N,3,21)->reshape to [N,63]? 
#   Actually reshape(*coords.shape[:-1], -1) => [N, 3*21=63] complex
if 63 < head_dim//2 (=64): pad with polar(ones, zeros)=1+0j to length 64 -> [N,64] complex
phases cached in spatial cache.
_rotary_embedding(x[N,12,128], phases[N,64]):
   x_complex = view_as_complex(x.float().reshape(N,12,64,2))   # [N,12,64] complex
   x_rotated = x_complex * phases.unsqueeze(-2)                # phases[N,1,64]
   x_embed = view_as_real(x_rotated).reshape(N,12,128).to(dtype)
return q_embed, k_embed
```
Concretely: head dim 128 = 64 complex pairs. The first 63 complex pairs get rotated by angle `coord_axis * freq`: pairs 0..20 use x-coord with freqs[0..20], pairs 21..41 use y-coord, pairs 42..62 use z-coord; pair 63 is unrotated (phase 0, identity, the padding). Rotation per complex pair (a,b) with angle θ: `(a*cosθ - b*sinθ, a*sinθ + b*cosθ)`. The phases depend ONLY on integer voxel xyz coords and are shared across q and k and all 12 heads (broadcast over head dim). Cache key `rope_phase_3d_freq1.0-10000.0_hd128`.

#### 4.4 Cross-attention (type="cross", full, qk_rms_norm_cross, no rope)
- `self.to_q = Linear(1536, 1536, bias=True)`
- `self.to_kv = Linear(ctx_channels=1024, 3072, bias=True)`  (2*1536)
- `q_rms_norm/k_rms_norm = SparseMultiHeadRMSNorm(128,12)` (because qk_rms_norm_cross=True)
- `self.to_out = Linear(1536, 1536)`

forward(x, context):  context = cond VarLenTensor (DINOv3 tokens, ctx_channels=1024)
```
q = to_q(x.feats); q = reshape -> [N,12,128]
kv = to_kv(context.feats); kv = _fused_pre(num_fused=2) -> [T,2,12,128]   (T=total cond tokens)
q = q_rms_norm(q); k,v = kv.unbind(-3); k = k_rms_norm(k)
h = sparse_scaled_dot_product_attention(q, k, v)   # 3-arg form, block-diagonal: q grouped by voxel-batch, k/v by cond-batch
h = reshape [N,1536]; h = to_out(h)
```
Cross-attention pairs each batch's voxels (queries) with that batch's DINOv3 tokens (keys/values), block-diagonal by batch. For B=1 inference: dense attention, N queries x T cond tokens. Scale 1/sqrt(128).

### 5. SparseFeedForwardNet (`transformer/blocks.py`)
```
mlp = Sequential(
   SparseLinear(1536, 8192),       # int(1536*5.3334)=8192
   SparseGELU(approximate="tanh"), # tanh-approx GELU
   SparseLinear(8192, 1536),
)
```
Applied to feats `[N,1536]`.

### 6. LayerNorm32 (`modules/norm.py`)
Casts input to fp32, runs nn.LayerNorm, casts back. norm1/norm3 in blocks: affine=False (no weight/bias). norm2: affine=True. eps=1e-6 for all block norms. Final forward LN (step 11) is `F.layer_norm` with eps=1e-5 default, no affine.

### 7. Conditioning input (DINOv3)
`cond` is a list of `[T_i, 1024]` tensors (one per batch image; DINOv3 ViT-L/16 patch tokens, image_size 512) -> packed to VarLenTensor. Cross-attn projects 1024->K/V. Classifier-free guidance is applied OUTSIDE this module (trainer p_uncond=0.1); the denoiser itself is unconditional of CFG. The C++ pipeline must run the model twice (cond + uncond/empty tokens) and combine, unless guidance handled elsewhere.

### 8. Inference vs training
- Training-only: `initialize_weights`, `ElasticSLatFlowModel`/`SparseTransformerElasticMixin`, `use_checkpoint` gradient checkpointing (`_forward` vs `forward`), the `initialization` branches. Port only `_forward` paths.
- Inference: forward as in section 2.3. dtype bf16 in blocks, fp32 elsewhere; attention via flash/xformers varlen (port as dense block-diagonal SDPA).

## Weight key patterns

Top-level (SLatFlowModel root; checkpoint may be under EMA but key suffixes are these):

t_embedder.mlp.0.weight  [1536,256], t_embedder.mlp.0.bias [1536]
t_embedder.mlp.2.weight  [1536,1536], t_embedder.mlp.2.bias [1536]
adaLN_modulation.1.weight [9216,1536], adaLN_modulation.1.bias [9216]   (share_mod=True; index 1 because [0]=SiLU, [1]=Linear)
input_layer.weight  [1536, in_channels]  (in=32 img2shape, in=64 imgshape2tex), input_layer.bias [1536]
out_layer.weight  [32,1536], out_layer.bias [32]

Per block i in 0..29 (prefix "blocks.{i}."):
  modulation                         [9216]          (nn.Parameter, share_mod=True; NO blocks.{i}.adaLN_modulation.* keys)
  norm1: NONE (elementwise_affine=False -> no weight/bias)
  norm2.weight [1536], norm2.bias [1536]            (elementwise_affine=True)
  norm3: NONE
  self_attn.to_qkv.weight [4608,1536], self_attn.to_qkv.bias [4608]
  self_attn.to_out.weight [1536,1536], self_attn.to_out.bias [1536]
  self_attn.q_rms_norm.gamma [12,128]
  self_attn.k_rms_norm.gamma [12,128]
  (self_attn.rope has NO learnable params; freqs is a non-persistent runtime tensor, not in state_dict)
  cross_attn.to_q.weight [1536,1536], cross_attn.to_q.bias [1536]
  cross_attn.to_kv.weight [3072,1024], cross_attn.to_kv.bias [3072]
  cross_attn.to_out.weight [1536,1536], cross_attn.to_out.bias [1536]
  cross_attn.q_rms_norm.gamma [12,128]
  cross_attn.k_rms_norm.gamma [12,128]
  mlp.mlp.0.weight [8192,1536], mlp.mlp.0.bias [8192]
  mlp.mlp.2.weight [1536,8192], mlp.mlp.2.bias [1536]
  (mlp.mlp.1 = GELU, no params)

Notes:
- SparseLinear is nn.Linear subclass -> standard .weight/.bias keys.
- Linear weights stored [out,in] (PyTorch); GGML ggml_mul_mat expects the weight as-is for x*W^T.
- to_qkv produces fused QKV; split as q=[0:1536],k=[1536:3072],v=[3072:4608] of output, then reshape each [...,12,128]. to_kv splits k=[0:1536],v=[1536:3072].
- If checkpoint nests under "denoiser." or "models.denoiser." or an EMA dict ("ema."/"module."), strip that prefix; the leaf structure above is authoritative.

## GGML notes

Per-op mapping to GGML:

- input_layer / out_layer / all to_qkv,to_q,to_kv,to_out, mlp linears, t_embedder, adaLN_modulation: plain ggml_mul_mat + ggml_add (bias). Operate on the dense feats matrix [N,C]; coords are metadata only, never touched by these ops. SparseLinear == ordinary matmul on feats.

- sparse_cat(dim=-1) for imgshape2tex: ggml_concat of feats along channel dim (x noise [N,32] || shape latent [N,32] -> [N,64]); coords identical so just concat feats. Caller must ensure both SparseTensors share identical coords ordering.

- LayerNorm: ggml_norm (eps 1e-6 blocks, 1e-5 final) over last dim 1536; norm2 multiply by weight + add bias; norm1/norm3/final have no affine. Compute in fp32.

- TimestepEmbedder sinusoidal: precompute freqs[128] on host; build [B,256] = concat(cos,sin); then 2 matmuls + SiLU (ggml_silu). For inference t is a scalar/per-batch, trivial.

- adaLN modulation broadcast: modulation[9216] + t_emb_proj[B,9216] -> chunk into 6 x [B,1536]. The per-voxel gather (other[batch_boardcast_map]) = ggml gather/repeat by batch index. For B=1 this is a simple broadcast of a [1536] vector across all N rows (ggml_add with broadcasting / ggml_repeat). For B>1 needs a gather op indexed by per-voxel batch id (build an [N] index, ggml_get_rows on the [B,1536] modulation chunks). Recommend implementing for B=1 first.

- SparseMultiHeadRMSNorm (qk_rms_norm): per-head over 128 dims. = ggml_rms_norm-like but it's F.normalize (L2 normalize, NOT mean-subtracted RMS): out = x / ||x||_2 * gamma * sqrt(128). ggml_rms_norm computes x/sqrt(mean(x^2)+eps) = x/(||x||/sqrt(D))= x*sqrt(D)/||x||, which equals F.normalize*sqrt(D). So with D=128: ggml_rms_norm(x, eps~0) gives x*sqrt(128)/||x|| = normalize(x)*sqrt(128). Then multiply by gamma[12,128]. So: ggml_rms_norm over last dim with eps≈0 (or small) THEN ggml_mul by gamma. Reshape feats to [N,12,128] first. Confirm eps: PyTorch F.normalize uses eps=1e-12; ggml_rms_norm default eps differs — set eps very small (1e-12) to match. The *scale=sqrt(128) is absorbed since ggml_rms_norm already includes sqrt(D); just ensure no double-counting (do NOT also multiply by sqrt(128) after ggml_rms_norm — gamma multiply only).
  CAUTION: verify ggml_rms_norm normalization constant matches sqrt(D)/||x|| exactly; if your ggml uses x/sqrt(mean(x^2)+eps), that = x*sqrt(D)/sqrt(sum(x^2)) = normalize(x)*sqrt(D) — correct.

- 3D RoPE (CUSTOM op needed): phases depend on integer voxel coords. Precompute per-voxel on host or a custom kernel: for each voxel, angle[axis,j] = coord[axis] * freqs[j], freqs[j]=1/10000^(j/21), j=0..20, axes x,y,z -> 63 complex pairs + 1 identity pair = 64 pairs for head_dim 128. Build cos/sin tables [N,64] (host-side from coords, cheap). Apply rotary to q,k [N,12,128] viewing as 64 (a,b) pairs: a'=a*cos-b*sin, b'=a*sin+b*cos, broadcast cos/sin over 12 heads. This is a standard ggml rope-pair rotation but with NON-contiguous, coord-derived angles — easiest as a custom op (ggml_map_custom) or precompute cos/sin [N,128] and do ggml_mul + a paired-swap (ggml_view/ggml_concat to build rotated-half) like NeoX rope. Pair layout is interleaved (reshape(...,64,2)) i.e. real,imag adjacent — GGML_ROPE_TYPE default (not NEOX) interleaving. Implement as custom map op for clarity.

- Attention (self & cross): standard GGML attention (ggml_mul_mat QK^T, scale 1/sqrt(128), softmax, *V). For B=1 it is fully dense over N tokens (self) or N x T (cross) — maps directly to flash/standard attention, NO sparse op. For B>1, block-diagonal per batch (mask cross-batch). Build per-batch attention or an additive mask [N,N] (-inf off-diagonal-block). Confirmed: attention is dense-over-gathered-tokens, NOT windowed/serialized (attn_mode="full", no Hilbert/Morton serialization in this model). Cross-attn keys/values come from DINOv3 cond [T,1024]->kv. head_dim=128.

- GELU: tanh-approx (ggml_gelu = tanh approximation in ggml — matches approximate="tanh").

- Memory: N = active voxels, up to max_tokens 8192 per sample. Feats [N,1536] bf16. QKV [N,4608]. MLP hidden [N,8192]. RoPE cos/sin [N,128] fp32. cond tokens T (DINOv3 ViT-L/16 @512 = ~1024 tokens +regs). All modest; single GPU buffer.

Custom ops required: (1) 3D coord-based RoPE phase generation + application; (2) per-voxel batch-gather for modulation when B>1; (3) block-diagonal attention mask when B>1. Everything else is standard ggml primitives.

## Open questions

1. CFG / guidance: this module is the bare denoiser; classifier-free guidance and the flow-matching sampler/scheduler (sigma_min=1e-5, uniform t) live in the trainer/sampler, NOT here. The C++ port must implement the sampler separately and decide cond vs uncond passes. Need to read the sampler/pipeline (`trellis2/pipelines/*` and trainer flow-matching) to confirm how `t` is scaled (0..1 vs 0..1000) and how empty/uncond cond tokens are formed.

2. SLat normalization: feats are normalized with per-channel mean/std (the JSON "normalization"/"shape_slat_normalization"/"pbr_slat_normalization" arrays, 32 values each). This (de)normalization happens around the VAE/dataset, not inside SLatFlowModel. The port must apply: input latents normalized (x-mean)/std before the flow model, output denormalized. For imgshape2tex the concat shape latent uses shape_slat_normalization, the texture target uses pbr_slat_normalization. Confirm exact placement by reading the pipeline.

3. RoPE pair interleaving: code uses view_as_complex(reshape(...,-1,2)) => adjacent (real,imag) pairs (interleaved), NOT split-half (NeoX). Port must use interleaved rotation. Double-check against any ggml rope mode you reuse.

4. F.normalize eps in RMSNorm (1e-12) vs ggml_rms_norm eps — verify numerically equal enough; pick eps so x*sqrt(D)/sqrt(sum x^2+eps*D)≈normalize*sqrt(D).

5. Whether checkpoints store EMA weights and the exact key prefix (denoiser./models.denoiser./ema.) — inspect the actual .safetensors to confirm before mapping.

6. concat_cond ordering: confirm `sp.sparse_cat([x, concat_cond], dim=-1)` requires x and shape-latent to have IDENTICAL coords in identical order (it just cats feats; coords taken from neither explicitly in dim=-1 path -> uses inputs[0].replace(feats), so coords = x's). The shape latent must be resampled to x's active-voxel set. Verify in pipeline how the shape latent voxels are aligned to the texture noise voxels.
