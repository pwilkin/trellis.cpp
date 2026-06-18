# ss_flow

		# SparseStructureFlowModel (ss_flow) — DENSE 3D DiT on 16^3 grid

Config (from `/tmp/TRELLIS.2/configs/gen/ss_flow_img_dit_1_3B_64_bf16.json`, denoiser):
`resolution=16, in_channels=8, out_channels=8, model_channels=1536, cond_channels=1024, num_blocks=30, num_heads=12, num_head_channels=64(unused since num_heads given), mlp_ratio=5.3334, pe_mode="rope", share_mod=true, initialization="scaled", qk_rms_norm=true, qk_rms_norm_cross=true`. Runtime dtype = bf16 (file name `_bf16`; weights converted via `convert_to`). `rope_freq` is NOT in config → default `(1.0, 10000.0)`.

## Derived constants
- `head_dim = model_channels / num_heads = 1536 / 12 = 128`
- `seq_len L = resolution^3 = 16^3 = 4096`
- `mlp_hidden = int(model_channels * mlp_ratio) = int(1536 * 5.3334) = 8192`
- `6 * model_channels = 9216` (adaLN produces 6 modulation chunks)
- RoPE: `freq_dim = head_dim // 2 // dim = 128 // 2 // 3 = 21`; phases per axis = 21, ×3 axes = 63, padded to `head_dim//2 = 64`.

## Forward pipeline (file `models/sparse_structure_flow.py`, `forward(x, t, cond)`)

Inputs:
- `x`: `[B, 8, 16, 16, 16]` (float; latent of sparse-structure VAE). Asserted shape `[B, in_channels, 16,16,16]`.
- `t`: `[B]` scalar timesteps (fractional allowed).
- `cond`: `[B, Lc, 1024]` DINOv3 image features (Lc = number of image tokens; enters via cross-attention only).

Steps:
1. **Patchify (flatten, no conv):** `h = x.view(B, 8, -1).permute(0,2,1).contiguous()` → `[B, 4096, 8]`. Each of the 4096 voxels becomes a token of dim 8. Voxel ordering is C-order over (X,Y,Z) i.e. index = ((x*16)+y)*16+z, matching `meshgrid(arange(16)^3, indexing='ij').reshape(-1,3)`.
2. **Input projection:** `h = self.input_layer(h)` — `nn.Linear(8 → 1536)` → `[B, 4096, 1536]`.
3. **APE add (skipped for rope):** since `pe_mode=="rope"`, NO additive pos-emb here. (`self.rope_phases` buffer is used inside attention instead.)
4. **Timestep embedding:** `t_emb = self.t_embedder(t)`:
   - `timestep_embedding(t, dim=256, max_period=10000)`: `half=128`; `freqs = exp(-log(10000) * arange(0,128)/128)`; `args = t[:,None]*freqs[None]`; `emb = cat([cos(args), sin(args)], -1)` → `[B, 256]`. (cos first, then sin; dim even so no pad.)
   - MLP: `Linear(256→1536)` → `SiLU` → `Linear(1536→1536)` → `[B, 1536]`.
5. **Shared modulation (share_mod=true):** `t_emb = self.adaLN_modulation(t_emb)` = `SiLU` then `Linear(1536 → 9216)` → `[B, 9216]`. This single shared 9216-vector is passed to EVERY block as `mod`.
6. Cast `t_emb`, `h`, `cond` to model dtype (bf16) via `manual_cast`.
7. **30 × ModulatedTransformerCrossBlock:** `h = block(h, t_emb, cond, self.rope_phases)`. (see block detail below.)
8. **Final layer:** cast `h` back to input dtype (`x.dtype`); `h = F.layer_norm(h, h.shape[-1:])` — a parameterless (no weight/bias) LayerNorm over last dim (1536), eps default 1e-5; then `h = self.out_layer(h)` = `nn.Linear(1536 → 8)` → `[B, 4096, 8]`.
9. **Unpatchify:** `h = h.permute(0,2,1).view(B, 8, 16,16,16).contiguous()` → `[B, 8, 16, 16, 16]`. Returned as the velocity/flow prediction.

## RoPE phases buffer (precompute once, register_buffer "rope_phases")
Built in `__init__` (pe_mode=="rope"):
- `pos_embedder = RotaryPositionEmbedder(head_dim=128, dim=3, rope_freq=(1.0,10000.0))`.
- `coords = meshgrid(arange(16),arange(16),arange(16), indexing='ij'); stack(-1).reshape(-1,3)` → `[4096, 3]` integer coords.
- `rope_phases = pos_embedder(coords)` → complex tensor `[4096, 64]` (64 = head_dim//2).
RotaryPositionEmbedder internals (`modules/attention/rope.py`):
- `freqs = arange(21)/21`; `freqs = 1.0 / (10000 ** freqs)` → 21 values.
- `_get_phases(indices)`: `phases = outer(indices_flat, freqs)`; `phases = polar(ones, phases)` (complex unit vectors, magnitude 1, angle = idx*freq).
- forward: flatten coords `[4096*3]`, get phases `[4096*3, 21]`, reshape `[4096, 3, 21]` then `[4096, 63]`. Since `63 < 64`, pad with 1 extra phase of `polar(1, 0) = 1+0j` → `[4096, 64]`.
- So per token, the 64 complex phases = [axis0: 21 phases][axis1: 21][axis2: 21][1 dummy=1+0j].

**apply_rotary_embedding(x, phases)** (static, applied to q and k of self-attn):
- `x` is `[B, L, H, 128]`. `x_complex = view_as_complex(x.float().reshape(B,L,H,64,2))` → `[B,L,H,64]` complex.
- `x_rotated = x_complex * phases.unsqueeze(-2)` (phases `[L,64]` broadcast over H).
- `x_embed = view_as_real(x_rotated).reshape(B,L,H,128)`.
- i.e. consecutive pairs (x[...,2i], x[...,2i+1]) are rotated by angle of phases[...,i]: `out_2i = x_2i*cos - x_{2i+1}*sin; out_{2i+1} = x_2i*sin + x_{2i+1}*cos`. **GGML: this is interleaved-pair (NEOX=false style) rope with per-axis frequency blocks; custom 3D rope needed.**

## ModulatedTransformerCrossBlock (modules/transformer/modulated.py) — per block i
ln_affine for the three norms in this class: norm1=False, **norm2=True**, norm3=False. eps=1e-6. All are `LayerNorm32` (compute in fp32, cast back).

Modulation (share_mod=true path): block has its own learnable `self.modulation` parameter `[9216]` (init `randn(9216)/sqrt(1536)`). Combined with shared mod from model:
`shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)` — wait mod is `[B,9216]`, modulation is `[9216]` broadcast → `[B,9216]`, chunk(6, dim=1) → six `[B,1536]`. (NOTE: with share_mod there is NO `adaLN_modulation` Sequential inside the block; modulation is a raw Parameter ADDED to the model-level adaLN output.)

Forward `_forward(x, mod, context, phases)`:
1. `h = norm1(x)` (no affine)
2. `h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)`  (unsqueeze(1) → `[B,1,1536]` broadcast over L)
3. `h = self.self_attn(h, phases=phases)`  (self MHA with rope + qk_rms_norm)
4. `h = h * gate_msa.unsqueeze(1)`
5. `x = x + h`
6. `h = norm2(x)` (**affine, has weight+bias**)
7. `h = self.cross_attn(h, context)`  (cross MHA, no rope, qk_rms_norm_cross)
8. `x = x + h`
9. `h = norm3(x)` (no affine)
10. `h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)`
11. `h = self.mlp(h)`
12. `h = h * gate_mlp.unsqueeze(1)`
13. `x = x + h`; return x.

## MultiHeadAttention (modules/attention/modules.py)
Self-attn (type="self", attn_mode="full", use_rope=True, qk_rms_norm=True):
- `to_qkv = nn.Linear(1536 → 4608)` (channels*3), bias=True.
- `qkv = to_qkv(x).reshape(B, L, 3, 12, 128)`; unbind dim=2 → q,k,v each `[B,L,12,128]`.
- qk_rms_norm: `q = q_rms_norm(q); k = k_rms_norm(k)` (MultiHeadRMSNorm).
- rope: `q = apply_rotary_embedding(q, phases); k = apply_rotary_embedding(k, phases)`.
- `h = scaled_dot_product_attention(q,k,v)` → `[B,L,12,128]`; reshape `[B,L,1536]`.
- `to_out = nn.Linear(1536 → 1536)`, bias=True → `[B,L,1536]`.

Cross-attn (type="cross", attn_mode="full", qk_rms_norm_cross=True, no rope):
- `to_q = nn.Linear(1536 → 1536)`, bias=True.
- `to_kv = nn.Linear(ctx_channels=1024 → 3072)` (channels*2 = 1536*2), bias=True.
- `q = to_q(x).reshape(B, L, 12, 128)`; `kv = to_kv(context).reshape(B, Lc, 2, 12, 128)`.
- qk_rms_norm_cross: `q = q_rms_norm(q)`; `k,v = kv.unbind(2)`; `k = k_rms_norm(k)`; `h = sdpa(q,k,v)`.
- reshape `[B,L,1536]`; `to_out = nn.Linear(1536→1536)`.

**scaled_dot_product_attention** (full_attn.py): standard SDPA, `scale = 1/sqrt(head_dim)=1/sqrt(128)`, softmax over keys, no mask, no causal. Layout [N,L,H,C] → permute to [N,H,L,C] for the kernel.

**MultiHeadRMSNorm** (per-head RMS norm on q/k): `scale = head_dim**0.5 = sqrt(128)`; `gamma` param `[num_heads=12, head_dim=128]`. forward: `(F.normalize(x.float(), dim=-1) * gamma * scale).to(dtype)`. Note `F.normalize` here divides by L2 norm (not RMS-by-sqrt(n)); combined with `*sqrt(128)` it is equivalent to RMSNorm: result ≈ x/rms(x)*gamma. Applied per (head,dim) vector of length 128. **GGML: custom — L2-normalize over last dim, multiply by gamma broadcast over [H,128] and by sqrt(128).**

## TimestepEmbedder
- `mlp[0] = Linear(256 → 1536)`, `mlp[1]=SiLU`, `mlp[2]=Linear(1536 → 1536)`.
- `frequency_embedding_size = 256`.

## Model-level adaLN (share_mod)
- `adaLN_modulation = Sequential(SiLU(), Linear(1536 → 9216, bias=True))`.

## FeedForwardNet (mlp)
- `mlp[0] = Linear(1536 → 8192)`, `mlp[1] = GELU(approximate="tanh")`, `mlp[2] = Linear(8192 → 1536)`. All bias=True.

## NOT used at inference
- `initialize_weights`, both `vanilla`/`scaled` branches — training only.
- `use_checkpoint` path (gradient checkpointing) — training only; use `_forward` directly.
- AbsolutePositionEmbedder / pos_emb buffer — only if pe_mode=="ape"; NOT used here.
- The non-share_mod `adaLN_modulation` per-block Sequential — NOT present (share_mod=true uses raw `modulation` Parameter).

## Weight key patterns

		Top-level (module `SparseStructureFlowModel`, typical prefix `denoiser.` or none depending on checkpoint packaging):

- `t_embedder.mlp.0.weight` [1536, 256], `t_embedder.mlp.0.bias` [1536]
- `t_embedder.mlp.2.weight` [1536, 1536], `t_embedder.mlp.2.bias` [1536]
- `adaLN_modulation.1.weight` [9216, 1536], `adaLN_modulation.1.bias` [9216]   (index 1 = the Linear; index 0 = SiLU has no params)
- `input_layer.weight` [1536, 8], `input_layer.bias` [1536]
- `out_layer.weight` [8, 1536], `out_layer.bias` [8]
- `rope_phases` — registered buffer, COMPLEX [4096, 64]. May or may not be saved in safetensors (complex dtype often not serialized; recompute at runtime). NOT a learnable param.
- (NO `pos_emb` buffer since rope.)

Per block i in 0..29, prefix `blocks.{i}.`:
- `blocks.{i}.modulation` [9216]   (raw Parameter, share_mod path; NO `blocks.{i}.adaLN_modulation.*`)
- `blocks.{i}.norm1.*` — elementwise_affine=False → NO weight/bias keys.
- `blocks.{i}.norm2.weight` [1536], `blocks.{i}.norm2.bias` [1536]   (affine=True)
- `blocks.{i}.norm3.*` — affine=False → NO weight/bias keys.
- Self attention:
  - `blocks.{i}.self_attn.to_qkv.weight` [4608, 1536], `blocks.{i}.self_attn.to_qkv.bias` [4608]
  - `blocks.{i}.self_attn.q_rms_norm.gamma` [12, 128]
  - `blocks.{i}.self_attn.k_rms_norm.gamma` [12, 128]
  - `blocks.{i}.self_attn.to_out.weight` [1536, 1536], `blocks.{i}.self_attn.to_out.bias` [1536]
- Cross attention:
  - `blocks.{i}.cross_attn.to_q.weight` [1536, 1536], `blocks.{i}.cross_attn.to_q.bias` [1536]
  - `blocks.{i}.cross_attn.to_kv.weight` [3072, 1024], `blocks.{i}.cross_attn.to_kv.bias` [3072]
  - `blocks.{i}.cross_attn.q_rms_norm.gamma` [12, 128]
  - `blocks.{i}.cross_attn.k_rms_norm.gamma` [12, 128]
  - `blocks.{i}.cross_attn.to_out.weight` [1536, 1536], `blocks.{i}.cross_attn.to_out.bias` [1536]
- MLP:
  - `blocks.{i}.mlp.mlp.0.weight` [8192, 1536], `blocks.{i}.mlp.mlp.0.bias` [8192]
  - `blocks.{i}.mlp.mlp.2.weight` [1536, 8192], `blocks.{i}.mlp.mlp.2.bias` [1536]   (index 1 = GELU, no params)

Note: nn.Linear stores weight as [out_features, in_features]; GGML ggml_mul_mat wants the weight as-is (it multiplies x[in] by W[out,in]).

## GGML notes

		Per-op mapping to GGML:

- Patchify/unpatchify: pure reshape+permute (ggml_reshape / ggml_permute / ggml_cont). x [B,8,16,16,16] -> flatten spatial -> [B,4096,8]. Be careful: PyTorch C-order voxel index ((x*16+y)*16+z). When loading the [8,16,16,16] tensor into ggml (which is column-major / ne[0] fastest), arrange so token index maps to (x,y,z) in 'ij' order consistent with rope_phases ordering. Easiest: precompute rope_phases on the SAME ordering used to flatten.
- input_layer / out_layer / all Linears: ggml_mul_mat + ggml_add (bias). Standard.
- Timestep sinusoidal embedding: precompute on CPU (freqs = exp(-log(10000)*arange(128)/128)), then cos/sin; or small custom op. cat order is [cos(128), sin(128)] -> 256. Then 2 Linears with SiLU (ggml_silu).
- adaLN shared modulation: SiLU + Linear(1536->9216). Add per-block `modulation` Parameter, then chunk into 6 x 1536. Implement chunk as views with offsets. broadcast over L: the [B,1536] vectors apply per-token (unsqueeze(1)).
- LayerNorm32: ggml_norm (eps 1e-6) over last dim 1536. norm1/norm3 NO affine (just normalize); norm2 affine (mul weight + add bias). Final F.layer_norm before out_layer: eps 1e-5, NO affine. All done in fp32 (cast up then back) — in ggml do norm in f32.
- adaLN apply: h*(1+scale)+shift = ggml_mul + ggml_add with broadcasting; gate: ggml_mul.
- Self-attention: to_qkv -> split into q,k,v (views). Reshape to [B,L,12,128].
  * qk_rms_norm (MultiHeadRMSNorm): L2-normalize each 128-vector over last dim, multiply by gamma[12,128] (broadcast over B,L) and by scalar sqrt(128). Custom: ggml_rms_norm is RMS (divide by sqrt(mean(x^2))) NOT L2. To match F.normalize exactly use L2: x / sqrt(sum(x^2)+eps). Easiest correct path: out = ggml_rms_norm(x, eps) gives x/sqrt(mean(x^2)) = x*sqrt(128)/sqrt(sum x^2); then multiply by gamma ONLY (the *sqrt(128) is already supplied by rms_norm's 1/sqrt(mean) = sqrt(N)/sqrt(sum)). i.e. MultiHeadRMSNorm == ggml_rms_norm(eps=0) * gamma. (Verify: F.normalize = x/sqrt(sumsq); *scale(sqrt128) = x*sqrt128/sqrt(sumsq) = x/sqrt(sumsq/128) = x/sqrt(mean) = ggml_rms_norm output. So MultiHeadRMSNorm(x) == ggml_rms_norm(x)*gamma. Use eps≈1e-12.) gamma broadcasts per-head [12,128].
  * 3D RoPE: CUSTOM op. Interleaved pairs (x_2i, x_2i+1) rotated by phase angle of rope_phases[token, i]. rope_phases are precomputed complex [4096,64]; pass as two f32 tensors cos[4096,64], sin[4096,64]. Apply: out_2i = x_2i*cos_i - x_{2i+1}*sin_i; out_{2i+1}=x_2i*sin_i + x_{2i+1}*cos_i. This is NOT ggml's standard rope (which is GPT-NeoX or GPT-J split). Frequencies are per-axis blocks (21,21,21,+1 identity). Write a custom kernel or precompute full cos/sin [4096,128] (duplicate each of 64 to two slots interleaved) and do elementwise: out = x*cos_full + rotate_pairs(x)*sin_full, where rotate_pairs swaps and negates within each pair. rotate_pairs can be done via reshape [.,64,2], stack(-x1,x0). Doable with ggml views + concat, or a small custom op.
  * SDPA: ggml_soft_max with scale 1/sqrt(128); or use ggml flash attn. Heads=12, head_dim=128, L=4096 (no mask). Standard.
  * to_out: Linear.
- Cross-attention: to_q(x), to_kv(context [B,Lc,1024]) -> k,v. qk_rms_norm_cross same MultiHeadRMSNorm on q and k (no rope). SDPA q[L=4096] x kv[Lc]. to_out.
- FeedForwardNet: Linear(1536->8192), GELU tanh-approx (ggml_gelu uses tanh approx — matches GELU(approximate="tanh")), Linear(8192->1536).
- Residual adds: ggml_add.
- Memory: 30 blocks, model_channels 1536, mlp 8192, L=4096. Activations ~ [B,4096,8192] for mlp hidden are the biggest. bf16 weights; norms/rms/rope in f32. Total params ~1.3B. rope_phases buffer can be recomputed (no need to load). cond Lc depends on DINOv3 (typically a few hundred to ~1k tokens).
- Inference is a flow-matching velocity predictor: called repeatedly by an ODE sampler (outside this module). This module is purely deterministic forward.

## Open questions

		1. `cond` token count Lc and exact DINOv3 feature layout (1024-dim) are defined by the image_feature_extractor / pipeline, not this module — confirm in modules/image_feature_extractor.py and the pipeline. Cross-attn only requires last-dim=1024.
2. Whether `rope_phases` complex buffer is actually serialized in the safetensors checkpoint or must be recomputed. Recommend recomputing on CPU from the documented formula (safer; complex dtype rarely stored). Verify by inspecting the actual checkpoint keys.
3. Checkpoint prefix: keys may be bare (`blocks.0...`) or namespaced (`denoiser.blocks.0...`) depending on how the pipeline saves/loads — confirm against the real .safetensors.
4. Voxel flatten ordering must exactly match rope_phases coord ordering (meshgrid 'ij' over X,Y,Z, C-order). Confirm the latent tensor axis order from the SS-VAE so (channel, x, y, z) maps correctly when porting to ggml's memory layout.
5. `int(1536*5.3334)` = 8192 exactly in CPython (5.3334*1536=8192.0064 -> int=8192). Confirmed via python; mlp hidden = 8192.
6. MultiHeadRMSNorm == ggml_rms_norm*gamma equivalence assumes eps negligible; F.normalize default eps=1e-12. Use matching tiny eps to be safe.
7. Final pre-out LayerNorm `F.layer_norm(h, h.shape[-1:])` uses default eps=1e-5 (NOT 1e-6) and has no affine params — note the eps differs from the LayerNorm32 blocks (1e-6).
