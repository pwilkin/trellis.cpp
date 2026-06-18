# slat_flow (SLatFlowModel: img2shape in=32→out=32 and imgshape2tex in=64→out=32)

## SLatFlowModel — exact architecture and algorithm

### 0. Confirmation: SAME block math as the validated dense SS flow

`SLatFlowModel` (`/tmp/TRELLIS.2/trellis2/models/structured_latent_flow.py`) is byte-for-byte the same DiT torso as the dense `SparseStructureFlowModel`, only the tensor container changes (sparse `feats[N,1536]` instead of dense `[B,L,1536]`). Proof:

- It builds `num_blocks` × `ModulatedSparseTransformerCrossBlock` (`modulated.py:81`). Compare `_forward` (`modulated.py:142-160`) with the dense `ModulatedTransformerCrossBlock._forward` (`/tmp/TRELLIS.2/trellis2/modules/transformer/modulated.py:140-158`) — the op sequence is identical:
  1. mod split into 6 chunks (share_mod path: `(self.modulation + mod).chunk(6, dim=1)`),
  2. `h = norm1(x)` (LayerNorm, NO affine, eps 1e-6); `h = h*(1+scale_msa)+shift_msa`; `h = self_attn(h)`; `h = h*gate_msa`; `x = x+h`,
  3. `h = norm2(x)` (LayerNorm, **affine=True**, eps 1e-6 → has `norm2.weight`,`norm2.bias`); `h = cross_attn(h, context)`; `x = x+h`,
  4. `h = norm3(x)` (LayerNorm, NO affine); `h = h*(1+scale_mlp)+shift_mlp`; `h = mlp(h)`; `h = h*gate_mlp`; `x = x+h`.
- The ONLY differences from dense: (a) modulation broadcast — dense does `scale_msa.unsqueeze(1)` to broadcast over the L axis of `[B,L,C]`; sparse omits `.unsqueeze` because `feats` is `[N,C]` and the broadcast is along N via VarLenTensor `__elemwise__` (`basic.py:201-212`). Numerically identical: every token gets the same `scale/shift/gate` vector. (b) RoPE phases are derived from real voxel coords instead of a fixed dense grid. (c) `out_layer` is preceded by `F.layer_norm(h.feats, h.feats.shape[-1:])` (final LayerNorm, no affine, default eps **1e-5**) — same as dense.

This is the validated `src/dit.cpp` graph. **No new attention/MLP/AdaLN math is required.**

### 1. Config (img2shape / shape DiT; ckpt `slat_flow_img2shape_dit_1_3B_512`)
From the key dump and constructor signature:
- `resolution = 512` (only used for `pe_mode=='ape'`; this model is rope, see below)
- `in_channels = 32` (`input_layer.weight [1536,32]`), `out_channels = 32` (`out_layer.weight [32,1536]`)
- `model_channels d_model = 1536`
- `cond_channels d_cond = 1024` (`cross_attn.to_kv.weight [3072,1024]`)
- `num_blocks = 30` (blocks.0 … blocks.29)
- `num_head_channels = 64`? **NO** — head_dim is **128**: `to_qkv.weight [4608,1536]` → 4608 = 3·1536; per-head gamma is `[12,128]` → `num_heads = 12`, `head_dim = 1536/12 = 128`. So `num_heads=12`, `head_dim=128` (NOT 64). `num_head_channels` in config must be 128 (or num_heads=12 passed explicitly).
- `mlp_ratio`: `mlp.mlp.0.weight [8192,1536]` → d_mlp = **8192** = 1536·5.3333 (mlp_ratio≈5.3333, NOT 4). This differs from the `DiTParams` default `d_mlp=8192` already set in `dit.h`, so it is correct.
- `pe_mode = "rope"` (there is NO `pos_emb`/`pos_embedder` key, and self_attn carries RoPE; rope is the only PE), `rope_freq = (1.0, 10000.0)`.
- `share_mod = True` (presence of top-level `adaLN_modulation.1.weight [9216,1536]` AND per-block `blocks.i.modulation [9216]`; 9216 = 6·1536). This is the share_mod branch in both the model and the block.
- `qk_rms_norm = True`, `qk_rms_norm_cross = True` (both self_attn and cross_attn have `q_rms_norm.gamma`/`k_rms_norm.gamma [12,128]`).
- All weights bf16.

For **imgshape2tex / tex DiT**: identical torso, only `in_channels = 64` (`input_layer.weight` will be `[1536,64]`) and `out_channels = 32`. Confirm by dumping that ckpt's keys (see open questions).

### 2. Forward op sequence (inference, B=1)
Input: SparseTensor `x` with `feats[N,in_ch]`, `coords[N,4]` (col0=batch idx=0, cols1..3 = voxel x,y,z ∈ [0,1023]). Timestep `t` (scalar, scaled to ×1000). Image cond `cond[N_img,1024]` (VarLenTensor; for B=1 a single dense `[N_img,1024]`).

```
# (texture model only) concat_cond BEFORE input_layer:
if concat_cond is not None:                      # imgshape2tex
    x = sparse_cat([x, concat_cond], dim=-1)     # feats: [N, 32(noise) | 32(shape_slat)] = [N,64]
h = input_layer(x.feats)                          # SparseLinear: [N,in_ch]->[N,1536]
t_freq = timestep_embedding(t, 256)               # sinusoidal (see §4)
t_emb  = t_embedder.mlp(t_freq)                   # Linear256->1536, SiLU, Linear1536->1536
mod    = adaLN_modulation(t_emb) = Linear1536->9216( SiLU(t_emb) )   # [9216] (share_mod)
# pe_mode == rope -> NO additive pos_emb. RoPE applied INSIDE self_attn.
for block in blocks(30):
    h = block(h, mod, cond)                        # see §0 step 1-4
h = F.layer_norm(h, [1536])                        # final LN, no affine, eps 1e-5
h = out_layer(h)                                   # SparseLinear 1536->out_ch  => velocity feats [N,out_ch]
return SparseTensor(feats=h, coords=x.coords)      # same coords as input
```

### 3. Self-attention with 3D RoPE from voxel coords (the ONLY sparse-specific change)
`SparseMultiHeadAttention.forward` (self) (`modules.py:100-126`):
- `qkv = to_qkv(feats)` → `[N,4608]`, reshape to `[N,3,12,128]`, unbind → q,k,v each `[N,12,128]`.
- `q = q_rms_norm(q)`, `k = k_rms_norm(k)` (SparseMultiHeadRMSNorm: `F.normalize(x,dim=-1) * gamma * sqrt(128)`, computed in fp32 — `modules.py:11-24`).
- `q,k = rope(q,k)` (SparseRotaryPositionEmbedder, `rope.py`).
- `attn_mode='full'` → `sparse_scaled_dot_product_attention(qkv)`: a SINGLE dense self-attention over **all N active voxels of the batch** (B=1 ⇒ one full N×N attention, no windowing, no block-diagonal masking beyond per-batch which is moot for B=1). `attn_mode='full'` is hard-set in `structured_latent_flow.py:71`. Scale = 1/sqrt(128).
- `to_out` Linear 1536->1536.

**RoPE math (sparse), exactly (`rope.py`):**
- `head_dim=128`, `dim=3`, `freq_dim = 128//2//3 = 21`.
- `freqs[j] = rope_freq[0] / rope_freq[1]^(j/freq_dim) = 1.0 / 10000^(j/21)`, j=0..20.
- For a voxel with integer coords `(cx,cy,cz)` = `coords[i,1:4]`:
  - `phases = outer([cx,cy,cz].flatten over the 3 axis-blocks], freqs)` → for axis a, block of 21 complex phases `exp(i * coord_a * freqs[j])`.
  - The per-token phase vector length = 3·21 = 63 < head_dim//2 = 64, so it is PADDED with 1 trailing identity phase `polar(1,0)=1+0i` (cos=1,sin=0). Final length 64.
  - `_rotary_embedding`: view q as complex pairs `[...,64,2]→64 complex`, multiply by `phases.unsqueeze(-2)` (broadcast over heads), back to real. This is **interleaved-pair** rotation (real/imag are adjacent elements 2k,2k+1), identical to the dense `apply_rotary_embedding` and to the validated `apply_rope` in `src/dit.cpp:37`.

So the cos/sin table layout `data[token*64 + pair]` with:
- pair 0..20  → axis x: `ang = cx * freqs[pair]`
- pair 21..41 → axis y: `ang = cy * freqs[pair-21]`
- pair 42..62 → axis z: `ang = cz * freqs[pair-42]`
- pair 63     → `ang = 0` (cos=1, sin=0)  (padding)

This is EXACTLY the loop in `flow_runner.cpp:44-58` with `fd = half/3 = 64/3 = 21` and `freqs[j]=1/pow(10000,j/21)` — the only change is that `(cx,cy,cz)` come from `coords[i,1:4]` instead of being derived from the dense token index (`cx=tok/(R*R)` etc.). Identical numerics.

### 4. Timestep embedding (`sparse_structure_flow.TimestepEmbedder`, reused by import)
`half=128`; `freqs[j]=exp(-ln(10000)*j/128)`; `emb=[cos(t·freqs), sin(t·freqs)]` length 256; then MLP Linear256→1536, SiLU, Linear1536→1536. Identical to `flow_runner.cpp:13-20` `timestep_embedding` and the dense t_embedder. `t` passed in is the FlowEuler scaled timestep `1000*t` (see `flow_runner.cpp:94`).

### 5. Cross-attention (`modules.py:127-138`, qk_rms_norm_cross=True path)
- `q = to_q(h)` → `[N,1536]` reshape `[N,12,128]`; `q = q_rms_norm(q)`.
- `kv = to_kv(cond)` → `[N_img,3072]` reshape `[N_img,2,12,128]`, unbind → k,v; `k = k_rms_norm(k)`.
- full attention `sdpa(q,k,v)` over the N_img cond tokens (B=1 single block). NO rope on cross. scale 1/sqrt(128).
- `to_out` 1536->1536. Identical to `cross_attn` in `dit.cpp:84` (which already applies q/k rms_norm).

### 6. imgshape2tex 64-ch input construction (ORDER matters)
From `trellis2_texturing.py:244-252` (`sample_tex_slat`):
```
in_channels = flow_model.in_channels                         # 64
noise = shape_slat.replace(feats = randn(N, in_channels - shape_slat.feats.shape[1]))  # 64-32 = 32 ch of N(0,1)
slat = sampler.sample(flow_model, noise, concat_cond=shape_slat, **cond)
```
Inside `SLatFlowModel.forward` (`structured_latent_flow.py:177-178`):
```
x = sparse_cat([x, concat_cond], dim=-1)    # x == noise(32), concat_cond == shape_slat(32)
```
So the 64-dim feature is **`[ tex_noise(32) , shape_slat(32) ]`** in that order (the sampled/denoised variable occupies channels 0..31, the conditioning shape latent occupies 32..63). `shape_slat` is normalized first: `shape_slat = (shape_slat - mean)/std` using `shape_slat_normalization` BEFORE the cat. The sampler integrates ONLY the first 32 channels (noise); concat_cond is re-concatenated fresh every step at the top of forward (it is not part of the integrated state). After sampling, output 32-ch slat is denormalized with `tex_slat_normalization`.

### 7. FlowEuler sampler / CFG
Identical to `sample_dense` in `flow_runner.cpp:78`: warped schedule `t_i`, `tscaled=1000*t`, classifier-free guidance with `guidance_interval [gi0,gi1]`, `guidance_strength`, optional `guidance_rescale`, Euler update `sample -= (t-tprev)*pred`. The ONLY adaptation: `cond` and `neg_cond` are now image-feature token sets `[N_img,1024]` (not the dense SS cond), and the integrated state is the sparse feats vector of length `N*out_ch` (out_ch=32). For imgshape2tex, the concat_cond channels are NOT integrated — only the 32 noise channels are stepped; rebuild the 64-ch input each forward by concatenating the (fixed, normalized) shape_slat.

## Weight key map

All keys bf16. d_model=1536, n_heads=12, head_dim=128, d_mlp=8192, d_cond=1024, n_blocks=30. (img2shape ckpt: `ckpts__slat_flow_img2shape_dit_1_3B_512_bf16.keys.txt`.)

GLOBAL:
- input_layer.weight [1536, in_ch]  (img2shape in_ch=32; imgshape2tex in_ch=64)   -> ggml mul_mat (transpose: ne=[in_ch,1536])
- input_layer.bias   [1536]
- out_layer.weight   [out_ch, 1536]  (out_ch=32)                                  -> ne=[1536,out_ch]
- out_layer.bias     [out_ch]
- t_embedder.mlp.0.weight [1536,256] , t_embedder.mlp.0.bias [1536]
- t_embedder.mlp.2.weight [1536,1536], t_embedder.mlp.2.bias [1536]
- adaLN_modulation.1.weight [9216,1536], adaLN_modulation.1.bias [9216]   (share_mod top-level; 9216=6*1536)
  (the SiLU is adaLN_modulation.0, no params)

PER BLOCK i in 0..29 (prefix "blocks.i."):
- modulation [9216]                                  # share_mod per-block learned offset; final mod = modulation + adaLN_out
- self_attn.to_qkv.weight [4608,1536], .to_qkv.bias [4608]      (4608 = 3*1536)
- self_attn.q_rms_norm.gamma [12,128]   (per-head RMS gamma, head_dim=128)
- self_attn.k_rms_norm.gamma [12,128]
- self_attn.to_out.weight [1536,1536],  self_attn.to_out.bias [1536]
- norm2.weight [1536], norm2.bias [1536]   (the ONLY affine LN — corresponds to norm2 elementwise_affine=True; norm1/norm3/final have no params)
- cross_attn.to_q.weight  [1536,1536],  cross_attn.to_q.bias  [1536]
- cross_attn.to_kv.weight [3072,1024],  cross_attn.to_kv.bias [3072]   (kv from 1024-dim cond; 3072=2*1536)
- cross_attn.q_rms_norm.gamma [12,128], cross_attn.k_rms_norm.gamma [12,128]
- cross_attn.to_out.weight [1536,1536], cross_attn.to_out.bias [1536]
- mlp.mlp.0.weight [8192,1536], mlp.mlp.0.bias [8192]    (FFN up; GELU-tanh is mlp.mlp.1, no params)
- mlp.mlp.2.weight [1536,8192], mlp.mlp.2.bias [1536]    (FFN down)

These names are IDENTICAL to the dense SS-flow naming consumed by src/dit.cpp (input_layer, out_layer, t_embedder.mlp.{0,2}, adaLN_modulation.1, blocks.i.{modulation, self_attn.*, norm2.*, cross_attn.*, mlp.mlp.{0,2}}). The dense ckpt additionally has pos_emb (ape) — the slat ckpts do NOT (rope, no pos buffer). No norm1/norm3 params (affine=False). No top-level final-norm params (F.layer_norm without affine).

NOT PRESENT (training-only / absent): pos_emb, rope_phases buffers (computed on the fly), any decoder/encoder keys.

## GGML plan

### Reuse `build_dit_dense` almost verbatim; the graph is shape-agnostic in N.

The existing `build_dit_dense(ctx, m, p, h0, tfreq, cond, cos, sin, inter)` (`src/dit.cpp:134`) already implements EXACTLY this torso (share_mod via `blocks.i.modulation + mod`, QK-RMSNorm self+cross, interleaved RoPE, affine norm2, GELU-tanh MLP, final LN, out_layer). It treats the token axis as `h->ne[1]` (call it L). For sparse we simply set **L = N (number of active voxels)** instead of `R^3`. No per-op change needed in dit.cpp.

Concrete plan — add a `SparseDitRunner` (mirror of `DenseDitRunner`, `flow_runner.cpp:22`):

1. DiTParams: set `in_ch=32` (shape) or `64` (tex), `out_ch=32`, `n_blocks=30`, `n_heads=12`, `head_dim=128`, `d_model=1536`, `d_mlp=8192`, `d_cond=1024`, `ln_eps=1e-6`, `final_ln_eps=1e-5`, `rms_eps=1e-12`. (Defaults in dit.h are already these except confirm head_dim=128 — the struct default is 128, good.)

2. Inputs (all ggml_set_input):
   - `gh0_  = F32 [in_ch, N]`   (channel-major feats; transpose of torch [N,in_ch])
   - `gtf_  = F32 [256]`
   - `gcond_= F32 [d_cond, N_img]`   (image cond tokens, channel-major)
   - `gcos_/gsin_ = F32 [1, 64, 1, N]`   (3D RoPE tables; 64 = head_dim/2)
   N (and N_img) are runtime values: build the graph AFTER coords are known (per generation), or rebuild when N changes. Since SLat sampling fixes the active-voxel set for the whole flow (coords come from Stage ① and are constant across all sampler steps), build ONCE per generation with that N and reuse across all 25/50 steps — exactly like DenseDitRunner builds once for fixed R.

3. RoPE table fill — replace the dense index→coord derivation (`flow_runner.cpp:48-58`) with the real coords array `coords[N][3]`:
```
const int half = 64, fd = half/3; // 21
float freqs[21]; for j: freqs[j] = 1.f/powf(10000.f, (float)j/fd);
for (tok=0; tok<N; ++tok) {
  int cx=coords[tok][0], cy=coords[tok][1], cz=coords[tok][2];
  for (pp=0; pp<half; ++pp){
    float ang = pp<fd ? cx*freqs[pp] : pp<2*fd ? cy*freqs[pp-fd] : pp<3*fd ? cz*freqs[pp-2*fd] : 0.f;
    rcos[tok*half+pp]=cosf(ang); rsin[tok*half+pp]=sinf(ang);
  }
}
```
(coords are the int voxel indices from Stage ①, cols 1..3.) Everything else (`apply_rope`, sdpa, rms_gamma) is unchanged.

4. input_layer: feed `gh0_` as `[in_ch,N]`; `lin(c,m,"input_layer",h0)` already does mul_mat→`[1536,N]`. For the tex model, build the 64-ch h0 on the host by concatenating `[noise(32) ; shape_slat_norm(32)]` per voxel in channel-major order BEFORE upload (channels 0..31 = current sample, 32..63 = shape_slat). Re-pack every step (sample changes; shape_slat constant).

5. Final LN + out_layer: already in build_dit_dense (`final_ln_eps=1e-5`, then `out_layer`). Output `[out_ch,N]` → transpose to torch `[N,out_ch]` on readback.

6. Sampler: reuse `sample_dense`/`SamplerParams` logic unchanged. The integrated vector length is `N*out_ch` (the velocity). For tex, only integrate the 32 noise channels; concat shape_slat fresh each `forward`. Guidance: pass image `cond` and `neg_cond` token blocks of shape `[1024, N_img]` and `[1024, N_img_neg]`.

7. Attention cost: self-attn is full N×N; for N up to ~tens of thousands of active voxels this is the dominant cost. ggml SDPA path in `sdpa()` builds `kq=[N,N,12]` — fine for moderate N; if N is large, consider `ggml_flash_attn_ext`. Cross-attn is N×N_img (small). No masking needed (B=1).

NEW code = thin runner (input packing, coord-based RoPE fill, per-step tex concat). REUSED = entire `build_dit_dense` graph, `apply_rope`, `sdpa`, `rms_gamma`, `layernorm`, `lin`, `timestep_embedding`, `sample_dense`, SamplerParams.

## Reuse

REUSE AS-IS (no change):
- `build_dit_dense` (src/dit.cpp) — the full 30-block torso is identical for SLat. The token axis ne[1] just becomes N (active voxels) rather than R^3. share_mod, QK-RMSNorm (self+cross), interleaved-pair RoPE, affine norm2, GELU-tanh MLP, final LN(1e-5), out_layer all already correct and validated.
- `apply_rope` (dit.cpp:37): interleaved-pair complex rotation == sparse `_rotary_embedding`. Identical.
- `rms_gamma` (dit.cpp:31): `ggml_rms_norm * gamma` == SparseMultiHeadRMSNorm (`F.normalize*gamma*sqrt(d)`), since ggml_rms_norm already multiplies by sqrt(d). Validated in SS flow.
- `sdpa`, `lin`, `layernorm`, `cross_attn`, `self_attn` helpers — unchanged.
- `timestep_embedding` (flow_runner.cpp:13) — identical TimestepEmbedder (the slat model imports TimestepEmbedder from sparse_structure_flow).
- `sample_dense` + `SamplerParams` (flow_runner.cpp:78) — FlowEuler + CFG identical; just swap cond tensors and state length.
- Weight key names identical to dense → existing Model loader/getter works; only ckpt file + in_ch/out_ch differ.

GENUINELY NEW (small):
- A `SparseDitRunner` that (a) sizes inputs by runtime N / N_img, (b) fills RoPE cos/sin from real `coords[:,1:4]` instead of dense grid index math, (c) for the tex model packs the 64-ch input as channel-major `[noise(32); shape_slat_norm(32)]` each step.
- Host-side sparse plumbing: carry the int coords array (from Stage ① active voxels), normalize shape_slat with shape_slat_normalization (mean/std) for the tex path, denormalize outputs.
- `sparse_cat([x, concat_cond], dim=-1)` becomes a host channel concat (no ggml op needed; do it before upload).

NOT NEEDED: SparseLayerNorm/SparseGroupNorm per-batch loops (those are for B>1 / GroupNorm; the flow torso uses plain LayerNorm over the channel axis which is per-token and B-agnostic — ggml_norm over ne0 is correct). Windowed/double-windowed attention paths (attn_mode is hard 'full'). AbsolutePositionEmbedder/pos_emb (rope mode). torchsparse/spconv backends. All training/init code (`initialize_weights`, scaled init) is training-only.

## Open questions

1. Confirm the imgshape2tex ckpt key dump (not provided here): expect `input_layer.weight [1536,64]`, `out_layer.weight [32,1536]`, everything else identical to img2shape. Dump `ckpts__slat_flow_imgshape2tex*_dit_*.keys.txt` to verify in_ch=64 and that there are no extra keys.

2. Cond token dim/source: the dump shows cross_attn.to_kv expects 1024-dim cond. Confirm the image cond for SLat is the SAME 1024-dim DINOv3/feature tokens used by Stage ① (the task states cond[N_img,1024]). Verify N_img and whether positive and negative (CFG) conds are precomputed feature blocks (and whether neg cond is zeros or a learned/empty token set) by reading trellis2_image_to_3d.py get_cond / the sampler call.

3. Normalization constants: shape_slat_normalization and tex_slat_normalization {mean,std} are per-channel vectors loaded from the pipeline config JSON. Need their exact values (and channel count 32) from the pipeline args to implement the tex concat normalize/denormalize. Dump from the pipeline config.

4. RoPE coord range: coords are stated ∈[0,1023]; freqs use raw integer coords (no normalization by resolution). The dense path used raw grid indices too, so this matches — but confirm Stage ① emits coords in the same absolute voxel space the flow was trained on (i.e. not downscaled). A quick check: dump one block's q after RoPE for a known coord and compare to a torch reference.

5. final LayerNorm eps: torch `F.layer_norm(h.feats, h.feats.shape[-1:])` uses default eps 1e-5; dit.h sets final_ln_eps=1e-5 — good, but double-check the dense validation used 1e-5 (the per-block norms use 1e-6).

6. Whether the sampler integrates in fp32 over feats while the torso runs bf16/f16 (manual_cast(x.dtype) around the torso). The dense runner casts weights optionally to f32 (cast_f32). Confirm acceptable precision for SLat or set cast_f32 as in validated SS flow.
