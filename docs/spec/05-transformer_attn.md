# transformer_attn

## transformer_attn — Dense (non-sparse) attention primitives

This component covers the shared attention building blocks used by the DENSE DiTs (e.g. `SparseStructureFlowModel`). The sparse variants (`SparseMultiHeadAttention`, `SparseRotaryPositionEmbedder`) live under `trellis2/modules/sparse/attention/` and are a separate component; they are noted here only for contrast.

Files: `trellis2/modules/attention/{modules,full_attn,rope,config,__init__}.py`, `trellis2/modules/norm.py`, helper `trellis2/modules/utils.py::manual_cast`.

All code below is INFERENCE-relevant. `config.py` backend selection, `torch.utils.checkpoint` wrappers in the blocks, and the xformers/flash_attn branches are runtime-only choices; the math is identical to the `naive`/`sdpa` paths.

---

### 1. `MultiHeadRMSNorm` (qk_rms_norm) — modules.py:9-16

```
self.scale = dim ** 0.5            # dim = head_dim
self.gamma = nn.Parameter(torch.ones(heads, dim))   # shape [num_heads, head_dim]
forward(x): (F.normalize(x.float(), dim=-1) * self.gamma * self.scale).to(x.dtype)
```

Exact formula, applied to the LAST dim (head_dim) of a tensor shaped `[B, L, H, head_dim]`:
- `F.normalize(x, dim=-1)` = `x / max(||x||_2 over last dim, eps)`, with PyTorch default `eps=1e-12` and `p=2`. So `y_i = x_i / sqrt(sum_j x_j^2 + 1e-12)` — note this is L2 normalization, NOT mean-square RMSNorm (no `1/sqrt(dim)` inside the norm; that factor is reintroduced externally as `self.scale = sqrt(head_dim)`).
- Then multiply elementwise by `gamma` (broadcast over B,L; gamma indexed by [H, head_dim]) and by scalar `scale = sqrt(head_dim)`.
- Computation is done in float32 (`x.float()`), result cast back to input dtype.
- Net effect: `out = (x / ||x||_2) * sqrt(head_dim) * gamma`. Because each head vector is L2-normalized to unit length then scaled by sqrt(head_dim), its post-norm L2 norm is `sqrt(head_dim)*||gamma elementwise||`-ish; this is the standard "normalize then learnable per-channel scale" QK-norm.
- `gamma` is the ONLY learnable parameter; initialized to ones.

### 2. `MultiHeadAttention` — modules.py:19-102

Constructor params (defaults): `channels`, `num_heads`, `ctx_channels=None`, `type="self"`, `attn_mode="full"`, `window_size=None`, `shift_window=None`, `qkv_bias=True`, `use_rope=False`, `rope_freq=(1.0,10000.0)`, `qk_rms_norm=False`.

Asserts: `channels % num_heads == 0`; cross-attn forces `attn_mode=="full"`; `attn_mode=="windowed"` raises NotImplementedError (so only "full" is reachable).

Derived: `head_dim = channels // num_heads`; `ctx_channels = ctx_channels or channels`.

Submodules / weight-bearing layers:
- type == "self": `self.to_qkv = nn.Linear(channels, channels*3, bias=qkv_bias)`
- type == "cross": `self.to_q = nn.Linear(channels, channels, bias=qkv_bias)`, `self.to_kv = nn.Linear(ctx_channels, channels*2, bias=qkv_bias)`
- if qk_rms_norm: `self.q_rms_norm = MultiHeadRMSNorm(head_dim, num_heads)`, `self.k_rms_norm = MultiHeadRMSNorm(head_dim, num_heads)`
- always: `self.to_out = nn.Linear(channels, channels)` (bias=True, default)

NOTE: RoPE is NOT instantiated inside this dense module. `use_rope` only gates whether `apply_rotary_embedding` is called on q,k using externally-supplied `phases`. The `RotaryPositionEmbedder` lives in the parent model (`sparse_structure_flow.py`) as a non-parameter buffer `rope_phases`. (Contrast: the SPARSE module DOES own `self.rope = SparseRotaryPositionEmbedder(...)`.)

#### forward(x, context=None, phases=None) — self-attention path (type=="self"):
1. `B, L, C = x.shape`.
2. `qkv = self.to_qkv(x)` → `[B, L, 3C]`.
3. `qkv = qkv.reshape(B, L, 3, num_heads, head_dim)` (the `-1` resolves to head_dim). Layout: the projection output channel dim is interpreted as `(3, num_heads, head_dim)` in that order — i.e. the qkv weight rows are grouped first by q/k/v, then by head, then by head_dim.
4. If `qk_rms_norm or use_rope`: `q,k,v = qkv.unbind(dim=2)` each `[B, L, H, head_dim]`.
   - if qk_rms_norm: `q = q_rms_norm(q)`, `k = k_rms_norm(k)` (v untouched).
   - if use_rope: `q = apply_rotary_embedding(q, phases)`, `k = apply_rotary_embedding(k, phases)` (v untouched). `phases` must be provided.
   - `h = scaled_dot_product_attention(q, k, v)` (3-arg form).
   - else (neither): `h = scaled_dot_product_attention(qkv)` (1-arg packed form).
5. `h` returned as `[B, L, H, head_dim]`.

#### forward — cross-attention path (type=="cross"):
1. `Lkv = context.shape[1]`.
2. `q = self.to_q(x)` → reshape `[B, L, num_heads, head_dim]`.
3. `kv = self.to_kv(context)` → reshape `[B, Lkv, 2, num_heads, head_dim]`.
4. if qk_rms_norm: `q = q_rms_norm(q)`; `k,v = kv.unbind(dim=2)`; `k = k_rms_norm(k)`; `h = sdpa(q,k,v)`. else: `h = sdpa(q, kv)` (2-arg form). RoPE is never applied in the cross path of the dense module.

#### output projection (both paths):
6. `h = h.reshape(B, L, -1)` → `[B, L, C]` (merges H*head_dim).
7. `h = self.to_out(h)` → `[B, L, C]`. Return.

Weight mapping note: q/k/v are stored as a single fused `to_qkv` (out=3C) for self-attn; cross-attn splits into `to_q` (out=C) and fused `to_kv` (out=2C). The C++ port must slice the fused matmul output into the (3,H,Dh) / (2,H,Dh) layout described above.

### 3. `scaled_dot_product_attention` — full_attn.py

Overloaded by arg count:
- 1 arg `qkv` `[N,L,3,H,C]` → unbind dim2 into q,k,v.
- 2 args `q [N,L,H,C]`, `kv [N,L,2,H,C]` → unbind kv.
- 3 args `q,k,v` each `[N,L,H,C]` (q may have Ci, v may have Co, but in practice all = head_dim).

The reference math (`_naive_sdpa`, and the `sdpa` backend are equivalent):
```
q,k,v: [N, L, H, C]  ->  permute to [N, H, L, C]
scale_factor = 1 / sqrt(C)          # C = head_dim (last dim of q)
attn_weight = (q @ k^T) * scale_factor      # [N,H,L,L]
attn_weight = softmax(attn_weight, dim=-1)
out = attn_weight @ v               # [N,H,L,C]
out = permute back -> [N, L, H, C]
```
Softmax scale = `1/sqrt(head_dim)`. Plain (non-causal) full attention, no mask, no dropout at inference. Note for the rms-norm path the q/k have already been scaled by sqrt(head_dim) inside MultiHeadRMSNorm, AND sdpa still applies its own `1/sqrt(head_dim)` — both factors are present (not redundant; rms-norm scale normalizes magnitude, sdpa scale is the standard attention temperature).

Backends (config.BACKEND, default `'flash_attn'`, env `ATTN_BACKEND` ∈ {xformers, flash_attn, flash_attn_3, sdpa, naive}): all produce numerically equivalent non-causal softmax attention. For the C++ port implement the `naive`/`sdpa` math.

### 4. `RotaryPositionEmbedder` — rope.py (DENSE, 3D)

Constructor: `head_dim` (must be even), `dim=3` (number of spatial axes), `rope_freq=(1.0, 10000.0)` = (theta_min/base scale, theta_max/base).

Precomputed at init (NOT a learnable param; a plain tensor / registered as buffer by the parent via `register_buffer("rope_phases", ...)`):
```
freq_dim = head_dim // 2 // dim          # frequencies per axis
freqs = arange(freq_dim, float32) / freq_dim          # [0, 1/freq_dim, ...]
freqs = rope_freq[0] / (rope_freq[1] ** freqs)        # = 1.0 / 10000^(i/freq_dim)
```
So `freqs[i] = 1.0 / (10000 ** (i/freq_dim))`, length `freq_dim`.

`_get_phases(indices)` (indices 1-D after flatten): `phases = outer(indices, freqs)` then `phases = polar(ones, phases)` = `exp(i * indices*freqs)` = complex unit vectors `(cos, sin)`. Shape `[len(indices), freq_dim]` complex.

`forward(indices)` — indices `[..., N, dim]` (last dim must == dim==3):
1. flatten indices to 1-D, compute phases, reshape to `[..., N, dim*freq_dim]` complex (concatenating the per-axis phase blocks: for each token, axis0's freq_dim phases, then axis1's, then axis2's).
2. If `dim*freq_dim < head_dim//2`, pad the remainder with `polar(ones, zeros)` = complex `1+0i` (no rotation) up to `head_dim//2`. This padding handles the case where `head_dim/2` is not divisible by `dim` (the leftover rotary slots are identity).
Returns `phases` complex `[..., N, head_dim//2]`.

`apply_rotary_embedding(x, phases)` (staticmethod) — x `[B, L, H, head_dim]`:
```
x_complex = view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))  # [B,L,H,head_dim/2] complex
x_rotated = x_complex * phases.unsqueeze(-2)        # phases [B,L,head_dim/2] -> [B,L,1,head_dim/2], broadcast over heads
x_embed = view_as_real(x_rotated).reshape(...last dim back to head_dim).to(x.dtype)
```
Key facts for the port:
- Rotary is applied to the FULL head_dim (rotary_dim == head_dim); the padding makes high slots identity rotations.
- Pairing convention: `view_as_complex` pairs ADJACENT elements `(x[2j], x[2j+1])` into real/imag. So rotation acts on consecutive pairs (interleaved), NOT split-half. The C++ op must rotate pairs (2k,2k+1): `out[2k]=x[2k]*cos - x[2k+1]*sin`, `out[2k+1]=x[2k]*sin + x[2k+1]*cos`, where `(cos,sin)=phases[...,k]`.
- phases broadcast over the head axis (same rotation for all heads; differs per token and per channel-pair).
- For the dense `SparseStructureFlowModel` the coords are a full dense grid: `meshgrid(arange(resolution)^3) -> reshape(-1,3)` (resolution from config). `rope_phases` computed once at init for all `resolution^3` tokens and registered as buffer; passed to every block's self-attn. So for the C++ port, phases can be precomputed on host from integer grid coords.

### 5. Norms — norm.py

- `LayerNorm32(nn.LayerNorm)`: casts input to float32 via `manual_cast`, runs standard `nn.LayerNorm` (formula `(x-mean)/sqrt(var+eps) * weight + bias`, mean/var over normalized_shape, default `eps=1e-5`, elementwise_affine default True), casts output back to input dtype. Params: `weight`, `bias` of shape `[normalized_shape]`. Used in transformer blocks (see blocks.py) typically as `LayerNorm32(channels, elementwise_affine=False, eps=1e-6)` for the modulated adaLN norm — confirm exact eps/affine at block instantiation (separate component).
- `GroupNorm32(nn.GroupNorm)`: same float32 cast wrapper; standard GroupNorm `(num_groups, num_channels, eps=1e-5, affine=True)`; params `weight`,`bias` `[num_channels]`. Normalizes per-group over (C/groups, spatial).
- `ChannelLayerNorm32(LayerNorm32)`: permutes channel dim to last (`[B,C,*] -> [B,*,C]`), applies LayerNorm32 over C, permutes back. For applying LayerNorm to channel-first tensors.
- `manual_cast(tensor, dtype)`: returns `tensor.type(dtype)` UNLESS torch autocast is enabled (then no-op). At inference without autocast it always casts. For the C++ port: simply compute these norms in float32.

### 6. config.py / __init__.py
- `BACKEND` default `'flash_attn'`, overridable by env `ATTN_BACKEND`. `DEBUG` from `ATTN_DEBUG`. Runtime-only.
- `__init__.py` re-exports everything from full_attn, modules, rope. Public names: `scaled_dot_product_attention`, `MultiHeadAttention`, `MultiHeadRMSNorm`, `RotaryPositionEmbedder`.

## Weight key patterns

For a `MultiHeadAttention` instance at module path `<P>` (e.g. `blocks.{i}.attn`, `blocks.{i}.self_attn`, `blocks.{i}.cross_attn`):

Self-attention (type=="self"):
- `<P>.to_qkv.weight`  shape `[3*channels, channels]`
- `<P>.to_qkv.bias`    shape `[3*channels]`        (present iff qkv_bias=True; default True)
- `<P>.to_out.weight`  shape `[channels, channels]`
- `<P>.to_out.bias`    shape `[channels]`
- if qk_rms_norm:
  - `<P>.q_rms_norm.gamma`  shape `[num_heads, head_dim]`
  - `<P>.k_rms_norm.gamma`  shape `[num_heads, head_dim]`
  (MultiHeadRMSNorm has NO bias and NO buffer; only `gamma`.)

Cross-attention (type=="cross"):
- `<P>.to_q.weight`   shape `[channels, channels]`
- `<P>.to_q.bias`     shape `[channels]`
- `<P>.to_kv.weight`  shape `[2*channels, ctx_channels]`
- `<P>.to_kv.bias`    shape `[2*channels]`
- `<P>.to_out.weight` shape `[channels, channels]`
- `<P>.to_out.bias`   shape `[channels]`
- if qk_rms_norm (qk_rms_norm_cross): `<P>.q_rms_norm.gamma [num_heads,head_dim]`, `<P>.k_rms_norm.gamma [num_heads,head_dim]`

RotaryPositionEmbedder: NO learnable parameters. In the DENSE model the precomputed phases are stored as a parent buffer `rope_phases` (e.g. `<model>.rope_phases`, complex/real tensor). `freqs` is a transient attribute, not registered, recomputed at init — NOT in state_dict. (Some checkpoints may persist `rope_phases` as a buffer; it can equally be recomputed on host from grid coords, so the port need not load it.)

Norm layers (when used by blocks): `<P>.weight`,`<P>.bias` for LayerNorm32/GroupNorm32/ChannelLayerNorm32 (absent if elementwise_affine/affine=False — the adaLN norms in modulated blocks are typically affine=False, so NO weight/bias keys there).

Interpretation of fused qkv weight for the C++ port: row index = `(qkv_idx * num_heads + head_idx) * head_dim + d`, where qkv_idx ∈ {0:q,1:k,2:v}. Same scheme for to_kv with index ∈ {0:k,1:v}.

## GGML notes

Per-op mapping to GGML:

- to_qkv / to_q / to_kv / to_out: plain `ggml_mul_mat` (+ `ggml_add` bias). PyTorch Linear weight is `[out,in]`; ggml_mul_mat(W, x) with W stored as `[in,out]` (ne0=in) gives `[out]`, matching after the usual transpose-on-load convention used elsewhere in llama.cpp ports.

- Fused-qkv split into (3,H,Dh): use `ggml_view`/`ggml_reshape` + `ggml_cont` to extract q,k,v sub-tensors. Plan the reshape `[B,L,3C] -> [Dh, H, 3, L, B]` carefully given ggml's reversed dim order (ne0 is innermost = head_dim). q,k,v are contiguous slices along the "3" dim.

- MultiHeadRMSNorm: this is L2-normalize over head_dim (NOT ggml_rms_norm, which divides by sqrt(mean(x^2))). Note ggml_rms_norm gives `x/sqrt(mean(x^2)+eps) = x/(||x||/sqrt(n))` = `x*sqrt(n)/||x||`. The TRELLIS formula is `x/||x|| * sqrt(head_dim) * gamma = x*sqrt(n)/||x|| * gamma`. So `ggml_rms_norm(x, eps=1e-12 approx)` ALREADY yields `x*sqrt(n)/||x||`, then just multiply by `gamma` (ggml_mul, broadcast [Dh,H]). So: `ggml_mul(ggml_rms_norm(x, 1e-6), gamma)` reproduces it exactly (since sqrt(head_dim) factor is baked into ggml_rms_norm). Pick eps small (1e-6) — original used F.normalize eps 1e-12. Compute in F32.

- RoPE (3D, interleaved adjacent pairs, broadcast over heads, identity-padded high slots): NOT directly ggml_rope (ggml_rope assumes a single position index and NEOX/standard layouts). This needs a CUSTOM op or precomputed cos/sin tables. Simplest: precompute on host two tensors `cos[L, head_dim/2]`, `sin[L, head_dim/2]` from integer grid coords and the freq formula `freqs[i]=1/10000^(i/freq_dim)` with 3-axis concatenation + identity pad. Then apply per pair (2k,2k+1): build via ggml elementwise: split even/odd via strided views, compute `out_even=e*cos - o*sin`, `out_odd=e*sin + o*cos`, interleave back. This is a custom kernel or a few ggml_mul/ggml_add with strided views. Pairing is INTERLEAVED (torch view_as_complex pairs adjacent), so emulating with ggml_rope's NEOX (half-split) mode would be WRONG — must use interleaved or a custom op. Phases are the SAME across heads (broadcast over H), and across the batch for the dense model (static grid), so cos/sin tables are computed once.

- scaled_dot_product_attention: standard `ggml_mul_mat(k,q)` -> scale by 1/sqrt(head_dim) (ggml_scale) -> ggml_soft_max -> ggml_mul_mat(v,...). Or use ggml_flash_attn_ext with scale=1/sqrt(head_dim), no mask, non-causal. Permutes [B,L,H,Dh]->[B,H,L,Dh] map to ggml_permute. No attention mask at inference.

- LayerNorm32: ggml_norm (eps 1e-5 default, or 1e-6 if block uses it) then optional ggml_mul/ggml_add for weight/bias (skip if affine=False). Compute F32. GroupNorm32: ggml_group_norm. ChannelLayerNorm32: permute channel-to-last then ggml_norm then permute back.

Memory: rope cos/sin tables `[L, head_dim/2]` per resolution (e.g. resolution^3 tokens) computed once on host, uploaded as constant buffers. Fused qkv is one big matmul — efficient. No KV cache (full bidirectional attention, single forward).

## Open questions

1. Exact LayerNorm eps and elementwise_affine used inside the transformer BLOCKS (blocks.py / modulated.py) — norm.py only defines the wrappers; the block instantiation (separate component) determines whether adaLN norms have weight/bias keys. From TRELLIS conventions these are almost certainly `eps=1e-6, elementwise_affine=False` for adaLN, but MUST be confirmed when speccing the blocks component.

2. F.normalize default eps is 1e-12; using ggml_rms_norm with a small eps (1e-6) is a close but not bit-exact match. Numerically negligible for normalized inputs but worth a tolerance check against the reference.

3. For the DENSE model the RoPE `dim=3` and coords are an integer grid `meshgrid(arange(resolution))`. Confirm the actual `resolution`, `head_dim`, and whether `head_dim/2` is divisible by 3 (determines how many identity-padded rotary slots exist) from the SparseStructureFlowModel config (separate component). With head_dim divisible-by-2 guaranteed but head_dim/2 maybe not divisible by 3, the trailing identity pad WILL be present and must be reproduced.

4. `rope_phases` is registered as a buffer; need to confirm whether the published safetensors actually contains it or whether it must be recomputed on host (recomputation is straightforward and recommended).

5. Whether any DiT actually sets `qk_rms_norm=True` / `use_rope=True` in the shipped TRELLIS.2 configs (defaults are False). The model configs (sparse_structure_flow / structured_latent_flow) must be read to know which weight keys actually exist. sparse_structure_flow uses `use_rope=(pe_mode=="rope")` so it depends on the config's pe_mode.

6. The 2-arg `scaled_dot_product_attention(q, kv)` path (cross-attn without qk_rms_norm) — for the flash_attn_3 backend it unbinds kv; semantics identical. No issue for the port (always implement as plain q,k,v).
