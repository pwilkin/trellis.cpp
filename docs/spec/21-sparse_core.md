# sparse_core

## sparse_core — SparseTensor abstraction + primitive ops (TRELLIS.2 GGML port)

Authoritative source files (read in full): `/tmp/TRELLIS.2/trellis2/modules/sparse/basic.py`, `linear.py`, `norm.py`, `nonlinearity.py`, `config.py`, `__init__.py`, `spatial/basic.py`, `spatial/spatial2channel.py`, `spatial/__init__.py`. Consumers verified: `/tmp/TRELLIS.2/trellis2/models/structured_latent_flow.py`, `/tmp/TRELLIS.2/trellis2/models/sc_vaes/sparse_unet_vae.py`, `/tmp/TRELLIS.2/trellis2/modules/norm.py`, `/tmp/TRELLIS.2/trellis2/modules/utils.py`.

### 0. The two data tensors (THE layout — transcribe exactly)
A `SparseTensor` is fully described at runtime by two parallel CPU/GPU tensors plus a derived layout:

- `coords`: `torch.int32`, shape `[N, 4]`, columns = `(batch, x, y, z)`. Column 0 is the batch index; columns 1..3 are integer voxel coordinates. **Coords are in `[0, 1023]`** (10-bit per axis; see class docstring "Coords should be in [0, 1023]"). For DIM=3 the trailing 3 columns are spatial.
- `feats`: float (bf16/f16/f32 — see dtype notes), shape `[N, C]` (or `[N, C, ...]` but the flow/VAE path always uses `[N, C]`). Row `i` of `feats` corresponds to row `i` of `coords`. There is NO separate spatial buffer — feats is a flat list of occupied voxels.
- `N` = number of occupied voxels across the whole batch (`feats.shape[0] == coords.shape[0]`).

**Row ordering (CRITICAL invariant):** Rows are NOT morton/z-order sorted by the SparseTensor class itself. The ONLY ordering invariant enforced is **per-batch contiguity**: all rows with `coords[:,0]==0` come first (a contiguous block), then all rows with `coords[:,0]==1`, etc. See `__init__` DEBUG assert lines 419-432: `assert torch.all(self.coords[self.layout[i], 0] == i)` ("The data of batch i is not contiguous"). Within a batch, the order is whatever order the producer emitted (insertion order from the SS decoder / active-voxel extraction). Downsample/spatial2channel re-derive order via `code.unique()` which returns sorted-ascending unique codes, so AFTER a downsample the new coords ARE sorted by the packed linear code (see §6). For our pipeline the active-voxel coords come from stage ① already; we must preserve whatever order the producer gives within batch 0.

**Layout** (`__cal_layout`, lines 467-471): `seq_len = bincount(coords[:,0], minlength=B)`; `offset = cumsum(seq_len)`; `layout[i] = slice(offset[i]-seq_len[i], offset[i])`. So `layout` is just the per-batch row ranges. Cached per-scale in `_spatial_cache`.

**shape** (`__cal_shape`, 461-465): `torch.Size([B, C, ...])` where `B = coords[:,0].max()+1`, rest from `feats.shape[1:]`. **spatial_shape** (`__cal_spatial_shape`, 473-474): `coords[:,1:].max(0)+1` per axis (the dense bounding grid size).

### 1. B=1 simplifications (our entire pipeline)
For the TRELLIS.2 i2-3D path the batch is always 1 (single image -> single object).
- `coords[:,0]` is all zeros. `layout = [slice(0, N)]`. `seqlen=[N]`, `cum_seqlen=[0,N]`, `batch_boardcast_map = zeros(N)`.
- We can store coords as `[N,3]` (drop the batch column entirely) and treat the whole feats buffer as one contiguous block. The batch column only matters for `unique`-code packing in downsample, where it contributes `c0 * OFFSET[0]`; with c0=0 it drops out.
- All the per-batch `for k in range(shape[0])` loops collapse to a single pass.
- `sparse_cat`/`sparse_unbind`/`__getitem__`/`from_tensor_list`/`to_tensor_list`/`VarLenTensor.*`/`reduce`/`to_dense`/`full` are NOT needed at inference for the flow+VAE forward path — skip them.

### 2. SparseLinear (`linear.py`) — INFERENCE: plain dense matmul on feats
`SparseLinear(nn.Linear).forward(input)` = `input.replace(super().forward(input.feats))`. It is exactly `y = feats @ W^T + b` over the `[N,C]` feats matrix, identical to the dense `Linear` you already implemented in `src/dit.cpp`. Coords pass through unchanged. Weight `weight: [out,in]`, `bias: [out]`. In `structured_latent_flow.py`: `input_layer = SparseLinear(in_channels, model_channels)` and `out_layer = SparseLinear(model_channels, out_channels)`. In VAE: `to_subdiv = SparseLinear(channels, 8)`, `to_latent`, `from_latent`, `output_layer`, skip connections. **All are just dense Linear on the feats matrix — reuse the existing Linear helper directly with feats as the `[N,C]` input.**

### 3. Norms — IMPORTANT inference-vs-deadcode distinction
`norm.py` defines `SparseGroupNorm/SparseLayerNorm/SparseGroupNorm32/SparseLayerNorm32`. **These classes are DEAD CODE in TRELLIS.2 models** — `grep` across `trellis2/models/` finds ZERO uses of `SparseLayerNorm`/`SparseGroupNorm`. Do NOT implement their (per-batch, normalize-over-N) semantics for the model graph.

What is ACTUALLY used for normalization on sparse feats (verified in `sparse_unet_vae.py` and the modulated transformer block):
- **`LayerNorm32` applied to `x.feats` directly** (`modules/norm.py` lines 6-11): standard `nn.LayerNorm(C)` over the LAST dim (channels), per row, in float32 then cast back. e.g. `h = x.replace(self.norm1(x.feats))` with `norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)` and `norm2 = LayerNorm32(out, elementwise_affine=False, eps=1e-6)` (affine=False => no weight/bias). This is **identical to the per-token LayerNorm you already have**, just over the `[N,C]` feats matrix instead of `[B,L,C]`.
- **`F.layer_norm(h.feats, h.feats.shape[-1:])`** (lines 380, 499 of unet): non-affine LayerNorm over channels, eps default 1e-5.
- The flow DiT (`ModulatedSparseTransformerCrossBlock`, `transformer/modulated.py`) uses the SAME AdaLN/QK-RMSNorm/affine-cross-norm machinery as the dense DiT you already validated, but operating on the varlen `[N,C]` feats. So **norms reuse your existing LayerNorm + MultiHeadRMSNorm helpers verbatim, applied to feats `[N,C]` (treat N as the token axis, no batch padding since B=1).**

If you ever DID need the dead `SparseLayerNorm`: it permutes feats of one batch to `[1, C, N_b]`, applies `nn.LayerNorm(normalized_shape)` — but note `normalized_shape` there is set to the SHAPE passed in and it normalizes over the channel dim after reshape; it is effectively per-(batch) instance norm over the spatial axis. Not needed.

### 4. Nonlinearities (`nonlinearity.py`) — elementwise on feats
`SparseReLU/SparseSiLU/SparseGELU/SparseActivation` all = `input.replace(act(input.feats))`. Elementwise on `[N,C]`. The flow/VAE path mostly uses raw `F.silu(h.feats)` and GELU-tanh inside the MLP (same as dense DiT). Reuse your existing SiLU/GELU-tanh ops on the feats buffer. No coord interaction.

### 5. config (`config.py`)
Backends: `CONV='flex_gemm'` default, `ATTN='flash_attn'`, `DEBUG=False`. For the C++ port these are irrelevant — we implement our own conv (already have `ggml_conv_3d`) and our own attention. The `config.CONV != torchsparse/spconv` branch in `SparseTensor.__init__` falls into the `else: self.data = {'feats','coords'}` dict path (lines 400-403, 504-505, 522-523) — i.e. plain dict storage, which matches our `{feats,coords}` C++ struct.

### 6. SparseSpatial2Channel — downsample by factor=2, pack 2^DIM block -> ×2^DIM channels (`spatial2channel.py` lines 7-55)
DIM=3, factor f=2, so a 2×2×2=8 block. Exact index math (transcribe):

Given input coords `[N,4]=(b,x,y,z)`, feats `[N,C]`:
1. **Coarse coords**: `coord[i+1] = coords[:,i+1] // f` for i in 0..DIM-1 (batch col unchanged). i.e. `cx=x//2, cy=y//2, cz=z//2`.
2. **Sub-index within block**: `subidx = sum_i (coords[:,1+i] % f) * f**i` = `(x%2)*1 + (y%2)*2 + (z%2)*4` ∈ [0,7]. (NOTE the ordering: x is least-significant, z most-significant.)
3. **Linear code for uniquing** (groups voxels sharing a coarse cell): `MAX[i] = ceil(spatial_shape[i] / f)` (coarse grid dims). `OFFSET = cumprod(reversed(MAX))` reversed, then `+ [1]` appended → list of length DIM+1. `code = b*OFFSET[0] + cx*OFFSET[1] + cy*OFFSET[2] + cz*OFFSET[3]` (with the OFFSET[i] = product of MAX of axes after i; OFFSET[-1]=1 for z). `code, idx = code.unique(return_inverse=True)` → `code` is the sorted unique coarse-cell codes (length `Nc`), `idx[n]` ∈ [0,Nc) maps input row n to its coarse cell.
4. **New coords** (decode code back, length Nc): `new_coords = stack([code//OFFSET[0], (code//OFFSET[1])%MAX[0], (code//OFFSET[2])%MAX[1], (code//OFFSET[3])%MAX[2]])` → `(b, cx, cy, cz)`. Sorted by code.
5. **Scatter feats into packed buffer**: allocate `new_feats_flat = zeros(Nc * 8, C)`. Scatter: `new_feats_flat[idx[n]*8 + subidx[n]] = feats[n]`. Missing sub-positions stay zero. Then `reshape(Nc, 8*C)`.
6. Output `SparseTensor(new_feats[Nc, 8C], new_coords[Nc,4])`, channels ×8, scale ×2.

Inverse mapping is cached: `channel2spatial_{f}` cache stores `(x.coords, idx, subidx)` for the paired upsample.

### 7. SparseChannel2Spatial — upsample, inverse of §6 (`spatial2channel.py` lines 58-93)
Two modes:
- **Paired (cache present)**: uses cached `(new_coords, idx, subidx)` from the matching down step → `new_feats = x.feats.reshape(Nx*8, C/8)[idx*8 + subidx]`; new_coords = the original fine coords; channels /8, scale /2. This is the exact inverse gather.
- **Standalone (cache None, `subdivision` provided)** — the inference path in the VAE decoder. Given `subdivision.feats` `[N, 8]` (a boolean mask per coarse cell of which of the 8 sub-positions are occupied, from `to_subdiv` → `> 0`):
  1. `N_leaf = sub.sum(-1)` per coarse cell (# children).
  2. `subidx = sub.nonzero()[:,-1]` → flattened list of occupied sub-positions, length = total children M.
  3. `new_coords = x.coords.clone(); new_coords[:,1:] *= f` then `repeat_interleave(new_coords, N_leaf)` → M rows.
  4. For i in 0..DIM-1: `new_coords[:,1+i] += (subidx // f**i) % f` → decode sub-position back to fine offset (same x-LSB ordering as §6 step 2).
  5. `idx = repeat_interleave(arange(N), N_leaf)` → parent index per child.
  6. `x_feats = x.feats.reshape(N*8, C/8)`; `new_feats = x_feats[idx*8 + subidx]`. channels /8, scale /2.

### 8. SparseDownsample / SparseUpsample (avg-pool / nearest) — `spatial/basic.py`
- **SparseDownsample(factor, mode='mean'|'max')**: identical coarse-coord + code-unique machinery as §6 steps 1,3,4, but instead of packing into 8C it does `scatter_reduce(zeros[Nc,C], dim=0, index=idx, src=feats, reduce=mean|max, include_self=False)`. Output channels UNCHANGED, scale ×factor.
- **SparseUpsample(factor)**: nearest. Paired-cache: `new_feats = x.feats[idx]` (gather, channels unchanged). Standalone w/ subdivision: same coord-expansion as §7 standalone steps 1-5, then `new_feats = x.feats[idx]` (just repeat parent feats to each child). scale /factor.
These are used in the `nearest` resample_mode of `SparseResBlock3d`. The TRELLIS.2 SS/SLAT decoders we target use `spatial2channel` mode for resampling per the f16c32 checkpoints — confirm against the specific VAE config you load (see open questions).

### 9. `subdivision` / `to_subdiv` (upsample driver)
On upsample blocks: `subdiv = self.to_subdiv(x)` where `to_subdiv = SparseLinear(channels, 8)` → predicts an 8-logit occupancy for the 8 children of each voxel; `subdiv.replace(subdiv.feats > 0)` thresholds at 0 to a bool `[N,8]` mask fed to `SparseChannel2Spatial`/`SparseUpsample` standalone path. This is how the decoder GROWS the active-voxel set (each occupied coarse voxel may spawn up to 8 fine voxels). This is the key generative step for the VAE decoder and must be implemented exactly (threshold > 0, x-LSB sub-position ordering).

## Weight key map

Parameterized submodules in sparse_core and their weight keys (cross-checked vs `/devel/alt/trellis.cpp/docs/spec/keys/`):

SPARSE PRIMITIVES THEMSELVES (Linear/Norm/Spatial) — note which carry weights:
- `SparseLinear` -> `nn.Linear` => keys `<prefix>.weight [out,in]`, `<prefix>.bias [out]`. In SLAT flow DiT (`ckpts__slat_flow_*_dit_1_3B_512_bf16.keys.txt`): `input_layer.weight/bias` ([model_channels=1024 (verify), in_channels], +bias) and `out_layer.weight/bias`. (The SS flow `ckpts__ss_flow_img_dit_1_3B_64_bf16` is the DENSE DiT already done — its input/out are dense Linear, but topologically identical.)
- `SparseSpatial2Channel`, `SparseChannel2Spatial`, `SparseDownsample`, `SparseUpsample`, `SparseReLU/SiLU/GELU`: **NO learnable parameters** — pure index/gather/scatter ops. No keys.
- `SparseGroupNorm/SparseLayerNorm*` (norm.py): DEAD CODE, no keys in any checkpoint. The affine norms that DO appear are dense `LayerNorm32`/`F.layer_norm`:
  - In the VAE ResBlocks: `*.norm1.weight/bias` (elementwise_affine=True), `*.norm2` has NO weight (elementwise_affine=False). These live in the SS/SLAT VAE decoder checkpoints, NOT in this sparse_core component per se but consumed via `.feats`.

VAE-decoder keys that EXERCISE the spatial ops (for the component that wraps sparse_core — confirm prefixes when implementing the decoder, here for traceability), from `ckpts__shape_dec_next_dc_f16c32_fp16.keys.txt` / `ckpts__tex_dec_next_dc_f16c32_fp16.keys.txt`:
- ResBlock subdiv predictor: `<block>.to_subdiv.weight [8, channels]`, `<block>.to_subdiv.bias [8]` (this is a SparseLinear; the only learnable param attached to an upsample).
- `<block>.conv1.*`, `<block>.conv2.*` (SparseConv3d — separate `conv` component, reuse your `ggml_conv_3d` channels-last helper), `<block>.skip_connection.weight/bias` (SparseLinear when channels change; lambda/reshape when spatial2channel — no weights), `<block>.norm1.weight/bias`.

ACTION: grep the exact decoder keys files for `to_subdiv`, `skip_connection`, `norm1`, `input_layer`, `out_layer`, `to_latent`, `from_latent` to bind concrete shapes:
  `grep -E 'to_subdiv|skip_connection|norm1|input_layer|out_layer|to_latent|from_latent' /devel/alt/trellis.cpp/docs/spec/keys/ckpts__*_dec_*.keys.txt /devel/alt/trellis.cpp/docs/spec/keys/ckpts__slat_flow_*.keys.txt`
I did not exhaustively bind every decoder key here because sparse_core itself is mostly weightless; the weighted pieces (Linear, norm affine, conv) belong to the consuming flow/VAE components and reuse existing helpers.

## GGML plan

### C++ / GGML representation of SparseTensor
Define a small struct (NOT a ggml op):
```
struct SparseTensor {
  ggml_tensor* feats;   // [C, N] in ggml ne order (ne0=C, ne1=N) == torch [N,C]
  std::vector<int32_t> coords; // length 3*N (x,y,z); drop batch col for B=1
  int N, C;
  int scale[3];         // Fraction numerator; track for cache keys (optional)
};
```
Store feats with ggml ne0=C (contiguous channels) so every Linear/Norm/activation is a normal 2D op over the leading dim — exactly how your dense DiT treats `[C, L]`. N is the "token" axis (ne1).

### Op-by-op plan (reuse vs new)
1. **SparseLinear** => REUSE existing `Linear` helper unchanged: `ggml_mul_mat(W, feats)` + bias, where feats is `[C_in, N]`. Output `[C_out, N]`. (Same code path as dense DiT token Linear.)
2. **LayerNorm32 on feats / F.layer_norm** => REUSE existing `LayerNorm` helper on `[C, N]` (norm over ne0=C). For affine=False (`norm2`, `F.layer_norm`) pass null weight/bias. eps: 1e-6 for ResBlock norm1/norm2, 1e-5 for the bare `F.layer_norm`. Use `ggml_norm` (not rms) then optional affine.
3. **SiLU/GELU-tanh** => REUSE existing elementwise ops on feats.
4. **Modulated sparse transformer block** => REUSE the entire `ModulatedTransformerCrossBlock` you built in `src/dit.cpp` (AdaLN share_mod, QK-RMSNorm, 3D interleaved RoPE, affine cross-norm, GELU-tanh MLP). The ONLY difference vs dense: tokens = N active voxels (B=1, no padding/mask), and **RoPE positions come from `coords` (x,y,z)** instead of a dense grid. Build the per-token position triplet from the int32 coords array and feed your existing 3D interleaved-pair RoPE. SDPA is full attention over all N tokens (B=1) — reuse your SDPA helper with seq=N. No windowed/serialized attention needed for the SLAT flow (attn_mode='full' in structured_latent_flow.py).
5. **SparseConv3d** => SEPARATE component; reuse `ggml_conv_3d` channels-last. Not part of sparse_core.
6. **SparseSpatial2Channel (down, f=2, 8×)** — NEW host-side index kernel (do on CPU, it's cheap and topology-defining):
   - Compute coarse coords `(x>>1,y>>1,z>>1)` and `subidx=(x&1)|((y&1)<<1)|((z&1)<<2)` per row.
   - Build linear code with `MAX[i]=ceil(spatial[i]/2)` and OFFSET as in spec; sort-unique to get `Nc` cells + `idx[n]`. (Use std::sort on (code,row) or a hash map keyed by packed code; emit unique codes in ASCENDING order to match torch `.unique`.)
   - Allocate `new_feats [8C, Nc]` ggml tensor zero-init; scatter each input column `feats[:,n]` into `new_feats[subidx[n]*C : subidx[n]*C+C, idx[n]]`. (Channel block layout: row r of packed = `subidx*C + c`, matching `reshape(Nc, 8*C)` of a `[Nc,8,C]` torch buffer where dim1=subidx — verify ordering by the reshape: torch packs `[Nc*8, C].reshape(Nc, 8C)` so within a coarse cell the 8 sub-blocks are contiguous, sub-block s occupies channels `[s*C,(s+1)*C)`. Confirm with a reference dump.)
   - Emit `new_coords` from the unique codes. Cache `(idx, subidx, fine_coords)` for the paired upsample.
   - Implement as a custom op via `ggml_map_custom`/precomputed index tensors + `ggml_get_rows`/`ggml_set_rows`, OR simplest: compute indices on host, build a gather/scatter index tensor, and use `ggml_get_rows` on a transposed feats `[N, ...]` view. A scatter (set_rows) is needed for the zero-fill packing; if no set_rows, precompute a dense `[8C, Nc]` placement by building a per-output-row source index (-1 = zero) and use a masked gather.
7. **SparseChannel2Spatial (up, standalone w/ subdivision)** — NEW host-side kernel:
   - Run `to_subdiv` (SparseLinear -> `[8, N]`), threshold `>0` to bool mask.
   - For each parent n, for each s in 0..7 with mask: emit child row with coords `(2x + (s&1), 2y + ((s>>1)&1), 2z + ((s>>2)&1))` and feats = `x_feats[s*C':(s+1)*C', n]` where `C' = C/8` (gather sub-block s of the parent's packed channels). Build host index list `(parent n, subblock s)` -> output row; realize with `ggml_get_rows` on a reshaped `[C', 8*N]`-style view. Output `new_feats [C', M]`, `new_coords [3,M]`.
8. **SparseDownsample(mean)/SparseUpsample(nearest)** — only if the loaded VAE uses `resample_mode='nearest'`. Mean-pool = host idx + `ggml_get_rows` + segmented mean (or scatter-add then divide by counts). Nearest up = pure `ggml_get_rows` gather. Implement on demand.

### Memory / dtype notes
- feats dtype: SLAT flow checkpoints are bf16 (`*_bf16`), VAE decoders fp16 (`*_fp16`). Norms cast to f32 internally (LayerNorm32/manual_cast) — do the norm reductions in f32 then cast back, matching `manual_cast`.
- Index kernels (coords math, unique, subidx) are integer host-side; keep coords as int32 std::vector. Only feats live in ggml.
- B=1 lets us drop the batch column and all layout slicing; N is the only ragged dim and there is no padding.
- The unique-sort in spatial2channel must reproduce torch `.unique` ascending order, because `new_coords` row order = ascending code order, and that order propagates into the next block's RoPE positions and conv neighbor lookups. Validate by dumping a reference SparseTensor.coords after one down step from the Python model.

## Reuse

REUSE (verbatim or near-verbatim) from existing `src/dit.cpp` / `src/flow_runner.cpp` / `src/ss_decoder.cpp`:
- Linear helper -> SparseLinear (input_layer, out_layer, to_subdiv, skip_connection, to_latent, from_latent). Zero new code; just feed feats `[C,N]`.
- LayerNorm helper -> LayerNorm32-on-feats and F.layer_norm-on-feats (per-row channel LN). Pass null affine for elementwise_affine=False norms (norm2, bare F.layer_norm).
- MultiHeadRMSNorm helper -> QK-RMSNorm inside the sparse transformer block (unchanged).
- Interleaved 3D RoPE helper -> sparse self-attn RoPE, but FED FROM coords(x,y,z) instead of a dense meshgrid. The rotation math is identical; only the position source changes.
- SDPA helper -> sparse full self-attn (B=1, seq=N, no mask) and cross-attn to 1024-dim cond (unchanged).
- GELU-tanh MLP, AdaLN share_mod modulation, the whole ModulatedTransformerCrossBlock -> directly reusable as ModulatedSparseTransformerCrossBlock; the sparse version differs ONLY in operating on the varlen feats matrix (token axis = N) and RoPE-from-coords. (Confirmed: structured_latent_flow.py builds ModulatedSparseTransformerCrossBlock with attn_mode='full', share_mod, qk_rms_norm — same flags as the dense SS DiT you validated.)
- ggml_conv_3d channels-last helper -> SparseConv3d (separate component, but already have the primitive).

GENUINELY NEW (no existing analog):
- SparseTensor host struct (feats ggml + int32 coords vector + N,C,scale).
- spatial2channel down-pack: coarse-coord + subidx + linear-code unique-sort + scatter into 8C buffer (with zero-fill for missing sub-positions). Index/scatter kernel + ordering must match torch .unique ascending.
- channel2spatial up-unpack standalone: to_subdiv -> threshold>0 -> child coord expansion (x-LSB sub ordering) + sub-block gather. This is the generative voxel-growth step.
- (Conditional) SparseDownsample mean-pool scatter-reduce and SparseUpsample nearest gather, only if the loaded VAE config uses resample_mode='nearest'.
- Reproducing torch `.unique(return_inverse=True)` ascending semantics in C++ (sort by packed code, dedup, build inverse map).

EXPLICITLY SKIP (dead at inference / training-only):
- norm.py SparseGroupNorm/SparseLayerNorm classes (unused by any model — verified by grep over trellis2/models).
- VarLenTensor and all its arithmetic/reduce/to_dense/elemwise broadcast machinery.
- sparse_cat/sparse_unbind/__getitem__/from_tensor_list/to_tensor_list/full (multi-batch / data-loading helpers).
- `self.training` branches in spatial ops (the `subdivision` bool tensor registered only when training=True in Down/Spatial2Channel; at inference the up step instead reads to_subdiv predictions). The PAIRED cache path is also effectively a training/autoencode convenience; the decoder uses the standalone subdivision path.
- spconv/torchsparse backend branches (config.CONV); we use the plain dict path semantics.

## Open questions

1. **VAE resample_mode actually loaded**: `SparseResBlock3d` supports `resample_mode in {'nearest','spatial2channel'}`. The `*_next_dc_f16c32` decoder name and `to_subdiv`+spatial2channel pairing strongly suggest spatial2channel, but I did not open the model-construction config that instantiates the decoder. CONFIRM by grepping the decoder config / keys: `grep -E 'to_subdiv|resample|spatial2channel|conv1' /devel/alt/trellis.cpp/docs/spec/keys/ckpts__shape_dec_next_dc_f16c32_fp16.keys.txt` and checking conv1 out-channel ratios (×8 up / ÷8 down indicates spatial2channel). If 'nearest', I must implement SparseDownsample/Upsample instead of Spatial2/Channel2Spatial.

2. **Channel packing order in spatial2channel reshape**: torch does `new_feats[idx*8+subidx]=feats` on a `[Nc*8, C]` buffer then `.reshape(Nc, 8*C)`. I asserted sub-block s occupies output channels `[s*C,(s+1)*C)`. This should be verified against a reference tensor dump (run the Python `SparseSpatial2Channel` on a tiny coords set and dump `out.feats[0]`), because getting it wrong silently corrupts every downstream conv/linear.

3. **subidx axis ordering (x LSB vs z LSB)**: code uses `sum(subidx[...,i]*f**i for i in range(DIM))` with column order (x,y,z) => x is LSB. I propagated this to channel2spatial decode. Confirm the SS decoder's coordinate convention (x,y,z vs the [D,W,H,C] channels-last conv layout you already use in ss_decoder.cpp) so coords passed to conv neighbor lookup and RoPE match. Dump one decoder-stage coords from Python and diff against the C++ output.

4. **Coord range / int width**: docstring says coords in [0,1023] (10-bit). Confirm max grid size for the SLAT VAE (64^3 for SS, larger for SLAT?) so the linear-code packing (MAX/OFFSET products) doesn't overflow int32 at the largest scale; use int64 for the packed code if MAX product can exceed 2^31.

5. **Whether SLAT flow uses RoPE-from-coords or APE**: structured_latent_flow.py picks `pe_mode` ('rope' or 'ape'); the block sets `use_rope=(pe_mode=='rope')`. Confirm the loaded config's pe_mode and rope_freq to wire the right position embedding (you have RoPE; if 'ape' you need AbsolutePositionEmbedder added with coords as positions).
