# sparse_core


# TRELLIS.2 `sparse_core` — Implementation Spec (GGML/C++ port)

This covers `trellis2/modules/sparse/{basic,linear,norm,nonlinearity,config}.py`, `sparse/__init__.py`, and `sparse/spatial/{basic,spatial2channel,__init__}.py`. It excludes sparse conv and attention (separate components). All of these modules are pure tensor ops with **no learned parameters** except `SparseLinear`, `SparseGroupNorm`, `SparseLayerNorm` (which inherit `nn.Linear`/`nn.GroupNorm`/`nn.LayerNorm`).

## 0. Backend config (`config.py`)
- Module-level globals, set at import and overridable via env vars:
  - `CONV = 'flex_gemm'` (DEFAULT). Env `SPARSE_CONV_BACKEND` (one of `none|spconv|torchsparse|flex_gemm`).
  - `ATTN = 'flash_attn'` (DEFAULT). Env `SPARSE_ATTN_BACKEND` or fallback `ATTN_BACKEND`.
  - `DEBUG = False`. Env `SPARSE_DEBUG == '1'`.
- **CRITICAL for the port:** `CONV` controls the internal storage of `SparseTensor.data`. For the default `'flex_gemm'` (and `'none'`), the code falls into the `else` branch everywhere: `self.data` is a plain Python dict `{'feats': Tensor, 'coords': Tensor}`. There is NO torchsparse/spconv object. So a SparseTensor is logically just **(feats, coords, _shape, _scale, _spatial_cache)**. The C++ port should model it exactly that way: a struct holding `feats` (N×C), `coords` (N×4 int32), a logical batch shape, a scale triple, and a cache map.

## 1. Data structure: `SparseTensor` (extends `VarLenTensor`)

### Members
- `feats`: `[N, C]` (C may be multi-dim spatial e.g. `[N, C1, C2]` in general, but in this codebase always `[N, C]`). dtype float16/bfloat16/float32.
- `coords`: `[N, 4]` int32, columns are **(batch, x, y, z)**. Comment in code: "Coords should be in [0, 1023]" (10-bit per axis). Column 0 is batch index `b`; columns 1..3 are spatial `(x,y,z)`.
- `_shape`: optional `torch.Size`. `shape[0]` = batch size B = `coords[:,0].max()+1`; `shape[1:]` = feats channel dims.
- `_scale`: tuple of 3 `Fraction`s, default `(Fraction(1,1), Fraction(1,1), Fraction(1,1))`. Tracks cumulative spatial stride per axis. Used **only** as the cache key (`str(self._scale)`) — different resolution levels store their caches under different scale strings. Down/upsample multiply/divide it.
- `_spatial_cache`: dict keyed by `str(scale)` → dict of cached derived tensors (layout, seqlen, downsample maps, etc.).

### Layout / ordering (NOT z-order at the data level)
- **Data layout requirement: rows belonging to the same batch must be contiguous** ("Data corresponding to a same batch should be contiguous"). Within a batch the row order is whatever the producer emitted — there is no enforced z-order/Morton ordering in the SparseTensor container itself. (Morton/serialization ordering exists separately in `serialize.py` / attention, not here.)
- `layout` = list of `slice`s, one per batch element, computed by `__cal_layout`:
  - `seq_len = torch.bincount(coords[:,0], minlength=B)` → count per batch.
  - `offset = cumsum(seq_len)`; `layout[i] = slice(offset[i]-seq_len[i], offset[i])`.
  - i.e. contiguous runs. This relies on the contiguity invariant above.
- `seqlen` `[B]` int64 = per-batch row counts. `cum_seqlen` `[B+1]` = `[0, cumsum(seqlen)]`. `batch_boardcast_map` `[N]` = `repeat_interleave(arange(B), seqlen)` → maps each row to its batch index. These three are the workhorses for any per-batch broadcast/reduce.
- `spatial_shape` = `torch.Size((coords[:,1:].max(0)+1))` → per-axis extent `(X,Y,Z)`.

### Construction
- `SparseTensor(feats, coords, shape=None)` — default path (`flex_gemm`/`none`) stores dict. Coords expected int32. No reordering done.
- `SparseTensor.from_tensor_list(feats_list, coords_list)`: cats feats; for each coord list element, **overwrites column 0** with batch index `i`: `coord = cat([full_like(coord[:, :1], i), coord[:, 1:]], dim=1)`; cats. So input coords[:, 1:] are spatial; col0 replaced.
- `to_tensor_list()` → splits by `layout`.
- `SparseTensor.full(aabb, dim=(N,C), value, ...)`: builds dense grid of coords over `aabb=[x0,y0,z0,x1,y1,z1]` inclusive via meshgrid `indexing='ij'`, ordering `(x,y,z)`, replicated across N batches, feats filled with `value`. int32 coords.

### `replace(feats, coords=None)`
- Default path: new dict `{'feats': feats, 'coords': old or new coords}`. Carries `_scale` and `_spatial_cache` forward. New `_shape = [old shape[0]] + feats.shape[1:]`. **Spatial cache is shared/forwarded** — important: ops that don't change coords reuse the cache (so e.g. a Linear keeps the layout cache).

### `to_dense()` (default path)
```
ret = zeros(B, C, *spatial_shape)   # note: channel before spatial
idx = [coords[:,0], slice(None)] + unbind(coords[:,1:], 1)   # (b, :, x, y, z)
ret[idx] = feats
```
Output dense layout: `[B, C, X, Y, Z]`.

### Indexing `__getitem__(idx)` (batch selection)
- `idx` int / slice / int-list / bool-mask / int-tensor → selects batch elements. For each selected old batch, clones its coord rows and **renumbers col0 to the new contiguous batch index**, re-cats feats/coords, rebuilds layout cache. (Reindexes batches 0..len(idx)-1.)

### Elementwise ops (`__add__`, `__sub__`, `__mul__`, `__truediv__`, and r-variants, `__neg__`)
- Inherited via `__elemwise__`. If `other` is a dense `torch.Tensor`, it tries `broadcast_to(self.shape)` then indexes with `batch_boardcast_map` → i.e. a per-batch scalar/vector of shape `[B, ...]` is expanded to `[N, ...]` by repeating each batch's value across its rows. If `other` is a (Var)SparseTensor, uses its `.feats`. Then `op(self.feats, other)` and `replace`. For the port: elementwise on feats; for "[B,...]" operands gather via batch_boardcast_map.

### Reductions (`reduce/mean/sum/prod/std`) — inherited from VarLenTensor
- `reduce(op, dim, keepdim)`: first reduces `feats` over `dim` (torch semantics over the flat feats `[N, ...]`). If `dim is None` or includes 0 → global reduce returned directly. Otherwise (reducing non-batch dims) applies `torch.segment_reduce(red, reduce=op, lengths=self.seqlen)` to reduce **within each batch segment** → output `[B, ...]`. `std` computed as `sqrt(mean(x^2) - mean(x)^2)`.

### cat / unbind
- `sparse_cat(inputs, dim=0)`: cat feats; for coords, offset each input's col0 by running batch count `start += input.shape[0]`. dim≠0 → cat feats along feature dim, `replace`.
- `sparse_unbind(input, dim=0)` → `[input[i] for i in range(B)]`.

### Spatial cache API
- `register_spatial_cache(key, value)` / `get_spatial_cache(key)` keyed under `str(self._scale)`. `clear_spatial_cache()` resets. This is how down/upsample share the gather/scatter index maps between paired modules at the same scale.

## 2. `SparseLinear` (`linear.py`)
- `class SparseLinear(nn.Linear)`. `forward(input) -> input.replace(super().forward(input.feats))`.
- It's just `nn.Linear` applied row-wise to `feats [N, in] -> [N, out]`. Bias optional (default True).
- **Weights:** standard `nn.Linear` → `{prefix}.weight [out_features, in_features]`, `{prefix}.bias [out_features]`. GGML: `ggml_mul_mat(W, feats) + bias` per row (W stored transposed as usual).

## 3. Norms (`norm.py`) — IMPORTANT semantics
All four normalize **per batch element across that batch's tokens AND channels jointly** by reshaping feats into a fake `[1, C, L]` tensor (channel = num_channels, L = number of tokens in that batch).

### `SparseGroupNorm(num_groups, num_channels, eps=1e-5, affine=True)` (extends `nn.GroupNorm`)
```
for k in range(B):
    bfeats = feats[layout[k]]                       # [L_k, C]
    bfeats = bfeats.permute(1,0).reshape(1, C, -1)  # [1, C, L_k]
    bfeats = nn.GroupNorm.forward(bfeats)           # GN over groups, stats over (C/G channels × L_k)
    bfeats = bfeats.reshape(C, -1).permute(1,0)     # [L_k, C]
    nfeats[layout[k]] = bfeats
```
So GroupNorm here computes mean/var over **all spatial tokens within the batch, grouped by channel groups** (standard GroupNorm with spatial=L_k). `num_channels = C`, `num_groups = G`. affine weight/bias shape `[C]`.

### `SparseLayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True)` (extends `nn.LayerNorm`)
- Same reshape to `[1, C, L_k]` then `nn.LayerNorm.forward`. **NOTE the subtlety:** because it reshapes to `[1, C, L]` and `normalized_shape` is the LayerNorm's configured shape, LN normalizes over the trailing dims matching `normalized_shape`. Given the reshape produces `[1, C, L_k]`, with `normalized_shape == [C, L_k]`? No — `normalized_shape` is fixed at construction (cannot equal variable L_k). In practice this LN is constructed with `normalized_shape = C`? **OPEN: must confirm how it's instantiated** (see open_questions). If `normalized_shape=C`, LN over the last dim of `[1,C,L]` = over L (the token axis) per (1,C) — which would be unusual. Most likely usage normalizes over C; the permute makes C the middle dim, so it would NOT be the normalized axis. This norm's exact axis is the single biggest risk; verify against an actual constructed module + a numeric reference run. (Real code is quoted above verbatim; do not guess in C++ — replicate the reshape + call torch reference to capture the axis, then hardcode.)

### `SparseGroupNorm32` / `SparseLayerNorm32`
- Subclasses that cast feats to float32 before, run parent forward, cast back to original dtype. Cast uses `manual_cast(tensor, dtype)` which is `tensor.type(dtype)` unless torch autocast enabled (`utils.py:68`). For the port (no autocast): always cast to fp32 for the norm math, cast back. Stats MUST be computed in fp32.
- eps default `1e-5`.

### GGML notes for norms
- Cannot use a single global GGML norm op directly because normalization is **per-batch-segment** (variable length). Implement as a loop/segmented op over `layout[k]` segments, or precompute per-batch mean/var via the `batch_boardcast_map`/`seqlen` segment-reduce. GroupNorm additionally groups channels. This is a **custom segmented norm op**.

## 4. Nonlinearities (`nonlinearity.py`)
- `SparseReLU(nn.ReLU)`, `SparseSiLU(nn.SiLU)`, `SparseGELU(nn.GELU)`, `SparseActivation(activation)`. All just apply the activation to `feats` then `replace`. No params.
- SiLU: `x*sigmoid(x)`. GELU: torch default = exact `0.5*x*(1+erf(x/sqrt2))` (no `approximate='tanh'` specified → exact). GGML: `ggml_silu`, `ggml_gelu` (note ggml_gelu is tanh-approx; for exact use `ggml_gelu_erf` if available — verify which GELU variant is needed).

## 5. Spatial down/up sampling — THE INDEX MATH (heart of the port)

There are two families, both downsample by `factor` (used with `factor=2` everywhere in the VAE):
- `spatial/basic.py`: `SparseDownsample` (pool, channels unchanged) + `SparseUpsample` (nearest copy).
- `spatial/spatial2channel.py`: `SparseSpatial2Channel` (pack 2×2×2 → 8C) + `SparseChannel2Spatial` (inverse).

`DIM = coords.shape[-1] - 1 = 3` (x,y,z).

### 5.1 Common coarse-coordinate computation (used by both Downsample and Spatial2Channel)
```
coord = list(coords.unbind(-1))            # [b, x, y, z]
for i in range(DIM): coord[i+1] //= factor # floor-divide spatial by factor  -> coarse (b, X', Y', Z')
MAX = [ceil(s/factor) for s in spatial_shape]                   # coarse grid extent per axis (X',Y',Z')
OFFSET = cumprod(reverse(MAX)).tolist()[::-1] + [1]
       # For MAX=[X',Y',Z']: cumprod of [Z',Y',X'] = [Z', Z'*Y', Z'*Y'*X'], reversed -> [Z'*Y'*X', Z'*Y', Z'] then +[1]
       # => OFFSET = [Y'*Z', Z', 1] for the SPATIAL part... but note OFFSET has DIM+1=4 entries:
       #    OFFSET[0] = X'*Y'*Z' (multiplies batch b), OFFSET[1]=Y'*Z', OFFSET[2]=Z', OFFSET[3]=1
code = sum(c*o for c,o in zip(coord, OFFSET))   # linear code = b*(X'Y'Z') + X'*(Y'Z') + Y'*Z' + Z'
code, idx = code.unique(return_inverse=True)    # unique coarse cells; idx[N] maps each fine row -> coarse-cell index
new_coords = stack([code//OFFSET[0],                                    # b
                    (code//OFFSET[1]) % MAX[0],                         # X' (=x)
                    (code//OFFSET[2]) % MAX[1],                         # Y'
                    (code//OFFSET[3]) % MAX[2]], dim=-1)                # Z'
```
- `idx`: `[N]` int — for each fine voxel, the index of its coarse cell in `new_coords`. This is the scatter/gather map.
- `code.unique` returns **sorted** unique codes → `new_coords` is in ascending linear-code order, i.e. lexicographic by (b, X', Y', Z'). This defines the coarse ordering deterministically (port must sort the same way: by `b*X'Y'Z' + x*Y'Z' + y*Z' + z`). **Within-batch contiguity is preserved** because b is the most-significant term.

### 5.2 `SparseDownsample(factor, mode='mean'|'max')` — `spatial/basic.py`
- Channels unchanged. Pools fine voxels into their coarse cell:
```
new_feats = scatter_reduce(zeros(M, C), dim=0,
              index=idx[:,None].expand(-1,C), src=feats,
              reduce=mode ('mean'|'max'), include_self=False)
```
  M = number of coarse cells. `mean` = average of fine feats in each cell; `max` = elementwise max.
- `out._scale = scale * factor` (per axis). `out._shape = x._shape`.
- Caches (under fine scale on x): `downsample_{factor} = (new_coords, idx)`. On out (coarse scale): `upsample_{factor} = (x.coords, idx)`, `shape = MAX`.
- `subdivision` cache only built `if self.training` → **inference can skip**. (subdivision `[M, factor**DIM]` bool marking which of the 8 sub-cells are occupied, via `subidx = sum(coords[:,1:]%factor [i] * factor**i)` = local index within block, ordering **x fastest? No: factor**i with i over (x,y,z) → x has weight 1, y weight factor, z weight factor**2**. i.e. subidx = lx + factor*ly + factor**2*lz.)

### 5.3 `SparseUpsample(factor)` — `spatial/basic.py`
- Inverse of Downsample. Nearest-neighbor copy: each fine voxel takes its coarse cell's feats.
- Cached path (paired with Downsample at same scale): reads `upsample_{factor} = (new_coords=x.coords_fine, idx)`; `new_feats = x.feats[idx]` (gather: fine row i copies coarse row idx[i]). out coords = the cached fine coords.
- Uncached path (needs `subdivision` SparseTensor): reconstructs fine coords from subdivision bitmask:
```
sub = subdivision.feats              # [M, factor**DIM] bool
N_leaf = sub.sum(-1)                 # occupied children per coarse cell
subidx = sub.nonzero()[:, -1]       # local child index for each occupied child
new_coords = x.coords.clone(); new_coords[:,1:] *= factor
new_coords = repeat_interleave(new_coords, N_leaf, dim=0)
for i in range(DIM): new_coords[:, i+1] += (subidx // factor**i) % factor   # decode local x,y,z
idx = repeat_interleave(arange(M), N_leaf)        # gather index into coarse feats
new_feats = x.feats[idx]
```
- `out._scale = scale / factor`. Channels unchanged. `out._shape = x._shape`.

### 5.4 `SparseSpatial2Channel(factor=2)` — `spatial2channel.py` (downsample, C → C*factor**DIM)
- Coarse coords identical to 5.1. Additionally computes per-voxel sub-block index:
```
subidx = coords[:,1:] % factor                       # [N,3] local (lx,ly,lz)
subidx = sum(subidx[...,i] * factor**i for i in range(DIM))   # = lx + factor*ly + factor**2*lz  (x fastest)
```
- Scatters fine feats into a packed buffer, then reshapes so each coarse cell has `factor**DIM` slots × C channels:
```
new_feats = zeros(M * factor**DIM, C)
new_feats[idx * factor**DIM + subidx] = feats        # place each fine voxel into its slot
out_feats = new_feats.reshape(M, factor**DIM * C)     # 8*C for factor=2,DIM=3
```
  **Empty sub-slots are left as zeros** (occupied-only fine voxels are written). Channel packing order: slot s = `lx + 2*ly + 4*lz`, and within reshape the layout is `[slot, C]` flattened → final channel index = `slot*C + c`.
- `out._shape = [B, C * factor**DIM]`. `out._scale = scale * factor`.
- Caches: on x `spatial2channel_{factor} = (new_coords, idx, subidx)`; on out `channel2spatial_{factor} = (x.coords, idx, subidx)`, and `shape = MAX`. `subdivision` only if training.

### 5.5 `SparseChannel2Spatial(factor=2)` — `spatial2channel.py` (upsample, C*factor**DIM → C)
- Inverse of 5.4. Unpacks the `factor**DIM` slots back into fine voxels.
- Cached path (paired): reads `channel2spatial_{factor} = (new_coords=fine coords, idx, subidx)`:
```
x_feats = x.feats.reshape(N_coarse * factor**DIM, C)   # split packed channels into slots
new_feats = x_feats[idx * factor**DIM + subidx]        # gather the right slot per fine voxel
```
- Uncached path: same subdivision reconstruction as 5.3 to get `new_coords`, `idx`, plus `subidx = sub.nonzero()[:,-1]`, then the same `x_feats[idx*factor**DIM + subidx]` gather.
- `out._shape = [B, C // factor**DIM]`. `out._scale = scale / factor`.

### 5.6 Worked numbers (factor=2, DIM=3)
- `factor**DIM = 8`. A coarse cell aggregates the 2×2×2 block of fine voxels whose fine coords floor-divide to it.
- Local slot for fine voxel with local coords (lx,ly,lz)∈{0,1}³: `slot = lx + 2*ly + 4*lz` (x fastest, z slowest).
- Spatial2Channel out channel for fine channel c in slot s: `s*C + c`.
- Coarse linear ordering for sorting M cells: `b*X'Y'Z' + x*(Y'Z') + y*Z' + z` (ascending), X'=ceil(X/2) etc.

## 6. Exports surface (`__init__.py`)
Lazy-loaded attributes (module map): from `basic`: `VarLenTensor, varlen_cat, varlen_unbind, SparseTensor, sparse_cat, sparse_unbind`; from `norm`: the 4 norms; from `nonlinearity`: 4 activations; from `linear`: `SparseLinear`; from `spatial`: `SparseDownsample, SparseUpsample, SparseSubdivide, SparseSpatial2Channel, SparseChannel2Spatial, sparse_nearest_interpolate, sparse_trilinear_interpolate`. (Some of these — Subdivide, interpolate, serialize — live in files not in this component's read list.)


## Weight key patterns


Only three of these modules carry learned parameters; the rest are stateless tensor ops (no keys).

SparseLinear (inherits nn.Linear) — keys at its attribute prefix:
  {prefix}.weight   shape [out_features, in_features]
  {prefix}.bias     shape [out_features]   (present unless bias=False)

SparseGroupNorm / SparseGroupNorm32 (inherit nn.GroupNorm) — if affine=True (default):
  {prefix}.weight   shape [num_channels]
  {prefix}.bias     shape [num_channels]
  (no running stats — GroupNorm has none)

SparseLayerNorm / SparseLayerNorm32 (inherit nn.LayerNorm) — if elementwise_affine=True (default):
  {prefix}.weight   shape = normalized_shape (typically [C])
  {prefix}.bias     shape = normalized_shape

The "32" subclasses add NO extra keys (only override forward to cast fp32). So a SparseGroupNorm32 still serializes as plain {prefix}.weight / {prefix}.bias.

Stateless (NO keys, do not appear in state_dict):
  SparseReLU, SparseSiLU, SparseGELU, SparseActivation,
  SparseDownsample, SparseUpsample, SparseSpatial2Channel, SparseChannel2Spatial,
  SparseTensor / VarLenTensor (data containers).

{prefix} is determined by the enclosing module's attribute name (e.g. a block holding self.norm = SparseGroupNorm32(...) and self.proj = SparseLinear(...) yields "....norm.weight", "....norm.bias", "....proj.weight", "....proj.bias"). The exact prefixes come from the consuming models (sparse_unet_vae.py etc.), not from this component.


## GGML notes


Storage model (default CONV='flex_gemm'/'none'): a SparseTensor is just (feats [N,C] tensor, coords [N,4] int32, batch B, scale triple, cache map). No torchsparse/spconv object — the C++ struct mirrors this directly. coords columns = (b, x, y, z), int32, values in [0,1023]. Rows of the same batch MUST stay contiguous (layout via bincount/cumsum relies on it).

Per-op GGML mapping:
- SparseLinear: ggml_mul_mat(W, feats) + broadcast bias. Trivial (row-wise dense matmul). Standard.
- SparseSiLU/ReLU/GELU: ggml_silu / ggml_relu / ggml_gelu(_erf). Note torch nn.GELU here is EXACT erf (no tanh approx) — prefer ggml_gelu_erf; verify availability.
- SparseGroupNorm/LayerNorm (+32 variants): CUSTOM segmented norm. Normalization is per-batch-segment (variable token count) with the reshape feats[layout[k]] -> [1, C, L_k]. GroupNorm groups channels (G groups over C, stats over (C/G)×L_k). Not expressible with a single ggml_norm/ggml_group_norm because segments are ragged. Implement either: (a) loop over batch segments using seqlen/cum_seqlen offsets and call a dense group/layer norm per segment, or (b) a custom op computing per-(batch,group) mean/var via segment reduction using batch_boardcast_map + seqlen, then normalize+affine. MUST compute stats in fp32 (the *32 classes cast up; do this for all to match). eps=1e-5. The LayerNorm normalized axis is ambiguous due to the permute-to-[1,C,L] trick — capture from a torch reference numeric run before hardcoding (see open_questions).
- Reductions (mean/sum/prod/std) and elementwise broadcast of [B,...] operands: need segment-reduce (torch.segment_reduce / scatter_reduce) and a gather by batch_boardcast_map. Custom small kernels or CPU loops; inference path may not need most of these.

Down/Up sampling — the critical custom ops (all are gather/scatter by precomputed index maps):
- Coarse-cell assignment: compute linear code = b*X'Y'Z' + x*Y'Z' + y*Z' + z with X'=ceil(X/2) etc., then a UNIQUE-with-inverse (sorted) to get new_coords (ascending code order) and idx[N] (fine->coarse map). Need a sort+unique+inverse primitive (CPU or custom CUDA). This is the only non-trivial part; everything else is gather/scatter using idx.
- SparseDownsample: scatter_reduce over idx with reduce=mean|max, include_self=False (i.e. uninitialized cells start empty; mean = sum/count, max = max over members). Custom scatter-reduce op (CUDA atomics or sorted-segment reduce). Channels unchanged.
- SparseUpsample: pure gather new_feats = feats[idx_fine] (each fine voxel copies its coarse cell). ggml_get_rows-style gather.
- SparseSpatial2Channel (C->8C): scatter feats into buffer[idx*8 + subidx] (subidx = lx + 2*ly + 4*lz), zero-fill empty slots, reshape [M, 8*C]. Scatter (no reduction; each target written once). Empty slots remain 0. Custom scatter op + view.
- SparseChannel2Spatial (8C->C): view feats as [N*8, C], gather rows [idx*8 + subidx]. ggml_get_rows after reshape.
- include_self=False semantics for mean: divide by actual member count, not member_count+1; max ignores the zero init. Replicate exactly.

Index/cache reuse: paired down+up modules share idx via _spatial_cache keyed by str(scale). In the C++ port, precompute idx/new_coords once per resolution transition and reuse for the inverse (decoder) — no need to recompute unique. The 'subdivision' bitmask path is TRAINING-ONLY (guarded by self.training); inference uses the cached (coords, idx) path, so the subdivision reconstruction (nonzero/repeat_interleave) can be omitted in the port unless decoder runs without a matching encoder cache.

Memory: feats are the bulk; coords are N×4 int32 (cheap). spatial2channel temporarily allocates M*8*C zeros then reshapes — size out = M*8*C, comparable to input N*C since M ≈ N/(occupied fraction). Keep dtype = feats dtype except norms (fp32 internally).

Inference-only vs training-only: everything here is inference-safe EXCEPT the `if self.training:` subdivision-cache blocks in SparseDownsample/SparseSpatial2Channel (skip). config.DEBUG asserts are debug-only (skip).


## Open questions


1. SparseLayerNorm axis ambiguity (HIGH RISK). norm.py reshapes feats[layout[k]] -> permute(1,0) -> [1, C, L_k] then calls nn.LayerNorm.forward, then reshapes back. The normalized axis depends on the LayerNorm's `normalized_shape` set at construction, which cannot equal the variable L_k. Need to find where SparseLayerNorm/SparseLayerNorm32 is instantiated (grep the models, e.g. transformer/elastic blocks) to learn `normalized_shape`, and run a torch numeric reference to confirm whether it normalizes over C (channels) or over L (tokens). Do NOT hardcode in C++ until confirmed. (SparseGroupNorm is unambiguous: standard GroupNorm with spatial=L_k.)

2. GELU variant: nn.GELU() default is exact (erf). Confirm no module sets approximate='tanh'. Pick ggml_gelu_erf vs ggml_gelu accordingly.

3. Confirm coords dtype/range at runtime: code asserts [0,1023] (10-bit). For the linear-code packing in down/upsample, the code uses ACTUAL coarse extents MAX=ceil(spatial_shape/2), not a fixed 1024 — so packing is data-dependent. Ensure the C++ unique/sort uses the same MAX-derived offsets (per-tensor), not a global constant, to match new_coords ordering bit-for-bit.

4. Which CONV backend the shipped checkpoint/runtime uses. Default is 'flex_gemm' (dict storage path). If the released inference config sets SPARSE_CONV_BACKEND=spconv/torchsparse, SparseTensor.data is a different object but feats/coords semantics are identical for THIS component (only conv differs). Safe to assume dict model for the port; verify the launch env / config does not force a backend that reorders coords.

5. SparseDownsample default mode: 'mean'. Confirm the VAE/encoder uses 'mean' vs 'max' at each call site (sparse_unet_vae.py uses SparseDownsample(2) -> default 'mean'; double-check no 'max' usages elsewhere).

6. 'subidx' bit ordering: derived as sum(local[i]*factor**i) with i over (x,y,z) -> x fastest (weight 1), z slowest (weight 4). Same convention in Spatial2Channel scatter and in the training subdivision. Confirm the consuming SparseSubdivide/conv modules use the same ordering so channel packing aligns with weights (relevant when Spatial2Channel output feeds a Linear whose weights assume a specific 8*C channel order).

