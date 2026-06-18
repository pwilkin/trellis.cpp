# image_feature

## image_feature — DinoV3FeatureExtractor (DINOv3 ViT-L/16)

Source: `/tmp/TRELLIS.2/trellis2/modules/image_feature_extractor.py` (identical copy in `/tmp/TRELLIS.2/trellis2/trainers/flow_matching/mixins/image_conditioned.py`). The runtime image-conditioning path is `DinoV3FeatureExtractor`. `DinoV2FeatureExtractor` is dead code for TRELLIS.2 (all gen configs use DINOv3) — IGNORE it for the port.

### 0. Which class / model is used
All four image-conditioned generator configs select DINOv3:
- `configs/gen/ss_flow_img_dit_1_3B_64_bf16.json`, `slat_flow_img2shape_dit_1_3B_512_bf16(.../_ft1024).json`, `slat_flow_imgshape2tex_dit_1_3B_512_bf16(.../_ft1024).json`
- `image_cond_model`: `{ "name": "DinoV3FeatureExtractor", "args": { "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m", "image_size": 512 (or 1024 in ft1024 configs) } }`
- Loaded via HuggingFace `transformers.DINOv3ViTModel.from_pretrained(model_name)`, set to `.eval()`. (NOT timm, NOT torch.hub.)
- `image_size` (512 or 1024) is overwritten at inference: `trellis2/pipelines/trellis2_texturing.py:169` does `self.image_cond_model.image_size = resolution` right before calling. So the actual input resolution is the pipeline `resolution` argument; the config value is just a default.

### 1. Preprocessing (exact)
Input is a list of PIL images (the `torch.Tensor` branch is only used when caller already preprocessed). Per image, in order:
1. `img.resize((image_size, image_size), Image.LANCZOS)` — square resize to `image_size × image_size` (e.g. 512×512). LANCZOS interpolation.
2. `np.array(img.convert('RGB')).astype(np.float32) / 255` — RGB, scale to [0,1], dtype float32. Layout HWC.
3. `torch.from_numpy(i).permute(2, 0, 1).float()` → CHW.
4. `torch.stack(...)` → `(B, 3, image_size, image_size)`, moved to CUDA.
5. Normalization (`torchvision.transforms.Normalize`): `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` (standard ImageNet, per-channel: `(x-mean)/std`). NOTE: this is the ImageNet mean/std, NOT the DINOv3-native mean/std `[0.430,0.411,0.296]/[0.213,0.156,0.143]` from the HF preprocessor — TRELLIS deliberately uses ImageNet stats. Replicate exactly.
- No center crop. Square resize only. Pixels expected to already be RGB on (premultiplied) white→black background per pipeline preprocessing (`output[:, :, :3] * output[:, :, 3:4]`), but for the feature extractor itself the only thing that matters is the 5 steps above.

### 2. Patch / token counts
- patch_size = 16. For `image_size = S`, `num_patches = (S/16) × (S/16)`.
  - S=512 → 32×32 = 1024 patch tokens.
  - S=1024 → 64×64 = 4096 patch tokens.
- Prefix tokens: 1 CLS + 4 register tokens = 5 prefix tokens, prepended before patch tokens.
- Total sequence length N = 5 + num_patches (e.g. 1029 for S=512, 4101 for S=1024).

### 3. Forward path actually executed (`extract_features`)
```
image = image.to(patch_embeddings.weight.dtype)
hidden_states = embeddings(image, bool_masked_pos=None)        # (B, N, 1024)
position_embeddings = rope_embeddings(image)                    # (cos, sin), each (num_patches, head_dim=64)
for layer_module in <24 transformer layers>:
    hidden_states = layer_module(hidden_states, position_embeddings=position_embeddings)
return F.layer_norm(hidden_states, hidden_states.shape[-1:])    # final LN over last dim, NO learned weight/bias
```
CRITICAL: the model's own final `self.norm` LayerNorm (with learned weight/bias) is **NOT** applied. Instead a parameter-free `F.layer_norm(x, (1024,))` (eps default 1e-5, weight=None, bias=None) is applied to the raw last-block output (`x_prenorm`-style). The port must do mean/var normalization over the 1024 channel dim with no affine. The HF model's `norm.weight`/`norm.bias` weights exist in the checkpoint but are UNUSED.

### 4. Output tensor
- Shape `(B, N, 1024)` = `(B, 5+num_patches, 1024)`, e.g. `(B, 1029, 1024)` for S=512.
- ALL tokens are returned including CLS + 4 registers + patch tokens (the extractor returns the full sequence; downstream DiT cross-attention consumes the whole thing). No slicing of prefix tokens here.
- dtype = model weight dtype (the checkpoint is float32; pipeline may cast). Treat as fp32/bf16 per loaded weights.

### 5. DINOv3 ViT-L/16 architecture (must reimplement in GGML)
Config `facebook/dinov3-vitl16-pretrain-lvd1689m` (verified from config.json):
- hidden_size D = 1024
- num_hidden_layers = 24
- num_attention_heads = 16  → head_dim = 1024/16 = 64
- intermediate_size = 4096
- patch_size = 16, num_channels = 3, image_size (config default) = 224 (irrelevant; dynamic)
- num_register_tokens = 4
- hidden_act = "gelu"  → exact erf GELU (`0.5*x*(1+erf(x/sqrt(2)))`), NOT tanh approximation
- use_gated_mlp = false → plain 2-layer MLP (up_proj → gelu → down_proj), NO SwiGLU
- mlp_bias = true
- layerscale_value = 1.0 (init; learned per-channel scale, stored in checkpoint)
- rope_theta = 100.0
- layer_norm_eps = 1e-5
- query_bias = true, key_bias = FALSE, value_bias = true, proj_bias = true
- attention_dropout = 0, drop_path_rate = 0 (both inference-irrelevant, identity)
- initializer_range = 0.02 (training only)

#### 5a. Embeddings (`DINOv3ViTEmbeddings`)
- `patch_embeddings`: `nn.Conv2d(3, 1024, kernel_size=16, stride=16)` → non-overlapping patches. Output `(B,1024,H/16,W/16)`, then `.flatten(2).transpose(1,2)` → `(B, num_patches, 1024)`. In GGML this is im2col + matmul (or a strided conv) producing per-patch 1024-vectors in row-major (h then w) order.
- `cls_token`: param `(1,1,1024)`, expanded to batch.
- `register_tokens`: param `(1,4,1024)`, expanded to batch.
- `mask_token`: param `(1,1,1024)` — inference-only NOT used (bool_masked_pos=None). Skip.
- Concatenation order: `cat([cls_token, register_tokens, patch_embeddings], dim=1)` → CLS first, then 4 registers, then patches.
- NO additive learned positional embedding (DINOv3 uses RoPE on patches only).

#### 5b. RoPE (`DINOv3ViTRopePositionEmbedding`) — CUSTOM op needed
Computed in float32 regardless of model dtype. head_dim = 64.
- `inv_freq = 1 / base ** torch.arange(0, 1, 4/head_dim, dtype=float32)` → exponents 0, 4/64, 8/64, ... up to <1 → length head_dim/4 = 16. base=rope_theta=100.0. So `inv_freq[j] = 100.0 ** (-(4j/64))`, j=0..15.
- Patch center coords (`get_patches_center_coordinates`), float32, normalized to [-1,+1]:
  - `coords_h = arange(0.5, num_patches_h) / num_patches_h`, same for w.
  - meshgrid(h,w, indexing="ij") → stack last dim → `(H*W, 2)`, then `coords = 2*coords - 1`. So each patch has (y,x) in [-1,1]; ordering is row-major h-outer, w-inner (matches conv flatten order).
  - Training-only `augment_patches_center_coordinates` (shift/jitter/rescale) is gated by `self.training` → SKIP for inference. `pos_embed_rescale=2.0` etc. are training-only.
- Angles: `angles = 2*pi * patch_coords[:, :, None] * inv_freq[None,None,:]` → shape `(num_patches, 2, 16)`; `.flatten(1,2)` → `(num_patches, 32)` (interleave: first 16 from y-axis, next 16 from x-axis); `.tile(2)` → `(num_patches, 64)` (duplicate the 32 to fill head_dim).
- `cos = cos(angles)`, `sin = sin(angles)`, each `(num_patches, 64)`. Returned cast to model dtype.

Application (`apply_rotary_pos_emb`), only to PATCH tokens (prefix CLS+registers NOT rotated):
- Split q,k along seq into prefix (first 5) and patches (rest).
- `rotate_half(x)`: `x1=x[...,:32]; x2=x[...,32:]; return cat([-x2, x1], -1)`. This is the "half-split" (GPT-NeoX) RoPE convention, NOT interleaved.
- `q_patches = q_patches*cos + rotate_half(q_patches)*sin`; same for k. cos/sin broadcast over (B, heads) dims, shape `(num_patches,64)`.
- Recombine `cat([prefix, rotated_patches], dim=seq)`.
GGML: implement as a custom RoPE that (a) is 2D (y and x sub-blocks), (b) uses half-split rotation, (c) applies only to the patch sub-range of the sequence, (d) precompute cos/sin tables on CPU at init since image_size is fixed per run.

#### 5c. Attention (`DINOv3ViTAttention`) — per layer
- Separate projections (NOT fused qkv): `q_proj`, `k_proj`, `v_proj`, `o_proj`, each `nn.Linear(1024,1024)`.
- biases: q_proj bias YES, k_proj bias NO, v_proj bias YES, o_proj bias YES.
- reshape to `(B, 16, N, 64)`, transpose(1,2).
- apply RoPE to q,k (patch part only).
- scaling = head_dim**-0.5 = 64**-0.5 = 0.125.
- `softmax(q·kᵀ * scaling) · v`, full bidirectional (no mask, is_causal=False).
- transpose back, reshape `(B,N,1024)`, `o_proj`.

#### 5d. LayerScale (`DINOv3ViTLayerScale`)
- `lambda1`: param `(1024,)`, multiplied elementwise. Two per layer (`layer_scale1` after attn, `layer_scale2` after mlp).

#### 5e. MLP (`DINOv3ViTMLP`, non-gated)
- `up_proj`: Linear(1024,4096, bias=true); `act_fn` = gelu; `down_proj`: Linear(4096,1024, bias=true).
- `forward = down_proj(gelu(up_proj(x)))`.

#### 5f. Block forward (`DINOv3ViTLayer`) — exact residual order
```
r = x
x = norm1(x)                       # LayerNorm(1024, eps=1e-5), affine
x = attention(x, pos_emb)
x = layer_scale1(x)
x = drop_path(x) + r               # drop_path = Identity at inference
r = x
x = norm2(x)                       # LayerNorm(1024, eps=1e-5), affine
x = mlp(x)
x = layer_scale2(x)
x = drop_path(x) + r
```
Pre-norm transformer. norm1/norm2 are standard LayerNorm with learned weight+bias, eps=1e-5.

### 6. Inference-only vs training-only
- Training only (skip): `mask_token`/`bool_masked_pos`, `augment_patches_center_coordinates` (shift/jitter/rescale), drop_path, attention_dropout, all `_init_weights`.
- The unused final `self.norm` (model's LayerNorm) — present in checkpoint but NOT applied; the extractor uses parameter-free `F.layer_norm` instead.


## Weight key patterns

DINOv3 ViT-L/16 checkpoint is the standard HF `facebook/dinov3-vitl16-pretrain-lvd1689m` safetensors (single `model.safetensors`, fp32). The state_dict keys follow the `nn.Module` attribute hierarchy of HF `DINOv3ViTModel`.

IMPORTANT VERSION CAVEAT: TRELLIS.2's `extract_features` iterates `self.model.layer` and uses `self.model.embeddings`/`self.model.rope_embeddings` directly. In the CURRENTLY-INSTALLED transformers (4.56.x) the encoder is nested as `self.model.model` (base_model_prefix="model"), so layers live under `model.model.layer.{i}` and `self.model.layer` would raise AttributeError. TRELLIS.2 was written against an EARLIER transformers DINOv3 layout where `DINOv3ViTModel` inlined the layer list as `self.layer` (no intermediate encoder module). The C++ port should NOT depend on transformers; load the actual safetensors and map keys. Expect the checkpoint keys to be ONE of these two layouts — inspect the real file. The Facebook-published checkpoint (4.56.0.dev0) uses the NESTED encoder layout:

Embeddings (no `model.` prefix at top level in DINOv3ViTModel; top-level submodules are `embeddings`, `rope_embeddings`, `model` (encoder), `norm`):
- `embeddings.cls_token`                       (1, 1, 1024)
- `embeddings.mask_token`                       (1, 1, 1024)   [unused at inference]
- `embeddings.register_tokens`                  (1, 4, 1024)
- `embeddings.patch_embeddings.weight`          (1024, 3, 16, 16)
- `embeddings.patch_embeddings.bias`            (1024,)

RoPE: `rope_embeddings.inv_freq` is a non-persistent buffer (persistent=False) → NOT in the safetensors. Recompute it: inv_freq[j]=100.0**(-(4j/64)), j=0..15.

Per-layer i = 0..23, prefix `model.layer.{i}.` (nested-encoder layout) — if the older inlined layout, prefix is `layer.{i}.`:
- `...norm1.weight` (1024,), `...norm1.bias` (1024,)
- `...attention.q_proj.weight` (1024,1024), `...attention.q_proj.bias` (1024,)
- `...attention.k_proj.weight` (1024,1024)            [NO bias — key_bias=false]
- `...attention.v_proj.weight` (1024,1024), `...attention.v_proj.bias` (1024,)
- `...attention.o_proj.weight` (1024,1024), `...attention.o_proj.bias` (1024,)
- `...layer_scale1.lambda1` (1024,)
- `...norm2.weight` (1024,), `...norm2.bias` (1024,)
- `...mlp.up_proj.weight` (4096,1024), `...mlp.up_proj.bias` (4096,)
- `...mlp.down_proj.weight` (1024,4096), `...mlp.down_proj.bias` (1024,)
- `...layer_scale2.lambda1` (1024,)
  (If use_gated_mlp were true there'd also be `mlp.gate_proj.*`, but it is FALSE here — no gate_proj.)

Final:
- `norm.weight` (1024,), `norm.bias` (1024,)   [PRESENT but UNUSED by extractor — do not load/apply]

nn.Linear weight convention: stored as (out_features, in_features); GGML matmul `ggml_mul_mat(W, x)` with W as (in,out) ne layout works directly since ggml stores transposed — load row-major (out,in) and use ggml_mul_mat which treats first arg as [in, out].

To resolve the layout ambiguity definitively the implementer should run `safetensors` key dump on the actual downloaded checkpoint (see open_questions).

## GGML notes

Per-op mapping to GGML primitives:
- Resize LANCZOS + ImageNet normalize: do on CPU (host) before upload. LANCZOS is not in GGML; use a CPU image lib (e.g. stb_image_resize with Lanczos, or implement separable Lanczos) to match `PIL.Image.resize(..., LANCZOS)`. Mismatch here causes feature drift; consider matching PIL's Lanczos exactly. Output (B,3,S,S) fp32, then `(x-mean)/std` per channel.
- Patch embed Conv2d(stride=16, kernel=16, no overlap): implement as im2col into (num_patches, 3*16*16=768) then `ggml_mul_mat` with reshaped weight (768,1024) + bias. Equivalent to ggml_conv_2d but a plain reshape+matmul is simpler and exact for non-overlapping patches. Patch ordering must be row-major (h outer, w inner) to match `.flatten(2).transpose(1,2)`.
- Token concat [CLS, 4 reg, patches]: ggml_concat along seq dim. CLS/registers are just parameter rows broadcast to batch.
- 2D RoPE (CUSTOM): ggml_rope does not support DINOv3's scheme (2D patch-center coords in [-1,1], inv_freq with step 4/head_dim giving 16 freqs, layout = [16 y-angles | 16 x-angles] tiled ×2 to 64, half-split rotate_half, applied ONLY to patch tokens not the 5 prefix tokens). Easiest: precompute cos/sin (num_patches,64) on CPU at init (image_size fixed), upload as constants, and implement application with elementwise mul + a rotate_half built from ggml_view/ggml_neg/ggml_concat (split last dim 64 into 32/32). Apply only to the patch sub-view of q,k (offset 5 tokens). This is the single trickiest op.
- Attention: standard. Separate q/k/v/o matmuls (note k has NO bias). scale 0.125, softmax via ggml_soft_max, no mask. head_dim 64, 16 heads. Can use flash-attn kernel if available; otherwise ggml_mul_mat + soft_max + mul_mat.
- LayerNorm (norm1/norm2): ggml_norm(eps=1e-5) then mul by weight + add bias.
- LayerScale: elementwise ggml_mul by lambda1 vector (broadcast).
- GELU: exact erf gelu = ggml GGML_UNARY_OP_GELU (which is the tanh approx in ggml!) — MUST use ggml_gelu_erf / GGML_UNARY_OP_GELU_ERF to match HF "gelu" (erf), not ggml_gelu (tanh). Verify the ggml build exposes gelu_erf; if not, add it. This is a correctness trap.
- MLP: mul_mat up_proj + bias → gelu_erf → mul_mat down_proj + bias.
- Final parameter-free LayerNorm: ggml_norm(eps=1e-5) with NO weight/bias. Do NOT use the checkpoint's norm.weight/bias.
- DropPath / dropout / mask_token: omit (inference identity / training-only).

Memory: at S=1024, N=4101 tokens × 1024 dims × 24 layers; attention is N²=~16.8M per head × 16 heads — sizable. Consider fp16/bf16 activations and flash attention for the 1024-res path. Output (B, N, 1024) is the cross-attn conditioning; keep on device for the DiT. Weights ~300M params (ViT-L) fp32 ≈ 1.2GB; consider fp16 load.

## Open questions

1. EXACT safetensors key prefix layout: the installed transformers (4.56.x) nests the encoder as `model.model.layer.{i}` while TRELLIS.2 code assumes `self.model.layer`. The implementer MUST dump keys from the actual downloaded `facebook/dinov3-vitl16-pretrain-lvd1689m/model.safetensors` (e.g. `python -c "from safetensors import safe_open; ..."`) to confirm whether layer keys are `model.layer.{i}.attention.q_proj.weight` vs `layer.{i}.attention.q_proj.weight` and whether top prefix is bare (`embeddings.*`, `norm.*`) or `model.*`. Spec lists both; pick by inspection. The repo is gated on HF — needs auth token to download.

2. LANCZOS resize fidelity: PIL Lanczos has a specific kernel/antialias behavior. How exact must the C++ resize match? Small differences propagate through DINOv3. Recommend matching PIL's implementation or accepting minor drift; verify against reference features.

3. Confirm the conditioning consumer uses ALL 1029/4101 tokens (CLS+registers+patches) vs only patch tokens. The extractor returns the full sequence; verify in the DiT cross-attention module (image_conditioned mixin / DiT) that no prefix-token slicing happens downstream before cross-attention. (Out of scope for this file but affects nothing here — extractor returns full sequence.)

4. Pipeline preprocessing before the extractor (alpha premultiply, bbox crop in trellis2_texturing.py get_image around lines 150-157) determines the actual pixels; for the shape/img2shape pipelines confirm the analogous preprocessing path feeds the same normalized RGB. Not in this file.

5. inv_freq buffer is non-persistent (not in checkpoint) — must be recomputed; confirmed from `register_buffer(..., persistent=False)`. No action needed beyond recompute.

6. The two DINOv2 paths (518 resize, x_prenorm, torch.hub) are unused by TRELLIS.2 configs — confirmed all gen configs use DINOv3. Safe to not port DINOv2.
