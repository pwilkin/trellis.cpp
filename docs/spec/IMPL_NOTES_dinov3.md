# DINOv3 ViT-L/16 conditioning — GGML recipe (from timm eva.py + pos_embed_sincos.py)

Golden: tools/ref_dinov3.py (timm `vit_large_patch16_dinov3`, weights = our timm safetensors,
pre-final-norm + manual non-affine F.layer_norm) → /media/.../ref/dinov3/{cond.npy [1,1029,1024], input_chw.npy [1,3,512,512]}.
Validation is permutation-invariant in token order (cross-attn keys) — but to compare per-token, emit
tokens in timm order: [cls, reg0..3, patch0..1023].

## Config (vit_large_patch16_dinov3)
patch=16, img=512 → 32×32=1024 patches; embed=1024; depth=24; heads=16; head_dim=64;
qkv_bias=False; proj has bias; LayerScale init_values=1e-5 (gamma_1/gamma_2 per block);
norm eps=1e-5; mlp ratio 4 (fc1 1024→4096, fc2 4096→1024), act = GELU (exact/erf — timm default `gelu`, use ggml_gelu_erf);
num_reg_tokens=4; num_prefix_tokens = 1(cls)+4(reg) = 5; use_abs_pos_emb=False (rope only); use_fc_norm=False.

## Preprocess (matches TRELLIS extract_features)
PIL→resize(512,512, LANCZOS)→/255→ normalize mean[.485,.456,.406] std[.229,.224,.225] → [1,3,512,512].

## Forward
1. patch_embed: Conv2d(3→1024, k16, s16) on [3,512,512] → [1024,32,32] → flatten to [1024 tokens,1024].
   timm key: `patch_embed.proj.weight [1024,3,16,16]`, `.bias [1024]`.
2. prepend tokens: `cls_token [1,1,1024]` then `reg_token [1,4,1024]` (storage_tokens) → sequence [1029,1024].
   timm keys: `cls_token`, `reg_token` (shape [1,4,1024]). Order: [cls, reg(4), patches(1024)].
   (No abs pos embed added.)
3. 24 blocks (EvaBlock, key `blocks.{i}`):
   - h = x + gamma_1 * attn(norm1(x), rope)        # LayerScale gamma_1 = blocks.i.gamma_1 [1024]
   - h = h + gamma_2 * mlp(norm2(h))               # gamma_2 = blocks.i.gamma_2 [1024]
   norm1/norm2 = LayerNorm(1024, eps 1e-5) affine (blocks.i.norm1.{weight,bias}, norm2.*).
   attn: qkv = Linear(1024→3072, NO bias) (blocks.i.attn.qkv.weight [3072,1024]); reshape [N,3,16,64];
         q,k,v [16 heads,64]. RoPE applied to q,k on PATCH tokens only (indices ≥5; prefix skipped).
         SDPA scale 1/sqrt(64). proj = Linear(1024→1024, bias) (attn.proj.{weight,bias}).
         (NO qk-norm in DINOv3, unlike the DiT.)
   mlp: fc1 1024→4096 (+bias), GELU(erf), fc2 4096→1024 (+bias). keys mlp.fc1.*, mlp.fc2.*.
4. final: `F.layer_norm(h, [1024])` NON-affine eps 1e-5 (TRELLIS's own; the model's `norm` head is NOT used → set timm m.norm=Identity in ref). Output cond [1029,1024].

## DINOv3 RoPE (the only non-standard bit) — NeoX rotate_half, patches only
head_dim D=64. periods: `dim4 = D//4 = 16`; `exponents = 2*arange(16)/(D//2=32) = arange(16)/16`;
`periods[p] = temperature^exponents[p] = 100^(p/16)`, p=0..15.  (temperature=100)
Patch (h,w), h,w∈[0,32): 0.5-centered, normalize 'separate', map to [-1,1]:
  ch = ((h+0.5)/32)*2 - 1 ;  cw = ((w+0.5)/32)*2 - 1.    (grid_indexing 'ij': token = h*32 + w)
angles (length 32) = [ for p in 0..15: 2π*ch/periods[p] ] ++ [ for p in 0..15: 2π*cw/periods[p] ]  (axis-major: all h-angles, then all w-angles).
tile×2 → length 64: angles64 = [angles32, angles32].  cos=cos(angles64), sin=sin(angles64).
Apply (rotate_half/NeoX) to q,k [D=64] per patch token:
  out[i] = q[i]*cos[i] + rh[i]*sin[i],   rh = concat(-q[32:64], q[0:32])
  (i.e. for i<32: rh=-q[i+32]; i>=32: rh=q[i-32]; cos/sin tiled so cos[i]=cos[i-32] for i>=32.)
Prefix tokens (0..4): NO rope (identity).

GGML plan: precompute cos[64, 1024]/sin[64,1024] host tables for the 1024 patches; for the 5 prefix tokens
use cos=1,sin=0 (identity) so a single uniform apply over all 1029 tokens works. rotate_half (half-split)
differs from the DiT's interleaved apply_rope — implement a second `apply_rope_half`: view q [64,H,N] →
split halves [32,..]+[32,..], rh = concat(-second, first), out = q*cos + rh*sin (cos/sin [64,1,N]).
patch_embed Conv2d via ggml_conv_2d (k16 s16) — verify CUDA support, or im2col. cls/reg are learned
[.,1024] params prepended on host or as ggml tensors concatenated along token axis.

Decouple validation: optionally dump timm's rope table (model.rope... or recompute) and feed it, OR
implement the formula above and unit-check cos/sin vs a python dump for the 32×32 grid before the full ViT.

## Weight key remap (timm → our gguf, convert dinov3 component)
timm names already in our dinov3.gguf (converted verbatim): patch_embed.proj.{weight,bias}, cls_token,
reg_token, blocks.{i}.{norm1,norm2}.{weight,bias}, blocks.{i}.attn.qkv.weight (no bias),
blocks.{i}.attn.proj.{weight,bias}, blocks.{i}.gamma_1, blocks.{i}.gamma_2, blocks.{i}.mlp.fc1.{weight,bias},
blocks.{i}.mlp.fc2.{weight,bias}, norm.{weight,bias} (unused — final LN is non-affine manual).
NOTE patch_embed.proj.weight is 4D [1024,3,16,16] → ggml ne [16,16,3,1024], fine for ggml_conv_2d
(no 5D issue). qkv has NO bias (try_get returns null → skip).
