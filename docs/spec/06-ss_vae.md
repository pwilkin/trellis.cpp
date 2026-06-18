# ss_vae

## Sparse Structure VAE — DECODER (`ss_dec_conv3d_16l8_fp16`)

Source: `/tmp/TRELLIS.2/trellis2/models/sparse_structure_vae.py` (class `SparseStructureDecoder`, lines 210-306). Weights from `microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors`. This is a DENSE 3D conv decoder (no sparse ops). Inference-only for the C++ port.

### Role in pipeline
From `trellis2/pipelines/trellis2_image_to_3d.py:188-235` (`sample_sparse_structure`):
1. SS flow model samples latent `z_s` of shape `(num_samples, 8, 16, 16, 16)` float (the flow runs in bf16/fp32; `z_s` is the denoised sample). `in_channels=8`, `resolution=16` (confirmed by `configs/gen/ss_flow_img_dit_1_3B_64_bf16.json`: resolution 16, in_channels 8, out_channels 8).
2. `decoded = decoder(z_s) > 0` → decoder outputs `(num_samples, 1, 64, 64, 64)` float; **threshold is `> 0` (strictly greater than zero)**, giving a bool occupancy grid `(N,1,64,64,64)`.
3. Optional downsample: `if resolution != decoded.shape[2]` (decoded res = 64): `ratio = 64 // resolution`; `decoded = F.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5`. For the standard `resolution=64` path this branch is SKIPPED (ratio would be 1, and condition is false). It only triggers if a coarser SS resolution is requested.
4. `coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()` → active voxel coords. `argwhere` over shape `(N,1,64,64,64)` returns indices `(batch, channel, x, y, z)`; column 1 (channel, always 0) is dropped, keeping `[batch, x, y, z]`. Result shape `(num_active, 4)` int32, columns = (batch_index, X, Y, Z) in the 64³ grid.

### Constructor args (published config `ss_dec_conv3d_16l8_fp16.json`)
```
SparseStructureDecoder(
  out_channels = 1,
  latent_channels = 8,
  num_res_blocks = 2,
  channels = [512, 128, 32],
  num_res_blocks_middle = 2,
  norm_type = "layer",   # NOT in config -> uses default "layer" => ChannelLayerNorm32
  use_fp16 = True,
)
```
With `channels=[512,128,32]` there are `len-1 = 2` UpsampleBlocks → 16 → 32 → 64 (4x total, matches expected). Channel progression: input 512 ch @16³ → after up0 128 ch @32³ → after up1 32 ch @64³ → head → 1 ch @64³.

### norm_type detail (IMPORTANT)
`norm_type` defaults to `"layer"` (line 230). `norm_layer("layer", C)` returns `ChannelLayerNorm32(C)` (norm.py line 16-17). ChannelLayerNorm32 is an `nn.LayerNorm` over the channel dim implemented as: permute `(B,C,H,W,D)->(B,H,W,D,C)`, LayerNorm over last dim (C) in float32, permute back. It has learnable `weight` and `bias` of shape `(C,)`, `eps=1e-5` (LayerNorm default). GroupNorm32 (group path, 32 groups) is NOT used here unless config overrides — flag as the single config uncertainty (default is layer).

### Forward op order (lines 295-306)
```
h = input_layer(x)                 # Conv3d(8 -> 512, k3, pad1) ; x:(N,8,16,16,16) -> (N,512,16,16,16)
h = h.type(dtype=fp16)             # cast to fp16 for the torso (convs/norm cast internally to fp32)
h = middle_block(h)                # Sequential of 2x ResBlock3d(512,512) @16^3
for block in self.blocks: h = block(h)
h = h.type(x.dtype)                # cast back to input dtype (fp32/bf16)
h = out_layer(h)                   # norm + SiLU + Conv3d(32 -> 1, k3, pad1) -> (N,1,64,64,64)
```
NOTE on fp16: `use_fp16=True` only halves the Conv3d weights (via `convert_module_to_f16`, applied to `blocks` and `middle_block` only — NOT input_layer or out_layer). The norm layers (ChannelLayerNorm32) always upcast activations to fp32 internally. For a C++ port you may run the whole decoder in fp32; precision-equivalent within threshold tolerance. `input_layer` and `out_layer` weights stay fp32 in the checkpoint.

### `self.blocks` expansion (ModuleList, lines 250-259)
Built by: for each i, ch in enumerate([512,128,32]): append `num_res_blocks` (=2) ResBlock3d(ch,ch); if i<2 append UpsampleBlock3d(ch, channels[i+1]). Linear index in ModuleList:
- blocks.0  = ResBlock3d(512,512)   @16³
- blocks.1  = ResBlock3d(512,512)   @16³
- blocks.2  = UpsampleBlock3d(512->128)  @16³ -> 32³
- blocks.3  = ResBlock3d(128,128)   @32³
- blocks.4  = ResBlock3d(128,128)   @32³
- blocks.5  = UpsampleBlock3d(128->32)   @32³ -> 64³
- blocks.6  = ResBlock3d(32,32)     @64³
- blocks.7  = ResBlock3d(32,32)     @64³

### ResBlock3d (lines 22-47)
Attributes: `norm1`, `norm2` (ChannelLayerNorm32), `conv1` Conv3d(C->Cout,k3,pad1), `conv2` Conv3d(Cout->Cout,k3,pad1) [zero-initialized at train, but loaded from ckpt at inference], `skip_connection` = Conv3d(C->Cout,k1) if C!=Cout else Identity. In this decoder every ResBlock has C==Cout, so skip_connection = Identity (no params). Forward:
```
h = norm1(x); h = SiLU(h); h = conv1(h)
h = norm2(h); h = SiLU(h); h = conv2(h)
return h + skip_connection(x)   # skip = identity here
```
SiLU(x) = x * sigmoid(x).

### UpsampleBlock3d (lines 75-98), mode default "conv"
Attribute `conv` = Conv3d(in_ch, out_ch*8, k3, pad1). Forward: `x = conv(x); return pixel_shuffle_3d(x, 2)`.
- blocks.2.conv: Conv3d(512 -> 128*8=1024, k3, pad1), then pixel_shuffle_3d scale 2 → 128 ch, 2x spatial.
- blocks.5.conv: Conv3d(128 -> 32*8=256, k3, pad1), then pixel_shuffle → 32 ch, 2x.

`pixel_shuffle_3d(x, s=2)` (spatial.py lines 4-13): input `(B,C,H,W,D)`, `C_=C//8`; reshape to `(B,C_,2,2,2,H,W,D)`; permute `(0,1,5,2,6,3,7,4)` → `(B,C_,H,2,W,2,D,2)`; reshape `(B,C_,2H,2W,2D)`. The 8 sub-channels are laid out as the (sx,sy,sz) offsets; channel ordering in reshape is `(s0,s1,s2)` = (along H, along W, along D) as the 3 size-2 dims AFTER C_. Exact unravel: for output channel-major weight slice index `c8 = ((c_ * 8) + sx*4 + sy*2 + sz)` maps to output spatial `(2h+sx, 2w+sy, 2d+sz)`. This must be reproduced exactly (custom reshape/permute, see GGML notes).

### out_layer (Sequential, lines 261-265)
- out_layer.0 = ChannelLayerNorm32(32)  (weight/bias shape (32,))
- out_layer.1 = nn.SiLU() (no params)
- out_layer.2 = nn.Conv3d(32 -> 1, k3, pad1)
Output `(N,1,64,64,64)` in input dtype. Then thresholded `>0` by the pipeline.

### Constants
- All ResBlock/out convs: kernel 3, stride 1, padding 1 (size-preserving).
- Upsample inner conv: kernel 3, stride 1, padding 1 then pixel-shuffle x2.
- LayerNorm eps = 1e-5; SiLU standard.
- Occupancy threshold: strictly `> 0` on raw logit (no sigmoid applied).
- input_layer Conv3d weight shape (512,8,3,3,3); out_layer.2 Conv3d weight (1,32,3,3,3).

## Weight key patterns

Prefix in safetensors is the decoder's own state_dict (loaded via from_pretrained with strict=False). Top-level module attrs: `input_layer`, `middle_block`, `blocks`, `out_layer`.

INPUT LAYER (Conv3d 8->512,k3,p1):
- input_layer.weight  (512, 8, 3, 3, 3)
- input_layer.bias    (512,)

MIDDLE BLOCK (nn.Sequential of 2 ResBlock3d(512,512)); indices 0,1; each ResBlock:
- middle_block.{m}.norm1.weight (512,)   middle_block.{m}.norm1.bias (512,)   # ChannelLayerNorm32
- middle_block.{m}.norm2.weight (512,)   middle_block.{m}.norm2.bias (512,)
- middle_block.{m}.conv1.weight (512,512,3,3,3)   middle_block.{m}.conv1.bias (512,)
- middle_block.{m}.conv2.weight (512,512,3,3,3)   middle_block.{m}.conv2.bias (512,)
  (no skip_connection params: Identity since in==out)
for m in {0,1}.

BLOCKS (nn.ModuleList), per index listed in spec:
ResBlock3d entries (blocks.0,1 -> 512; blocks.3,4 -> 128; blocks.6,7 -> 32):
- blocks.{i}.norm1.weight (C,)   blocks.{i}.norm1.bias (C,)
- blocks.{i}.norm2.weight (C,)   blocks.{i}.norm2.bias (C,)
- blocks.{i}.conv1.weight (C,C,3,3,3)   blocks.{i}.conv1.bias (C,)
- blocks.{i}.conv2.weight (C,C,3,3,3)   blocks.{i}.conv2.bias (C,)
  C = 512 for i in {0,1}; 128 for {3,4}; 32 for {6,7}. No skip_connection params (Identity).

UpsampleBlock3d entries (blocks.2, blocks.5), mode "conv" => attr `conv`:
- blocks.2.conv.weight (1024, 512, 3, 3, 3)   blocks.2.conv.bias (1024,)     # 128*8=1024
- blocks.5.conv.weight (256, 128, 3, 3, 3)    blocks.5.conv.bias (256,)      # 32*8=256

OUT LAYER (nn.Sequential[ ChannelLayerNorm32(32), SiLU, Conv3d(32->1,k3,p1) ]):
- out_layer.0.weight (32,)   out_layer.0.bias (32,)
- out_layer.2.weight (1, 32, 3, 3, 3)   out_layer.2.bias (1,)
  (out_layer.1 is SiLU, no params)

dtype: input_layer / out_layer params are float32 in ckpt; blocks.* and middle_block.* conv weights are float16 (use_fp16=True converts only Conv3d params in blocks & middle_block). norm (LayerNorm) weights/bias remain float32. The C++ loader should read each tensor's stored dtype; safe to upcast all to fp32 for compute.

## GGML notes

All ops are DENSE; no sparse conv needed for this decoder. Per-op mapping:

- Conv3d (k3, s1, p1) and (k1): GGML has no native conv3d. Implement as im2col-style or as a custom kernel. Simplest correct approach: reuse the existing custom 3D conv used elsewhere in this port, or do 3D conv via 27-tap (3x3x3) gather + matmul. Tensors are small (max 512ch @16³, 32ch @64³), so an im2col + ggml_mul_mat is feasible. Memory at 64³: 32*64^3 = 8.4M floats = ~34MB fp32 per activation; the *8 upsample conv intermediate at 32³ is 256*32^3=8.4M too. Manageable.
  Conv weight layout in GGML: PyTorch Conv3d weight is (Cout,Cin,kd,kh,kw); flatten to (Cout, Cin*27) for matmul against im2col (Cin*27, N_spatial). Watch index order of the 27 kernel taps must match (d,h,w) iteration.

- GroupNorm/LayerNorm: this decoder uses ChannelLayerNorm32 = LayerNorm over the CHANNEL dim only (not spatial). Implement as: for each spatial voxel, normalize across C channels (mean/var over C), scale+shift by per-channel weight/bias. NOT ggml_group_norm. Easiest: permute so channels are contiguous innermost, apply ggml_norm (which normalizes over dim0/rows) then mul weight + add bias, permute back. eps=1e-5. Must be done in fp32.

- SiLU: ggml_silu (native).

- pixel_shuffle_3d (x2): custom reshape+permute, NO learnable params. Input (B,C=out*8,H,W,D) -> (B,out,2H,2W,2D). Implement with ggml_reshape + ggml_permute + ggml_cont reproducing: reshape (B,C_,2,2,2,H,W,D), permute to (B,C_,H,2,W,2,D,2), reshape (B,C_,2H,2W,2D). GGML max 4 dims — must fold batch (B=num_samples, usually 1) and emulate the 8D reshape via staged 4D reshapes/permutes, or write a small custom op (a gather with a precomputed index map is cleanest and least error-prone). Verify channel-to-offset mapping: sub-index = sx*4+sy*2+sz over the three size-2 dims in order (H,W,D).

- Threshold + argwhere: output (N,1,64,64,64); compute raw float, then `>0` to bool, then enumerate active voxel coords as int (batch,X,Y,Z). This is a CPU post-process (gather of nonzero indices); no GGML op needed — read the 64^3 logits back to host and scan. Output coords feed the downstream sparse SLat stage.

- residual add: ggml_add. Skip connections are Identity (no conv) since all ResBlocks have in==out.

- The optional max_pool3d branch (line 232) is only used when SS resolution < 64; for the standard 64 path it's skipped, so a max_pool3d op is not required initially.

fp16: blocks/middle_block conv weights stored fp16; recommend upcasting to fp32 on load for a single-precision compute path (norms already force fp32). Numerical threshold `>0` is robust to this.

## Open questions

1. norm_type: the published `ss_dec_conv3d_16l8_fp16.json` config (loaded from HF, NOT present on local disk anywhere — confirmed by filesystem search) does not appear to set `norm_type`, so the code default `"layer"` (ChannelLayerNorm32) applies. This must be confirmed by inspecting the actual downloaded JSON or by checking checkpoint key shapes: ChannelLayerNorm32 stores `norm1.weight/bias` of shape (C,) (LayerNorm), identical key names to GroupNorm32 (`weight`,`bias` shape (C,)). KEY DISAMBIGUATION: GroupNorm32 uses 32 groups but the param shapes are the same (C,), so key names/shapes alone do NOT distinguish layer vs group. The two differ in the normalization math (LayerNorm over channels-per-voxel vs GroupNorm over 32 channel-groups across all spatial). MUST verify by reading the real JSON config from microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.json, or by running the reference decoder once. I have assumed default "layer". (Original TRELLIS-image-large repo's ss_dec is documented with norm_type layer default; but verify.)

2. channels=[512,128,32], num_res_blocks=2, num_res_blocks_middle=2, out_channels=1, latent_channels=8 are taken from the standard published config + name encoding (16=res, l8=8 latent ch). Confirmed consistent with flow model (in/out 8 ch, res 16) and the 16->64 (4x=2 upsample) requirement. Still should be cross-checked against the actual downloaded JSON for exact channel list (an alternative could be e.g. [512,128,32] vs other — name only encodes l8). HIGH CONFIDENCE but verify channel list from the real .json.

3. Exact pixel_shuffle channel->spatial-offset ordering: derived from spatial.py permute (0,1,5,2,6,3,7,4); re-verify against a tiny numeric test when porting (a transposed offset map would silently produce a mirrored/garbled 64^3 grid).

4. Whether `z_s` fed to decoder needs a latent scale/normalization (mean/std) before decode. In this file the decoder takes the raw flow sample directly (`decoder(z_s)`, pipeline line 227) with no scaling — but check the SS latent dataset (`trellis2/datasets/sparse_structure_latent.py`) for any `latent_std`/normalization applied at TRAIN time that the flow model already learned to produce in normalized space (it appears decode is direct, no division). Recommend confirming no scale factor is silently applied between flow output and decoder input.
