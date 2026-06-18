# SS conv3d decoder recipe (SparseStructureDecoder) — next to implement

GGUF: `gguf/ss_dec.gguf` (arch trellis2-ss-dec, 74 tensors). Config:
`out_channels=1, latent_channels=8, num_res_blocks=2, num_res_blocks_middle=2, channels=[512,128,32]`, fp16, norm_type=**layer** (ChannelLayerNorm).

Input: SS-flow latent z_s [B,8,16,16,16]. Output: occupancy logits [B,1,64,64,64]; `>0` → active voxels; `argwhere(decoded)[:,[0,2,3,4]]` → coords [N,(b,x,y,z)] @res64. For pipeline_type '512'/'1024_cascade', ss_res=32: `max_pool3d(decoded.float(),2)>0.5` first → coords@res32.

## Forward (all Conv3d are k=3,pad=1 unless noted; ggml has native `ggml_conv_3d`)
1. `input_layer`: Conv3d(8→512) @16³
2. `middle_block.{0,1}`: ResBlock3d(512,512)
3. blocks (ModuleList, flat index — see key names):
   - ch=512: 2× ResBlock3d(512,512); UpsampleBlock3d(512→128) → 32³
   - ch=128: 2× ResBlock3d(128,128); UpsampleBlock3d(128→32) → 64³
   - ch=32:  2× ResBlock3d(32,32)
4. `out_layer`: ChannelLayerNorm(32) → SiLU → Conv3d(32→1) → logits [B,1,64,64,64]

## ResBlock3d (norm_type=layer)
```
h = ChannelLayerNorm(x); h = silu(h); h = conv1(h)          # conv1: Conv3d(C->Cout)
h = ChannelLayerNorm(h); h = silu(h); h = conv2(h)          # conv2: zero-init at train, real weights now
x_skip = (C==Cout) ? x : skip_connection(x)                 # skip: Conv3d k=1 if channel change
return h + x_skip
```
Keys per block: `*.norm1.{weight,bias}`, `*.conv1.{weight,bias}`, `*.norm2.*`, `*.conv2.*`, `*.skip_connection.*` (only when C!=Cout). For middle/same-channel resblocks there is NO skip_connection (Identity).

## UpsampleBlock3d(in→out) : `*.conv` = Conv3d(in → out*8, k=3,p=1) then pixel_shuffle_3d(·,2)
pixel_shuffle_3d (TRELLIS uses [B,C,H,W,D]):
```
x[B, C=out*8, H,W,D] -> reshape [B,out,2,2,2,H,W,D] -> permute(0,1,5,2,6,3,7,4) -> [B,out,H,2,W,2,D,2] -> reshape [B,out,2H,2W,2D]
```
i.e. channel decomposes as out·(s0·4+s1·2+s2); s0→H, s1→W, s2→D fastest-interleaved. **Verify this offset order with a labeled numeric test vs torch (risk #10 in synthesis).**

## ChannelLayerNorm32 (norm_type=layer)
LayerNorm over the CHANNEL dim per voxel: permute channel to last, F.layer_norm over [C], permute back. eps default 1e-5. weight/bias [C]. In ggml on a [C,X,Y,Z] (ne0=C) tensor this is just ggml_norm over ne0 + affine — but note the conv tensors are laid [W?,...]; arrange so channel is ne0 for the norm then back for conv. (Conv3d in ggml expects a specific layout — check ggml_conv_3d signature: dims order and weight [OC,IC,kd,kh,kw] vs ggml.)

## Validation
Dump reference: feed z_s = ref/ss_sample/samples.npy (or a fresh z) through SparseStructureDecoder on GPU, save logits [1,1,64,64,64] and the coords. C++: feed SAME z_s, compare logits (rel<2e-2 f16) and the active-voxel coord set (should match exactly after >0 threshold since decoder is deterministic — this is the clean validation, independent of sampler stochasticity per [[trellis2-validation-strategy]]).
Key thing: ggml_conv_3d weight layout + the ChannelLayerNorm axis are the two things to get right; pixel_shuffle interleave is the correctness trap.
