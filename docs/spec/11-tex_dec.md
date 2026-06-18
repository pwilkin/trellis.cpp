# tex_dec

## tex_dec — SparseUnetVaeDecoder (texture/PBR SLAT decoder)

Class: `SparseUnetVaeDecoder` in `/tmp/TRELLIS.2/trellis2/models/sc_vaes/sparse_unet_vae.py` (lines 398-522).
Authoritative config: `/tmp/TRELLIS.2/configs/scvae/tex_vae_next_dc_f16c32_fp16.json` (decoder block, lines 43-83). A `_ft_512` variant exists with the same model args (only training/dataset differ).

### Role in pipeline
In `Trellis2TexturingPipeline.decode_tex_slat` (`/tmp/TRELLIS.2/trellis2/pipelines/trellis2_texturing.py:267-285`):
```
ret = self.models['tex_slat_decoder'](slat) * 0.5 + 0.5
```
Input `slat` is a `SparseTensor` with `feats` of shape `[N, 32]` (the de-normalized texture SLAT; `slat = slat*std + mean` using `tex_slat_normalization`). Output is a `SparseTensor` with `feats` of shape `[N_out, 6]` mapped from [-1,1] to [0,1] by `*0.5+0.5`. N_out is larger than N because of channel→spatial upsampling (factor 2 per up block, 4 up blocks => 16x linear / 4096x voxel-count increase in active voxels at the leaves, gated by subdivision).

### The 6 output channels (PBR), confirmed by layout
Layout (identical in pipeline `pbr_attr_layout` lines 60-65, in `MeshWithPbrMaterial.layout` base.py:173-178, used in renderer 338-341):
- channels [0:3] = `base_color` (albedo RGB, linear-ish, stored as 0..1; renderer applies `**2.2` to convert to linear before shading)
- channel [3:4] = `metallic`
- channel [4:5] = `roughness`
- channel [5:6] = `alpha` (opacity)

NOT RGB+normal. There is no normal/occlusion channel in the SLAT; normals are computed geometrically from the mesh in the renderer, occlusion is computed via SSAO at render time.

### Decoder architecture (exact, for pred_subdiv=false)
Config args:
- out_channels=6, latent_channels=32
- model_channels = [1024, 512, 256, 128, 64]
- num_blocks = [4, 16, 8, 4, 0]
- block_type = ["SparseConvNeXtBlock3d"] * 5 (only the first 4 stages have any, last is 0)
- up_block_type = ["SparseResBlockC2S3d"] * 4
- block_args (use_checkpoint, irrelevant at inference) = [F, F, F, T, T]
- use_fp16 = true, pred_subdiv = false

Module construction (lines 423-444):
- `self.from_latent = SparseLinear(latent_channels=32, model_channels[0]=1024)`  (32 -> 1024)
- `self.output_layer = SparseLinear(model_channels[-1]=64, out_channels=6)`  (64 -> 6)
- `self.blocks` = `nn.ModuleList` of 5 stage-`ModuleList`s. For stage i: append `num_blocks[i]` instances of `block_type[i](model_channels[i], **block_args[i])`; then, if i<4, append one `up_block_type[i](model_channels[i], model_channels[i+1], pred_subdiv=False, **block_args[i])`.

Resulting per-stage contents (channels):
- blocks[0]: 4x ConvNeXt(1024) + 1x C2S(1024->512)  => 5 modules
- blocks[1]: 16x ConvNeXt(512) + 1x C2S(512->256)    => 17 modules
- blocks[2]: 8x ConvNeXt(256) + 1x C2S(256->128)     => 9 modules
- blocks[3]: 4x ConvNeXt(128) + 1x C2S(128->64)      => 5 modules
- blocks[4]: 0 ConvNeXt + (no up block, i==4 is last) => 0 modules

This is the MIRROR of the encoder (`SparseUnetVaeEncoder`, config lines 3-41): encoder uses model_channels [64..1024], num_blocks [0,4,8,16,4], down_block_type `SparseResBlockS2C3d`. Encoder ends with `to_latent = SparseLinear(1024, 2*32=64)` (mean+logvar). So decoder is shape-symmetric to encoder but uses channel→spatial upsampling instead of spatial→channel downsampling, and emits 6 channels instead of 6-in.

### forward() op order (inference, pred_subdiv=false; guide_subs=None at inference)
File lines 478-507. With pred_subdiv=False and guide_subs default None:
1. `h = self.from_latent(x)`  (SparseLinear 32->1024 applied to feats)
2. `h = h.type(self.dtype)`   (cast feats to fp16)
3. For i, stage in enumerate(blocks); for j, block in enumerate(stage):
   - If `i < 4 and j == last_of_stage` (the up block): `h = block(h, subdiv=None)`  (guide_subs is None)
   - Else: `h = block(h)`  (ConvNeXt block)
4. `h = h.type(x.dtype)`  (cast feats back to fp32)
5. `h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))`  — final LayerNorm over the 64-channel feature dim, NO affine (no weight/bias), eps default 1e-5.
6. `h = self.output_layer(h)`  (SparseLinear 64->6)
Returns the SparseTensor (then pipeline does `*0.5+0.5`).

NOTE the up block C2S receives `subdiv=None` at inference and pred_subdiv=False, so inside it `self.pred_subdiv` is False AND the passed subdiv is None => `subdiv_binarized = None`. Then `SparseChannel2Spatial.forward(x, subdivision=None)` requires the spatial cache `channel2spatial_2` to exist (lines 70-73 raise if cache missing and subdivision None). At inference this cache must be supplied — it is produced by the matching encoder S2C pass OR (in the texturing pipeline) the SLAT already carries the coords/scale so the upsampling pattern is deterministic. IMPORTANT for the C++ port: when there is no spatial cache and no subdiv prediction, the decoder cannot invent which sub-voxels are occupied. In the texturing pipeline the texture SLAT coords come from the shape SLAT (same active voxels), so the decoder's upsampling is driven by the known target voxel grid. The C++ implementation must reconstruct the channel->spatial mapping from the target coordinate set (see open questions).

### SparseConvNeXtBlock3d (block_type, lines 265-294)
Attributes: `norm` (LayerNorm32, elementwise_affine=True, eps=1e-6), `conv` (SparseConv3d(C,C,3)), `mlp` = nn.Sequential(Linear(C, 4C), SiLU, Linear(4C, C)) — mlp_ratio=4.0 default; last Linear is zero-initialized but a real trained param.
_forward:
1. `h = self.conv(x)`  (sparse 3x3x3 conv, submanifold, C->C)
2. `h = h.replace(self.norm(h.feats))`  (LayerNorm over C, affine)
3. `h = h.replace(self.mlp(h.feats))`  (Linear C->4C, SiLU, Linear 4C->C; per-voxel dense MLP)
4. `return h + x`  (residual on feats; coords unchanged)
Note: order is conv->norm->mlp->residual (norm AFTER conv, unlike classic ConvNeXt). No DWConv; it is the SparseConv3d that mixes spatial neighbors.

### SparseResBlockC2S3d (up_block_type, lines 217-262) — channel-to-spatial upsample
Attributes: `norm1` (LayerNorm32 affine eps1e-6), `norm2` (LayerNorm32 elementwise_affine=False eps1e-6, i.e. NO weight/bias), `conv1` (SparseConv3d(channels, out_channels*8, 3)), `conv2` (zero_module SparseConv3d(out_channels, out_channels, 3)), `skip_connection` (lambda: repeat_interleave on feats — NOT a parameter), `updown` (SparseChannel2Spatial(2)). `to_subdiv` (SparseLinear(channels,8)) ONLY exists if pred_subdiv=True — for tex_dec pred_subdiv=False so `to_subdiv` is ABSENT (no weight key).
_forward(x, subdiv=None) with pred_subdiv=False:
1. (skip to_subdiv since pred_subdiv False; subdiv stays as passed arg = None at inference)
2. `h = x.replace(norm1(x.feats))`; `h = silu(h)`
3. `h = self.conv1(h)`  (SparseConv3d channels -> out_channels*8, e.g. 1024 -> 512*8=4096)
4. `subdiv_binarized = subdiv>0 if subdiv is not None else None`  (None at inference)
5. `h = self.updown(h, subdiv_binarized)`  (SparseChannel2Spatial: reshape feats [N, 8*out] into [8N, out] scattered to child voxels per subdivision/cache; halves spatial scale, multiplies voxel count by up-to-8)
6. `x = self.updown(x, subdiv_binarized)`  (upsample the residual path too)
7. `h = h.replace(norm2(h.feats))`; `h = silu(h)`  (norm2 non-affine)
8. `h = self.conv2(h)`  (zero-init SparseConv3d out->out, 3x3x3)
9. `h = h + self.skip_connection(x)` where skip = `x.feats.repeat_interleave(out_channels // (channels//8), dim=1)`. channels//8 is the channel count after C2S of x (x had `channels` feats; C2S of x reshapes [N, channels] -> [8N, channels/8]); repeat_interleave brings channels/8 up to out_channels.
10. pred_subdiv False -> `return h` (no subdiv returned).

### SparseChannel2Spatial(factor=2) (spatial2channel.py:58-93) — the actual upsample op
DIM=3, factor=2, factor**DIM=8.
- Needs spatial cache `channel2spatial_2` = (new_coords, idx, subidx) OR a `subdivision` SparseTensor with feats `[N, 8]` boolean.
- From subdivision: sub=[N,8] bool; N_leaf=sub.sum(-1); subidx = nonzero last index; new_coords = x.coords*2 (on spatial dims 1:), repeat_interleaved by N_leaf; for i in 0..2: new_coords[:,i+1] += (subidx // 2**i) % 2; idx = repeat_interleave(arange(N), N_leaf).
- feats: `x_feats = x.feats.reshape(N*8, -1)` (i.e. split the C=8*out channels into 8 spatial children each of `out` channels), then `new_feats = x_feats[idx*8 + subidx]`. Scale divided by 2 (finer grid). This is a gather/scatter; needs custom GGML op.

### LayerNorm32 (modules/norm.py)
Subclass of nn.LayerNorm that casts input to fp32, runs standard LayerNorm, casts back. Formula: standard layer norm over last dim, eps as given (1e-6 in res/convnext blocks; 1e-5 default in the final F.layer_norm at decoder end). Affine when elementwise_affine=True (weight,bias present); norm2 in res blocks has NO affine params.

### SparseConv3d weights (flex_gemm backend — the inference backend)
`/tmp/TRELLIS.2/trellis2/modules/sparse/conv/conv_flex_gemm.py`: builds `self.weight = nn.Parameter(empty(out_channels, in_channels, Kd, Kh, Kw))` then permutes to `(Co, Kd, Kh, Kw, Ci) = (out, 3, 3, 3, in)` contiguous and re-wraps as Parameter (line 34). `self.bias = nn.Parameter(empty(out_channels))` (kernel_size 3 => 3x3x3 submanifold conv, stride 1, dilation 1). So stored checkpoint weight shape for SparseConv3d is `[out, 3, 3, 3, in]`. (The spconv/torchsparse backends nest under `.conv.weight`; flex_gemm is flat `.weight`/`.bias` — confirm which the released checkpoint uses; see open questions.)

### Renderer at inference (PbrMeshRenderer, pbr_mesh_renderer.py)
Two consumption paths from the 6-ch voxel/texture:
1. Texture baking (the actual i2-3D output) is in the PIPELINE `postprocess_mesh` (texturing.py:287-371), NOT the renderer. Steps: UV-unwrap mesh (cumesh `uv_unwrap` if no UVs), rasterize UV layout to a `texture_size`x`texture_size` (default 2048; run() passes 2048) image via nvdiffrast, interpolate object-space positions `pos`, then `grid_sample_3d(pbr_voxel.feats, pbr_voxel.coords, shape=[N,6, *spatial_shape], grid=((pos+0.5)*resolution), mode='trilinear')` to fetch per-texel PBR (resolution=1024 default). Split by layout into base_color(3)/metallic(1)/roughness(1)/alpha(1), scale *255 uint8, cv2.INPAINT_TELEA to fill non-UV regions. Pack into a trimesh PBRMaterial: baseColorTexture = RGB+alpha; metallicRoughnessTexture = [0, roughness, metallic] (G=roughness, B=metallic, per glTF), doubleSided, OPAQUE. Swap Y/Z axes, invert. This is the GGML port's primary target for "mapping per-voxel PBR onto the mesh surface".
2. The renderer `PbrMeshRenderer.render` (used for preview/eval, not the GLB export) handles `MeshWithVoxel` by per-pixel `grid_sample_3d(mesh.attrs, coords, voxel_shape, xyz_in_voxel_space, 'trilinear')` then splits via `mesh.layout` into gb_basecolor/metallic/roughness/alpha. Shading: gb_basecolor clamped & `**2.2`, builds `gb_orm = cat([zeros, roughness, metallic])`, calls EnvMap.shade (split-sum IBL via nvdiffrec EnvironmentLight). Compositing across depth-peel layers weighted by alpha, then SSAO multiply, ACES tonemap + gamma. This renderer is inference/eval-only and depends on nvdiffrast + nvdiffrec — likely NOT ported to GGML for the texturing output (the GLB is produced by postprocess_mesh's bake).

### Inference-only vs training-only
- `pred_subdiv=False` path is the deployed one; `subs_gt`/`subs`/`get_spatial_cache('subdivision')` branches (lines 489-493) are training-only (`self.training`).
- `decoder.upsample()` method (lines 509-522) requires pred_subdiv=True — NOT used by tex_dec.
- zero_module on conv2/mlp final linear is init-only.
- `initialize_weights`, `convert_to_fp16`, `use_checkpoint`/checkpoint wrappers are not needed for forward correctness.

## Weight key patterns

Top-level (state_dict prefix typically the saved decoder module; HF safetensors likely under `decoder.` or just bare — confirm):

Linears (nn.Linear subclasses, standard `.weight [out,in]`, `.bias [out]`):
- `from_latent.weight` [1024, 32], `from_latent.bias` [1024]
- `output_layer.weight` [6, 64], `output_layer.bias` [6]

Blocks stage i, module j: prefix `blocks.{i}.{j}.`

SparseConvNeXtBlock3d (block_type) keys under `blocks.{i}.{j}.`:
- `norm.weight` [C], `norm.bias` [C]   (C = model_channels[i])
- conv (flex_gemm backend, FLAT): `conv.weight` [C, 3, 3, 3, C], `conv.bias` [C]
  - (spconv/torchsparse backend would be `conv.conv.weight` [3,3,3,C,C] and `conv.conv.bias` — verify backend)
- `mlp.0.weight` [4C, C], `mlp.0.bias` [4C]
- `mlp.2.weight` [C, 4C], `mlp.2.bias` [C]   (mlp.1 is SiLU, no params)

SparseResBlockC2S3d (up_block, the last module in stages 0..3) under `blocks.{i}.{last}.` with channels=model_channels[i], out=model_channels[i+1]:
- `norm1.weight` [channels], `norm1.bias` [channels]
- norm2: NO params (elementwise_affine=False)
- conv1 (flex_gemm flat): `conv1.weight` [out*8, 3, 3, 3, channels], `conv1.bias` [out*8]
- conv2 (flex_gemm flat): `conv2.weight` [out, 3, 3, 3, out], `conv2.bias` [out]
- skip_connection: NONE (it is a Python lambda, not a Module — no params)
- to_subdiv: ABSENT for tex_dec (pred_subdiv=False)
- updown (SparseChannel2Spatial): no params

Concrete enumerated up blocks:
- `blocks.0.4.*`  C2S 1024->512 : conv1.weight [4096,3,3,3,1024], conv2.weight [512,3,3,3,512]
- `blocks.1.16.*` C2S 512->256  : conv1.weight [2048,3,3,3,512],  conv2.weight [256,3,3,3,256]
- `blocks.2.8.*`  C2S 256->128  : conv1.weight [1024,3,3,3,256],  conv2.weight [128,3,3,3,128]
- `blocks.3.4.*`  C2S 128->64   : conv1.weight [512,3,3,3,128],   conv2.weight [64,3,3,3,64]

Concrete ConvNeXt counts:
- blocks.0.{0..3}: ConvNeXt(1024): conv.weight [1024,3,3,3,1024], mlp.0 [4096,1024], mlp.2 [1024,4096]
- blocks.1.{0..15}: ConvNeXt(512): conv.weight [512,3,3,3,512], mlp.0 [2048,512], mlp.2 [512,2048]
- blocks.2.{0..7}: ConvNeXt(256): conv.weight [256,3,3,3,256], mlp.0 [1024,256], mlp.2 [256,1024]
- blocks.3.{0..3}: ConvNeXt(128): conv.weight [128,3,3,3,128], mlp.0 [512,128], mlp.2 [128,512]
- blocks.4: empty (no keys)

No params for the final F.layer_norm (decoder forward step 5) — it is functional, non-affine.

Encoder (for completeness, mirror): `input_layer.weight` [64,6]; `to_latent.weight` [64,1024]; stages use `SparseResBlockS2C3d` with `conv1.weight` [out//8,3,3,3,channels], `conv2`, `norm1` affine, `norm2` non-affine; ConvNeXt as above.

## GGML notes

Per-op GGML mapping for tex_dec decode:

1. SparseLinear (from_latent 32->1024, output_layer 64->1024->6, mlp Linears): trivial dense matmul on the `[N, Cin]` feats matrix + bias broadcast. ggml_mul_mat + ggml_add. feats is just a 2D dense tensor [N, C]; coords carried alongside as int32 [N,4] (batch,x,y,z) outside ggml.

2. LayerNorm32 (affine and non-affine) and final F.layer_norm: ggml_norm (eps 1e-6 in blocks, 1e-5 at final) over channel dim, then optionally *weight + bias (ggml_mul + ggml_add). Cast-to-fp32 behavior is automatic if you keep feats fp32; the reference forces fp32 for the norm even when running fp16 torso — replicate by computing norm in fp32.

3. SiLU: ggml_silu. Standard.

4. SparseConv3d (submanifold 3x3x3, stride 1): CUSTOM OP. This is the hard one. Submanifold sparse conv: output occupied set == input occupied set (coords unchanged); for each active voxel, gather the up-to-27 neighbor active voxels (by hashing coords), accumulate weight[kd,kh,kw] @ neighbor_feat. Weight layout in checkpoint is [Co, Kd, Kh, Kw, Ci]. Implement as: build a neighbor index table (a [N, 27] gather map of source row indices, -1 for empty) ONCE per coordinate set (shared across all convs at a given resolution), then conv = sum over 27 taps of (gathered_feats[:, k, :] matmul weight[:,k,:,:]^T) + bias. Needs: (a) a hash/sort to find neighbors (host-side precompute is fine, produces the [N,27] int32 table), (b) a gather op (ggml_get_rows per tap or a custom batched gather), (c) 27 matmuls + accumulate. Memory: store the 27-tap index map (N*27 int32) per resolution level. There are 4 distinct resolutions (one per stage) -> 4 neighbor tables. This dominates the implementation effort.

5. SparseChannel2Spatial (upsample factor 2, the up block updown): CUSTOM gather/scatter. Given input feats [N, 8*out] and a known target child-voxel set, reshape to [N,8,out] conceptually and emit one output row per occupied child = gather rows by (parent_idx*8 + subchild_code). The (idx, subidx) mapping is data-dependent: at inference with pred_subdiv=False there is NO subdivision tensor and NO spatial cache unless paired with the encoder S2C. KEY ISSUE for the port: the target occupied child voxels are determined by the SLAT/shape coords. The C++ port must, given the parent coords and the known finer-resolution target coords (from the shape SLAT / voxel grid at the next resolution), compute for each target child its parent index and its 3-bit subchild code (child_xyz - 2*parent_xyz packed as bit0=x,bit1=y,bit2=z; matches `subidx//2**i % 2` decode), then gather feats. Implement as host-precomputed gather index [N_child] into the [N*8, out] reshaped buffer, plus a ggml_get_rows. The residual skip `repeat_interleave` is a channel-tiling gather on the upsampled-x path (also host index + get_rows or ggml_repeat patterns).

6. grid_sample_3d trilinear (postprocess_mesh baking + renderer): CUSTOM 3D trilinear sample from a sparse voxel grid. Given query xyz in voxel index space, find the 8 surrounding voxels in the sparse set (hash lookup; absent voxels contribute 0), trilinear blend their 6-ch attrs. Needed for texture baking. Implement with the same coordinate hash as the conv neighbor table.

7. Pipeline tail `*0.5+0.5`: ggml_scale + add (or fold into output bias). Then clamp 0..1 happens at uint8 conversion.

Custom ops summary (must write): submanifold SparseConv3d (neighbor-gather + 27 matmuls), SparseChannel2Spatial gather/scatter, sparse trilinear grid_sample_3d, coordinate hashing for neighbor/child resolution. Everything else (linear, layernorm, silu, residual add) is stock ggml. No attention, no rope in this decoder. Recommend running the whole decoder in fp32 for the port (fp16 is only a perf/memory choice in reference; norms are fp32 anyway).

Memory: largest activation is at the finest resolution after all 4 upsamples — N grows ~up to *4096 in voxel count vs input but channels drop to 64; feats buffer ~[N_final, 64] fp32. Pre-size neighbor tables per level. UV bake texture is 2048x2048x6 float in postprocess (run() default texture_size=2048; docstring default 1024).

## Open questions

1. CRITICAL — channel->spatial upsampling target at inference with pred_subdiv=False and no spatial cache: `SparseChannel2Spatial.forward` raises if both subdivision and `channel2spatial_2` cache are None. In `decode_tex_slat` no subdiv is passed. So either (a) the released `tex_slat_decoder` checkpoint actually has pred_subdiv=True (contradicting the config flag — must verify the shipped pipeline.json / loaded model args), or (b) the SLAT SparseTensor arrives carrying spatial caches from the encoder S2C pass, or (c) the up block's `to_subdiv` is present and used. RESOLVE by inspecting the actual released checkpoint keys: presence of `blocks.{i}.{last}.to_subdiv.weight` would mean pred_subdiv=True at runtime. This determines how the C++ port decides which child voxels to populate. Recommend dumping the safetensors key list of the real tex decoder.

2. SparseConv3d backend in the released weights: flex_gemm (flat `conv.weight` shape [Co,Kd,Kh,Kw,Ci]) vs spconv/torchsparse (nested `conv.conv.weight`, different permutation [Kd,Kh,Kw,Ci,Co]). `config.CONV` selects at runtime. Must confirm against the actual checkpoint to get key names AND the weight axis order right for the conv kernel.

3. Top-level state_dict prefix for the decoder within the pipeline checkpoint (e.g. `tex_slat_decoder.` vs separate file vs bare). Confirm from the HF repo layout / from_pretrained logic in pipelines/base.py.

4. The exact `resolution` used in grid_sample during baking: `postprocess_mesh` uses `resolution` (1024 default) to scale `(pos+0.5)*resolution`, while voxel `spatial_shape` comes from the decoded SparseTensor. Confirm pbr_voxel.spatial_shape equals the final decoder grid so query coords align (off-by-scale risk).

5. Whether the final non-affine `F.layer_norm` eps is the torch default 1e-5 (not specified in code) — assumed 1e-5; verify it doesn't matter numerically or matches the affine 1e-6 used elsewhere.

6. Does the C++ port need the EnvMap/nvdiffrec renderer at all? The GLB texturing output is produced entirely by `postprocess_mesh` (UV rasterize + grid_sample bake + inpaint), so `PbrMeshRenderer.render` (IBL shading, SSAO, depth peeling) appears preview-only. Confirm the product requirement (bake-only vs also real-time shaded preview).
