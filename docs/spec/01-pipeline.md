# pipeline

## TRELLIS.2 Image-to-3D Pipeline — End-to-End Inference Spec

Entry class: `Trellis2ImageTo3DPipeline` in `/tmp/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py`. Base: `Pipeline` in `/tmp/TRELLIS.2/trellis2/pipelines/base.py`. Driver: `/tmp/TRELLIS.2/example.py`. Real config read from `/run/media/ilintar/D_SSD/models/trellis2/pipeline.json` — concrete values below.

### 0. Top-level call graph (example.py)
```
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()                       # low_vram=True default => only sets _device='cuda'
image = Image.open(...)
mesh  = pipeline.run(image)[0]        # List[MeshWithVoxel], take [0]
mesh.simplify(16777216)               # CuMesh quadric simplify (CUDA, external)
glb   = o_voxel.postprocess.to_glb(vertices,faces,attr_volume=mesh.attrs,coords=mesh.coords,
            attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-.5]*3,[.5]*3],
            decimation_target=1000000, texture_size=4096, remesh=True, ...)  # external C++ lib
glb.export("sample.glb", extension_webp=True)
```

`run(image, num_samples=1, seed=42, ...params..., preprocess_image=True, return_latent=False, pipeline_type=None, max_num_tokens=49152)`:
```
pipeline_type = pipeline_type or self.default_pipeline_type        # default '1024_cascade'
if preprocess_image: image = preprocess_image(image)
torch.manual_seed(seed)
cond_512  = get_cond([image], 512)                                 # {'cond','neg_cond'}
cond_1024 = get_cond([image], 1024)  if pipeline_type != '512' else None
ss_res = {'512':32,'1024':64,'1024_cascade':32,'1536_cascade':32}[pipeline_type]
coords = sample_sparse_structure(cond_512, ss_res, num_samples, ss_params)   # [N,4] int
# branch on pipeline_type (see §5/§6), produces shape_slat (SparseTensor) and res
tex_slat = sample_tex_slat(cond_1024_or_512, tex_flow, shape_slat, tex_params)
torch.cuda.empty_cache()
out_mesh = decode_latent(shape_slat, tex_slat, res)                # List[MeshWithVoxel]
```

### 1. Loading (`from_pretrained`)
`Pipeline.from_pretrained` reads `pipeline.json` -> `args`. For each `k,v` in `args['models']` with `k in model_names_to_load`, loads `models.from_pretrained(f"{path}/{v}")`. Each model: `models/__init__.py:from_pretrained` reads `{v}.json` (`'name'`=class, `'args'`=ctor kwargs) and `{v}.safetensors` (`load_state_dict(..., strict=False)`).

Concrete model map (relpaths under repo root):
```
sparse_structure_decoder    : microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16   (class SparseStructureDecoder, fp16)
sparse_structure_flow_model : ckpts/ss_flow_img_dit_1_3B_64_bf16        (SparseStructureFlowModel, reso 64, bf16)
shape_slat_decoder          : ckpts/shape_dec_next_dc_f16c32_fp16       (sc_vae decoder, 32-ch latent, fp16)
shape_slat_flow_model_512   : ckpts/slat_flow_img2shape_dit_1_3B_512_bf16
shape_slat_flow_model_1024  : ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16
tex_slat_decoder            : ckpts/tex_dec_next_dc_f16c32_fp16         (32-ch latent, fp16)
tex_slat_flow_model_512     : ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16
tex_slat_flow_model_1024    : ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16
```
"1_3B" = each flow DiT is ~1.3B params (total ~4B). "f16c32" = latent 32 channels. Subclass also builds samplers, normalization, image_cond_model, rembg_model, `low_vram=True` (default, not in json), `default_pipeline_type='1024_cascade'`.

`pbr_attr_layout = {'base_color': 0:3, 'metallic': 3:4, 'roughness': 4:5, 'alpha': 5:6}` (6 PBR channels).

`to(device)` low_vram: only sets `_device`; weights moved to GPU per-stage (`m.to(device)` ... `m.cpu()`).

### 2. `preprocess_image(PIL) -> PIL`
1. RGBA & not-all-opaque -> `has_alpha=True`.
2. `scale=min(1,1024/max(size))`; LANCZOS resize if `<1`.
3. `has_alpha`: keep; else `convert('RGB')`, `output=rembg_model(input)` (BiRefNet `briaai/RMBG-2.0`).
4. `alpha=np.array(output)[:,:,3]`; `bbox=argwhere(alpha>0.8*255)` -> (xmin,ymin,xmax,ymax).
5. square: `center=mid`, `size=max(w,h)`, bbox=`center +/- size//2`; `crop`.
6. float/255; **premultiply** `rgb*alpha[...,None]`; -> uint8 RGB PIL (black bg, centered).

### 3. `get_cond([PIL], resolution, include_neg_cond=True) -> {'cond','neg_cond'}`
- `image_cond_model.image_size = resolution` (512 then 1024).
- `cond = image_cond_model(image)` -> `(B,N,D)` patch tokens (DINOv3 ViT-L/16, D=1024). low_vram: to(device) then cpu.
- `neg_cond = torch.zeros_like(cond)` — unconditional embedding is ALL ZEROS, same shape.
- Spread as `**cond` (keys `cond`,`neg_cond`) into sampler.

DINOv3 `DinoV3FeatureExtractor` (`model_name='facebook/dinov3-vitl16-pretrain-lvd1689m'`, HF `DINOv3ViTModel`): list PIL -> `resize((image_size,image_size),LANCZOS)` -> `/255` CHW stack -> Normalize(mean=[.485,.456,.406], std=[.229,.224,.225]). `extract_features`: cast to patch-embed dtype; `hidden=model.embeddings(image,bool_masked_pos=None)`; `pos=model.rope_embeddings(image)`; loop `model.layer[i](hidden, position_embeddings=pos)`; return `F.layer_norm(hidden, hidden.shape[-1:])` (manual final LN, NOT model's norm head). Patch16 => 512/16=32 -> 32*32=1024 patch tokens (+CLS+register tokens). DINOv3 internals = separate component.

### 4. `sample_sparse_structure(cond_512, resolution=ss_res, num_samples, params) -> coords [N,4] int`
- `flow=models['sparse_structure_flow_model']`; `reso=flow.resolution` (=64); `C=flow.in_channels`.
- `noise = randn(num_samples, C, 64,64,64)` dense 5D.
- `z_s = sparse_structure_sampler.sample(flow, noise, cond=..., neg_cond=..., steps=12, guidance_strength=7.5, guidance_rescale=0.7, guidance_interval=[0.6,1.0], rescale_t=5.0).samples`.
- `decoder=models['sparse_structure_decoder']`; `decoded = decoder(z_s) > 0` bool `[B,1,D,D,D]` (D = decoder output res, 64 for this checkpoint -> after 16x conv decode this is the dense occupancy; here `decoder` is the conv3d SS decoder so D likely 64).
- If `resolution != decoded.shape[2]`: `ratio=D//resolution`; `decoded = max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5`. (ss_res=32 with D=64 -> ratio 2 downsample.)
- `coords = argwhere(decoded)[:, [0,2,3,4]].int()` -> `[N,(batch,X,Y,Z)]`.

### 5. Shape SLAT non-cascade (`sample_shape_slat`) — '512'/'1024'
- `noise = SparseTensor(feats=randn(N, flow.in_channels), coords=coords)`.
- `slat = shape_slat_sampler.sample(flow, noise, cond, neg_cond, steps=12, guidance_strength=7.5, guidance_rescale=0.5, guidance_interval=[0.6,1.0], rescale_t=3.0).samples`.
- Denorm: `slat = slat*std[None] + mean[None]` (per-channel, C=32 vectors below).

### 6. Shape SLAT cascade (`sample_shape_slat_cascade`) — '1024_cascade' (default) / '1536_cascade'
Args `(cond_512, cond_1024, flow_512, flow_1024, lr_resolution=512, resolution=1024 or 1536, coords, params, max_num_tokens=49152)`:
1. LR: `noise=SparseTensor(randn(N, flow_512.in_channels), coords)`; `slat=shape_slat_sampler.sample(flow_512, noise, **cond_512, **shape_params).samples`; denorm.
2. `models['shape_slat_decoder'].low_vram=True`; `hr_coords = shape_slat_decoder.upsample(slat, upsample_times=4)`; reset low_vram.
3. quantize loop (`hr_resolution=resolution`, step -=128):
   ```
   quant = cat([hr_coords[:,:1], ((hr_coords[:,1:]+0.5)/512*(hr_resolution//16)).int()], 1)
   coords = quant.unique(dim=0); num_tokens = coords.shape[0]
   if num_tokens < 49152 or hr_resolution==1024: break
   hr_resolution -= 128
   ```
4. HR: `noise=SparseTensor(randn(coords.shape[0], flow_1024.in_channels), coords)`; `slat=shape_slat_sampler.sample(flow_1024, noise, **cond_1024, **shape_params).samples`; denorm.
5. return `(slat, hr_resolution)`; `res=hr_resolution`.

### 7. Texture SLAT (`sample_tex_slat`)
- Renormalize shape slat: `shape_slat = (shape_slat - mean[None])/std[None]` (SHAPE normalization).
- `in_channels = flow.in_channels` (tex flow). `noise = shape_slat.replace(feats=randn(N, in_channels - shape_slat.feats.shape[1]))` -> tex-only channels.
- `slat = tex_slat_sampler.sample(flow, noise, concat_cond=shape_slat, **cond_1024, steps=12, guidance_strength=1.0, guidance_rescale=0.0, guidance_interval=[0.6,0.9], rescale_t=3.0).samples`. NOTE guidance_strength=1.0 -> CFG mixin returns cond-only (single forward), neg_cond unused for tex.
- Denorm with TEX normalization: `slat = slat*std[None]+mean[None]`.

### 8. Decode (`decode_latent`, no_grad)
```
meshes, subs = decode_shape_slat(shape_slat, res)   # decoder.set_resolution(res); decoder(slat,return_subs=True)
tex_voxels   = decode_tex_slat(tex_slat, subs)      # tex_decoder(slat, guide_subs=subs)*0.5+0.5  -> [0,1]
for m,v: m.fill_holes(); MeshWithVoxel(m.vertices,m.faces, origin=[-.5,-.5,-.5],
            voxel_size=1/res, coords=v.coords[:,1:], attrs=v.feats,
            voxel_shape=Size([*v.shape,*v.spatial_shape]), layout=pbr_attr_layout)
```
tex decoder output 6-ch PBR in [-1,1] -> `*0.5+0.5` -> [0,1]. `subs` = intermediate sparse guidance from shape decoder fed to tex decoder. `MeshWithVoxel.query_attrs(xyz)` trilinearly samples attrs at world pos (used by GLB texture bake).

### 9. Sampler math (`flow_euler.py`, all 3 = `FlowEulerGuidanceIntervalSampler`, sigma_min=1e-5)
`sample`:
```
sample=noise; t_seq=linspace(1,0,steps+1); t_seq = rescale_t*t_seq/(1+(rescale_t-1)*t_seq)
for (t,t_prev): pred_v=_inference_model(model,sample,t,cond,neg_cond,...); sample = sample-(t-t_prev)*pred_v
return sample
```
`_inference_model(model,x_t,t,cond)`: `model(x_t, tensor([1000*t]*B), cond, **kw)`.
GuidanceInterval: if `gi[0]<=t<=gi[1]` apply CFG(gs) else gs=1 (cond-only).
CFG: gs==1 -> cond only; gs==0 -> neg only; else `pred=gs*pos+(1-gs)*neg`; if guidance_rescale>0 std-match rescale of x0 (`_pred_to_xstart`/`_xstart_to_pred`). SS uses rescale 0.7, shape 0.5, tex 0.0.

### 10. pipeline_type
- '512': SS res32; shape+tex via *_512 + cond_512; res=512.
- '1024': SS res64; shape+tex via *_1024 + cond_1024; res=1024.
- '1024_cascade' (default): SS res32; cascade shape (flow_512/cond_512 LR -> upsample x4 -> flow_1024/cond_1024 HR, target 1024); tex via tex_1024/cond_1024; res<=1024.
- '1536_cascade': cascade target 1536 (still flow_1024/tex_1024/cond_1024; budget loop floors 1024).

### 32-channel normalization vectors (exact, from pipeline.json)
shape_slat_normalization.mean = [0.781296,0.018091,-0.495192,-0.558457,1.060530,0.093252,1.518149,-0.933218,-0.732996,2.604095,-0.118341,-2.143904,0.495076,-2.179512,-2.130751,-0.996944,0.261421,-2.217463,1.260067,-0.150213,3.790713,1.481266,-1.046058,-1.523667,-0.059621,2.220780,1.621212,0.877230,0.567247,-3.175944,-3.186688,1.578665]
shape_slat_normalization.std = [5.972266,4.706852,5.445010,5.209927,5.320220,4.547237,5.020802,5.444004,5.226681,5.683095,4.831436,5.286469,5.652043,5.367606,5.525084,4.730578,4.805265,5.124013,5.530808,5.619001,5.103930,5.417670,5.269677,5.547194,5.634698,5.235274,6.110351,5.511298,6.237273,4.879207,5.347008,5.405691]
tex_slat_normalization.mean = [3.501659,2.212398,2.226094,0.251093,-0.026248,-0.687364,0.439898,-0.928075,0.029398,-0.339596,-0.869527,1.038479,-0.972385,0.126042,-1.129303,0.455149,-1.209521,2.069067,0.544735,2.569128,-0.323407,2.293000,-1.925608,-1.217717,1.213905,0.971588,-0.023631,0.106750,2.021786,0.250524,-0.662387,-0.768862]
tex_slat_normalization.std = [2.665652,2.743913,2.765121,2.595319,3.037293,2.291316,2.144656,2.911822,2.969419,2.501689,2.154811,3.163343,2.621215,2.381943,3.186697,3.021588,2.295916,3.234985,3.233086,2.260140,2.874801,2.810596,3.292720,2.674999,2.680878,2.372054,2.451546,2.353556,2.995195,2.379849,2.786195,2.775190]

Both latents are 32-channel. tex flow in_channels = 64 (32 tex noise + 32 shape concat) — confirm from tex flow JSON; noise width = tex_in - 32.

## Weight key patterns

Pipeline holds NO learnable params. 8 sub-model safetensors (each with companion `.json` giving `{"name":Class,"args":{...}}`):

- `ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors` (SparseStructureFlowModel, dense 3D DiT, resolution=64, bf16): `input_layer.*`, `t_embedder.mlp.{0,2}.{weight,bias}`, `blocks.{i}.norm1/norm2.*`, `blocks.{i}.attn.qkv.{weight,bias}`, `blocks.{i}.attn.proj.*`, image cross-attn `blocks.{i}.cross_attn.{to_q,to_kv,to_out}.*`, `blocks.{i}.mlp.{fc1,fc2}.*`, AdaLN `blocks.{i}.adaLN_modulation.*`, `out_layer.*`. attrs `resolution`, `in_channels`.
- `microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors` (SparseStructureDecoder, conv3d, fp16): `*.conv*`, `*.norm*`, resnet/upsample blocks, occupancy logit head. `decoder(z)>0`.
- `slat_flow_img2shape_dit_1_3B_{512,1024}_bf16.safetensors` (SLatFlowModel/ElasticSLatFlowModel, sparse DiT, in_channels=32): `input_layer.*`, `t_embedder.*`, sparse `blocks.{i}.attn.qkv/proj.*`, image cross-attn, `blocks.{i}.mlp.*`, `adaLN*`, `out_layer.*`. Possible 3D-RoPE / serialized-window attention params.
- `shape_dec_next_dc_f16c32_fp16.safetensors` (sc_vae decoder, latent c=32, fp16): sparse U-Net up/down blocks; exposes methods `set_resolution(res)`, `upsample(slat, upsample_times=4)`, `__call__(slat, return_subs=True)`, attr `low_vram`. Keys `*.up/down*`, sparse conv weights, `upsample_head.*` / coord-prediction head.
- `slat_flow_imgshape2tex_dit_1_3B_{512,1024}_bf16.safetensors` (SLatFlowModel, in_channels=64 = 32 tex + 32 shape concat): same DiT key layout; accepts `concat_cond`.
- `tex_dec_next_dc_f16c32_fp16.safetensors` (sc_vae decoder, c=32, fp16): `__call__(slat, guide_subs=subs)`, 6-ch PBR head (tanh) -> caller `*0.5+0.5`.

Per-model `.json` `args` supply every hyperparameter (in_channels, resolution, channels/depth/heads, latent dim) and MUST be read by the C++ loader. The 32-length mean/std vectors live in pipeline.json (verbatim above), not in safetensors.

## GGML notes

Pipeline-level ops (model internals are separate components):

- RNG order (PyTorch global, seed=42): SS dense noise `[1,C,64,64,64]`; then cascade LR shape noise `[N,32]`; HR shape noise `[N',32]`; tex noise `[N',tex_in-32]`. For bit-exactness must mirror PyTorch RNG; otherwise accept visual-equivalence and document.

- Flow Euler: host scalar t-schedule `rescale_t*t/(1+(rescale_t-1)t)` over linspace(1,0,steps+1); per step 1-2 model forwards (CFG) + `sample = sample - (t-t_prev)*pred_v` (ggml_sub+scale). CFG combine `gs*pos+(1-gs)*neg` elementwise. guidance_rescale std-match (SS 0.7, shape 0.5): per-sample std over all non-batch dims, scale x0, blend — needs a reduction op; tex uses 0.0 (skip). GuidanceInterval is host control flow on t.

- Latent affine normalize/denormalize: per-channel `x*std+mean` / `(x-mean)/std` with the 32-vectors above broadcast over channel axis of sparse `[N,32]` feats. ggml_mul/add with 1x32 const tensors.

- SS->coords: elementwise `>0`; optional 3D max_pool3d(stride=ratio) on dense bool volume (CUSTOM 3D max-pool); `argwhere`/nonzero with column reorder `[0,2,3,4]` -> `[N,4]` int (CUSTOM host gather, cheap).

- Cascade: `shape_slat_decoder.upsample` is a model forward (custom sparse op — separate component). Host-side quant `((hr_coords[:,1:]+0.5)/512*(hr_res//16)).int()` + `unique(dim=0)` row-dedup (CUSTOM CPU sort+unique over int4 rows) + budget while-loop (-128, floor 1024).

- SparseTensor.replace/concat: host bookkeeping; tex input = `[tex_noise(32) | shape_slat(32)]` via model `concat_cond` -> ensure channel order noise-first.

- Heavy custom-op components (own specs): DINOv3 ViT-L/16 (3D/axial RoPE, patch embed, LN), 3 flow DiTs (sparse attention, AdaLN, image cross-attn, possibly serialized-window/3D-RoPE), 2 sc-vae decoders (sparse conv, gather/scatter, coord upsample). `neg_cond=zeros_like(cond)` trivial.

- External (out of GGML scope): BiRefNet/RMBG-2.0 rembg (heavy seg net — prefer requiring pre-masked RGBA input; preprocess_image already bypasses via has_alpha), LANCZOS resize, CuMesh fill_holes/simplify (CUDA), `o_voxel.postprocess.to_glb` (remesh/UV/texture-bake/GLB+WebP). Memory: low_vram moves one network to GPU per stage then back — mirror this to fit ~4B; only one net + running latent resident.

## Open questions

RESOLVED from `/run/media/ilintar/D_SSD/models/trellis2/pipeline.json`:
- All three samplers = `FlowEulerGuidanceIntervalSampler`, sigma_min=1e-5. Params: SS {steps12, gs7.5, grescale0.7, ginterval[0.6,1.0], rescale_t5.0}; shape {steps12, gs7.5, grescale0.5, ginterval[0.6,1.0], rescale_t3.0}; tex {steps12, gs1.0, grescale0.0, ginterval[0.6,0.9], rescale_t3.0}.
- Latents are 32-channel; exact mean/std vectors captured verbatim.
- DINOv3 = `facebook/dinov3-vitl16-pretrain-lvd1689m` (ViT-L/16, D=1024). rembg = BiRefNet class with `briaai/RMBG-2.0` weights.
- SS flow resolution=64.

STILL TO CONFIRM from per-model `.json`/safetensors:
1. tex flow `in_channels` (expected 64 = 32 tex + 32 shape concat) -> sets tex noise width. Verify in `slat_flow_imgshape2tex_*_*.json`.
2. SS flow `in_channels` C (channels of SS latent) and SS decoder output resolution D (to confirm ratio=D//ss_res, likely D=64 -> ratio 2 for ss_res=32).
3. `shape_slat_decoder` semantics of `upsample_times=4` and the `hr_resolution//16` latent-grid factor (decoder component).
4. DINOv3 token count incl. CLS + register tokens for image_size 512 vs 1024 (16px patch -> 1024 vs 4096 patch tokens) and how flow models cross-attend.
5. Whether tex flow is a single Module or list (code has `flow_model[0].in_channels` fallback) — config says single SLatFlowModel per resolution.
6. RNG bit-exactness vs PyTorch (likely target visual equivalence).
7. SS decoder is from the OLDER TRELLIS-image-large repo path — confirm its config matches the trellis2 `SparseStructureDecoder` class signature.
