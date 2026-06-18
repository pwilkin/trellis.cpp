# Consolidated implementation plan (synthesis)

## Implementation order

Build bottom-up so each layer is independently testable against dumped PyTorch references. Dependency-aware order:

PHASE 0 — Infrastructure (no model math):
1. safetensors loader + JSON config loader (parse 8-byte-len header + JSON; map dtypes F16/BF16/F32). Already verified header format works. Handle flat `conv.weight` flex_gemm layout [Co,Kd,Kh,Kw,Ci].
2. SparseTensor C++ struct: {feats [N,C], coords [N,4] int32 (b,x,y,z), B, scale triple, spatial_cache map}. Plus layout/seqlen/cum_seqlen/batch_broadcast_map derived from bincount+cumsum (rows contiguous-per-batch invariant). VarLenTensor (cond, no coords).
3. ggml context/backend bootstrap (reuse llama.cpp/ggml-cuda scaffolding in the repo).

PHASE 1 — Shared DiT primitives (used by all 3 flow models): build & validate ONCE.
4. Linear(+bias), LayerNorm (affine + non-affine, eps param), SiLU, GELU-tanh (mlp), GELU-erf (DINOv3 only).
5. MultiHeadRMSNorm (qk-norm): ggml_rms_norm(eps~1e-12) * gamma[H,Dh] — sqrt(D) factor already baked into rms_norm. Validate equality vs F.normalize*sqrt(D)*gamma.
6. SDPA: mul_mat QK^T -> scale 1/sqrt(128) -> soft_max -> mul_mat V. (flash-attn optional later.)
7. TimestepEmbedder (sinusoidal cos|sin 256 -> Linear/SiLU/Linear), AdaLN shared modulation (SiLU+Linear->9216) + per-block raw `modulation` Parameter add + chunk-6.
8. 3D interleaved-pair RoPE (custom): precompute cos/sin tables from integer coords; freqs[j]=1/10000^(j/freq_dim), 3 axis blocks of 21 + identity pad to 64. Interleaved (view_as_complex adjacent-pair) rotation, broadcast over heads. This is the single trickiest shared op — covers BOTH dense (ss_flow grid coords) and sparse (slat voxel coords) by feeding different coord arrays.

PHASE 2 — Sampler (pure host scalar math, no ggml graph):
9. FlowEulerGuidanceIntervalSampler: linspace(1,0,steps+1) -> Mobius rescale_t warp -> per-step interval-gated CFG -> Euler x -= (t-t_prev)*v. CFG combine + optional std-match rescale (host-pulled scalars for B=1). Params verified from pipeline.json (SS 12/7.5/0.7/[0.6,1.0]/5.0; shape 12/7.5/0.5/[0.6,1.0]/3.0; tex 12/1.0/0.0/[0.6,0.9]/3.0; sigma_min=1e-5).

PHASE 3 — First end-to-end vertical slice (MVP, see mvp_path):
10. DINOv3 ViT-L/16 image encoder (image_feature). Validate token output [B,1029,1024] @512.
11. SparseStructureFlowModel (dense 3D DiT, 30 blocks) — uses Phase-1 primitives on dense [B,4096,1536]. Validate velocity output.
12. SparseStructureDecoder (dense conv3d, fp16 torso) -> occupancy -> argwhere coords. Needs dense Conv3d (im2col+matmul) + pixel_shuffle_3d + ChannelLayerNorm. First custom-conv work but DENSE (simpler than sparse).

PHASE 4 — Sparse core + sparse conv (the hard middle):
13. sparse_core ops: SparseLinear, segmented norms (per-batch), SparseSpatial2Channel/Channel2Spatial gather-scatter with linear-code unique/sort, SparseDownsample/Upsample.
14. Submanifold SparseConv3d: build [N,27] neighbor map via coord hashmap (cache per scale), pad-zero-row trick for missing neighbors, 27x (get_rows + mul_mat) accumulate + bias.
15. SLatFlowModel (img2shape, in=32) sparse DiT — reuses Phase-1 DiT primitives on feats[N,1536] + sparse 3D RoPE + block-diagonal attention (dense for B=1).

PHASE 5 — Decoders & geometry:
16. FlexiDualGridVaeDecoder (shape_dec): C2S up-blocks WITH to_subdiv (confirmed pred_subdiv=True, 8 to_subdiv keys), returns (h7ch, subs). Then flexible_dual_grid_to_mesh (pure CPU geometry: hashmap, edge-quad gather, diagonal split, vertex world-pos).
17. SLatFlowModel (imgshape2tex, in=64): sparse_cat([tex_noise32, shape_slat32]) -> 64. concat_cond ordering noise-first.
18. SparseUnetVaeDecoder (tex_dec): pred_subdiv=FALSE (confirmed 0 to_subdiv keys) — DRIVEN by guide_subs from shape decoder. Critical: tex C2S consumes subs[i] (raw-logit SparseTensors, binarized >0) so voxel tree is IDENTICAL to shape decode.

PHASE 6 — Export:
19. ovoxel_mesh GLB export: CPU UV-raster + sparse trilinear grid_sample_3d bake + seam dilation + glTF2 writer (axis swap y,z->z,-y; V-flip). Stub remesh/decimate initially (bake on raw mesh, doubleSided=true).

PHASE 7 — Cascade & pipeline glue:
20. Pipeline orchestration: preprocess (require pre-masked RGBA to skip rembg), get_cond@512/@1024, cascade shape (LR flow_512 -> shape_dec.upsample x4 -> quant/unique/budget loop -> HR flow_1024), denorm with the verified 32-vectors, low-vram per-stage GPU residency. Driven from text via stable-diffusion.cpp upstream.

## Shared primitives

REUSABLE GGML BUILDING BLOCKS (build once in Phase 1, instantiate everywhere):

A. DiT block primitive set (shared by ALL 3 flow DiTs — ss_flow dense, img2shape sparse, imgshape2tex sparse; all are identical 30-block, model_channels=1536, num_heads=12, head_dim=128, mlp_hidden=8192, mlp_ratio=5.3334, share_mod=true, qk_rms_norm=true, qk_rms_norm_cross=true, pe_mode=rope — VERIFIED from all 3 JSON configs). The ONLY difference is dense [B,L,1536] vs sparse feats[N,1536] tensor shape and in_channels (8/32/64). Write the block ONCE parameterized by whether tensor is dense-3D or sparse-feats:
  - ModulatedTransformerCrossBlock: norm1(no affine,eps1e-6) -> adaLN shift/scale -> self_attn(rope+qk_rms) -> gate -> +res -> norm2(AFFINE,eps1e-6) -> cross_attn(qk_rms,no rope) -> +res -> norm3(no affine) -> adaLN -> mlp(GELU-tanh) -> gate -> +res.
  - Self-attn: to_qkv[4608,1536] fused -> split q/k/v -> qk_rms_norm -> 3D RoPE -> SDPA -> to_out. Keys: blocks.{i}.self_attn.{to_qkv,to_out}.{weight,bias}, .{q,k}_rms_norm.gamma[12,128].
  - Cross-attn: to_q[1536,1536], to_kv[3072,1024] fused -> qk_rms -> SDPA(q=N, kv=cond tokens) -> to_out.
  - AdaLN: shared adaLN_modulation.1.{weight[9216,1536],bias} + per-block modulation[9216] Parameter, add then chunk-6 -> shift/scale/gate x{msa,mlp}.

B. MultiHeadRMSNorm = ggml_rms_norm(eps~1e-12) then ggml_mul by gamma[H,Dh]. Identical formula in dense and sparse paths.

C. 3D interleaved-pair RoPE generator+applicator: cos/sin tables from integer coords (dense: meshgrid grid; sparse: voxel coords[:,1:]). Same freq formula, freq_dim=21, 3 axes + 1 identity pad to 64. Reused by all self-attn.

D. TimestepEmbedder + sinusoidal embedding: identical across all flows.

E. SparseTensor container + derived (layout/seqlen/batch_broadcast_map) + SparseLinear (= dense matmul on feats) + elementwise/replace: shared by sparse_core, both VAE decoders, all sparse flows.

F. Submanifold SparseConv3d (neighbor-map build + 27-tap gather-matmul): shared by BOTH shape_dec and tex_dec (and shape_dec.upsample). Neighbor map cached per scale (level1 reuses across 16 ConvNeXt blocks).

G. SparseChannel2Spatial / SparseSpatial2Channel gather-scatter: shared by both decoders' C2S up-blocks. CRITICAL SHARED STATE: tex_dec reuses the EXACT subdivision masks (subs) emitted by shape_dec, so the C2S child-voxel index maps computed in shape decode can be cached and replayed in tex decode (same coords, same growth tree).

H. Coordinate hashmap (packed b,x,y,z -> row idx): shared by sparse conv neighbor-map, C2S child indexing, and the export trilinear grid_sample_3d.

I. Conv weight loader: flex_gemm layout [Co,3,3,3,Ci] confirmed in actual checkpoints (F16); reorder to [27,Ci,Co] tap-major at load for all sparse convs and the dense SS decoder convs.

J. LayerNorm family: affine/non-affine variants, eps switchable (1e-6 blocks, 1e-5 final F.layer_norm, 1e-5 ChannelLayerNorm); compute in fp32.

K. Latent affine (de)normalize: per-channel x*std+mean over 32 channels with the 4 verified 32-vectors (shape_slat + pbr/tex), broadcast over [N,32]. Shared by shape and tex stages.

## Hardest parts

Ranked by difficulty/risk, with concrete GGML/C++ approach:

1. SUBMANIFOLD SPARSE CONV 3x3x3 (dominant effort, two VAE decoders + shape upsample). Output coords == input coords (no growth). Approach: (a) build a coord->row hashmap (pack b + 10-bit x,y,z into uint64); (b) for each active voxel and each of 27 offsets d in {-1,0,1}^3 look up p+d, producing nmap[N,27] int32 (-1 if absent), cached per scale under a key like SubMConv3d_3x3x3. (c) Append one zero row to feats, remap -1 -> N (zero contribution). (d) conv = bias + sum_{t=0..26} ggml_get_rows(feats, nmap[:,t]) @ W[t] where W reordered to [27,Ci,Co]; tap t -> (kd,kh,kw)=(t/9,(t/9)%3,t%3) offset -1..1 — VERIFY offset enumeration against flex_gemm so W[t] aligns (the [Co,Kd,Kh,Kw,Ci] permute implies Kd outer). De-risk: dump one reference conv layer's input feats+coords+output and match. Memory: nmap N*27 int32 reused across all convs at a scale.

2. SPARSE CHANNEL->SPATIAL UPSAMPLE with EXTERNAL subdivision (C2S). The shape decoder predicts subdiv via to_subdiv (confirmed present); the TEX decoder has NO to_subdiv (confirmed 0 keys) and is driven by guide_subs from the shape decoder. Approach: For each parent voxel, binarize subdiv[N,8]>0; for each set octant s in 0..7, emit child coord = parent*2 + (s&1, (s>>1)&1, (s>>2)&1) and gather feats from reshape(feats,[N*8,Cout])[parent*8+s]. Need host prefix-sum to allocate child rows + emit new coords. CRITICAL: cache the (child_coords, gather_idx) computed during SHAPE decode and REPLAY in TEX decode — identical subs => identical tree. This coupling is the key correctness lever and the biggest "gotcha". De-risk: assert shape-decode and tex-decode produce identical coords at each level.

3. 3D INTERLEAVED-PAIR RoPE (correctness trap). NOT ggml_rope (which is NeoX/GPT-J half-split). view_as_complex pairs ADJACENT (x[2k],x[2k+1]); rotate per pair by coord-derived angle. freq blocks: pairs 0..20 = x-coord*freqs, 21..41 = y, 42..62 = z, pair 63 = identity. Approach: precompute cos/sin[N or L, 64] on host from int coords; custom map op or build rotate-pairs via strided views (reshape [.,64,2], swap+negate). Broadcast over 12 heads. De-risk: unit-test against torch apply_rotary_embedding on random q + known coords.

4. flexible_dual_grid_to_mesh (CPU geometry, shape_dec head). 7 channels: [0:3] vertices=2*sigmoid-0.5 (voxel_margin .5), [3:6] intersected=logit>0, [6] quad_lerp=softplus. For each active cell+axis with intersected flag, gather 4-neighbor quad (3 fixed offset tables), hashmap-lookup all 4 active, emit quad; diagonal by sw[0]*sw[2] vs sw[1]*sw[3]; vertex world-pos=(coord+offset)*(1/256)-0.5. NOTE coords are at 512-scale (4 up-blocks) but normalized by grid_size=256 — VERIFIED config resolution=256 with 4 C2S; the (-0.5,1.5) vertex range from 2*sigmoid-0.5 is consistent with dual-grid-on-512-through-256-voxel-size. Pure scalar C++, O(N).

5. DENSE Conv3d + pixel_shuffle_3d (SS decoder). No ggml conv3d: im2col (Cin*27, Nspatial) + mul_mat against [Cout,Cin*27]; small tensors (max 512ch@16^3). pixel_shuffle_3d x2: precompute gather index map (c8=c_*8+sx*4+sy*2+sz -> out (2h+sx,2w+sy,2d+sz)); verify offset order (permute 0,1,5,2,6,3,7,4). ChannelLayerNorm = LN over channel dim per voxel.

6. UV-raster + sparse trilinear grid_sample_3d (export). CPU scanline raster of UV triangles -> per-texel barycentric 3D pos -> trilinear sample 8 neighbor voxels via hashmap (0 if absent), grid units=(world+0.5)*res. Seam fill via dilation (Telea substitute acceptable). De-risk: confirm grid_sample align convention (voxel-center vs corner) against flex_gemm.

7. CFG std-match rescale (SS 0.7, shape 0.5). Per-sample std over all non-batch dims; for B=1 pull two std scalars to host, scale x0, blend. For sparse the std is over all voxel*feature elements. Minor but needed for SS/shape parity.

## End-to-end data-flow shapes

TEXT -> IMAGE (external, stable-diffusion.cpp): prompt -> RGBA PIL HxWx4 uint8. Require pre-masked RGBA to skip BiRefNet rembg. preprocess -> square crop, premultiply rgb*alpha, black bg, centered, uint8 RGB.

IMAGE -> DINOv3 cond (get_cond @512 and @1024):
- resize LANCZOS to SxS (S=512 or 1024), /255, ImageNet normalize -> [3,S,S] f32.
- patch embed Conv2d k16s16 -> [num_patches=(S/16)^2, 1024]; +CLS+4reg -> hidden [1, 5+np, 1024] (1029@512, 4101@1024).
- 24 ViT layers (RoPE on patch tokens only, head_dim64, 16 heads) -> parameter-free F.layer_norm -> cond [B, N=1029/4101, 1024] f32. neg_cond = zeros_like(cond).

SPARSE STRUCTURE (cond_512):
- noise [num_samples=1, 8, 64,64,64]... NOTE flow resolution=16 so latent grid is 16^3: noise dense [1,8,16,16,16] f32 (SS flow in_channels=8, resolution=16 VERIFIED).
- ss_flow forward per Euler step: patchify [1,8,16,16,16]->[1,4096,8] -> input_layer ->[1,4096,1536] -> 30 DiT blocks (self-attn 3D-RoPE over 4096 tokens, cross-attn to cond[1,1029,1024]) -> F.layer_norm -> out_layer [1,4096,8] -> unpatchify [1,8,16,16,16]. 12 steps, CFG 2 passes in interval.
- z_s [1,8,16,16,16] -> SS decoder: input_layer Conv3d 8->512 @16^3 -> 2 mid ResBlocks -> blocks (Up 16->32->64, channels 512->128->32) -> out_layer Conv3d 32->1 -> logits [1,1,64,64,64] -> >0 -> coords argwhere[:,[0,2,3,4]] -> [N,4] int32 (b,x,y,z) at res 64. (1024_cascade uses ss_res=32: max_pool3d ratio2 -> [1,1,32,32,32] -> coords at res32.)

SHAPE SLAT cascade (default 1024_cascade):
- LR: noise SparseTensor feats[N,32] @coords(res32); shape_flow_512 (in=32) 12 Euler steps with cond_512 -> slat feats[N,32]; denorm *std+mean (shape vectors). 
- shape_slat_decoder.upsample(slat, x4) -> hr_coords. quant=((hr_coords[:,1:]+0.5)/512*(hr_res//16)).int(); unique -> coords[Nhr,4]; budget loop (-128, floor 1024) until <49152 tokens.
- HR: noise feats[Nhr,32] @hr_coords; shape_flow_1024 (in=32) + cond_1024 -> slat[Nhr,32]; denorm. res=hr_resolution(<=1024).

TEX SLAT (cond_1024):
- renorm shape_slat (x-mean)/std. tex noise feats[Nhr, 64-32=32] (imgshape2tex in_channels=64 VERIFIED). noise = shape_slat.replace(randn[Nhr,32]).
- tex_flow forward: sparse_cat([noise32, shape_slat32]) -> feats[Nhr,64] -> input_layer ->[Nhr,1536] -> 30 blocks (concat_cond shape) -> out_layer [Nhr,32]. guidance_strength=1.0 -> single pass, 12 steps. -> denorm *std+mean (tex/pbr vectors) -> tex_slat[Nhr,32].

DECODE:
- shape_dec.set_resolution(256); forward(shape_slat[Nhr,32], return_subs=True): from_latent 32->1024 -> 4 stages ConvNeXt + 4 C2S up-blocks (each emits subdiv[Nlevel,8], grows voxels 32->64->128->256->512 scale) -> F.layer_norm -> output_layer ->[M,7]. subs = list of 4 SparseTensors (raw logits). Mesh via flexible_dual_grid_to_mesh -> vertices[V,3] f32, faces[F,3] int.
- tex_dec(tex_slat[Nhr,32], guide_subs=subs): from_latent 32->1024 -> SAME 4-stage growth driven by subs[i]>0 (NO own to_subdiv, VERIFIED) -> output_layer ->[M,6] -> *0.5+0.5 -> tex_voxels coords[M,3]+feats[M,6] (base_color3, metallic1, roughness1, alpha1).
- MeshWithVoxel(vertices, faces, voxel_size=1/res, coords, attrs[M,6]).

EXPORT to_glb: UV unwrap -> rasterize UV to texture_sizextexture_size -> per-texel 3D pos -> trilinear sparse-sample attrs[M,6] -> split/quantize uint8 -> inpaint seams -> PBRMaterial(baseColor RGBA, metallicRoughness [0,rough,metal]) -> axis swap (y,z)->(z,-y), V-flip -> Trimesh -> .glb (WebP textures).

## MVP path

MVP = simplest non-cascade '512' path producing an untextured (or flat-color) GLB, deferring all cascade/texture complexity.

MVP SCOPE (pipeline_type='512', res=512):
1. Require pre-masked RGBA input (skip BiRefNet entirely).
2. DINOv3 @512 cond only (1029 tokens). neg_cond=zeros.
3. SS: ss_flow (dense DiT) 12 steps CFG -> SS decoder -> coords @ res32 (ss_res=32 for '512', max_pool3d ratio2 from 64). This exercises: shared DiT primitives, 3D RoPE, sampler, dense conv3d, pixel_shuffle.
4. Shape: shape_flow_512 (in=32) non-cascade 12 steps CFG -> denorm. Exercises sparse DiT + sparse 3D RoPE + block-diag attention (dense B=1).
5. shape_dec.set_resolution(256) forward -> mesh (vertices/faces) via flexible_dual_grid_to_mesh. Exercises sparse conv, C2S with own to_subdiv, geometry head. SKIP tex entirely.
6. Export geometry-only GLB: no UV/texture, just vertices+faces, flat baseColorFactor. Skip remesh/decimate/UV-bake (raw mesh, doubleSided). This proves the whole geometry stack end-to-end.

This MVP touches every hard op EXCEPT: cascade upsample/quant/budget loop, tex flow concat_cond, tex decoder guide_subs replay, and UV texture baking — all deferred.

MVP -> FULL increments (each independently shippable):
A. Add texture: tex_flow + tex_dec(guide_subs) + trilinear bake + PBR GLB. (Biggest value-add after geometry.)
B. Add cascade ('1024_cascade' default): shape_dec.upsample x4, quant/unique/budget loop, HR flow_1024, cond_1024. Higher res geometry.
C. Add remesh/decimate/proper UV unwrap (xatlas) + Telea-quality inpaint for production-grade GLB.
D. Perf: bf16 activations, flash-attn, low-vram per-stage GPU residency (one ~1.3B net at a time to fit 4B total), neighbor-map caching.

Fidelity note: accept visual-equivalence (not bit-exact) vs PyTorch RNG — document. LANCZOS resize and grid_sample align conventions are the two places to match carefully; everything else tolerates fp32-vs-fp16 drift given the >0 / softmax robustness.

## Risks

Top risks (highest first) with de-risking via dumped reference tensors:

1. SUBMANIFOLD CONV offset->weight-tap mapping. The [Co,Kd,Kh,Kw,Ci] permute means tap enumeration order matters; a transposed mapping silently garbles geometry. DE-RISK: dump one reference SparseConv3d layer's (feats_in, coords, weight, bias, feats_out) from a torch run; match exactly before building decoders. Confirm center tap (d=0) always present (p is active).

2. TEX-DECODER guide_subs COUPLING. tex_dec has NO to_subdiv (VERIFIED 0 keys) — it 100% depends on shape_dec's subs to know which child voxels to materialize. If the C++ shape decode and tex decode diverge in voxel-tree growth, tex feats land on wrong coords -> corrupt texture. DE-RISK: assert identical coords at every C2S level between the two decode passes; dump shape_dec subs[i] and tex_dec intermediate coords from reference and diff.

3. SHAPE-DEC resolution reconcile (RESOLVED but verify numerically). 4 C2S up-blocks (VERIFIED: blocks.{0.4,1.16,2.8,3.4} all have to_subdiv) take scale 32->512, but flexible_dual_grid uses grid_size=256, voxel_size=1/256. DE-RISK: run reference decoder, print h.coords.max() and _scale before output_layer; confirm 512-scale coords + /256 normalization (vertex range -0.5..1.5 matches 2*sigmoid-0.5).

4. 3D RoPE interleaved-vs-half-split. Using ggml_rope NeoX mode would be WRONG (code uses view_as_complex adjacent pairs). DE-RISK: unit-test custom RoPE op vs torch apply_rotary_embedding on random input + known coords; check first/last freq blocks + identity pad-pair.

5. RNG bit-exactness vs PyTorch (seed=42, global RNG order: SS dense -> LR shape -> HR shape -> tex). Mirroring torch RNG exactly is hard. DE-RISK: accept visual equivalence; document. Optionally seed from dumped reference noise tensors for validation runs.

6. grid_sample_3d align convention (export). voxel-center vs corner shifts samples 0.5 voxel. DE-RISK: read flex_gemm grid_sample CUDA kernel OR dump (attr_volume, query grid, output) from reference bake and match. Two callers use (pos-aabb0)/voxel_size vs (pos+0.5)*res — confirm equivalent (they are for aabb=+-0.5).

7. MultiHeadRMSNorm eps + ggml_rms_norm normalization constant. F.normalize eps=1e-12; ggml_rms_norm must yield x*sqrt(D)/||x||. DE-RISK: numeric check ggml_rms_norm(x)*gamma == F.normalize(x)*sqrt(128)*gamma within tol.

8. GELU variant trap: DINOv3 uses ERF gelu (use ggml_gelu_erf), DiT mlp uses TANH gelu (ggml_gelu). Mixing them drifts features. DE-RISK: confirm ggml build exposes gelu_erf; per-component pick correct variant.

9. cumesh remesh/decimate/uv_unwrap are compiled-CUDA black boxes. Reproducing exactly is infeasible; substitute xatlas + skip remesh. GLB geometry will differ from reference but texture stays correct (BVH re-samples original surface). DE-RISK: confirm product accepts non-identical mesh topology (texture parity is the requirement).

10. pixel_shuffle_3d channel->offset order (SS decoder). Wrong permute -> mirrored 64^3 grid -> wrong occupancy. DE-RISK: tiny numeric test of pixel_shuffle vs torch on a labeled tensor.

11. Memory at 4B params + 1024-res: 24-layer DINOv3 @4101 tokens N^2 attention, three 1.3B DiTs, 452M-param conv layer (blocks.0.4) in shape/tex decoders. DE-RISK: low-vram per-stage residency (one net on GPU at a time), bf16 weights, build neighbor maps once per scale, tile UV bake.

