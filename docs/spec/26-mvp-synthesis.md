## MVP build order

DECISION: Geometry-only MVP = DINOv3 cond -> SS stage (DONE) -> shape_slat flow (ALREADY WIRED) -> shape_dec sparse decoder + dual-grid mesh -> GLB (writer DONE). The two genuinely-remaining big builds are (1) DINOv3 conditioner and (2) the sparse shape decoder (sparse_conv + shape_dec + mesh extraction). sparse_core is absorbed into those (no standalone component). slat_flow needs only a real cond + glue, not new code.

Verified current state before ordering: src/dit.cpp::build_dit_dense + all helpers (lin/layernorm/rms_gamma/apply_rope/sdpa/self_attn/cross_attn/modulate/block) exist; src/flow_runner.cpp has DitRunner + make_sparse_runner + sample_flow (FlowEuler w/ guidance interval); src/ss_decoder.cpp has run_seg harness, conv3d, pixel_shuffle, ss_coords->vector<array<int,3>>; src/mesh_glb.cpp ALREADY implements write_glb(path,verts,V,faces,F) with the (x,z,-y) swap. test_slat_shape.cpp already runs full img2shape sampling vs golden. So slat_flow is effectively done barring real-cond wiring.

BUILD ORDER (dependency-aware):

STEP 1 — DINOv3 conditioner (src/dinov3.cpp). No upstream deps; unblocks real cond for slat_flow + SS. Build forward graph reusing lin/layernorm/sdpa. NEW code: im2col patch-embed, prefix concat (ggml_concat cls+reg+patches), HALF-SPLIT RoPE (not dit.cpp interleaved), gelu_erf, layerscale gamma_1/gamma_2, parameter-free final ggml_norm(1e-5). Converter: add a dinov3 path that passes timm names through, drops norm.weight/bias, and (RECOMMENDED) pre-splits fused qkv [3072,1024] into q/k/v to keep the graph simple. VALIDATE against tools/ref_dinov3.py: feed ref input_chw.npy [1,3,512,512] (bypass LANCZOS) -> match cond.npy [1,1029,1024]. Add CPU lanczos only after the graph matches.

STEP 2 — slat_flow real-cond integration (glue only, no new kernels). Feed STEP 1 [1024,N_img] cond into make_sparse_runner using SS-stage coords. Re-confirm test_slat_shape still passes with dumped cond, then with live DINOv3 cond. img2shape in_ch=out_ch=32. (Tex flow in_ch=64 is M5, skip for geometry MVP.) VALIDATE: rerun trellis-test-slat-shape; then a fresh ref dump using real SS coords + real cond.

STEP 3 — sparse infrastructure (src/sparse.{h,cpp}): SparseTensor struct {ggml_tensor* feats[C,N]; vector<array<int,3>> coords; int res;}, coord hashmap (unordered_map<uint64,int> packed x<<40|y<<20|z), [N,27] neighbor-table builder, submanifold conv as 27x (ggml_get_rows + ggml_mul_mat accumulate) with a zero sentinel row for missing neighbors. VALIDATE in isolation: dump one reference conv (blocks.3.0.conv) on a tiny known sparse input from Python; compare per-voxel output. This is the single highest-risk kernel — validate BEFORE building blocks on it.

STEP 4 — shape_dec decoder (src/shape_dec.cpp) on top of STEP 3. from_latent (Linear 32->1024); per-stage ConvNeXt blocks (conv -> rowwise LayerNorm affine eps1e-6 -> Linear C->4C -> SiLU -> Linear 4C->C -> residual); 4 C2S up-blocks (to_subdiv Linear C->8, threshold >0 on host; norm1 affine; conv1 C->8*Cout; host channel2spatial octant materialization growing N; norm2 NON-affine eps1e-6; conv2; skip = repeat_interleave x4). Host-orchestrated per-stage graphs (mirror run_seg) with readback of subdiv logits + coords at each up-block since M is data-dependent. Final non-affine ggml_norm(1e-5) + output_layer (Linear 64->7). VALIDATE stage-by-stage: dump per-stage feats+coords from Python; first verify coords (octant ordering) match after each up-block, then feats.

STEP 5 — dual-grid mesh extraction (CPU, in shape_dec.cpp or mesh.cpp). Split 7-ch output: vertices=2*sigmoid(f0:3)-0.5, intersected=f3:6>0, split_weight=softplus(f6); flexible_dual_grid_to_mesh with the edge_neighbor_voxel_offset[3][4][3] tables + diagonal split. vertex world = (coord+vert)/512 - 0.5. VALIDATE: dump Python mesh.vertices/faces (min/max in [-0.5,0.5]); compare vertex positions + face count.

STEP 6 — wire end-to-end + GLB. Feed STEP 5 verts/faces into existing write_glb (already correct). Visual inspect the goblin GLB. This is the MVP milestone.

Defer to post-MVP: tex flow (in_ch=64), tex decoder, PBR bake, fill_holes, 1024 cascade, BiRefNet (geometry tolerates non-cut image; or stub).

## Reuse map

DINOv3 (src/dinov3.cpp, NEW file): REUSE src/dit.cpp lin (handles no-bias via try_get->null, used for qkv/fc1/fc2/proj), layernorm (affine norm1/norm2; param-free final via w=b=nullptr), sdpa (scale 0.125 for head_dim 64, no mask), fused-qkv view-split pattern. REUSE ggml builtins: ggml_concat (prefix prepend), ggml_im2col+reshape (patch conv, family of ss_decoder conv), ggml_gelu_erf, ggml_mul (layerscale). NEW: half-split/NeoX RoPE applied ONLY to patch sub-range (offset 5..N), prefix 5 untouched (dit.cpp apply_rope is interleaved-pair AND rotates all — cannot reuse); 2D axial RoPE table (y-angles++x-angles, tile x2) from patch-center coords; host ImageNet preprocess + CPU LANCZOS.

slat_flow: REUSE EVERYTHING AS-IS. src/dit.cpp::build_dit_dense unchanged; src/flow_runner.cpp::make_sparse_runner + sample_flow + DitRunner unchanged; timestep_embedding + fill_rope unchanged. feats [C,N] layout == dense [d_model,L]; key prefixes (input_layer/out_layer/t_embedder.mlp.0/2/adaLN_modulation.1/blocks.i.*) map 1:1. ZERO new compute code; only a driver supplying real cond + SS coords. Confirmed by working test_slat_shape.cpp.

sparse_core: NO standalone component. Container struct is NEW but tiny. SparseLinear->dit.cpp lin verbatim on feats[C,N]; per-row SparseLayerNorm->dit.cpp layernorm verbatim (ggml_norm over ne0=C); activations->ggml_silu/ggml_gelu_erf/ggml_relu. spatial2channel/channel2spatial gather->ggml_get_rows on reshaped feats + host index math. SparseGroupNorm NOT needed (shape/tex decoders use LayerNorm only — keys show *.norm.weight LayerNorm, no GroupNorm).

sparse_conv + shape_dec (src/sparse.cpp + src/shape_dec.cpp, NEW): REUSE dit.cpp lin (from_latent/output_layer/to_subdiv/mlp.0/mlp.2 and as the GEMM inside the 27-tap conv), dit.cpp layernorm (ConvNeXt norm affine 1e-6, up-block norm1 affine, norm2 non-affine -> pass w=b=nullptr, final non-affine 1e-5), ggml_silu. REUSE src/ss_decoder.cpp run_seg/gallocr per-stage subgraph harness + fp16->fp32 load path. NEW: coord hashmap, [N,27] neighbor table, submanifold conv via get_rows+mul_mat accumulate (NOT ggml_conv_3d — that dense path in ss_decoder is the wrong primitive; do NOT reuse conv3d() or pixel_shuffle() which assume all 8 children exist), host channel2spatial octant expansion, dynamic per-stage N growth, CPU dual-grid mesh extractor.

mesh_glb: DONE. src/mesh_glb.cpp::write_glb already implements (x,z,-y) swap, min/max, GLB chunk framing. Note its buffer order is positions-then-indices (spec said indices-first) — irrelevant since accessors point correctly. Consumes STEP 5 verts/faces directly. ZERO new code.

CONVERTER (tools/convert.py): NEW per-arch paths. (a) dinov3 entry exists (line 36) but needs a non-dense-conv branch: patch_embed.proj is a real 4D Conv2d [1024,3,16,16] OK, but must NOT hit the 5D dense reshape; drop norm.weight/bias; optionally pre-split qkv. (b) shape_dec arch trellis2-shape-dec: conv weights are [Co,3,3,3,Ci] ALREADY-permuted sparse layout — the existing generic 5D reshape [OC,IC,KD,KH,KW]->[OC*IC,...] (convert.py:84-85) would CORRUPT them into [Co*3,3,3,Ci]. MUST add an arch check so sparse conv weights are reshaped to ggml ne=[Ci,27,Co] (flatten kd*kh*kw) instead. Load-bearing converter bug to avoid.

## Hardest risks

RANKED by correctness risk x effort:

1. SUBMANIFOLD SPARSE CONV neighbor/weight/tap layout (HIGHEST). Two coupled ambiguities: (a) tap t=kd*9+kh*3+kw with offset (kd-1,kh-1,kw-1) and the (kd,kh,kw)<->(x,y,z) axis mapping; (b) the converter must load [Co,3,3,3,Ci] as ggml ne=[Ci,27,Co] WITHOUT hitting the existing dense 5D reshape (convert.py:84 would produce [Co*3,3,3,Ci] = silent corruption). flex_gemm is not installed so the axis convention is not source-verified. DE-RISK: dump from a working Python TRELLIS.2 install one reference conv (blocks.3.0.conv = smallest, C=128) applied to a tiny hand-made sparse input (e.g. 5 voxels in a known pattern): save input feats+coords and output feats. In C++ replicate and require exact match. Since build+load use the SAME convention, a self-consistent ordering is correct ONLY IF it matches training — the dump is the only way to confirm. Validate the converter reshape in the same test.

2. C2S subdiv index math: octant->channel-block AND octant->coord-offset (HIGH). channel2spatial reshape [N*8,C/8] means octant o consumes channels [o*(C/8):(o+1)*(C/8)]; coord offset = (o&1,(o>>1)&1,(o>>2)&1) on 2*parent. A transpose between these two scrambles geometry silently while keeping shapes valid. Tied to risk 1 axis convention (conv1 8*Cout output block ordering must match the octant decode). DE-RISK: dump input coords/feats and output coords/feats of ONE up-block (blocks.3.4, smallest) from Python; in C++ verify (i) output coord set matches exactly (octant->offset), then (ii) per-child feats match (octant->channel-block). Coords before feats.

3. DINOv3 half-split RoPE + prefix exclusion (MEDIUM-HIGH). rotate_half = cat([-x[32:64], x[0:32]]); 2D axial table = 16 y-freqs ++ 16 x-freqs then tile x2 -> 64; coords (i+0.5)/Hp mapped to [-1,1]; meshgrid('ij') row-major (h-outer/w-inner) MUST match the im2col patch order or RoPE silently misaligns with tokens. Applied ONLY to patches (seq 5..N); prefix 5 unrotated. dit.cpp RoPE is wrong on all three counts. DE-RISK: ref_dinov3.py already produces cond.npy; first match with input_chw.npy to remove LANCZOS variance; if mismatch, dump post-RoPE q for layer 0 from timm and compare one patch row + one prefix token.

4. Dual-grid mesh (MEDIUM). edge_neighbor_voxel_offset[3][4][3] tables, all-4-neighbors-active gate, diagonal choice sw[0]*sw[2] vs sw[1]*sw[3], vertex=(coord+2*sigmoid-0.5)/512-0.5. Off-by-one in offset tables or wrong grid_size (must be 512, the final coord grid, NOT a requested output res) gives a distorted/holey mesh. DE-RISK: dump Python mesh.vertices (assert min/max within [-0.5,0.5]) + faces; compare vertex positions and face count/topology.

5. data-dependent N growth control flow (MEDIUM, effort not correctness). M unknown until to_subdiv readback -> per-up-block graph rebuild + host readback (mirror run_seg). Risk is plumbing/memory (N up to ~1e6 at 512^3; use per-tap accumulate not materialized im2col at low-C/high-N stages). DE-RISK: covered by step-4/step-3 dumps; watch peak feats memory.

## Validation plan

PATTERN: extend tools/ref_*.py (ref_slat_shape.py is the template) — GPU, RAM-light, monkeypatch flash_attn->torch SDPA, load via safetensors, dump .npy, then a C++ test exe (mirror src/test_slat_shape.cpp) loads the gguf + npy and asserts max rel err < ~3e-2. All refs run on REF_DEV cuda:1, fp32 model for fidelity, dump only small tensors.

PER COMPONENT:

DINOv3 (tools/ref_dinov3.py EXISTS, correct): dump cond.npy [1,1029,1024] + input_chw.npy [1,3,512,512]. C++ test (new src/test_dinov3.cpp + CMake target trellis-test-dinov3): feed input_chw.npy to bypass LANCZOS, run dinov3 graph, compare cond.npy. Add intermediate dumps (post-patch-embed tokens[0..2], post-layer0 x, post-RoPE q layer0) ONLY if final mismatch, to localize (patch order vs RoPE vs gelu).

sparse conv kernel (NEW tools/ref_sparse_conv.py): construct a SparseTensor with ~5-20 hand-placed voxels at known coords, run ONE conv (blocks.3.0.conv, C=128) via the real SparseConv3d module (SPCONV_ALGO=native), dump in_feats/in_coords/out_feats. C++ test (src/test_sparse_conv.cpp): build coords+feats, run submanifold conv, exact compare. THIS GATES STEP 3/4 — do first.

shape_dec stages (NEW tools/ref_shape_dec.py): load shape_dec, run SparseUnetVaeDecoder.forward(x, return_subs=True) on a real (or dumped) shape_slat latent; dump per-stage (after each of the 4 up-blocks) coords[Mi,3] + feats sample, the subs logits, and the final [N4,7]. C++ test (src/test_shape_dec.cpp): compare coords FIRST (set equality + ordering) after each up-block, then feats, then final 7-ch. Also dump the input latent so C++ does not depend on slat_flow yet.

mesh (extend ref_shape_dec.py): also dump mesh.vertices [V,3] + mesh.faces [F,3] from flexible_dual_grid_to_mesh(train=False). C++ test: compare vertex min/max in [-0.5,0.5], vertex array, face count, and a sample of faces (allow ordering differences only if topology identical).

slat_flow w/ real cond (extend ref_slat_shape.py): re-dump using REAL SS-stage coords + REAL DINOv3 cond (not random) so the existing trellis-test-slat-shape validates the integrated path, not just flow math.

END-TO-END: no numeric golden for the full goblin GLB (Python full pipeline needs CUDA-only o-voxel/nvdiffrast that may not install — per ROADMAP validation strategy). Validate by: (1) each component passes its dump test, (2) visual inspection of the exported GLB (sane bounding box ~[-0.5,0.5], watertight-ish goblin silhouette). Keep all dumps small (single image, fp32, sample slices for big feats) to stay host-RAM-light; never dump full 512^3 dense tensors — only the sparse [N,C] active rows.

