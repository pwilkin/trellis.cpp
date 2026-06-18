# dinov3

DINOv3 ViT-L/16 image conditioner — IMPLEMENTABLE GGML spec

0. ROLE & GROUND TRUTH. TRELLIS.2 image conditioning = DinoV3FeatureExtractor.extract_features (/tmp/TRELLIS.2/trellis2/modules/image_feature_extractor.py:81-92). Produces cross-attn cond [B,N,1024] consumed by all 3 flow DiTs. CRITICAL: GGUF weights come from timm mirror timm/vit_large_patch16_dinov3.lvd1689m (MEMORY workaround), NOT HF DINOv3ViTModel. Verified by dumping real safetensors (/media/ilintar/D_SSD/models/trellis2/dinov3/model.safetensors, 318 tensors). timm = Eva model (timm/models/eva.py:2886). Same Meta lvd1689m weights, numerically identical features (timm RotaryEmbeddingDinoV3 math == HF DINOv3ViTRopePositionEmbedding). The keys dump docs/spec/keys/dinov3__model.keys.txt is timm-named and authoritative for the loader. docs/spec/02-image_feature.md describes HF state_dict and is WRONG about names AND qkv bias (sec 6). Golden ref tools/ref_dinov3.py uses this exact timm path (img_size=512, m.norm=Identity, forward_features, then non-affine F.layer_norm); match cond.npy [1,1029,1024].

1. CONFIG (timm model_args eva.py:2890-2907 + shapes): hidden D=1024; depth=24; heads=16; head_dim=64; intermediate=4096; patch=16; channels=3; num_reg_tokens=4 -> num_prefix_tokens=5 (1 cls + 4 reg). qkv_bias=FALSE (fused qkv NO bias; q/k/v bias all zero, eva.py:2914 'QKV bias but zero'). proj(o_proj) bias=TRUE; mlp fc1/fc2 bias=TRUE. layerscale init 1e-5 (load learned gamma_1/gamma_2). rope temperature=100.0, rotate_half=TRUE, use_abs_pos_emb=FALSE. qk_norm=FALSE, scale_norm=FALSE. hidden_act=GELU erf (exact) -> ggml_gelu_erf. LayerNorm eps=1e-5; plain non-gated MLP.

2. PREPROCESS (host) ref_dinov3.py:33-38: (1) PIL resize((S,S),LANCZOS) square no crop, S=resolution (512 default;1024 ft1024). (2) np.array(RGB).float32/255 HWC[0,1]. (3) permute->CHW; stack->[B,3,S,S]. (4) ImageNet normalize (x-mean)/std, mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225] (deliberately ImageNet, not DINOv3-native). LANCZOS not in ggml -> CPU lanczos or use ref input_chw.npy for bit-exact validation.

3. TOKENS: num_patches=(S/16)^2. S=512->1024 patches N=1029; S=1024->4096 N=4101. Output [B,N,1024] ALL tokens (cls+4reg+patches), no slicing.

4. FORWARD. 4a Patch embed Conv2d k16 s16 non-overlap: out [B,1024,S/16,S/16], flatten(2).transpose(1,2)->[B,num_patches,1024] row-major h-outer/w-inner. GGML reshape weight [1024,3,16,16]->[768,1024]; im2col input->[768,num_patches]; mul_mat+bias[1024]. 4b Prefix concat cat([cls_token,reg_token,patches],seq)->[B,N,1024]. cls/reg broadcast over B. No additive pos-embed. mask_token absent. 4c RoPE tables (host, fixed image_size): q4=16. inv_freq[j]=100.0**(-(j/16)) j=0..15. coords_h[i]=(i+0.5)/Hp, coords_w likewise (Hp=Wp=S/16); meshgrid('ij')->[num_patches,2]=(y,x) row-major; 2*c-1->[-1,1]. angles=2*pi*coords[:,:,None]*inv_freq -> [num_patches,2,16]; flatten(1)->[num_patches,32] (16 y then 16 x); tile(2)->[num_patches,64]. cos=cos(angles),sin=sin(angles) f32. 4d Per-layer i=0..23 (pre-norm EvaBlock eva.py:269): r=x; y=LN(x,eps,norm1.w,norm1.b); y=Attn(y); y=y*gamma_1; x=r+y; r=x; y=LN(x,eps,norm2.w,norm2.b); y=fc2(gelu_erf(fc1(y))); y=y*gamma_2; x=r+y. 4e Attention (EvaAttention eva.py:197-262): qkv=qkv_w@x [3072,N] NO bias; reshape [B,N,3,16,64].permute(2,0,3,1,4)->q,k,v [B,16,N,64]; q/k norm Identity; RoPE half-split ONLY patches [:,:,5:,:]: q_p=q_p*cos+rotate_half(q_p)*sin, k_p same (cos/sin [num_patches,64] bcast B,heads); prefix 5 unrotated; attn=softmax((q@k^T)*0.125)@v no mask; x=transpose->[B,N,1024]; x=proj(x)+proj.bias. rotate_half(t): split ne 64 into t1[0:32],t2[32:64]; cat([-t2,t1]). 4f Final non-affine LayerNorm (CRITICAL): F.layer_norm(x,(1024,)) eps=1e-5 weight=None bias=None. checkpoint norm.weight/bias NOT applied. GGML ggml_norm(x,1e-5) only. Output [B,N,1024].

5. CONVENTIONS: nn.Linear [out,in]; ggml ne [in,out]; ggml_mul_mat(W,x) direct (dit.cpp lin). Compute f32 for fidelity.

6. INFERENCE vs TRAINING. SKIP: mask_token/bool_masked_pos; coord augs shift/jitter/rescale (self.training); drop_path; dropout; _init_weights. Final norm.{weight,bias} present but UNUSED. q/k/v biases absent (zero anyway).

7. VALIDATION: match ref cond.npy; use ref input_chw.npy [1,3,512,512] to bypass LANCZOS first pass then add CPU lanczos.

## Weight key map

Authoritative = timm names, verified vs real safetensors (318 tensors) and docs/spec/keys/dinov3__model.keys.txt. All f32.

Embeddings/prefix: patch_embed.proj.weight [1024,3,16,16] (reshape->[768,1024]); patch_embed.proj.bias [1024]; cls_token [1,1,1024] (prefix token 0); reg_token [1,4,1024] (prefix tokens 1..4, 4 registers); (NO mask_token; NO pos_embed; RoPE inv_freq/periods non-persistent -> recompute host sec4c).

Per block i=0..23 (prefix blocks.{i}.): norm1.weight [1024] + norm1.bias [1024] = LN pre-attn; attn.qkv.weight [3072,1024] (NO bias) = fused, split rows [0:1024]=q,[1024:2048]=k,[2048:3072]=v each [1024,1024]; attn.proj.weight [1024,1024] + attn.proj.bias [1024] = o_proj (bias yes); gamma_1 [1024] = layerscale1; norm2.weight [1024] + norm2.bias [1024] = LN pre-MLP; mlp.fc1.weight [4096,1024] + mlp.fc1.bias [4096] = up_proj; mlp.fc2.weight [1024,4096] + mlp.fc2.bias [1024] = down_proj; gamma_2 [1024] = layerscale2.

Final: norm.weight [1024] + norm.bias [1024] = PRESENT but UNUSED (do NOT apply; final norm parameter-free).

timm<->HF (reference only; loader uses timm names): patch_embed.proj<->embeddings.patch_embeddings; reg_token<->embeddings.register_tokens; attn.qkv<->attention.{q,k,v}_proj(fused); attn.proj<->attention.o_proj; gamma_1/gamma_2<->layer_scale1/2.lambda1; mlp.fc1/fc2<->mlp.up_proj/down_proj.

## GGML plan

REUSE from src/dit.cpp (share static helpers): lin(c,m,prefix,x)=ggml_mul_mat(weight,x)+optional bias (use for fc1/fc2 and attn.proj; qkv works since try_get bias->null). layernorm(c,x,eps,w,b)=ggml_norm+mul+add (norm1/norm2 affine; FINAL norm via w=b=nullptr parameter-free). sdpa(c,q,k,v,d_model)=attention core (scale 1/sqrt(64)=0.125, soft_max_ext, no mask; expects q,k,v [head_dim,n_heads,L]). fused-qkv split pattern from self_attn: reshape [3072,L]->[hd,nh,3,L]+ggml_view_4d+ggml_cont (split order q,k,v contiguous matches timm permute(2,0,3,1,4)).

GENUINELY NEW: 1. Patch embed: weight reshape [1024,3,16,16]->[768,1024]; ggml_im2col(k16,s16,p0) input[3,S,S]->[768,num_patches]; mul_mat+bias. Patch order h-outer/w-inner. 2. Prefix concat: ggml_concat cls,reg,patches along seq dim (ne1). 3. DINOv3 half-split RoPE (NEW — dit.cpp apply_rope is interleaved-pair, WRONG). Precompute cos/sin (ne0=64,num_patches) host. rotate_half via ggml_view split ne0 64->two 32 halves, ggml_scale(-1) second, ggml_concat([-x2,x1]) ne0; x*cos+rot*sin. Apply ONLY to patch sub-view of q/k (seq offset 5..N); prefix 5 untouched (concat unrotated-prefix+rotated-patch along seq). Broadcast cos/sin over heads (shape [64,1,num_patches]). 4. GELU: ggml_gelu_erf (ggml.h:1169, GGML_UNARY_OP_GELU_ERF:607). NOT ggml_gelu (tanh) — correctness trap; dit.cpp uses tanh which is WRONG here. 5. LayerScale: ggml_mul(x, gamma_1/gamma_2) broadcast [1024].

Memory: ~300M params fp32 ~1.2GB; load f32 (validate) then optionally f16. S=1024 N=4101 attention N^2 x16heads sizable -> consider flash-attn/f16 for 1024 path; S=512 (N=1029) cheap. Keep cond on device for DiT. Compute f32 first validation.

Converter (tools/convert.py): dinov3 entry present (line 36-37, arch dinov3-vitl16). (a) pass timm names through (already match keys dump); (b) drop or never-load norm.weight/bias (unused); (c) keep fused qkv [3072,1024], split in C++ (mirrors dit.cpp self_attn) OR pre-split q/k/v in converter for simpler graph.

## Reuse

REUSE verbatim from src/dit.cpp: lin (Linear via ggml_mul_mat+optional bias; handles no-bias qkv since try_get->null), layernorm (ggml_norm+affine; also parameter-free final norm via w=b=nullptr), sdpa (scale/softmax/matmul core; identical math, no mask, scale=0.125), fused-qkv view-split pattern (reshape [hd,nh,3,L]+ggml_view_4d+ggml_cont). REUSE concepts: ggml_concat for token prepend; ggml_im2col+reshape for non-overlapping patch conv (same family as ss_decoder.cpp conv, though that is conv_3d).

GENUINELY NEW (cannot reuse): Half-split (NeoX) RoPE — dit.cpp apply_rope is INTERLEAVED-PAIR (reshape [2,half], pick even/odd); DINOv3 is HALF-SPLIT cat([-x2,x1]) over 32/32 halves; write new. RoPE on ONLY seq sub-range (patches offset 5), prefix 5 unrotated — dit.cpp rotates all. 2D axial RoPE table (y-angles ++ x-angles, tile x2) from patch-center coords [-1,1] — dit.cpp tables are 3D voxel RoPE. GELU-erf vs GELU-tanh. LayerScale gamma_1/gamma_2 — no analog (DiT uses AdaLN). Patch-embed conv + ImageNet preprocess/LANCZOS resize on host. Reference harness tools/ref_dinov3.py already exists and is correct (timm path, non-affine final LN) — use cond.npy/input_chw.npy as golden.

## Open questions

1. RESOLVED: keys are timm layout (safetensors dump), fused qkv NO bias (qkv_bias=False). docs/spec/02-image_feature.md sec5c (separate q/k/v with q/v bias) is wrong for the weights but numerically equivalent (biases zero). Loader must use timm names. 2. LANCZOS fidelity: PIL 3-lobe Lanczos+antialias must be matched for bit-exact; first pass use ref input_chw.npy; then CPU lanczos (stb_image_resize) and verify drift. 3. Patch order in im2col must be h-outer/w-inner to match conv flatten AND RoPE meshgrid('ij'); a transpose bug silently misaligns RoPE vs tokens; verify one patch row vs ref embeddings. 4. Default S: pipeline overrides image_size at runtime (trellis2_texturing.py:169 sets =resolution); confirm actual resolution arg (512 vs 1024) so RoPE tables sized right; ref uses 512. 5. Confirm DiT cross-attn consumes full N (incl 5 prefix), no stripping; cross-attn permutation-invariant so order irrelevant, but token count must match training (full sequence).
