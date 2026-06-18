# DiT implementation recipe (authoritative — from /tmp/TRELLIS.2 source)

Shared by all 3 flow models. SS-flow = dense [B,L=res³,C]; SLAT flows = sparse feats[N,C]
(same block, tensor is [N,1536] instead of [B,L,1536], attention is over all N voxels of a batch).
model_channels=1536, num_heads=12, head_dim=128, mlp_hidden=8192 (int(1536*5.3334)),
num_blocks=30, share_mod=true, qk_rms_norm=true, qk_rms_norm_cross=true, pe_mode=rope.

## SS-flow forward (`SparseStructureFlowModel.forward(x[B,8,16,16,16], t[B], cond[B,Lc,1024])`)
1. patchify: `x.view(B,8,4096).permute(0,2,1)` → h [B,4096,8]
2. `h = input_layer(h)` Linear 8→1536  (SLAT: 32→1536 / 64→1536)
3. `t_emb = t_embedder(t)` → [B,1536]; then **shared** `t_emb = adaLN_modulation(t_emb)` = `Linear1536→9216(SiLU(t_emb))` → [B,9216]
4. for each of 30 blocks: `h = block(h, t_emb, cond, rope_phases)`
5. `h = F.layer_norm(h, [1536])`  (NON-affine, eps=1e-5 default)  — compute in f32
6. `h = out_layer(h)` Linear 1536→8 (SLAT shape/tex out: 1536→32)
7. unpatchify: `h.permute(0,2,1).view(B,8,16,16,16)`

## TimestepEmbedder
`half=128; freqs = exp(-ln(10000) * arange(128)/128)` → [128];  `args = t[:,None]*freqs[None]`
`emb = cat([cos(args), sin(args)], -1)` → [B,256]  (**cos first, then sin**)
`mlp`: Linear(256→1536) `t_embedder.mlp.0`, SiLU, Linear(1536→1536) `t_embedder.mlp.2`.

## Block (`ModulatedTransformerCrossBlock`, share_mod)
```
m = block.modulation[9216] + t_emb            # per-block Parameter + shared mod
shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp = m.chunk(6, dim=-1)   # each [B,1536]
h = norm1(x)                                   # LayerNorm NO affine, eps 1e-6, in f32
h = h*(1+scale_msa) + shift_msa
h = self_attn(h, rope_phases)
x = x + gate_msa*h
h = norm2(x)                                   # LayerNorm AFFINE (norm2.weight/bias), eps 1e-6
h = cross_attn(h, cond)                        # NO gate, NO modulation
x = x + h
h = norm3(x)                                   # LayerNorm NO affine, eps 1e-6
h = h*(1+scale_mlp) + shift_mlp
h = mlp(h)                                      # Linear1536→8192, GELU(tanh), Linear8192→1536
x = x + gate_mlp*h
```
mlp keys: `blocks.{i}.mlp.mlp.0.{weight,bias}`, `...mlp.2.{weight,bias}`.

## Self-attention (qk_rms_norm + rope)
`qkv = to_qkv(x)` [.,4608] → reshape [B,L,3,12,128] → unbind q,k,v.
`q=q_rms_norm(q); k=k_rms_norm(k)` then `q=rope(q,phases); k=rope(k,phases)`.
SDPA: softmax(qkᵀ / sqrt(128)) v. `to_out` Linear 1536→1536.
Keys: `self_attn.to_qkv.{w,b}`, `self_attn.{q,k}_rms_norm.gamma[12,128]`, `self_attn.to_out.{w,b}`.

## Cross-attention (qk_rms_norm_cross, NO rope)
`q=to_q(x)`[.,1536]→[B,L,12,128]; `kv=to_kv(cond)`[.,3072]→[B,Lc,2,12,128]→k,v.
`q=q_rms_norm(q); k=k_rms_norm(k)`; SDPA(q,k,v); `to_out`.
Keys: `cross_attn.to_q.{w,b}`, `cross_attn.to_kv.{w,b}` (1024→3072), `cross_attn.{q,k}_rms_norm.gamma`, `cross_attn.to_out.{w,b}`.

## MultiHeadRMSNorm  (== ggml_rms_norm then *gamma)
`F.normalize(x.float(),dim=-1) * gamma * sqrt(128)`. Over head_dim=128.
ggml: `ggml_rms_norm(x, eps≈1e-12)` already yields `normalize(x)*sqrt(128)` → just multiply by `gamma[128,12]`.

## 3D RoPE  (interleaved adjacent-pair — NOT ggml_rope/NeoX!)
`RotaryPositionEmbedder(head_dim=128, dim=3)`: `freq_dim = 128//2//3 = 21`;
`freqs[j] = 1.0 / 10000^(j/21)`, j=0..20.
phases for a token at integer coord (cx,cy,cz): 64 complex angles θ (pair index p=0..63):
  p in [0,21):  θ_p = cx * freqs[p]
  p in [21,42): θ_p = cy * freqs[p-21]
  p in [42,63): θ_p = cz * freqs[p-42]
  p == 63:      θ_p = 0     (identity pad: 63 = 3*21 < 64)
Apply to x[...,128] viewed as 64 adjacent pairs (x[2p], x[2p+1]):
  out[2p]   = x[2p]*cos θ_p − x[2p+1]*sin θ_p
  out[2p+1] = x[2p]*sin θ_p + x[2p+1]*cos θ_p
Broadcast over heads and batch; per-token θ from that token's coords.
SS-flow coords = meshgrid(16,16,16) row-major (ij), reshape [4096,3]. SLAT coords = voxel xyz.
GGML plan: precompute cos[128,L]/sin[128,L] host tensors (cos/sin repeated for both elems of a pair);
build swapped-neg r (r[2p]=-x[2p+1], r[2p+1]=x[2p]) via reshape-to-[2,64,..]+concat(neg(odd),even);
out = x*cos + r*sin.

## dtype
Reference runs torso in bf16 (manual_cast). We use f16 weights / f32 compute for norms, rms-norm,
layernorm, rope, softmax. Accept visual-equivalence; validate per-tensor with rtol~2e-2 vs f32 ref.
