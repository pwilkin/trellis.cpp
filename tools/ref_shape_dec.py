#!/usr/bin/env python3
"""Pure-torch reference for the FULL FlexiDualGrid shape decoder (gather-based,
no spconv). Synthetic shape latent at res 32 -> dual-grid head [M,7] at res 512.
    python tools/ref_shape_dec.py
"""
import os, json, struct
import numpy as np, torch, torch.nn.functional as F

CK = "/media/ilintar/D_SSD/models/trellis2/ckpts/shape_dec_next_dc_f16c32_fp16"
OUT = "/media/ilintar/D_SSD/models/trellis2/ref/shape_dec"; os.makedirs(OUT, exist_ok=True)

SD = {}
with open(CK + ".safetensors", "rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n)); base = 8+n
    for k, v in hdr.items():
        if k == "__metadata__": continue
        a, b = v["data_offsets"]; fh.seek(base+a); buf = fh.read(b-a)
        arr = np.frombuffer(buf, "<f2").astype(np.float32) if v["dtype"]=="F16" else np.frombuffer(buf,"<f4")
        SD[k] = torch.from_numpy(arr.reshape(v["shape"]).copy())
g = lambda k: SD[k]

def nbr_table(coords):
    cmap={(int(c[0]),int(c[1]),int(c[2])):i for i,c in enumerate(coords)}
    nb=torch.full((len(coords),27),-1,dtype=torch.long)
    for i,c in enumerate(coords):
        x,y,z=int(c[0]),int(c[1]),int(c[2])
        for kd in range(3):
          for kh in range(3):
            for kw in range(3):
                j=cmap.get((x+kd-1,y+kh-1,z+kw-1))
                if j is not None: nb[i,kd*9+kh*3+kw]=j
    return nb

def submconv(feats, coords, W, bias):
    Co=W.shape[0]; Ci=W.shape[4]; Wt=W.reshape(Co,27,Ci); nb=nbr_table(coords)
    fez=torch.cat([feats,torch.zeros(1,Ci)],0); out=bias.clone().repeat(feats.shape[0],1)
    for t in range(27):
        idx=nb[:,t].clone(); idx[idx<0]=feats.shape[0]; out=out+fez[idx]@Wt[:,t,:].T
    return out

def convnext(feats, coords, P):
    h=submconv(feats,coords,g(f"{P}.conv.weight"),g(f"{P}.conv.bias"))
    h=F.layer_norm(h,(h.shape[1],),g(f"{P}.norm.weight"),g(f"{P}.norm.bias"),eps=1e-6)
    h=h@g(f"{P}.mlp.0.weight").T+g(f"{P}.mlp.0.bias"); h=F.silu(h)
    h=h@g(f"{P}.mlp.2.weight").T+g(f"{P}.mlp.2.bias")
    return h+feats

def c2s(feats, coords, P, Cin, Cout):
    subdiv=feats@g(f"{P}.to_subdiv.weight").T+g(f"{P}.to_subdiv.bias"); mask=subdiv>0
    h=F.silu(F.layer_norm(feats,(Cin,),g(f"{P}.norm1.weight"),g(f"{P}.norm1.bias"),eps=1e-6))
    h=submconv(h,coords,g(f"{P}.conv1.weight"),g(f"{P}.conv1.bias"))   # [N,Cout*8]
    K=Cin//8; nc=[]; hn=[]; xs=[]
    for i in range(feats.shape[0]):
        x,y,z=int(coords[i,0]),int(coords[i,1]),int(coords[i,2])
        for o in range(8):
            if mask[i,o]:
                nc.append((2*x+(o&1),2*y+((o>>1)&1),2*z+((o>>2)&1)))
                hn.append(h[i,o*Cout:(o+1)*Cout]); xs.append(feats[i,o*K:(o+1)*K])
    nc=torch.tensor(nc,dtype=torch.int64); hn=torch.stack(hn); xs=torch.stack(xs)
    hn=F.silu(F.layer_norm(hn,(Cout,),None,None,eps=1e-6))
    hn=submconv(hn,nc,g(f"{P}.conv2.weight"),g(f"{P}.conv2.bias"))
    skip=xs.repeat_interleave(Cout//K,dim=1)
    return hn+skip, nc

# synthetic input at res 32
R0,N0=32,500
rng=np.random.default_rng(9); seen=set(); rows=[]
while len(rows)<N0:
    p=tuple(int(x) for x in rng.integers(0,R0,3))
    if p not in seen: seen.add(p); rows.append(p)
coords=torch.tensor(rows,dtype=torch.int64)
latent=torch.randn(N0,32,generator=torch.Generator().manual_seed(9))

with torch.no_grad():
    h=latent@g("from_latent.weight").T+g("from_latent.bias")     # [N0,1024]
    stages=[(1024,4,512,"0",4),(512,16,256,"1",16),(256,8,128,"2",8),(128,4,64,"3",4)]
    for (C,nb,Cout,s,c2si) in stages:
        for j in range(nb): h=convnext(h,coords,f"blocks.{s}.{j}")
        h,coords=c2s(h,coords,f"blocks.{s}.{c2si}",C,Cout)
        print(f"after stage {s}: N={h.shape[0]} C={h.shape[1]}")
    h=F.layer_norm(h,(h.shape[1],))            # final no-affine LN
    h=h@g("output_layer.weight").T+g("output_layer.bias")        # [M,7]

np.save(f"{OUT}/coords.npy", coords.numpy().astype(np.float32))
np.save(f"{OUT}/latent.npy", latent.numpy().astype(np.float32))
np.save(f"{OUT}/coords0.npy", torch.tensor(rows).numpy().astype(np.float32))
np.save(f"{OUT}/out7.npy", h.numpy().astype(np.float32))
print(f"FINAL M={h.shape[0]} res=512  out7 std={h.std():.4f} -> {OUT}")
