#!/usr/bin/env python3
"""Pure-torch reference for SparseResBlockC2S3d (channel->spatial ×2 up-block).
Validates the C++ C2S: to_subdiv>0 octant subdivision, conv1 on old coords, C2S
feature redistribution, conv2 on new coords, skip repeat_interleave.
    python tools/ref_c2s.py
"""
import os, json, struct
import numpy as np, torch, torch.nn.functional as F

CK = "/media/ilintar/D_SSD/models/trellis2/ckpts/shape_dec_next_dc_f16c32_fp16"
OUT = "/media/ilintar/D_SSD/models/trellis2/ref/c2s"; os.makedirs(OUT, exist_ok=True)
P = "blocks.3.4"   # C2S 128 -> 64

def load(path, keys):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n)); base = 8+n
        o = {}
        for k in keys:
            v = hdr[k]; a, b = v["data_offsets"]; fh.seek(base+a); buf = fh.read(b-a)
            arr = np.frombuffer(buf, "<f2").astype(np.float32) if v["dtype"]=="F16" else np.frombuffer(buf,"<f4")
            o[k] = torch.from_numpy(arr.reshape(v["shape"]).copy())
    return o

w = load(CK+".safetensors", [f"{P}.conv1.weight", f"{P}.conv1.bias", f"{P}.conv2.weight", f"{P}.conv2.bias",
    f"{P}.norm1.weight", f"{P}.norm1.bias", f"{P}.to_subdiv.weight", f"{P}.to_subdiv.bias"])
Cin = w[f"{P}.norm1.weight"].shape[0]
Cout = w[f"{P}.conv2.weight"].shape[0]
print("C2S", Cin, "->", Cout)

R, N = 16, 600
rng = np.random.default_rng(5); seen=set(); rows=[]
while len(rows)<N:
    p=tuple(int(x) for x in rng.integers(0,R,3))
    if p not in seen: seen.add(p); rows.append(p)
coords = torch.tensor(rows, dtype=torch.int64)
feats = torch.randn(N, Cin, generator=torch.Generator().manual_seed(5))

def nbr_table(coords):
    cmap={(int(c[0]),int(c[1]),int(c[2])):i for i,c in enumerate(coords)}
    M=len(coords); nb=torch.full((M,27),-1,dtype=torch.long)
    for i,c in enumerate(coords):
        x,y,z=int(c[0]),int(c[1]),int(c[2])
        for kd in range(3):
          for kh in range(3):
            for kw in range(3):
                j=cmap.get((x+kd-1,y+kh-1,z+kw-1))
                if j is not None: nb[i,kd*9+kh*3+kw]=j
    return nb

def submconv(feats, coords, weight, bias):
    Co,_,_,_,Ci=weight.shape; Wt=weight.reshape(Co,27,Ci)
    nb=nbr_table(coords); fez=torch.cat([feats,torch.zeros(1,Ci)],0)
    out=bias.clone().repeat(feats.shape[0],1)
    for t in range(27):
        idx=nb[:,t].clone(); idx[idx<0]=feats.shape[0]
        out=out+fez[idx]@Wt[:,t,:].T
    return out

with torch.no_grad():
    subdiv = feats @ w[f"{P}.to_subdiv.weight"].T + w[f"{P}.to_subdiv.bias"]   # [N,8]
    mask = subdiv > 0
    h = F.silu(F.layer_norm(feats, (Cin,), w[f"{P}.norm1.weight"], w[f"{P}.norm1.bias"], eps=1e-6))
    h = submconv(h, coords, w[f"{P}.conv1.weight"], w[f"{P}.conv1.bias"])       # [N, Cout*8]
    # C2S
    new_coords=[]; gidx=[]; pidx=[]
    for i in range(N):
        for o in range(8):
            if mask[i,o]:
                x,y,z=int(coords[i,0]),int(coords[i,1]),int(coords[i,2])
                new_coords.append((2*x+(o&1), 2*y+((o>>1)&1), 2*z+((o>>2)&1)))
                gidx.append((i,o)); pidx.append(i)
    Mc=len(new_coords); new_coords_t=torch.tensor(new_coords,dtype=torch.int64)
    h_new=torch.stack([h[i, o*Cout:(o+1)*Cout] for (i,o) in gidx])              # [M,Cout]
    K=Cin//8
    xs=torch.stack([feats[i, o*K:(o+1)*K] for (i,o) in gidx])                   # [M,K]
    h_new=F.silu(F.layer_norm(h_new, (Cout,), None, None, eps=1e-6))            # norm2 no affine
    h_new=submconv(h_new, new_coords_t, w[f"{P}.conv2.weight"], w[f"{P}.conv2.bias"])
    skip=xs.repeat_interleave(Cout//K, dim=1)                                   # [M,Cout]
    out=h_new+skip

np.save(f"{OUT}/coords.npy", coords.numpy().astype(np.float32))
np.save(f"{OUT}/feats_in.npy", feats.numpy().astype(np.float32))
np.save(f"{OUT}/new_coords.npy", new_coords_t.numpy().astype(np.float32))
np.save(f"{OUT}/out.npy", out.numpy().astype(np.float32))
print(f"N={N} M={Mc} Cin={Cin} Cout={Cout} out std={out.std():.4f} -> {OUT}")
