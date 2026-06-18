#!/usr/bin/env python3
"""Pure-torch reference for submanifold SparseConv3d + SparseConvNeXtBlock3d.

No spconv/flex_gemm needed (gather-based, matching the documented convention:
weight [Co,Kd,Kh,Kw,Ci], tap (kd,kh,kw) reads neighbor p+(kd-1,kh-1,kw-1) on (x,y,z)).
Validates the C++ submanifold conv for correctness; convention vs flex_gemm is
confirmed from conv_spconv.py (standard correlation) + the final visual mesh.

    python tools/ref_sparse_conv.py
"""
import os, sys, json, struct
import numpy as np, torch, torch.nn.functional as F

CK = "/media/ilintar/D_SSD/models/trellis2/ckpts/shape_dec_next_dc_f16c32_fp16"
OUT = "/media/ilintar/D_SSD/models/trellis2/ref/sparse_conv"; os.makedirs(OUT, exist_ok=True)
BLK = "blocks.3.0"   # ConvNeXt at C=128

def load_st_f32(path, keys):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n)); base = 8 + n
        out = {}
        for k in keys:
            v = hdr[k]; o0, o1 = v["data_offsets"]; fh.seek(base + o0); buf = fh.read(o1 - o0)
            dt = v["dtype"]
            if dt == "F16": a = np.frombuffer(buf, "<f2").astype(np.float32)
            elif dt == "BF16": a = (np.frombuffer(buf, "<u2").astype(np.uint32) << 16).view(np.float32)
            else: a = np.frombuffer(buf, "<f4")
            out[k] = torch.from_numpy(a.reshape(v["shape"]).copy())
    return out

w = load_st_f32(CK + ".safetensors", [f"{BLK}.conv.weight", f"{BLK}.conv.bias",
    f"{BLK}.norm.weight", f"{BLK}.norm.bias", f"{BLK}.mlp.0.weight", f"{BLK}.mlp.0.bias",
    f"{BLK}.mlp.2.weight", f"{BLK}.mlp.2.bias"])
Co, Kd, Kh, Kw, Ci = w[f"{BLK}.conv.weight"].shape
print("conv weight", list(w[f"{BLK}.conv.weight"].shape), "C=", Ci)

# synthetic sparse input: N distinct voxels in [0,R)^3
R, N = 16, 800
rng = np.random.default_rng(3)
seen = set(); rows = []
while len(rows) < N:
    p = tuple(int(x) for x in rng.integers(0, R, 3))
    if p not in seen: seen.add(p); rows.append(p)
coords = torch.tensor(rows, dtype=torch.int32)        # [N,3] (x,y,z)
feats = torch.randn(N, Ci, generator=torch.Generator().manual_seed(3))

# neighbor table [N,27]
cmap = {(int(c[0]), int(c[1]), int(c[2])): i for i, c in enumerate(coords)}
nbr = torch.full((N, 27), -1, dtype=torch.long)
for i, c in enumerate(coords):
    x, y, z = int(c[0]), int(c[1]), int(c[2])
    for kd in range(3):
        for kh in range(3):
            for kw in range(3):
                t = kd*9 + kh*3 + kw
                j = cmap.get((x+kd-1, y+kh-1, z+kw-1))
                if j is not None: nbr[i, t] = j

def submconv(feats, nbr, weight, bias):           # weight [Co,3,3,3,Ci]
    Co, _, _, _, Ci = weight.shape
    Wt = weight.reshape(Co, 27, Ci)
    fez = torch.cat([feats, torch.zeros(1, Ci)], 0)   # row N = zero (missing nbr)
    out = bias.clone().repeat(feats.shape[0], 1)      # [N,Co]
    for t in range(27):
        idx = nbr[:, t].clone(); idx[idx < 0] = feats.shape[0]
        out = out + fez[idx] @ Wt[:, t, :].T
    return out

with torch.no_grad():
    conv_out = submconv(feats, nbr, w[f"{BLK}.conv.weight"], w[f"{BLK}.conv.bias"])
    h = F.layer_norm(conv_out, (Ci,), w[f"{BLK}.norm.weight"], w[f"{BLK}.norm.bias"], eps=1e-6)
    h = h @ w[f"{BLK}.mlp.0.weight"].T + w[f"{BLK}.mlp.0.bias"]
    h = F.silu(h)
    h = h @ w[f"{BLK}.mlp.2.weight"].T + w[f"{BLK}.mlp.2.bias"]
    block_out = h + conv_out                          # residual on conv output? NO -> see note

# NOTE residual in SparseConvNeXtBlock3d is `h + x` where x is the BLOCK INPUT (pre-conv).
# Recompute correctly:
with torch.no_grad():
    block_out = h + feats

np.save(f"{OUT}/coords.npy", coords.numpy().astype(np.float32))
np.save(f"{OUT}/feats_in.npy", feats.numpy().astype(np.float32))
np.save(f"{OUT}/conv_out.npy", conv_out.numpy().astype(np.float32))
np.save(f"{OUT}/block_out.npy", block_out.numpy().astype(np.float32))
print(f"N={N} Ci={Ci} conv_out std={conv_out.std():.4f} block_out std={block_out.std():.4f} -> {OUT}")
