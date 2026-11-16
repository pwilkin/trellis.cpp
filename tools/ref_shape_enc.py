#!/usr/bin/env python3
"""Pure-torch reference for the FlexiDualGrid shape VAE ENCODER (gather-based, no spconv, CPU).

Mirrors trellis2/models/sc_vaes/{sparse_unet_vae.py,fdg_vae.py} (SparseUnetVaeEncoder,
SparseResBlockS2C3d, SparseConvNeXtBlock3d) and modules/sparse/spatial/spatial2channel.py, in
float32. The reference runs its blocks in fp16; trellis-test-shape-enc compares against this f32
run with the decoder test's tolerance.

Inputs (one of):
    --sphere G        synthetic surface shell at grid G with random 6-channel features
    --dual FILE.bin   a dual grid written by `trellis-test-shape-enc dump <mesh> <grid> FILE.bin`
                      (i64 N, i32 grid; i32 coords[N*3]; f32 dual[N*3]; u8 intersected[N*3])
Outputs ($OUT, default ref/shape_enc/): coords0.npy [N,3], feats0.npy [N,6] (the encoder input),
grid.npy, latent_coords.npy [NL,3], latent.npy [NL,32] (posterior mean), stage{i}_coords.npy /
stage{i}.npy (after each stage, for bisecting).

    CK=<...>/ckpts/shape_enc_next_dc_f16c32_fp16 python tools/ref_shape_enc.py --sphere 256
"""
import argparse
import json
import os
import struct

import numpy as np
import torch
import torch.nn.functional as F

CH = [64, 128, 256, 512, 1024]
NB = [0, 4, 8, 16, 4]


def save(path, a):
    np.save(path, np.ascontiguousarray(a, dtype=np.float32))


def load_sd(ck):
    sd = {}
    with open(ck + ".safetensors", "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        base = 8 + n
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            a, b = v["data_offsets"]
            fh.seek(base + a)
            buf = fh.read(b - a)
            dt = {"F16": "<f2", "F32": "<f4", "BF16": None}[v["dtype"]]
            if dt is None:
                arr = (np.frombuffer(buf, "<u2").astype(np.uint32) << 16).view(np.float32)
            else:
                arr = np.frombuffer(buf, dt).astype(np.float32)
            sd[k] = torch.from_numpy(arr.reshape(v["shape"]).copy())
    return sd


def nbr_table(coords):
    """[N,27] neighbour rows (-1 absent), tap t = kd*9 + kh*3 + kw on offsets (x,y,z) - 1."""
    c = coords.long() + 1
    M = int(c.max()) + 2
    key = (c[:, 0] * M + c[:, 1]) * M + c[:, 2]
    order = torch.argsort(key)
    sk = key[order]
    out = torch.full((coords.shape[0], 27), -1, dtype=torch.long)
    for kd in range(3):
        for kh in range(3):
            for kw in range(3):
                q = ((c[:, 0] + kd - 1) * M + (c[:, 1] + kh - 1)) * M + (c[:, 2] + kw - 1)
                pos = torch.searchsorted(sk, q).clamp(max=len(sk) - 1)
                hit = sk[pos] == q
                out[:, kd * 9 + kh * 3 + kw] = torch.where(hit, order[pos], torch.full_like(q, -1))
    return out


def submconv(feats, nb, W, bias):
    Co, Ci = W.shape[0], W.shape[4]
    Wt = W.reshape(Co, 27, Ci)
    fez = torch.cat([feats, torch.zeros(1, Ci)], 0)
    out = bias.repeat(feats.shape[0], 1)
    for t in range(27):
        idx = nb[:, t].clone()
        idx[idx < 0] = feats.shape[0]
        out = out + fez[idx] @ Wt[:, t, :].T
    return out


def convnext(g, feats, nb, P):
    h = submconv(feats, nb, g[f"{P}.conv.weight"], g[f"{P}.conv.bias"])
    h = F.layer_norm(h, (h.shape[1],), g[f"{P}.norm.weight"], g[f"{P}.norm.bias"], eps=1e-6)
    h = F.silu(h @ g[f"{P}.mlp.0.weight"].T + g[f"{P}.mlp.0.bias"])
    h = h @ g[f"{P}.mlp.2.weight"].T + g[f"{P}.mlp.2.bias"]
    return h + feats


def s2c_pack(feats, coords):
    """SparseSpatial2Channel(2): parents lexicographic, slot = lx + 2ly + 4lz."""
    pc = torch.div(coords, 2, rounding_mode="floor")
    M = int(pc.max()) + 1
    code = (pc[:, 0].long() * M + pc[:, 1]) * M + pc[:, 2]
    code, idx = code.unique(return_inverse=True)
    newc = torch.stack([code // (M * M), (code // M) % M, code % M], 1).int()
    sub = (coords[:, 0] % 2) + 2 * (coords[:, 1] % 2) + 4 * (coords[:, 2] % 2)
    C = feats.shape[1]
    nf = torch.zeros(newc.shape[0] * 8, C)
    nf[idx * 8 + sub.long()] = feats
    return nf.reshape(newc.shape[0], 8 * C), newc


def s2c_block(g, feats, coords, P, Cin, Cout):
    nb = nbr_table(coords)
    h = F.silu(F.layer_norm(feats, (Cin,), g[f"{P}.norm1.weight"], g[f"{P}.norm1.bias"], eps=1e-6))
    h = submconv(h, nb, g[f"{P}.conv1.weight"], g[f"{P}.conv1.bias"])     # [N, Cout/8]
    h, newc = s2c_pack(h, coords)
    x, _ = s2c_pack(feats, coords)
    h = F.silu(F.layer_norm(h, (Cout,), None, None, eps=1e-6))
    h = submconv(h, nbr_table(newc), g[f"{P}.conv2.weight"], g[f"{P}.conv2.bias"])
    skip = x.reshape(x.shape[0], Cout, Cin * 8 // Cout).mean(-1)
    return h + skip, newc


def encode(g, feats6, coords, out):
    h = feats6 @ g["input_layer.weight"].T + g["input_layer.bias"]
    C = CH[0]
    for i in range(5):
        if NB[i]:
            nb = nbr_table(coords)
            for j in range(NB[i]):
                h = convnext(g, h, nb, f"blocks.{i}.{j}")
        if i < 4:
            h, coords = s2c_block(g, h, coords, f"blocks.{i}.{NB[i]}", C, CH[i + 1])
            C = CH[i + 1]
        save(f"{out}/stage{i}.npy", h.numpy())
        save(f"{out}/stage{i}_coords.npy", coords.numpy())
        print(f"stage {i}: C={C} N={coords.shape[0]}")
    h = F.layer_norm(h, (C,), eps=1e-5)
    lat = h @ g["to_latent.weight"].T + g["to_latent.bias"]
    return lat[:, :32], coords


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sphere", type=int, default=0)
    ap.add_argument("--dual", default="")
    ap.add_argument("--out", default=os.environ.get("OUT", "ref/shape_enc"))
    ap.add_argument("--ck", default=os.environ.get("CK", ""))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    if a.dual:
        raw = open(a.dual, "rb").read()
        N = struct.unpack_from("<q", raw, 0)[0]
        grid = struct.unpack_from("<i", raw, 8)[0]
        o = 12
        coords = torch.from_numpy(np.frombuffer(raw, "<i4", N * 3, o).reshape(N, 3).copy()); o += N * 12
        dual = torch.from_numpy(np.frombuffer(raw, "<f4", N * 3, o).reshape(N, 3).copy()); o += N * 12
        inter = torch.from_numpy(np.frombuffer(raw, "u1", N * 3, o).reshape(N, 3).copy()).float()
        feats6 = torch.cat([dual - 0.5, inter - 0.5], 1)
    else:
        grid = a.sphere or 256
        r = torch.arange(grid, dtype=torch.float32) + 0.5
        X, Y, Z = torch.meshgrid(r, r, r, indexing="ij")
        d = ((X - grid / 2) ** 2 + (Y - grid / 2 * 0.9) ** 2 + (Z - grid / 2 * 1.05) ** 2).sqrt()
        shell = (d - grid * 0.33).abs() < 0.9
        coords = torch.nonzero(shell).int()
        feats6 = torch.cat([torch.rand(coords.shape[0], 3) - 0.5, (torch.rand(coords.shape[0], 3) < 0.4).float() - 0.5], 1)
    g = load_sd(a.ck)
    print(f"input: N={coords.shape[0]} grid={grid}")
    save(f"{a.out}/coords0.npy", coords.numpy())
    save(f"{a.out}/feats0.npy", feats6.numpy())
    save(f"{a.out}/grid.npy", np.array([grid]))
    with torch.no_grad():
        lat, lc = encode(g, feats6, coords, a.out)
    save(f"{a.out}/latent.npy", lat.numpy())
    save(f"{a.out}/latent_coords.npy", lc.numpy())
    print(f"latent: N={lc.shape[0]} mean={lat.mean():.4f} std={lat.std():.4f} -> {a.out}")


if __name__ == "__main__":
    main()
