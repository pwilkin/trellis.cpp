#!/usr/bin/env python3
"""Turn a rough guide mesh into a latent for `trellis-cli --ss-control`.

The shape guide starts TRELLIS.2's sparse-structure flow from the latent of a
guide shape instead of pure noise, so the generated object follows the guide's
volume (training-free blockout control, after SpaceControl). This script makes
that latent:

  1. load the mesh (.obj natively; .glb/.ply/... when `trimesh` is installed),
  2. centre it and scale its largest side to 0.95 of the [-0.5, 0.5] cube,
  3. mark the 64^3 voxels its surface passes through,
  4. encode them with the TRELLIS sparse-structure VAE encoder
     (ss_enc_conv3d_16l8_fp16 from microsoft/TRELLIS-image-large, MIT,
     exported to ONNX) -> 8 x 16^3 latent (posterior mean),
  5. write it as f32 [4096][8] (cell s = x*256 + y*16 + z, channel fastest).

The mesh is read in the same frame as the GLB trellis-cli writes: Y up, the
front of the object facing -Z (the way a generated object faces). Put the
guide's front where the generated object's front should be: a guide facing the
other way fights the image and turns the result around.

    pip install numpy onnxruntime        # (+ trimesh for non-OBJ meshes)
    python tools/ss_control_encode.py house.obj -o house.ssc
    trellis-cli -i house.png -o house.glb --ss-control house.ssc --ss-control-strength 0.6

The encoder (~237 MB) is downloaded on first use from
https://huggingface.co/fernandotonon/QtMeshEditor-trellis-ss-encoder-onnx
and checked against its SHA-256; pass --encoder to use a local copy.
"""
import argparse
import hashlib
import os
import sys
import urllib.request

import numpy as np

ENCODER_URL = ("https://huggingface.co/fernandotonon/QtMeshEditor-trellis-ss-encoder-onnx/"
               "resolve/main/ss_encoder.onnx")
ENCODER_SHA256 = "904d66260f36d4922b891d366abddfaa48f76adafbda80126137119f6e77e8c0"
RES = 64
EXTENT = 0.95


def load_mesh(path):
    """Vertices (N,3) float32 and triangles (M,3) int64."""
    if path.lower().endswith(".obj"):
        verts, faces = [], []
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                t = line.split()
                if not t:
                    continue
                if t[0] == "v" and len(t) >= 4:
                    verts.append([float(t[1]), float(t[2]), float(t[3])])
                elif t[0] == "f" and len(t) >= 4:
                    idx = []
                    for tok in t[1:]:
                        i = int(tok.split("/")[0])
                        idx.append(i - 1 if i > 0 else len(verts) + i)
                    for k in range(1, len(idx) - 1):   # fan-triangulate polygons
                        faces.append([idx[0], idx[k], idx[k + 1]])
        return np.asarray(verts, np.float32), np.asarray(faces, np.int64)
    try:
        import trimesh
    except ImportError:
        sys.exit(f"{path}: only .obj is read natively; `pip install trimesh` for other formats")
    m = trimesh.load(path, force="mesh")
    return np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int64)


def to_grid(verts):
    """GLB frame (Y up, front -Z) -> TRELLIS grid frame (Z up), centred and scaled."""
    lo, hi = verts.min(0), verts.max(0)
    side = float((hi - lo).max())
    if side <= 0:
        sys.exit("the guide mesh has no extent")
    p = (verts - 0.5 * (lo + hi)) * (EXTENT / side)
    # Measured with trellis-cli: a guide given as (X, Y, Z) -> grid (X, -Z, Y)
    # comes out facing the way the GLB's own generations do.
    return np.stack([p[:, 0], -p[:, 2], p[:, 1]], axis=1)


def voxelize_surface(v, f):
    """64^3 occupancy of every voxel a triangle passes through."""
    occ = np.zeros((RES, RES, RES), np.float32)
    h = 0.5 / RES   # sample spacing: half a voxel, so no crossed voxel is skipped
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    edge = np.maximum.reduce([np.linalg.norm(a - b, axis=1), np.linalg.norm(b - c, axis=1),
                              np.linalg.norm(c - a, axis=1)])
    steps = np.clip(np.ceil(edge / h), 1, 512).astype(int)
    for n in np.unique(steps):
        sel = steps == n
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = (i + j) <= n
        u = (i[keep] / n)[None, :, None]
        w = (j[keep] / n)[None, :, None]
        pts = u * a[sel][:, None] + w * b[sel][:, None] + (1 - u - w) * c[sel][:, None]
        idx = np.clip(np.floor((pts.reshape(-1, 3) + 0.5) * RES).astype(int), 0, RES - 1)
        occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    return occ


def encoder_path(given):
    if given:
        return given
    cache = os.path.join(os.path.expanduser("~"), ".cache", "trellis.cpp")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, "ss_encoder.onnx")
    if not os.path.exists(path):
        print(f"downloading the sparse-structure encoder (~237 MB) to {path}")
        urllib.request.urlretrieve(ENCODER_URL, path + ".part")
        os.replace(path + ".part", path)
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != ENCODER_SHA256:
        os.remove(path)
        sys.exit(f"{path}: SHA-256 mismatch (removed); run again to re-download")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mesh", help="guide mesh (.obj, or any trimesh format)")
    ap.add_argument("-o", "--out", required=True, help="latent file for --ss-control")
    ap.add_argument("--encoder", help="local ss_encoder.onnx (default: download + verify)")
    a = ap.parse_args()

    import onnxruntime as ort

    verts, faces = load_mesh(a.mesh)
    if len(faces) == 0:
        sys.exit(f"{a.mesh}: no triangles")
    occ = voxelize_surface(to_grid(verts), faces)
    sess = ort.InferenceSession(encoder_path(a.encoder), providers=["CPUExecutionProvider"])
    z = sess.run(None, {"occupancy": occ[None, None]})[0]          # [1, 8, 16, 16, 16]
    if z.shape != (1, 8, 16, 16, 16) or not np.isfinite(z).all():
        sys.exit(f"unexpected encoder output {z.shape}")
    latent = np.ascontiguousarray(z[0].reshape(8, 4096).T, np.float32)   # [4096][8]
    latent.tofile(a.out)
    print(f"{a.mesh}: {int(occ.sum())} surface voxels -> {a.out} (f32 [4096][8])")


if __name__ == "__main__":
    main()
