#!/usr/bin/env python3
"""Render a GLB from explicit multi-view cameras, and compare two sets of renders.

Cameras follow docs/spec/30-multiview.md: the viewer/glTF frame is X right, Y up, Z toward the
front camera; view (az, el) sits at d = (cos el sin az, sin el, cos el cos az), looks at the mesh
centre, and is orthographic. az=0 is the front (camera on +Z), az=90 the camera on +X, 180 the
back, 270 the left, el=+90 the top. Every view of one mesh uses the same framing (bbox centre and
the bbox diagonal), so silhouettes are comparable across angles.

Renders are RGBA PNGs with alpha = coverage, so trellis-cli keeps the matte (image_has_alpha ->
threshold path) and no BiRefNet is needed when they are fed back as views.

The rasteriser is a vectorised surface splatter: area-weighted points on every triangle, colour
from TEXCOORD_0 + the base-colour texture (PNG or EXT_texture_webp) or COLOR_0, z-buffered per
pixel, then a 3x3 hole fill. Lambert shading with a head light is optional (--shade).

    python tools/render_views.py render <glb> <outdir> [--az 0,90,180,270] [--el 0] [--size 768]
    python tools/render_views.py compare <dir_a> <dir_b> [--az ...] [--el ...]

compare reads <dir>/view_az<az>_el<el>.png from both directories, crops each silhouette to its
alpha bbox, rescales it to a common square (aspect kept), and reports per-view silhouette IoU and
the mean RGB error (0..255) over the pixels both cover.
"""
import argparse
import io
import json
import os
import struct
import sys

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- GLB

def _quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _node_matrix(n):
    if "matrix" in n:
        return np.array(n["matrix"], dtype=float).reshape(4, 4).T
    m = np.eye(4)
    if "scale" in n:
        m = np.diag(list(n["scale"]) + [1.0]) @ m
    if "rotation" in n:
        r = np.eye(4); r[:3, :3] = _quat_to_mat(n["rotation"]); m = r @ m
    if "translation" in n:
        t = np.eye(4); t[:3, 3] = n["translation"]; m = t @ m
    return m


def load_glb(path):
    d = open(path, "rb").read()
    if d[:4] != b"glTF":
        raise ValueError(f"{path}: not a GLB")
    jl = struct.unpack_from("<I", d, 12)[0]
    J = json.loads(d[20:20 + jl])
    p = 20 + jl
    BIN = b""
    if p + 8 <= len(d):
        bl = struct.unpack_from("<I", d, p)[0]
        BIN = d[p + 8:p + 8 + bl]

    def acc(i):
        a = J["accessors"][i]
        bv = J["bufferViews"][a["bufferView"]]
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        dt, sz = {5126: ("<f4", 4), 5125: ("<u4", 4), 5123: ("<u2", 2), 5121: ("u1", 1)}[a["componentType"]]
        nc = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
        stride = bv.get("byteStride", 0)
        if stride and stride != sz * nc:
            raw = np.frombuffer(BIN, dtype="u1", count=stride * (a["count"] - 1) + sz * nc, offset=off)
            rows = np.lib.stride_tricks.as_strided(raw, shape=(a["count"], sz * nc), strides=(stride, 1))
            arr = np.frombuffer(rows.copy().tobytes(), dtype=dt).reshape(a["count"], nc)
        else:
            arr = np.frombuffer(BIN, dtype=dt, count=a["count"] * nc, offset=off).reshape(a["count"], nc)
        if a.get("normalized") and dt in ("u1", "<u2"):
            arr = arr.astype(float) / (255.0 if dt == "u1" else 65535.0)
        return arr

    def texture_of(mat_idx):
        if mat_idx is None or "materials" not in J:
            return None, (1.0, 1.0, 1.0)
        pbr = J["materials"][mat_idx].get("pbrMetallicRoughness", {})
        factor = tuple(pbr.get("baseColorFactor", [1, 1, 1, 1])[:3])
        bct = pbr.get("baseColorTexture")
        if bct is None:
            return None, factor
        t = J["textures"][bct["index"]]
        src = t.get("source")
        if src is None:
            src = t.get("extensions", {}).get("EXT_texture_webp", {}).get("source")
        if src is None:
            return None, factor
        img = J["images"][src]
        bv = J["bufferViews"][img["bufferView"]]
        o = bv.get("byteOffset", 0)
        tex = np.asarray(Image.open(io.BytesIO(BIN[o:o + bv["byteLength"]])).convert("RGB"), dtype=np.float32)
        return tex, factor

    parts = []

    def visit(ni, parent):
        n = J["nodes"][ni]
        m = parent @ _node_matrix(n)
        if "mesh" in n:
            for prim in J["meshes"][n["mesh"]]["primitives"]:
                if prim.get("mode", 4) != 4:
                    continue
                P = acc(prim["attributes"]["POSITION"]).astype(np.float64)
                P = P @ m[:3, :3].T + m[:3, 3]
                I = acc(prim["indices"]).reshape(-1, 3).astype(np.int64) if "indices" in prim \
                    else np.arange(len(P)).reshape(-1, 3)
                UV = acc(prim["attributes"]["TEXCOORD_0"]).astype(np.float64) if "TEXCOORD_0" in prim["attributes"] else None
                COL = acc(prim["attributes"]["COLOR_0"])[:, :3].astype(np.float64) if "COLOR_0" in prim["attributes"] else None
                tex, factor = texture_of(prim.get("material"))
                parts.append((P, I, UV, COL, tex, factor))
        for c in n.get("children", []):
            visit(c, m)

    scene = J.get("scenes", [{"nodes": list(range(len(J.get("nodes", []))))}])[J.get("scene", 0)]
    for ni in scene["nodes"]:
        visit(ni, np.eye(4))
    if not parts:
        raise ValueError(f"{path}: no triangle primitives")
    return parts


# --------------------------------------------------------------------------- render

def camera(az, el):
    th, ph = np.radians(az), np.radians(el)
    d = np.array([np.cos(ph) * np.sin(th), np.sin(ph), np.cos(ph) * np.cos(th)])
    r = np.array([np.cos(th), 0.0, -np.sin(th)])        # screen right
    u = np.cross(d, r)                                   # screen up
    return d, r, u


def sample_surface(parts, density, rng):
    """Area-weighted points with colour. density = points per unit area (world)."""
    pts, cols = [], []
    for P, I, UV, COL, tex, factor in parts:
        A, B, C = P[I[:, 0]], P[I[:, 1]], P[I[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
        n = np.maximum(1, np.round(area * density).astype(np.int64))
        fi = np.repeat(np.arange(len(I)), n)
        r1 = np.sqrt(rng.random(len(fi)))
        r2 = rng.random(len(fi))
        w0, w1, w2 = 1 - r1, r1 * (1 - r2), r1 * r2
        p = w0[:, None] * A[fi] + w1[:, None] * B[fi] + w2[:, None] * C[fi]
        if tex is not None and UV is not None:
            uv = w0[:, None] * UV[I[fi, 0]] + w1[:, None] * UV[I[fi, 1]] + w2[:, None] * UV[I[fi, 2]]
            th, tw = tex.shape[:2]
            tx = np.clip((uv[:, 0] % 1.0) * tw, 0, tw - 1).astype(np.int64)
            ty = np.clip((uv[:, 1] % 1.0) * th, 0, th - 1).astype(np.int64)
            c = tex[ty, tx] * np.array(factor)
        elif COL is not None:
            c = (w0[:, None] * COL[I[fi, 0]] + w1[:, None] * COL[I[fi, 1]] + w2[:, None] * COL[I[fi, 2]])
            c = c * 255.0 if c.max() <= 1.0 + 1e-6 else c
        else:
            c = np.tile(np.array(factor) * 200.0, (len(fi), 1))
        nrm = np.cross(B - A, C - A)[fi]
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        pts.append(np.concatenate([p, nrm], axis=1))
        cols.append(c)
    return np.concatenate(pts), np.concatenate(cols)


def render(parts, views, size, shade=False, seed=0):
    allP = np.concatenate([p[0] for p in parts])
    lo, hi = allP.min(0), allP.max(0)
    centre = (lo + hi) / 2
    span = float(np.linalg.norm(hi - lo)) * 1.02          # same framing for every view
    px = span / size
    rng = np.random.default_rng(seed)
    pts, cols = sample_surface(parts, density=4.0 / (px * px), rng=rng)
    P, N = pts[:, :3] - centre, pts[:, 3:]
    out = []
    for az, el in views:
        d, r, u = camera(az, el)
        sx = ((P @ r) / span + 0.5) * size
        sy = (0.5 - (P @ u) / span) * size
        depth = P @ d                                     # larger = nearer the camera
        X, Y = np.floor(sx).astype(np.int64), np.floor(sy).astype(np.int64)
        ok = (X >= 0) & (X < size) & (Y >= 0) & (Y < size)
        X, Y, Z, C = X[ok], Y[ok], depth[ok], cols[ok]
        if shade:
            C = C * (0.35 + 0.65 * np.abs(N[ok] @ d))[:, None]
        pix = Y * size + X
        order = np.lexsort((Z, pix))                      # per pixel, nearest last
        pix_s = pix[order]
        last = np.r_[pix_s[1:] != pix_s[:-1], True]
        sel = order[last]
        rgb = np.zeros((size * size, 3), np.float32)
        a = np.zeros(size * size, np.uint8)
        rgb[pix[sel]] = C[sel]
        a[pix[sel]] = 255
        rgb, a = rgb.reshape(size, size, 3), a.reshape(size, size)
        # 3x3 hole fill: an uncovered pixel with >= 6 covered neighbours takes their mean.
        cov = (a > 0).astype(np.float32)
        pad_c = np.pad(cov, 1)
        pad_rgb = np.pad(rgb * cov[..., None], ((1, 1), (1, 1), (0, 0)))
        nb = sum(pad_c[1 + dy:1 + dy + size, 1 + dx:1 + dx + size] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        nrgb = sum(pad_rgb[1 + dy:1 + dy + size, 1 + dx:1 + dx + size] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        fill = (a == 0) & (nb >= 6)
        rgb[fill] = nrgb[fill] / nb[fill][:, None]
        a[fill] = 255
        img = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), a])
        out.append(img)
    return out


def view_name(az, el):
    return f"view_az{az:g}_el{el:g}.png"


# --------------------------------------------------------------------------- compare

def _norm_crop(img, size=256):
    a = img[..., 3] > 127
    ys, xs = np.nonzero(a)
    if len(xs) == 0:
        return np.zeros((size, size), bool), np.zeros((size, size, 3), np.float32)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h, w = y1 - y0, x1 - x0
    s = max(h, w)
    canvas = np.zeros((s, s, 4), np.uint8)
    oy, ox = (s - h) // 2, (s - w) // 2
    canvas[oy:oy + h, ox:ox + w] = img[y0:y1, x0:x1]
    r = np.asarray(Image.fromarray(canvas).resize((size, size), Image.NEAREST))
    return r[..., 3] > 127, r[..., :3].astype(np.float32)


def compare(dir_a, dir_b, views):
    rows = []
    for az, el in views:
        fa, fb = os.path.join(dir_a, view_name(az, el)), os.path.join(dir_b, view_name(az, el))
        A = np.asarray(Image.open(fa).convert("RGBA"))
        B = np.asarray(Image.open(fb).convert("RGBA"))
        ma, ca = _norm_crop(A)
        mb, cb = _norm_crop(B)
        inter, union = (ma & mb).sum(), (ma | mb).sum()
        iou = inter / max(1, union)
        both = ma & mb
        cerr = float(np.abs(ca[both] - cb[both]).mean()) if both.any() else float("nan")
        rows.append((az, el, iou, cerr))
        print(f"az {az:6g} el {el:5g}: IoU {iou:.4f}  colour err {cerr:6.2f}")
    m_iou = float(np.mean([r[2] for r in rows]))
    m_col = float(np.nanmean([r[3] for r in rows]))
    print(f"mean: IoU {m_iou:.4f}  colour err {m_col:6.2f}")
    return rows


def parse_list(s):
    return [float(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("glb"); r.add_argument("outdir")
    r.add_argument("--az", default="0,90,180,270"); r.add_argument("--el", default="0")
    r.add_argument("--size", type=int, default=768)
    r.add_argument("--shade", action="store_true", help="Lambert shading with a head light")
    c = sub.add_parser("compare")
    c.add_argument("dir_a"); c.add_argument("dir_b")
    c.add_argument("--az", default="0,90,180,270"); c.add_argument("--el", default="0")
    a = ap.parse_args()
    az, el = parse_list(a.az), parse_list(a.el)
    if len(el) == 1:
        el = el * len(az)
    if len(el) != len(az):
        print("--el must have one entry or one per azimuth", file=sys.stderr)
        return 2
    views = list(zip(az, el))
    if a.cmd == "render":
        parts = load_glb(a.glb)
        os.makedirs(a.outdir, exist_ok=True)
        for (vaz, vel), img in zip(views, render(parts, views, a.size, a.shade)):
            p = os.path.join(a.outdir, view_name(vaz, vel))
            Image.fromarray(img, "RGBA").save(p)
            print(f"saved {p}  coverage {(img[..., 3] > 0).mean():.3f}")
        return 0
    compare(a.dir_a, a.dir_b, views)
    return 0


if __name__ == "__main__":
    sys.exit(main())
