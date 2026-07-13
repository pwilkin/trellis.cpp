#!/usr/bin/env python3
"""CPU-only GLB mesh/UV/material metrics (numpy required, PIL optional).
Usage: python3 tools/glb_metrics.py <a.glb> [b.glb ...]"""
import base64
import io
import json
import struct
import sys

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

COMP_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5124: np.int32,
    5125: np.uint32,
    5126: np.float32,
}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
EPS_UV = 1e-6


def parse_glb(path):
    d = open(path, "rb").read()
    if d[:4] != b"glTF":
        raise ValueError("not a GLB file")
    n = len(d)
    p = 12
    gltf = None
    bin_data = b""
    while p + 8 <= n:
        clen, ctype = struct.unpack_from("<II", d, p)
        body = d[p + 8 : p + 8 + clen]
        if ctype == 0x4E4F534A:
            gltf = json.loads(body)
        elif ctype == 0x004E4942:
            bin_data = body
        p += 8 + clen + (-clen % 4)
    if gltf is None:
        raise ValueError("no JSON chunk")
    return gltf, bin_data


def read_accessor(gltf, bin_data, idx):
    a = gltf["accessors"][idx]
    ncomp = NCOMP.get(a["type"])
    dt = COMP_DTYPE.get(a["componentType"])
    if ncomp is None or dt is None:
        return None
    count = a["count"]
    itemsize = np.dtype(dt).itemsize
    tight = ncomp * itemsize
    if "bufferView" in a:
        bv = gltf["bufferViews"][a["bufferView"]]
        base = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = bv.get("byteStride", tight)
        if stride == tight:
            arr = np.frombuffer(bin_data, dtype=dt, count=count * ncomp, offset=base).reshape(count, ncomp)
        else:
            need = (count - 1) * stride + tight if count else 0
            raw = np.frombuffer(bin_data, dtype=np.uint8, count=need, offset=base)
            arr = np.lib.stride_tricks.as_strided(raw, shape=(count, tight), strides=(stride, 1))
            arr = np.ascontiguousarray(arr).view(dt).reshape(count, ncomp)
    else:
        arr = np.zeros((count, ncomp), dtype=dt)
    if "sparse" in a:
        try:
            sp = a["sparse"]
            sidt = COMP_DTYPE[sp["indices"]["componentType"]]
            sbv = gltf["bufferViews"][sp["indices"]["bufferView"]]
            soff = sbv.get("byteOffset", 0) + sp["indices"].get("byteOffset", 0)
            sidx = np.frombuffer(bin_data, dtype=sidt, count=sp["count"], offset=soff)
            vbv = gltf["bufferViews"][sp["values"]["bufferView"]]
            voff = vbv.get("byteOffset", 0) + sp["values"].get("byteOffset", 0)
            vals = np.frombuffer(bin_data, dtype=dt, count=sp["count"] * ncomp, offset=voff).reshape(sp["count"], ncomp)
            arr = arr.copy()
            arr[sidx.astype(np.int64)] = vals
        except Exception:
            pass
    if a.get("normalized") and np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        arr = np.maximum(arr.astype(np.float64) / info.max, -1.0)
    return arr


def quat_to_mat(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def node_local_matrix(n):
    if "matrix" in n:
        return np.array(n["matrix"], dtype=np.float64).reshape(4, 4).T
    M = np.eye(4)
    R = np.eye(4)
    S = np.eye(4)
    if "rotation" in n:
        R[:3, :3] = quat_to_mat(n["rotation"])
    if "scale" in n:
        S[:3, :3] = np.diag(n["scale"])
    M[:3, :3] = (R @ S)[:3, :3]
    if "translation" in n:
        M[:3, 3] = n["translation"]
    return M


def mesh_instances(gltf):
    nodes = gltf.get("nodes", [])
    inst = {}
    children = set()
    for n in nodes:
        children.update(n.get("children", []))
    scenes = gltf.get("scenes", [])
    if scenes:
        roots = []
        for s in scenes:
            roots.extend(s.get("nodes", []))
    else:
        roots = [i for i in range(len(nodes)) if i not in children]
    stack = [(r, np.eye(4)) for r in roots]
    while stack:
        ni, M = stack.pop()
        if ni >= len(nodes):
            continue
        n = nodes[ni]
        W = M @ node_local_matrix(n)
        if "mesh" in n:
            inst.setdefault(n["mesh"], []).append(W)
        for c in n.get("children", []):
            stack.append((c, W))
    for mi in range(len(gltf.get("meshes", []))):
        if mi not in inst:
            inst[mi] = [np.eye(4)]
    return inst


def gather_geometry(gltf, bin_data, notes):
    P_list, UV_list, F_list = [], [], []
    voff = 0
    inst = mesh_instances(gltf)
    n_prims = 0
    for mi, mesh in enumerate(gltf.get("meshes", [])):
        for prim in mesh.get("primitives", []):
            if prim.get("mode", 4) != 4:
                notes.append("skipped non-triangle primitive (mode %d)" % prim.get("mode", 4))
                continue
            attrs = prim.get("attributes", {})
            if "POSITION" not in attrs:
                notes.append("primitive without POSITION skipped")
                continue
            try:
                P = read_accessor(gltf, bin_data, attrs["POSITION"])
            except Exception as e:
                notes.append("POSITION accessor unreadable (%s)" % e)
                continue
            if P is None or P.shape[1] < 3:
                notes.append("POSITION accessor unsupported, primitive skipped")
                continue
            P = P[:, :3].astype(np.float64)
            UV = None
            if "TEXCOORD_0" in attrs:
                try:
                    UV = read_accessor(gltf, bin_data, attrs["TEXCOORD_0"])
                    if UV is not None:
                        UV = UV[:, :2].astype(np.float64)
                except Exception:
                    UV = None
            if UV is None:
                UV = np.full((len(P), 2), np.nan)
            if "indices" in prim:
                try:
                    I = read_accessor(gltf, bin_data, prim["indices"])
                except Exception as e:
                    notes.append("index accessor unreadable (%s)" % e)
                    continue
                if I is None:
                    notes.append("index accessor unsupported, primitive skipped")
                    continue
                I = I.reshape(-1).astype(np.int64)
                I = I[: (len(I) // 3) * 3].reshape(-1, 3)
            else:
                I = np.arange((len(P) // 3) * 3, dtype=np.int64).reshape(-1, 3)
            for W in inst.get(mi, [np.eye(4)]):
                Pw = P @ W[:3, :3].T + W[:3, 3]
                P_list.append(Pw)
                UV_list.append(UV)
                F_list.append(I + voff)
                voff += len(P)
            n_prims += 1
    if not P_list:
        return np.zeros((0, 3)), np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int64), 0
    return np.vstack(P_list), np.vstack(UV_list), np.vstack(F_list), n_prims


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def edge_stats(F, nv):
    D = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    lo = np.minimum(D[:, 0], D[:, 1])
    hi = np.maximum(D[:, 0], D[:, 1])
    key = lo * np.int64(nv) + hi
    fwd = D[:, 0] == lo
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    counts = np.diff(np.r_[starts, len(ks)])
    fsum = np.add.reduceat(fwd[order].astype(np.int64), starts) if len(ks) else np.zeros(0, dtype=np.int64)
    boundary = int((counts == 1).sum())
    nonmanifold = int((counts > 2).sum())
    m2 = counts == 2
    n2 = int(m2.sum())
    consistent = int(((fsum == 1) & m2).sum())
    return boundary, nonmanifold, n2, consistent, (order, starts, counts)


def geometry_metrics(P, F):
    m = {}
    nv, nf = len(P), len(F)
    m["vertices"] = nv
    m["faces"] = nf
    if nf == 0 or nv == 0:
        return m
    uf = UF(nv)
    union = uf.union
    for a, b, c in F:
        union(int(a), int(b))
        union(int(b), int(c))
    find = uf.find
    roots = np.array([find(int(v)) for v in F[:, 0]], dtype=np.int64)
    _, sizes = np.unique(roots, return_counts=True)
    m["components"] = len(sizes)
    m["top5"] = sorted(sizes.tolist(), reverse=True)[:5]

    b, nm, n2, cons, _ = edge_stats(F, nv)
    m["boundary_edges"] = b
    m["nonmanifold_edges"] = nm
    m["mult2_edges"] = n2
    m["winding_pct"] = 100.0 * cons / n2 if n2 else float("nan")
    m["watertight"] = b == 0 and nm == 0

    P32 = np.ascontiguousarray(P.astype(np.float32))
    _, W = np.unique(P32.view([("", np.float32)] * 3).ravel(), return_inverse=True)
    m["welded_vertices"] = int(W.max()) + 1 if len(W) else 0
    FW = W[F].astype(np.int64)
    wb, wnm, _, _, _ = edge_stats(FW, m["welded_vertices"])
    m["welded_boundary"] = wb
    m["welded_nonmanifold"] = wnm
    m["welded_watertight"] = wb == 0 and wnm == 0
    m["_W"] = W

    m["bbox_min"] = P.min(0)
    m["bbox_max"] = P.max(0)
    C = P - P.mean(0)
    cov = C.T @ C / nv
    _, vecs = np.linalg.eigh(cov)
    proj = C @ vecs
    ext = np.sort(proj.max(0) - proj.min(0))[::-1]
    m["principal_extents"] = ext
    return m


def tri_area2d(uv):
    e1 = uv[:, 1] - uv[:, 0]
    e2 = uv[:, 2] - uv[:, 0]
    return 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])


def tri_area3d(p):
    e1 = p[:, 1] - p[:, 0]
    e2 = p[:, 2] - p[:, 0]
    return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)


def chart_count(F, UV, W):
    valid = ~np.isnan(UV[F].reshape(len(F), -1)).any(1)
    Fv = F[valid]
    fids = np.flatnonzero(valid)
    if len(Fv) == 0:
        return None
    corners = [(0, 1), (1, 2), (2, 0)]
    keys, eface, uvlo, uvhi = [], [], [], []
    nvw = int(W.max()) + 1
    for i, j in corners:
        a = Fv[:, i]
        b = Fv[:, j]
        wa = W[a].astype(np.int64)
        wb = W[b].astype(np.int64)
        swap = wa > wb
        lo = np.where(swap, wb, wa)
        hi = np.where(swap, wa, wb)
        ua = UV[a]
        ub = UV[b]
        keys.append(lo * nvw + hi)
        eface.append(np.arange(len(Fv), dtype=np.int64))
        uvlo.append(np.where(swap[:, None], ub, ua))
        uvhi.append(np.where(swap[:, None], ua, ub))
    key = np.concatenate(keys)
    eface = np.concatenate(eface)
    uvlo = np.vstack(uvlo)
    uvhi = np.vstack(uvhi)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    counts = np.diff(np.r_[starts, len(ks)])
    uf = UF(len(Fv))
    two = counts == 2
    i0 = starts[two]
    e0 = order[i0]
    e1 = order[i0 + 1]
    match = (np.abs(uvlo[e0] - uvlo[e1]).max(1) < EPS_UV) & (np.abs(uvhi[e0] - uvhi[e1]).max(1) < EPS_UV)
    for a, b in zip(eface[e0[match]], eface[e1[match]]):
        uf.union(int(a), int(b))
    for s, c in zip(starts[counts > 2], counts[counts > 2]):
        grp = order[s : s + min(int(c), 12)]
        for x in range(len(grp)):
            for y in range(x + 1, len(grp)):
                ex, ey = grp[x], grp[y]
                if np.abs(uvlo[ex] - uvlo[ey]).max() < EPS_UV and np.abs(uvhi[ex] - uvhi[ey]).max() < EPS_UV:
                    uf.union(int(eface[ex]), int(eface[ey]))
    return len({uf.find(i) for i in range(len(Fv))})


def uv_metrics(P, UV, F, atlas_w, W):
    m = {}
    has_uv = ~np.isnan(UV).any(1)
    if not has_uv.any():
        m["has_uv"] = False
        return m
    m["has_uv"] = True
    m["uv_min"] = np.nanmin(UV, axis=0)
    m["uv_max"] = np.nanmax(UV, axis=0)
    m["charts"] = chart_count(F, UV, W)
    valid = has_uv[F].all(1)
    Fv = F[valid]
    if len(Fv) == 0:
        return m
    uv_a = tri_area2d(UV[Fv])
    w_a = tri_area3d(P[Fv])
    zero_uv = uv_a < 1e-12
    m["zero_uv_pct"] = 100.0 * zero_uv.sum() / len(Fv)
    T = atlas_w if atlas_w else 1
    m["density_T"] = T
    ok = (w_a > 1e-12) & ~zero_uv
    if ok.any():
        dens = uv_a[ok] * (T * T) / w_a[ok]
        m["density"] = {
            "mean": float(dens.mean()),
            "p5": float(np.percentile(dens, 5)),
            "p50": float(np.percentile(dens, 50)),
            "p95": float(np.percentile(dens, 95)),
            "cv": float(dens.std() / dens.mean()) if dens.mean() else float("nan"),
        }
    return m


def image_bytes(gltf, bin_data, img):
    if "bufferView" in img:
        bv = gltf["bufferViews"][img["bufferView"]]
        o = bv.get("byteOffset", 0)
        return bin_data[o : o + bv["byteLength"]]
    uri = img.get("uri", "")
    if uri.startswith("data:"):
        try:
            return base64.b64decode(uri.split(",", 1)[1])
        except Exception:
            return None
    return None


def dims_from_header(b):
    if b is None or len(b) < 30:
        return None, None, None
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", b[16:24])
        return "PNG", w, h
    if b[:2] == b"\xff\xd8":
        i = 2
        while i + 9 < len(b):
            if b[i] != 0xFF:
                i += 1
                continue
            marker = b[i + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h, w = struct.unpack(">HH", b[i + 5 : i + 9])
                return "JPEG", w, h
            if marker == 0x01 or 0xD0 <= marker <= 0xD9:
                i += 2
                continue
            i += 2 + struct.unpack(">H", b[i + 2 : i + 4])[0]
        return "JPEG", None, None
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        cc = b[12:16]
        if cc == b"VP8X":
            return "WebP", 1 + int.from_bytes(b[24:27], "little"), 1 + int.from_bytes(b[27:30], "little")
        if cc == b"VP8 " and b[23:26] == b"\x9d\x01\x2a":
            w = struct.unpack("<H", b[26:28])[0] & 0x3FFF
            h = struct.unpack("<H", b[28:30])[0] & 0x3FFF
            return "WebP", w, h
        if cc == b"VP8L":
            bits = int.from_bytes(b[21:25], "little")
            return "WebP", (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        return "WebP", None, None
    return "unknown", None, None


def decode_image(b):
    if Image is None or b is None:
        return None
    try:
        img = Image.open(io.BytesIO(b))
        img.load()
        return img
    except Exception:
        return None


def image_info(gltf, bin_data, image_idx):
    imgs = gltf.get("images", [])
    if image_idx is None or image_idx >= len(imgs):
        return {"format": "n/a", "mime": "n/a", "w": None, "h": None, "bytes": None}
    img = imgs[image_idx]
    b = image_bytes(gltf, bin_data, img)
    fmt, w, h = dims_from_header(b) if b else (None, None, None)
    pil = decode_image(b)
    if pil is not None:
        w, h = pil.size
    return {
        "format": fmt or ("external" if "uri" in img else "n/a"),
        "mime": img.get("mimeType", "n/a"),
        "w": w,
        "h": h,
        "bytes": b,
        "pil": pil,
        "uri": img.get("uri"),
    }


def mr_channel_stats(pil):
    if pil is None:
        return None
    try:
        im = pil.convert("RGB")
        im.thumbnail((64, 64))
        a = np.asarray(im, dtype=np.float64)
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        return {
            "r_const": bool(r.max() - r.min() <= 2),
            "r_mean": float(r.mean()),
            "g_var": float(g.var()),
            "b_var": float(b.var()),
        }
    except Exception:
        return None


def tex_source(tex):
    webp = tex.get("extensions", {}).get("EXT_texture_webp")
    if webp is not None and "source" in webp:
        return webp["source"]
    return tex.get("source")


def material_metrics(gltf, bin_data):
    out = {"materials": [], "textures": []}
    textures = gltf.get("textures", [])
    for ti, tex in enumerate(textures):
        src = tex_source(tex)
        info = image_info(gltf, bin_data, src)
        out["textures"].append((ti, info))
    for mi, mat in enumerate(gltf.get("materials", [])):
        pbr = mat.get("pbrMetallicRoughness", {})
        md = {
            "index": mi,
            "name": mat.get("name", ""),
            "doubleSided": mat.get("doubleSided", False),
            "metallicFactor": pbr.get("metallicFactor", 1.0),
            "roughnessFactor": pbr.get("roughnessFactor", 1.0),
            "baseColorTexture": pbr.get("baseColorTexture", {}).get("index"),
            "mrTexture": pbr.get("metallicRoughnessTexture", {}).get("index"),
            "normalTexture": mat.get("normalTexture", {}).get("index"),
        }
        out["materials"].append(md)
    return out


def tex_image_info(gltf, bin_data, mats, tex_index):
    textures = gltf.get("textures", [])
    if tex_index is None or tex_index >= len(textures):
        return None
    return image_info(gltf, bin_data, tex_source(textures[tex_index]))


def f3(v):
    return "[" + ", ".join("%.4g" % x for x in v) + "]"


def analyze(path):
    r = {"path": path, "notes": []}
    gltf, bin_data = parse_glb(path)
    P, UV, F, n_prims = gather_geometry(gltf, bin_data, r["notes"])
    r["n_prims"] = n_prims
    r["n_meshes"] = len(gltf.get("meshes", []))
    r["geo"] = geometry_metrics(P, F)
    mats = material_metrics(gltf, bin_data)
    r["mat"] = mats
    base_info = None
    mr_info = None
    for md in mats["materials"]:
        if base_info is None and md["baseColorTexture"] is not None:
            base_info = tex_image_info(gltf, bin_data, mats, md["baseColorTexture"])
        if mr_info is None and md["mrTexture"] is not None:
            mr_info = tex_image_info(gltf, bin_data, mats, md["mrTexture"])
    if base_info is None and mats["textures"]:
        base_info = mats["textures"][0][1]
        r["notes"].append("no baseColorTexture in materials; using texture[0] as atlas")
    r["atlas"] = base_info
    atlas_w = base_info["w"] if base_info else None
    W = r["geo"].get("_W")
    if W is None:
        W = np.arange(len(P))
    r["uv"] = uv_metrics(P, UV, F, atlas_w, W)
    r["mr_stats"] = mr_channel_stats(mr_info.get("pil")) if mr_info else None
    r["mr_info"] = mr_info
    if Image is None:
        r["notes"].append("PIL not available: texture decode skipped, dims from headers only")
    return r


def print_report(r):
    g = r["geo"]
    uv = r["uv"]
    print("=" * 72)
    print(r["path"])
    print("=" * 72)
    print("meshes: %d  triangle primitives: %d" % (r["n_meshes"], r["n_prims"]))
    print("GEOMETRY")
    print("  vertices: %d   faces: %d" % (g.get("vertices", 0), g.get("faces", 0)))
    if g.get("faces"):
        print("  components (index-based): %d   top-5 sizes (faces): %s" % (g["components"], g["top5"]))
        print("  boundary edges: %d   non-manifold edges: %d" % (g["boundary_edges"], g["nonmanifold_edges"]))
        print(
            "  winding consistency: %s (over %d mult-2 edges)"
            % ("%.2f%%" % g["winding_pct"] if g["mult2_edges"] else "n/a", g["mult2_edges"])
        )
        print("  watertight (index-based): %s" % ("yes" if g["watertight"] else "no"))
        print(
            "  welded (by position): %d verts, boundary %d, non-manifold %d, watertight %s"
            % (
                g["welded_vertices"],
                g["welded_boundary"],
                g["welded_nonmanifold"],
                "yes" if g["welded_watertight"] else "no",
            )
        )
        print("  bbox min: %s  max: %s" % (f3(g["bbox_min"]), f3(g["bbox_max"])))
        print("  principal extents: %s" % f3(g["principal_extents"]))
    else:
        print("  no triangle geometry: n/a")
    print("UV/ATLAS")
    atlas = r["atlas"]
    if atlas:
        dims = "%sx%s" % (atlas["w"], atlas["h"]) if atlas["w"] else "unknown dims"
        print("  atlas (baseColor): %s  format=%s  mime=%s" % (dims, atlas["format"], atlas["mime"]))
        if atlas["bytes"] is not None and atlas.get("pil") is None and Image is not None:
            print("  atlas decode: failed (format unsupported by PIL); dims from header bytes")
    else:
        print("  atlas: n/a (no texture)")
    if uv.get("has_uv"):
        print("  uv bbox: %s .. %s" % (f3(uv["uv_min"]), f3(uv["uv_max"])))
        print("  chart estimate (uv-space components): %s" % (uv["charts"] if uv["charts"] is not None else "n/a"))
        d = uv.get("density")
        if d:
            unit = "texels^2/unit^2 (T=%d)" % uv["density_T"]
            print(
                "  texel density %s: mean %.4g  p5 %.4g  p50 %.4g  p95 %.4g  CV %.3f"
                % (unit, d["mean"], d["p5"], d["p50"], d["p95"], d["cv"])
            )
        else:
            print("  texel density: n/a")
        print("  zero-UV-area faces: %.2f%%" % uv.get("zero_uv_pct", float("nan")))
    else:
        print("  UVs: n/a (no TEXCOORD_0)")
    print("MATERIAL")
    mats = r["mat"]
    if not mats["materials"]:
        print("  materials: n/a")
    for md in mats["materials"]:
        print(
            "  material[%d]%s: doubleSided=%s  metallicFactor=%.3g  roughnessFactor=%.3g"
            % (
                md["index"],
                " '%s'" % md["name"] if md["name"] else "",
                md["doubleSided"],
                md["metallicFactor"],
                md["roughnessFactor"],
            )
        )
    for ti, info in mats["textures"]:
        dims = "%sx%s" % (info["w"], info["h"]) if info["w"] else "dims n/a"
        print("  texture[%d]: format=%s  mime=%s  %s" % (ti, info["format"], info["mime"], dims))
    mr = r["mr_stats"]
    if mr:
        print(
            "  metallicRoughness channels (64px sample): R %s (mean %.1f)  G var %.2f  B var %.2f"
            % ("constant" if mr["r_const"] else "varying", mr["r_mean"], mr["g_var"], mr["b_var"])
        )
    elif r["mr_info"] is not None:
        print("  metallicRoughness channels: n/a (texture not decodable)")
    for n in r["notes"]:
        print("  note: %s" % n)
    print()


def cmp_val(r, keys, fmt="%s"):
    v = r
    for k in keys:
        if v is None:
            return "n/a"
        v = v.get(k) if isinstance(v, dict) else None
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return fmt % v
    return str(v)


def print_comparison(results):
    names = [r["path"].split("/")[-1] for r in results]
    rows = [
        ("vertices", lambda r: cmp_val(r, ["geo", "vertices"])),
        ("faces", lambda r: cmp_val(r, ["geo", "faces"])),
        ("components", lambda r: cmp_val(r, ["geo", "components"])),
        ("boundary edges", lambda r: cmp_val(r, ["geo", "boundary_edges"])),
        ("non-manifold edges", lambda r: cmp_val(r, ["geo", "nonmanifold_edges"])),
        ("winding %", lambda r: cmp_val(r, ["geo", "winding_pct"], "%.2f")),
        ("watertight (index)", lambda r: cmp_val(r, ["geo", "watertight"])),
        ("watertight (welded)", lambda r: cmp_val(r, ["geo", "welded_watertight"])),
        ("charts", lambda r: cmp_val(r, ["uv", "charts"])),
        ("zero-UV faces %", lambda r: cmp_val(r, ["uv", "zero_uv_pct"], "%.2f")),
        ("texel dens p50", lambda r: cmp_val(r, ["uv", "density", "p50"], "%.4g")),
        ("texel dens CV", lambda r: cmp_val(r, ["uv", "density", "cv"], "%.3f")),
        (
            "atlas",
            lambda r: "%sx%s %s" % (r["atlas"]["w"], r["atlas"]["h"], r["atlas"]["format"])
            if r.get("atlas") and r["atlas"].get("w")
            else "n/a",
        ),
        (
            "principal extents",
            lambda r: f3(r["geo"]["principal_extents"]) if "principal_extents" in r.get("geo", {}) else "n/a",
        ),
    ]
    table = [[label] + [fn(r) for r in results] for label, fn in rows]
    header = ["metric"] + names
    widths = [max(len(str(row[i])) for row in [header] + table) for i in range(len(header))]
    print("=" * 72)
    print("COMPARISON")
    print("=" * 72)
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    for row in table:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    results = []
    for path in sys.argv[1:]:
        try:
            r = analyze(path)
        except Exception as e:
            print("=" * 72)
            print(path)
            print("  ERROR: %s" % e)
            print()
            continue
        print_report(r)
        results.append(r)
    if len(results) >= 2:
        print_comparison(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
