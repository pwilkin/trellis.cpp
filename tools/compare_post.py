#!/usr/bin/env python3
"""Compare two --dump-post files (trellis-cli --dump-post PATH).

Layout (include/trellis_args.h, TrellisParams::dump_post):
    i32 V, F, Mv, res
    f32 verts[V*3]; i32 faces[F*3]; i32 coords[Mv*3]; f32 pbr6[Mv*6]

Reports exact equality, or the per-array differences: counts, max |dverts|, face/coord
mismatch counts and max |dpbr|. Exit status 0 when the files are bit-identical, 1 when
they differ, 2 on a read error.

    python tools/compare_post.py a.post b.post [--tol 1e-6]
"""
import argparse
import sys

import numpy as np


def load_post(path):
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) < 16:
        raise ValueError(f"{path}: truncated header")
    V, F, Mv, res = np.frombuffer(raw, dtype="<i4", count=4)
    off = 16

    def take(dtype, n):
        nonlocal off
        nbytes = int(n) * 4
        if off + nbytes > len(raw):
            raise ValueError(f"{path}: truncated payload")
        a = np.frombuffer(raw, dtype=dtype, count=int(n), offset=off)
        off += nbytes
        return a

    d = {
        "V": int(V), "F": int(F), "Mv": int(Mv), "res": int(res),
        "verts": take("<f4", V * 3).reshape(-1, 3),
        "faces": take("<i4", F * 3).reshape(-1, 3),
        "coords": take("<i4", Mv * 3).reshape(-1, 3),
        "pbr": take("<f4", Mv * 6).reshape(-1, 6),
    }
    if off != len(raw):
        raise ValueError(f"{path}: {len(raw) - off} trailing bytes")
    return d, raw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="report 'within tolerance' when every float |delta| <= tol and counts match")
    args = ap.parse_args()
    try:
        a, ra = load_post(args.a)
        b, rb = load_post(args.b)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if ra == rb:
        print(f"EXACT: identical ({len(ra)} bytes; V={a['V']} F={a['F']} Mv={a['Mv']} res={a['res']})")
        return 0

    print(f"counts  a: V={a['V']} F={a['F']} Mv={a['Mv']} res={a['res']}")
    print(f"        b: V={b['V']} F={b['F']} Mv={b['Mv']} res={b['res']}")
    same_shape = all(a[k] == b[k] for k in ("V", "F", "Mv", "res"))
    worst = 0.0
    if same_shape:
        dv = np.abs(a["verts"].astype(np.float64) - b["verts"])
        nf = int(np.any(a["faces"] != b["faces"], axis=1).sum())
        nc = int(np.any(a["coords"] != b["coords"], axis=1).sum())
        dp = np.abs(a["pbr"].astype(np.float64) - b["pbr"]) if a["Mv"] else np.zeros(1)
        worst = max(float(dv.max(initial=0)), float(dp.max(initial=0)))
        print(f"verts   max|d|={dv.max(initial=0):.3e} mean|d|={dv.mean() if dv.size else 0:.3e} "
              f"differing={int(np.any(dv > 0, axis=1).sum())}/{a['V']}")
        print(f"faces   differing rows={nf}/{a['F']}")
        print(f"coords  differing rows={nc}/{a['Mv']}")
        print(f"pbr     max|d|={dp.max(initial=0):.3e} mean|d|={dp.mean() if dp.size else 0:.3e}")
        if args.tol > 0 and nf == 0 and nc == 0 and worst <= args.tol:
            print(f"WITHIN TOLERANCE ({worst:.3e} <= {args.tol:g})")
            return 0
    else:
        # Different topology: compare what still has a meaning.
        print(f"delta   V {b['V'] - a['V']:+d}  F {b['F'] - a['F']:+d}  Mv {b['Mv'] - a['Mv']:+d}")
        if a["V"] and b["V"]:
            print(f"bbox a: {a['verts'].min(0)} .. {a['verts'].max(0)}")
            print(f"bbox b: {b['verts'].min(0)} .. {b['verts'].max(0)}")
        if a["Mv"] and b["Mv"]:
            sa = {tuple(c) for c in a["coords"]}
            sb = {tuple(c) for c in b["coords"]}
            inter = len(sa & sb)
            print(f"coords  IoU={inter / max(1, len(sa | sb)):.4f} (|a|={len(sa)} |b|={len(sb)})")
    print("DIFFERENT")
    return 1


if __name__ == "__main__":
    sys.exit(main())
