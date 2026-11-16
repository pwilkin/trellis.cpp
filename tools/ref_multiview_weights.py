#!/usr/bin/env python3
"""Reference multi-view blend weights for trellis-test-multiview (test T1).

Three formulas live here:

  ref_sparse / ref_dense   literal NumPy transcriptions of the ComfyUI-Trellis2 reference,
                           FlowEulerMultiViewSampler._compute_view_weights_sparse/_dense
                           (trellis2/pipelines/samplers/flow_euler.py). Cardinal named views
                           only, front_axis 'z' or 'x', no elevation, no yaw.
  gen_sparse / gen_dense   the generalised formula of docs/spec/30-multiview.md: arbitrary
                           azimuth/elevation, six front-axis presets, yaw offset, geometric or
                           reference normalisation.
  real (optional)          when MV_REF points at a ComfyUI-Trellis2 checkout and torch imports,
                           the reference's own torch functions, loaded under a stub package so
                           trellis2/pipelines/__init__ (CUDA-only imports) is never executed.

Self-checks (always): gen(ref-z|ref-x, cardinal views) == ref(...) to 1e-6, and, with MV_REF,
ref(...) == real(...) to 1e-6. The dumped golden for a reference-compatible case is the reference
result itself (real when available, else the transcription), so the C++ is compared against the
reference, not against the generalisation it implements.

Output: $OUT (default ref/mv_weights/) holds cases.txt (one case per line:
    name kind grid model_res preset yaw tau az,el;az,el;...
) plus <name>_coords.npy (float32 [N,3], sparse cases) and <name>_w.npy (float32 [V,N]).

    python tools/ref_multiview_weights.py [--out DIR]
"""
import argparse
import importlib.util
import os
import sys
import types

import numpy as np

# --------------------------------------------------------------------------- reference

REF_NAMES = {0.0: "front", 180.0: "back", 90.0: "right", 270.0: "left"}


def _softmax(scores, tau, axis):
    s = scores * tau
    s = s - s.max(axis=axis, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=axis, keepdims=True)


def ref_sparse(coords, R, views, front_axis="z", tau=2.0):
    """flow_euler.py _compute_view_weights_sparse. coords [N,3] = torch coords[:,1:4]."""
    coords = np.asarray(coords)
    z = (coords[:, 0].astype(np.float32) / R) * 2 - 1.0
    x = (coords[:, 2].astype(np.float32) / R) * 2 - 1.0
    if front_axis == "z":
        vv = {"front": z, "back": -z, "right": x, "left": -x}
    else:
        vv = {"front": x, "back": -x, "right": z, "left": -z}
    scores = [vv[v] if v in vv else np.full_like(z, -10.0) for v in views]
    scores = np.stack(scores, axis=1).astype(np.float32)        # [N, V]
    return _softmax(scores.astype(np.float64), tau, axis=1).T   # [V, N]


def ref_dense(D, views, front_axis="z", tau=2.0):
    """flow_euler.py _compute_view_weights_dense for a (1, C, D, D, D) tensor."""
    lin = np.linspace(-1, 1, D, dtype=np.float32)
    gz, gy, gx = np.meshgrid(lin, lin, lin, indexing="ij")
    if front_axis == "z":
        vs = {"front": gz, "back": -gz, "right": gx, "left": -gx}
    else:
        vs = {"front": gx, "back": -gx, "right": gz, "left": -gz}
    scores = [vs[v] if v in vs else np.full_like(gz, -10.0) for v in views]
    scores = np.stack(scores, axis=0).astype(np.float64)        # [V, D, D, D]
    return _softmax(scores, tau, axis=0).reshape(len(views), -1)  # token = x*D*D + y*D + z


# --------------------------------------------------------------------------- generalised

# grid axis a = sign * viewer[idx]; viewer 0=X right, 1=Y up, 2=Z toward the front camera
PRESETS = {
    "-y":    ((0, +1), (2, -1), (1, +1)),
    "+y":    ((0, -1), (2, +1), (1, +1)),
    "+x":    ((2, +1), (0, +1), (1, +1)),
    "-x":    ((2, -1), (0, -1), (1, +1)),
    "ref-z": ((2, +1), (1, -1), (0, +1)),
    "ref-x": ((0, +1), (1, +1), (2, +1)),
}


def view_dir_grid(az, el, preset, yaw=0.0):
    th = np.deg2rad(az + yaw)
    ph = np.deg2rad(el)
    d = np.array([np.cos(ph) * np.sin(th), np.sin(ph), np.cos(ph) * np.cos(th)])
    d[np.abs(d) < 1e-9] = 0.0
    return np.array([s * d[i] for i, s in PRESETS[preset]])


def _gen(p, views, preset, yaw, tau):
    if len(views) == 1:
        return np.ones((1, p.shape[0]))
    dirs = np.stack([view_dir_grid(az, el, preset, yaw) for az, el in views])  # [V,3]
    scores = p @ dirs.T                                                        # [N,V]
    return _softmax(scores, tau, axis=1).T


def gen_sparse(coords, grid, views, preset, yaw, tau, model_res=0):
    c = np.asarray(coords, dtype=np.float64)
    if preset.startswith("ref-"):
        p = c / (model_res if model_res > 0 else grid) * 2 - 1
    else:
        p = (c + 0.5) / grid * 2 - 1
    return _gen(p, views, preset, yaw, tau)


def gen_dense(D, views, preset, yaw, tau):
    i = np.arange(D, dtype=np.float64)
    lin = (-1 + 2 * i / (D - 1)) if preset.startswith("ref-") else ((i + 0.5) / D * 2 - 1)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    p = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    return _gen(p, views, preset, yaw, tau)


# --------------------------------------------------------------------------- real reference

def load_real_sampler(root):
    """Import flow_euler.py from a ComfyUI-Trellis2 checkout without its package __init__."""
    import torch  # noqa: F401  (fails fast when torch is missing)
    sdir = os.path.join(root, "trellis2", "pipelines", "samplers")
    pkg = types.ModuleType("mvref_samplers")
    pkg.__path__ = [sdir]
    sys.modules["mvref_samplers"] = pkg
    for name in ("base", "classifier_free_guidance_mixin", "guidance_interval_mixin", "flow_euler"):
        spec = importlib.util.spec_from_file_location(f"mvref_samplers.{name}", os.path.join(sdir, name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"mvref_samplers.{name}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["mvref_samplers.flow_euler"]


def real_sparse(fe, coords, R, views, front_axis, tau):
    import torch
    s = fe.FlowEulerMultiViewSampler(sigma_min=1e-5, resolution=R)
    c = torch.zeros((len(coords), 4), dtype=torch.int32)
    c[:, 1:] = torch.from_numpy(np.asarray(coords, dtype=np.int32))
    return s._compute_view_weights_sparse(c, views, front_axis, tau).numpy().astype(np.float64).T


def real_dense(fe, D, views, front_axis, tau):
    s = fe.FlowEulerMultiViewSampler(sigma_min=1e-5, resolution=D)
    w = s._compute_view_weights_dense((1, 8, D, D, D), "cpu", views, front_axis, tau)
    return w.numpy().astype(np.float64).reshape(len(views), -1)


# --------------------------------------------------------------------------- cases

def save_npy(path, a):
    np.save(path, np.ascontiguousarray(a, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.environ.get("OUT", "ref/mv_weights"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    fe = None
    if os.environ.get("MV_REF"):
        try:
            fe = load_real_sampler(os.environ["MV_REF"])
            print(f"[ref] using the real reference sampler from {os.environ['MV_REF']}")
        except Exception as e:  # noqa: BLE001
            print(f"[ref] MV_REF set but the reference could not be loaded ({e}); transcription only")

    rng = np.random.default_rng(1234)
    view_sets = {
        "v1":    [(0, 0)],
        "fb":    [(0, 0), (180, 0)],
        "card4": [(0, 0), (90, 0), (180, 0), (270, 0)],
        "six":   [(0, 0), (90, 0), (180, 0), (270, 0), (0, 90), (0, -90)],
        "arb3":  [(30, 10), (150, -20), (260, 45)],
    }
    taus = [0.5, 1.0, 2.0, 5.0]
    lines = []
    worst_self = 0.0
    worst_real = 0.0

    def emit(name, kind, grid, mres, preset, yaw, tau, views, w, coords=None):
        if coords is not None:
            save_npy(os.path.join(args.out, f"{name}_coords.npy"), coords)
        save_npy(os.path.join(args.out, f"{name}_w.npy"), w)
        vs = ";".join(f"{az:g},{el:g}" for az, el in views)
        lines.append(f"{name} {kind} {grid} {mres} {preset} {yaw:g} {tau:g} {vs}")

    n = 0
    sparse_coords = {g: rng.integers(0, g, size=(5000, 3)) for g in (32, 64, 96)}
    for preset in PRESETS:
        for yaw in (0.0, 37.0):
            for vname, views in view_sets.items():
                for tau in taus:
                    cardinal = yaw == 0.0 and all(el == 0 and az in REF_NAMES for az, el in views)
                    is_ref = preset.startswith("ref-")
                    fa = preset[-1] if is_ref else None
                    # sparse: grid 32/64/96; the reference preset normalises by a model resolution
                    for g in (32, 64, 96):
                        mres = 64 if (is_ref and g != 32) else (32 if is_ref else 0)
                        c = sparse_coords[g]
                        w = gen_sparse(c, g, views, preset, yaw, tau, mres)
                        if is_ref and cardinal and len(views) > 1:
                            names = [REF_NAMES[az] for az, _ in views]
                            r = ref_sparse(c, mres, names, fa, tau)
                            worst_self = max(worst_self, float(np.abs(r - w).max()))
                            if fe is not None:
                                rr = real_sparse(fe, c, mres, names, fa, tau)
                                worst_real = max(worst_real, float(np.abs(rr - r).max()))
                                r = rr
                            w = r
                        emit(f"s{n:04d}", "sparse", g, mres, preset, yaw, tau, views, w, c)
                        n += 1
                    # dense 16
                    w = gen_dense(16, views, preset, yaw, tau)
                    if is_ref and cardinal and len(views) > 1:
                        names = [REF_NAMES[az] for az, _ in views]
                        r = ref_dense(16, names, fa, tau)
                        worst_self = max(worst_self, float(np.abs(r - w).max()))
                        if fe is not None:
                            rr = real_dense(fe, 16, names, fa, tau)
                            worst_real = max(worst_real, float(np.abs(rr - r).max()))
                            r = rr
                        w = r
                    emit(f"d{n:04d}", "dense", 16, 16, preset, yaw, tau, views, w)
                    n += 1

    with open(os.path.join(args.out, "cases.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[ref] {len(lines)} cases -> {args.out}")
    print(f"[ref] self-check gen(ref-*) vs transcription: max |dw| = {worst_self:.3e}")
    if fe is not None:
        print(f"[ref] transcription vs real reference sampler: max |dw| = {worst_real:.3e}")
    ok = worst_self < 1e-6 and (fe is None or worst_real < 1e-6)
    print("[ref] SELF-CHECK " + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
