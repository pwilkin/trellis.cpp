#!/usr/bin/env python3
"""Multi-view sampler parity oracle for trellis-test-multiview (test T3).

Runs the reference FlowEulerMultiViewGuidanceIntervalSampler maths -- per view:
GuidanceIntervalSamplerMixin -> ClassifierFreeGuidanceSamplerMixin (CFG + rescale) ->
FlowEulerSampler._inference_model; then `pred_v = sum_i w_i * pred_v_i` and the Euler step --
over a deterministic fake model that trellis-test-multiview reimplements in C++:

    v(x, t_scaled, cond)[k] = (0.5 + t_scaled / 2000) * x[k] + 0.7 * mean(cond) + 0.1 * sin(0.37 * k)

k is the flat index in trellis.cpp's velocity layout [c + C*n]. Arrays are float32 and the
arithmetic is the reference's (float32 tensors, unbiased std over all non-batch dims).

By default the sampler maths below is a literal transcription of
trellis2/pipelines/samplers/{flow_euler,classifier_free_guidance_mixin,guidance_interval_mixin}.py
(ComfyUI-Trellis2). With MV_REF=<checkout> and torch installed, the real
FlowEulerMultiViewGuidanceIntervalSampler.sample() is run instead (dense cases only -- a sparse
x_t needs the SparseTensor class, which pulls in CUDA extensions) and the transcription is
checked against it.

Output: $OUT (default ref/mv_step/) with one directory per case:
    params.txt  "C N V steps gs gr gi0 gi1 rescale_t" ; x0.npy [N*C] ; conds.npy [V, L*D]
    w.npy [V, N] ; out.npy [N*C] (final sample) ; step_<i>.npy (per-step samples)

    python tools/ref_multiview_step.py [--out DIR]
"""
import argparse
import os
import sys

import numpy as np

SIGMA_MIN = 1e-5


def fake_model(x, ts, cond):
    """x: float32 [N*C] flat in the [c + C*n] layout."""
    k = np.arange(x.size, dtype=np.float32)
    a = np.float32(0.5) + np.float32(ts) / np.float32(2000.0)
    return (a * x + np.float32(0.7) * np.float32(cond.astype(np.float64).mean())
            + np.float32(0.1) * np.sin(np.float32(0.37) * k)).astype(np.float32)


def cfg_inference(x, t, cond, neg, gs, gr):
    """ClassifierFreeGuidanceSamplerMixin._inference_model (+ FlowEulerSampler's)."""
    ts = 1000.0 * np.float32(t)
    if gs == 1:
        return fake_model(x, ts, cond)
    if gs == 0:
        return fake_model(x, ts, neg)
    pos = fake_model(x, ts, cond)
    ng = fake_model(x, ts, neg)
    pred = np.float32(gs) * pos + np.float32(1 - gs) * ng
    if gr > 0:
        a = np.float32(1 - SIGMA_MIN)
        b = np.float32(SIGMA_MIN + (1 - SIGMA_MIN) * t)
        x0p = a * x - b * pos
        x0c = a * x - b * pred
        sp = x0p.std(ddof=1)
        sc = x0c.std(ddof=1)
        x0r = x0c * (sp / sc)
        x0 = np.float32(gr) * x0r + np.float32(1 - gr) * x0c
        pred = (a * x - x0) / b
    return pred.astype(np.float32)


def sample_transcription(x0, conds, neg, w, C, steps, gs, gr, gi, rescale_t):
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    x = x0.copy()
    trace = []
    N = x.size // C
    for i in range(steps):
        t, t_prev = float(t_seq[i]), float(t_seq[i + 1])
        g = gs if gi[0] <= t <= gi[1] else 1.0          # GuidanceIntervalSamplerMixin
        acc = np.zeros_like(x)
        for v in range(conds.shape[0]):
            pv = cfg_inference(x, t, conds[v], neg, g, gr)
            acc = acc + pv * np.repeat(w[v], C).astype(np.float32)
        x = (x - np.float32(t - t_prev) * acc).astype(np.float32)
        trace.append(x.copy())
    return x, trace


def sample_real(root, x0, conds, neg, w_dense, C, R, steps, gs, gr, gi, rescale_t, views):
    """Run the real reference sampler on a dense [1,C,R,R,R] tensor (MV_REF)."""
    import torch
    sys.path.insert(0, os.path.dirname(__file__))
    from ref_multiview_weights import load_real_sampler
    fe = load_real_sampler(root)

    def to_dense(flat):    # [c + C*n] -> [1,C,R,R,R], n = x*R*R + y*R + z
        return torch.from_numpy(flat.reshape(R * R * R, C).T.reshape(1, C, R, R, R).copy())

    def from_dense(t):
        return t.reshape(C, R * R * R).T.reshape(-1).numpy()

    class Fake(torch.nn.Module):
        def forward(self, x, t, cond):
            return to_dense(fake_model(from_dense(x), float(t[0]), cond.numpy()))

    s = fe.FlowEulerMultiViewGuidanceIntervalSampler(sigma_min=SIGMA_MIN, resolution=R)
    vc = {name: {"cond": torch.from_numpy(conds[i]), "neg_cond": torch.from_numpy(neg)}
          for i, name in enumerate(views)}
    out = s.sample(Fake(), to_dense(x0), conds=vc, views=views, steps=steps, rescale_t=rescale_t,
                   guidance_strength=gs, guidance_rescale=gr, guidance_interval=gi,
                   front_axis="z", blend_temperature=2.0, verbose=False)
    return from_dense(out.samples), [from_dense(t) for t in out.pred_x_t]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.environ.get("OUT", "ref/mv_step"))
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ref_multiview_weights import ref_dense, ref_sparse

    rng = np.random.default_rng(7)
    L, D = 6, 16
    cases = [
        # name, dense?, C, R/grid, views, steps, gs, gr, gi, rescale_t
        ("dense_cfg_rescale", True, 8, 16, ["front", "back", "right", "left"], 6, 7.5, 0.7, (0.6, 1.0), 5.0),
        ("dense_cfg",         True, 8, 16, ["front", "back"],                  4, 3.0, 0.0, (0.0, 1.0), 1.0),
        ("sparse_cfg",        False, 32, 32, ["front", "back", "left"],       5, 7.5, 0.0, (0.6, 1.0), 3.0),
        ("sparse_nocfg",      False, 32, 64, ["front", "right"],              4, 1.0, 0.0, (0.6, 0.9), 3.0),
        ("sparse_gs0",        False, 8, 32, ["front", "back"],                3, 0.0, 0.0, (0.0, 1.0), 1.0),
    ]
    worst_real = None
    for name, dense, C, R, views, steps, gs, gr, gi, rt in cases:
        if dense:
            N = R ** 3
            w = ref_dense(R, views, "z", 2.0)
            coords = None
        else:
            N = 3000
            coords = rng.integers(0, R, size=(N, 3))
            w = ref_sparse(coords, R, views, "z", 2.0)
        V = len(views)
        x0 = rng.standard_normal(N * C).astype(np.float32)
        conds = rng.standard_normal((V, L * D)).astype(np.float32)
        neg = np.zeros(L * D, dtype=np.float32)
        w32 = np.ascontiguousarray(w, dtype=np.float32)
        out, trace = sample_transcription(x0, conds, neg, w32, C, steps, gs, gr, gi, rt)
        if dense and os.environ.get("MV_REF"):
            try:
                rout, _ = sample_real(os.environ["MV_REF"], x0, conds, neg, w32, C, R, steps, gs, gr, gi, rt, views)
                d = float(np.abs(rout - out).max() / max(1e-12, np.abs(rout).max()))
                worst_real = d if worst_real is None else max(worst_real, d)
                out = rout
            except Exception as e:  # noqa: BLE001
                print(f"[ref] {name}: real sampler failed ({e}); transcription only")
        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "params.txt"), "w") as f:
            f.write(f"{C} {N} {V} {steps} {gs} {gr} {gi[0]} {gi[1]} {rt}\n")
        np.save(os.path.join(d, "x0.npy"), x0)
        np.save(os.path.join(d, "conds.npy"), conds)
        np.save(os.path.join(d, "w.npy"), w32)
        np.save(os.path.join(d, "out.npy"), out.astype(np.float32))
        for i, s in enumerate(trace):
            np.save(os.path.join(d, f"step_{i}.npy"), s.astype(np.float32))
        print(f"[ref] {name}: C={C} N={N} V={V} steps={steps} gs={gs} gr={gr} -> {d}")
    if worst_real is not None:
        print(f"[ref] transcription vs real sampler (dense): max rel = {worst_real:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
