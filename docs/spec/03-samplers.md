# samplers

## TRELLIS.2 Samplers — `FlowEulerGuidanceIntervalSampler`

Rectified-flow / flow-matching Euler sampler with classifier-free guidance (CFG), CFG-rescale, and a guidance interval. This is the **only** sampler class actually used by both `trellis2_image_to_3d.py` and `trellis2_texturing.py` (instantiated via `getattr(samplers, args['..._sampler']['name'])`; the name in every `pipeline.json` is `"FlowEulerGuidanceIntervalSampler"`).

The class is composed by MRO from three classes (left to right):
`class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler)`.
Method resolution for `_inference_model` is: `GuidanceIntervalSamplerMixin._inference_model` → `ClassifierFreeGuidanceSamplerMixin._inference_model` → `FlowEulerSampler._inference_model`. Each `super()._inference_model(...)` call walks one step down this chain. This layering is the whole behavior; reproduce it exactly.

There are **no trainable parameters** in any sampler file — this is pure inference math, no state_dict keys. The only constructor arg is `sigma_min` (a float, read from `pipeline.json` `args.sigma_min`).

---

### 0. Time convention (CRITICAL)

`t` runs from `1.0` (pure noise) down to `0.0` (clean sample). Internally throughout `flow_euler.py`, `t` is a plain Python `float` in `[0,1]` (per-step scalar), broadcast over the batch. The model, however, expects the timestep in `[0,1000]`: see `_inference_model`, which does `t = torch.tensor([1000 * t] * x_t.shape[0], dtype=float32)` before calling `model(x_t, t, cond, **kwargs)`. So model time = `1000 * t`.

The flow interpolant is `x_t = (1 - t) * x_0 + (sigma_min + (1 - sigma_min) * t) * eps`, where `x_0` is the clean target and `eps` is noise. The model predicts velocity `v` (the `pred` returned by `model(...)` IS the velocity `pred_v`).

---

### 1. Timestep schedule (`sample`, lines 114–118)

```
t_seq = np.linspace(1, 0, steps + 1)                       # steps+1 points: 1.0 ... 0.0 inclusive
t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)  # rescale transform, elementwise
t_seq = t_seq.tolist()
t_pairs = [(t_seq[i], t_seq[i+1]) for i in range(steps)]   # consecutive (t, t_prev) pairs
```

- `np.linspace(1, 0, steps+1)` gives `steps+1` evenly spaced values from `1.0` to `0.0` inclusive. For `steps=12` that is `[1.0, 11/12, 10/12, ..., 1/12, 0.0]`.
- **`rescale_t` transform** (applied to every element `u` of the linspace): `t = rescale_t * u / (1 + (rescale_t - 1) * u)`.
  - With `rescale_t == 1.0` this is identity (`t = u`).
  - It maps `0 → 0` and `1 → 1` (endpoints fixed) and warps intermediate points; larger `rescale_t` pushes mass toward larger `t` (spends more steps near the noisy end). This is a Möbius-style reparameterization.
- The schedule is **deterministic and stateless** — precompute the `steps+1` array of `t` values once on CPU in C++ (e.g. with doubles), then iterate over consecutive pairs.

### 2. Per-step Euler update (`sample_once`, lines 53–81; `sample` loop 120–125)

For each pair `(t, t_prev)` (note `t > t_prev`):
```
pred_x_0, pred_eps, pred_v = _get_model_prediction(model, sample, t, cond, **kwargs)
pred_x_prev = sample - (t - t_prev) * pred_v
sample = pred_x_prev
```
That is the entire Euler step: `x_{prev} = x_t - (t - t_prev) * v`, where `v` is the (possibly CFG-combined) velocity. After the last pair (whose `t_prev == 0.0`) `sample` is the final output, returned as `ret.samples`.

`_get_model_prediction` (lines 48–51): calls `pred_v = _inference_model(...)` then `pred_x_0, pred_eps = _v_to_xstart_eps(...)`. For the actual Euler update only `pred_v` matters; `pred_x_0` is bookkeeping returned in `ret.pred_x_0` (a list) and is NOT fed back. **For the C++ port you only need `pred_v` per step** unless you also want to expose intermediate x0 predictions.

### 3. Conversion formulas (`FlowEulerSampler`, lines 24–42)

(`s = sigma_min`, scalar `t`)
- `_eps_to_xstart`:  `x_0 = (x_t - (s + (1-s)*t) * eps) / (1 - t)`
- `_xstart_to_eps`:  `eps = (x_t - (1-t) * x_0) / (s + (1-s)*t)`
- `_v_to_xstart_eps`:
  - `eps = (1 - t) * v + x_t`
  - `x_0 = (1 - s) * x_t - (s + (1-s)*t) * v`
- `_pred_to_xstart`:  `x_0 = (1 - s) * x_t - (s + (1-s)*t) * pred`   (identical to the x_0 line of `_v_to_xstart_eps`; `pred` IS `v`)
- `_xstart_to_pred`:  `pred = ((1 - s) * x_t - x_0) / (s + (1-s)*t)`

Only `_pred_to_xstart` and `_xstart_to_pred` are needed at runtime, and ONLY when `guidance_rescale > 0` (CFG-rescale path). If `guidance_rescale == 0` (the default for SS, shape, tex per the UI), these are never called.

### 4. Classifier-free guidance (`ClassifierFreeGuidanceSamplerMixin._inference_model`, lines 9–29)

Signature: `_inference_model(self, model, x_t, t, cond, neg_cond, guidance_strength, guidance_rescale=0.0, **kwargs)`.

```
if guidance_strength == 1:
    return super()._inference_model(model, x_t, t, cond, **kwargs)          # cond only, no neg pass
elif guidance_strength == 0:
    return super()._inference_model(model, x_t, t, neg_cond, **kwargs)      # neg only
else:
    pred_pos = super()._inference_model(model, x_t, t, cond, **kwargs)
    pred_neg = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
    pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg

    if guidance_rescale > 0:                                                # CFG-rescale
        x_0_pos = _pred_to_xstart(x_t, t, pred_pos)
        x_0_cfg = _pred_to_xstart(x_t, t, pred)
        std_pos = x_0_pos.std(dim=[1..ndim-1], keepdim=True)               # per-sample std over all non-batch dims
        std_cfg = x_0_cfg.std(dim=[1..ndim-1], keepdim=True)
        x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
        x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
        pred = _xstart_to_pred(x_t, t, x_0)
    return pred
```

Notes for the port:
- **CFG formula**: `v = g * v_pos + (1 - g) * v_neg`, where `g = guidance_strength`. (Equivalent to `v_neg + g * (v_pos - v_neg)`.) This requires **two forward passes** of the DiT per step (one with `cond`, one with `neg_cond`) when `1 < g`.
- The `g == 1` / `g == 0` short-circuits skip the second pass. For tex (g default 1.0) this means **only the conditional pass runs** and no negative cond is used — important: tex needs only one DiT pass per step.
- **CFG-rescale** (`std_pos / std_cfg`): std is computed **per batch sample** over every dimension except batch (`dim=list(range(1, ndim))`), keepdim. For a SparseTensor (shape/tex stages) "all non-batch dims" must be interpreted as the feature/voxel dims for that sample — see open questions. With `guidance_rescale == 0` (UI defaults for all three stages) this block is skipped entirely.

### 5. Guidance interval gating (`GuidanceIntervalSamplerMixin._inference_model`, lines 9–13)

Signature: `_inference_model(self, model, x_t, t, cond, guidance_strength, guidance_interval, **kwargs)`.

```
if guidance_interval[0] <= t <= guidance_interval[1]:
    return super()._inference_model(model, x_t, t, cond, guidance_strength=guidance_strength, **kwargs)
else:
    return super()._inference_model(model, x_t, t, cond, guidance_strength=1, **kwargs)
```

- `t` here is the **per-step scalar float in `[0,1]`** (the warped schedule value), NOT `1000*t`.
- When `t` is **inside** `[lo, hi]` (inclusive both ends), pass the real `guidance_strength` → full CFG (two passes).
- When `t` is **outside** the interval, force `guidance_strength = 1` → conditional-only single pass (no CFG, no neg pass). This is the gating: CFG is only applied for `t` in the interval.
- `guidance_interval` is forwarded from `sample(...)` down through `kwargs` to this mixin. `guidance_rescale` (if present) continues down to the CFG mixin unchanged.

### 6. Full per-step pseudocode (transcribable)

```
# Precompute (once):
us = linspace(1.0, 0.0, steps+1)                       # steps+1 doubles
ts = [ rescale_t*u / (1 + (rescale_t-1)*u) for u in us ]

x = noise                                              # initial sample (Gaussian)
for i in 0 .. steps-1:
    t      = ts[i]
    t_prev = ts[i+1]

    # --- determine effective guidance strength via interval gating ---
    if guidance_interval_lo <= t <= guidance_interval_hi:
        g = guidance_strength
    else:
        g = 1.0

    # --- velocity prediction with CFG ---
    t_model = 1000.0 * t                               # broadcast to batch as float32
    if g == 1.0:
        v = DiT(x, t_model, cond)                      # single pass
    elif g == 0.0:
        v = DiT(x, t_model, neg_cond)
    else:
        v_pos = DiT(x, t_model, cond)
        v_neg = DiT(x, t_model, neg_cond)
        v = g * v_pos + (1 - g) * v_neg
        if guidance_rescale > 0:
            x0_pos = (1-sigma_min)*x - (sigma_min+(1-sigma_min)*t)*v_pos
            x0_cfg = (1-sigma_min)*x - (sigma_min+(1-sigma_min)*t)*v
            # per-sample std over all non-batch dims, keepdim
            std_pos = std(x0_pos); std_cfg = std(x0_cfg)
            x0_res  = x0_cfg * (std_pos/std_cfg)
            x0      = guidance_rescale*x0_res + (1-guidance_rescale)*x0_cfg
            v       = ((1-sigma_min)*x - x0) / (sigma_min+(1-sigma_min)*t)

    # --- Euler step ---
    x = x - (t - t_prev) * v

return x   # ret.samples
```

### 7. Pipeline parameters (per stage)

`sample(...)` signature defaults: `steps=50, rescale_t=1.0, guidance_strength=3.0, guidance_interval=(0.0,1.0)`. But at runtime the pipeline merges `{**self.X_sampler_params, **sampler_params}` where `self.X_sampler_params` comes from `pipeline.json` (`args['..._sampler']['params']`) and `sampler_params` is the per-call override (e.g. from `app.py`). `guidance_interval` is supplied **only** by `pipeline.json` `params` (the apps do not expose it), so it falls to whatever the checkpoint config sets.

**UI / per-call override defaults from `app.py` / `app_texturing.py`** (these are the production values; sliders `value=...`):

| Stage | steps | guidance_strength | guidance_rescale | rescale_t |
|---|---|---|---|---|
| **SS** (sparse_structure) | 12 | 7.5 | 0.7 | 5.0 |
| **shape** (shape_slat) | 12 | 7.5 | 0.5 | 3.0 |
| **tex** (tex_slat) | 12 | 1.0 | 0.0 | 3.0 |

Slider ranges (for context): steps 1–50; guidance_strength 1.0–10.0; guidance_rescale 0.0–1.0; rescale_t 1.0–6.0.

Consequences:
- **tex** has `guidance_strength == 1.0` and `guidance_rescale == 0.0` → exactly one DiT pass per step, no neg cond, no rescale, plain Euler. Cheapest stage.
- **SS** and **shape** use real CFG (two passes/step) plus CFG-rescale (`std_pos/std_cfg` reweighting on x0).
- `sigma_min` is a constructor arg from `pipeline.json` `args` (typical TRELLIS value `1e-5`; confirm from checkpoint config — see open questions).
- `guidance_interval` value is NOT in the apps; it comes from `pipeline.json` (commonly `[0.0, 1.0]` = always on, or a sub-interval). Must read from the shipped config — flagged below.

### 8. `FlowEulerCfgSampler` and `FlowEulerSampler` (not the production path)

- `FlowEulerSampler.sample(model, noise, cond, steps, rescale_t, verbose, **kwargs)` — plain Euler, no CFG. Same schedule + Euler step as above with `v = DiT(x, 1000*t, cond)`.
- `FlowEulerCfgSampler.sample(model, noise, cond, neg_cond, steps, rescale_t, guidance_strength, verbose, **kwargs)` — CFG but no interval gating (CFG every step). Delegates to `FlowEulerSampler.sample` with `neg_cond`/`guidance_strength` in kwargs. Only needed if you want to support these modes; the actual TRELLIS.2 pipelines use `FlowEulerGuidanceIntervalSampler` for all three stages.


## Weight key patterns

NONE — the sampler classes (FlowEulerSampler, FlowEulerCfgSampler, FlowEulerGuidanceIntervalSampler, and the two mixins) contain NO nn.Module, NO parameters, NO buffers. They produce zero safetensors/state_dict keys. The only configuration "weights" are scalar floats read from the checkpoint's pipeline.json:
  - args.sigma_min (float, constructor arg of FlowEulerSampler)
  - params.steps, params.guidance_strength, params.guidance_rescale, params.rescale_t, params.guidance_interval (sampler call params, dict per stage: sparse_structure_sampler, shape_slat_sampler, tex_slat_sampler).
These live under pipeline.json keys like:
  sparse_structure_sampler: { name: "FlowEulerGuidanceIntervalSampler", args: { sigma_min: <float> }, params: { steps, guidance_strength, guidance_rescale, guidance_interval, rescale_t } }
  shape_slat_sampler: { ... same shape ... }
  tex_slat_sampler: { ... same shape ... }
They are NOT tensors; load them as config scalars, not as a safetensors blob.

## GGML notes

All sampler math is elementwise scalar/tensor arithmetic on the host loop — no custom GGML op is needed for the sampler itself; the heavy lifting is the DiT forward pass (separate component).

Per-op mapping:
- Timestep schedule: compute on CPU in C++ with doubles (linspace + Mobius rescale). No GGML.
- Euler step `x = x - (t - t_prev) * v`: ggml_sub + ggml_scale (scalar `(t - t_prev)`), OR a single ggml_add of x and a scaled v. Operates on whatever tensor type the stage uses (dense [N,C,D,H,W] for SS; sparse feats [num_voxels, C] for shape/tex).
- CFG combine `g*v_pos + (1-g)*v_neg`: two ggml_scale + ggml_add (or ggml_add of v_neg and ggml_scale(v_pos - v_neg, g)). Trivial.
- _pred_to_xstart / _xstart_to_pred: ggml_scale + ggml_sub with scalar coefficients computed from (sigma_min, t). Only needed when guidance_rescale > 0 (SS and shape stages).
- CFG-rescale std: needs per-sample standard deviation over all non-batch dims. ggml has ggml_std-style reductions via mean/variance (compute mean, subtract, square, mean, sqrt) or use a small custom reduction. For batch=1 (typical inference) this is a single global std over the whole tensor — easy: mean -> center -> square -> mean -> sqrt. For SparseTensor stages, std must be taken over the sample's voxel+feature elements (all elements for batch=1). A tiny custom reduction op or a sequence (ggml_mean, ggml_sqr, ggml_mean, ggml_sqrt) suffices. Then x0_cfg * (std_pos/std_cfg) is ggml_scale by the scalar ratio (batch=1) — pull the two std scalars back to host and use them as scale factors to avoid a tensor-broadcast divide.

Memory: each CFG step (SS/shape) requires TWO DiT forward passes; budget activations for both, but they can run sequentially (compute v_pos, free its graph, compute v_neg) to keep peak memory to one forward at a time, since only the output velocity tensors must persist. tex stage is single-pass.

Determinism: noise is the only randomness (Gaussian init in the pipeline, not in the sampler). The sampler loop is fully deterministic given noise + cond. Seed handling lives in the pipeline, not here.

Inference-only: all of this is @torch.no_grad and inference-only; there is no training code in these files.

## Open questions

1. Exact values of `sigma_min` and `guidance_interval` per stage are in the shipped checkpoint `pipeline.json` (downloaded with the model via from_pretrained), which is NOT present in the repo at /tmp/TRELLIS.2. Original TRELLIS used sigma_min=1e-5 and guidance_interval like [0.0,1.0] (SS) / [0.5,1.0] (slat); TRELLIS.2 may differ. MUST read the actual pipeline.json from the downloaded checkpoint to confirm sigma_min and guidance_interval for sparse_structure / shape_slat / tex_slat. The other four params (steps, guidance_strength, guidance_rescale, rescale_t) are pinned by app.py/app_texturing.py defaults given in the spec table and should match pipeline.json defaults.

2. CFG-rescale std semantics for SparseTensor: `x_0.std(dim=list(range(1, x_0.ndim)), keepdim=True)` assumes a dense tensor with a leading batch dim. For SS (dense [N,C,D,H,W]) this is unambiguous (std over C,D,H,W per sample). For shape/tex the prediction is a SparseTensor; need to confirm how `.std(dim=...)` is implemented on the SparseTensor wrapper (whether it reduces over feature dim only, over all voxels+features, and what ndim/batch means there). Since shape stage uses guidance_rescale=0.5 (>0), this path IS active for shape and must be matched exactly. Read the SparseTensor class (.std/.ndim) to nail this down. (Tex uses guidance_rescale=0.0 so it is unaffected.)

3. Confirm whether any stage in the production pipeline_type variants ("512"/"1024_cascade"/"1536_cascade") overrides guidance_interval or passes additional kwargs (e.g. the cascade LR/HR passes in sample_shape_slat_cascade reuse shape_slat_sampler_params); verify no extra sampler kwargs are injected there.

4. The model forward signature `model(x_t, t, cond, **kwargs)` — confirm `cond`/`neg_cond` are dicts unpacked via `**cond` at the pipeline call site (they are: pipeline does `**cond` then sampler threads `cond=` and `neg_cond=`). Confirm the DiT accepts the integer-ish float timestep in [0,1000] as the time embedding input (matches DiT timestep embedding component).
