#!/usr/bin/env python3
"""Golden dump for the sparse SLAT shape flow (SLatFlowModel img2shape, in=out=32).
Random voxel coords + cond (validates flow math, like ref_ss_sample). GPU, RAM-light.
    GS=1.0 GR=0.0 OUT=.../ref/slat_shape_gs1 python tools/ref_slat_shape.py
"""
import os, sys, json
os.environ["ATTN_BACKEND"] = "sdpa"
os.environ["SPCONV_ALGO"] = "native"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/tmp/TRELLIS.2")
import numpy as np, torch
from safetensors.torch import load_file
from trellis2.models.structured_latent_flow import SLatFlowModel
from trellis2.modules.sparse import SparseTensor
from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler

# --- monkeypatch sparse attention to torch SDPA (no flash_attn); B=1 = one sequence ---
import torch.nn.functional as F
from trellis2.modules.sparse.attention import modules as _spm
def _feats(x): return x.feats if hasattr(x, "feats") else x
def _bhsd(x):  # [T,H,C] -> [1,H,T,C]
    return x.unsqueeze(0).transpose(1, 2)
def _sdpa(*args, **kwargs):
    n = len(args) + len(kwargs)
    if n == 1:
        qkv = args[0] if args else kwargs["qkv"]; s = qkv
        q, k, v = _feats(qkv).unbind(1)                 # each [T,H,C]
    elif n == 3:
        q = args[0]; k = args[1]; v = args[2]; s = q if hasattr(q, "feats") else None
        q = _feats(q); k = _feats(k).reshape(-1, *_feats(k).shape[-2:]); v = _feats(v).reshape(-1, *_feats(v).shape[-2:])
    else:
        raise ValueError(f"sdpa patch: unhandled n={n}")
    out = F.scaled_dot_product_attention(_bhsd(q), _bhsd(k), _bhsd(v))   # scale=1/sqrt(head_dim)
    out = out.transpose(1, 2).squeeze(0)                # [T,H,C]
    return s.replace(out) if s is not None else out
_spm.sparse_scaled_dot_product_attention = _sdpa

CK = "/media/ilintar/D_SSD/models/trellis2/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16"
OUT = os.environ.get("OUT", "/media/ilintar/D_SSD/models/trellis2/ref/slat_shape"); os.makedirs(OUT, exist_ok=True)
DEV = os.environ.get("REF_DEV", "cuda:1"); DT = torch.float32
GS = float(os.environ.get("GS", "7.5")); GR = float(os.environ.get("GR", "0.5"))
N = int(os.environ.get("N", "3000")); RES = 32; NIMG = 1029

cfg = json.load(open(CK + ".json"))["args"]
print("cfg:", {k: cfg[k] for k in ("resolution","in_channels","out_channels","model_channels","num_blocks","num_heads")})
with torch.device(DEV):
    model = SLatFlowModel(**cfg)
model.convert_to(DT)
model.load_state_dict({k: v for k, v in load_file(CK + ".safetensors", device=DEV).items()}, strict=False)
model.eval()

g = torch.Generator(device="cpu").manual_seed(11)
# N distinct random voxel coords in [0,RES)^3, batch 0
seen = set(); rows = []
gg = np.random.default_rng(11)
while len(rows) < N:
    x, y, z = int(gg.integers(0, RES)), int(gg.integers(0, RES)), int(gg.integers(0, RES))
    if (x, y, z) not in seen: seen.add((x, y, z)); rows.append((0, x, y, z))
coords = torch.tensor(rows, dtype=torch.int32, device=DEV)
gc = torch.Generator(device=DEV).manual_seed(11)
noise_feats = torch.randn(N, cfg["in_channels"], generator=gc, device=DEV, dtype=DT)
cond = torch.randn(1, NIMG, cfg["cond_channels"], generator=gc, device=DEV, dtype=DT) * 0.5
neg = torch.zeros_like(cond)
noise = SparseTensor(feats=noise_feats, coords=coords)

sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
with torch.no_grad():
    out = sampler.sample(model, noise, cond=cond, neg_cond=neg, steps=12, rescale_t=3.0,
                         guidance_strength=GS, guidance_rescale=GR, guidance_interval=(0.6, 1.0), verbose=False)
sl = out.samples.feats                          # [N,32]
def sv(n, a): np.save(f"{OUT}/{n}.npy", np.ascontiguousarray(a));
sv("coords", coords[:, 1:].cpu().numpy().astype(np.float32))
sv("noise", noise_feats.cpu().numpy().astype(np.float32))
sv("cond", cond[0].cpu().numpy().astype(np.float32))
sv("neg", neg[0].cpu().numpy().astype(np.float32))
sv("samples", sl.float().cpu().numpy().astype(np.float32))
print(f"N={N} samples [{sl.shape[0]},{sl.shape[1]}] std={sl.float().std():.5f}  -> {OUT}")
