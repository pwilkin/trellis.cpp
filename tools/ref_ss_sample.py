#!/usr/bin/env python3
"""Golden dump for the FlowEulerGuidanceIntervalSampler driving the SS-flow DiT.
Runs on GPU (VRAM), host-RAM-light. Saves noise, cond, neg_cond, final latent.
    /media/ilintar/D_SSD/trellis2-venv/bin/python tools/ref_ss_sample.py
"""
import os, sys, json
os.environ["ATTN_BACKEND"] = "sdpa"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/tmp/TRELLIS.2")
import numpy as np
import torch
from safetensors.torch import load_file
from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler

GS = float(os.environ.get("GS", "7.5"))
GR = float(os.environ.get("GR", "0.7"))
CKPT = "/media/ilintar/D_SSD/models/trellis2/ckpts/ss_flow_img_dit_1_3B_64_bf16"
OUT  = os.environ.get("OUT", "/media/ilintar/D_SSD/models/trellis2/ref/ss_sample"); os.makedirs(OUT, exist_ok=True)
DEV  = os.environ.get("REF_DEV", "cuda:1")
DT   = torch.float32

def save(n, t): a = np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy()); np.save(f"{OUT}/{n}.npy", a); print(f"  {n:10s} {list(a.shape)} std={a.std():.5f}")

cfg = json.load(open(CKPT + ".json"))["args"]
with torch.device(DEV):
    model = SparseStructureFlowModel(**cfg)
model.convert_to(DT)
model.load_state_dict({k: v for k, v in load_file(CKPT + ".safetensors", device=DEV).items()}, strict=False)
model.eval()

g = torch.Generator(device=DEV).manual_seed(7)
B, R, Cin = 1, cfg["resolution"], cfg["in_channels"]
noise = torch.randn(B, Cin, R, R, R, generator=g, device=DEV, dtype=DT)
cond  = torch.randn(B, 1029, cfg["cond_channels"], generator=g, device=DEV, dtype=DT) * 0.5
neg   = torch.zeros_like(cond)
save("noise", noise); save("cond", cond); save("neg_cond", neg)

# SS sampler params (pipeline.json)
sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
with torch.no_grad():
    out = sampler.sample(model, noise, cond=cond, neg_cond=neg,
                         steps=12, rescale_t=5.0, guidance_strength=GS,
                         guidance_rescale=GR, guidance_interval=(0.6, 1.0), verbose=False)
print("per-step std:", " ".join(f"{x.std():.4f}" for x in out.pred_x_t))
for i, st in enumerate(out.pred_x_t):
    np.save(f"{OUT}/step_{i}.npy", np.ascontiguousarray(st.detach().to(torch.float32).cpu().numpy()))
save("samples", out.samples)
print("DONE ->", OUT)
