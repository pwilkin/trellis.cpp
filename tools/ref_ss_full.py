#!/usr/bin/env python3
"""Ground-truth SS stage for the goblin: real DINOv3 cond -> SS sampler -> ss_decode.
Dumps the sampled z (torch [1,8,16,16,16]) + voxel count, and the per-step z trace.
    python tools/ref_ss_full.py
"""
import os, sys, json
os.environ["ATTN_BACKEND"] = "sdpa"; os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
sys.path.insert(0, "/tmp/TRELLIS.2")
import numpy as np, torch
from safetensors.torch import load_file
from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
from trellis2.models.sparse_structure_vae import SparseStructureDecoder
from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler

M = "/media/ilintar/D_SSD/models/trellis2"
OUT = M + "/ref/ss_full"; os.makedirs(OUT, exist_ok=True)
DEV = "cuda:1"; DT = torch.float32

cfg = json.load(open(M + "/ckpts/ss_flow_img_dit_1_3B_64_bf16.json"))["args"]
with torch.device(DEV): flow = SparseStructureFlowModel(**cfg)
flow.convert_to(DT); flow.load_state_dict(load_file(M+"/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",device=DEV),strict=False); flow.eval()

cond = torch.from_numpy(np.load(M+"/ref/dinov3/cond.npy")).to(DEV).float()   # [1,1029,1024]
neg = torch.zeros_like(cond)
print("cond", list(cond.shape), "std", float(cond.std()))
torch.manual_seed(42)
noise = torch.randn(1, 8, 16, 16, 16, device=DEV, dtype=DT)
np.save(f"{OUT}/noise.npy", noise.cpu().numpy())

sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
with torch.no_grad():
    out = sampler.sample(flow, noise, cond=cond, neg_cond=neg, steps=12, rescale_t=5.0,
                         guidance_strength=7.5, guidance_rescale=0.7, guidance_interval=(0.6,1.0), verbose=False)
z = out.samples
print("z std", float(z.std()), "per-step std:", " ".join(f"{x.std():.3f}" for x in out.pred_x_t))
np.save(f"{OUT}/z.npy", z.cpu().numpy())

dcfg = json.load(open(M+"/tilarge/ckpts/ss_dec_conv3d_16l8_fp16.json"))["args"]; dcfg["use_fp16"]=False
with torch.device(DEV): dec = SparseStructureDecoder(**dcfg)
dec.load_state_dict({k:v.float() for k,v in load_file(M+"/tilarge/ckpts/ss_dec_conv3d_16l8_fp16.safetensors",device=DEV).items()},strict=False); dec.eval()
with torch.no_grad():
    logits = dec(z); occ = logits > 0
print("DECODE: logits min/max %.2f/%.2f  active voxels @res64 = %d / %d" % (float(logits.min()), float(logits.max()), int(occ.sum()), occ.numel()))
