#!/usr/bin/env python3
"""Golden-tensor dump for the SS-flow DiT, to validate the C++/GGML port.

Runs the real SparseStructureFlowModel (sdpa attention backend) on a fixed,
seeded input and saves inputs + per-stage intermediates to .npy. Runs on GPU
(weights load straight to VRAM) to keep host RAM low — never a CPU forward of
the 1.3B net.

    /media/ilintar/D_SSD/trellis2-venv/bin/python tools/ref_ss_flow.py
"""
import os, sys, json
os.environ["ATTN_BACKEND"] = "sdpa"          # must precede trellis2 import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/tmp/TRELLIS.2")
import numpy as np
import torch
from safetensors.torch import load_file
from trellis2.models.sparse_structure_flow import SparseStructureFlowModel

CKPT = "/media/ilintar/D_SSD/models/trellis2/ckpts/ss_flow_img_dit_1_3B_64_bf16"
OUT  = "/media/ilintar/D_SSD/models/trellis2/ref/ss_flow"
os.makedirs(OUT, exist_ok=True)
DEV  = os.environ.get("REF_DEV", "cuda:1")     # 5060 Ti has the most free VRAM
DTYPE = torch.float32                          # clean f32 golden

def save(name, t):
    a = np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy())
    np.save(f"{OUT}/{name}.npy", a)
    print(f"  {name:22s} {str(list(a.shape)):20s} mean={a.mean():.5f} std={a.std():.5f} absmax={np.abs(a).max():.5f}")

cfg = json.load(open(CKPT + ".json"))["args"]
print("config:", {k: cfg[k] for k in ("resolution","in_channels","model_channels","num_blocks","num_heads")})

with torch.device(DEV):
    model = SparseStructureFlowModel(**cfg)    # params allocate directly on GPU
model.convert_to(DTYPE)                         # undo the bf16 torso -> f32
sd = load_file(CKPT + ".safetensors", device=DEV)   # bf16 on GPU; copy_ casts to f32
missing, unexpected = model.load_state_dict(sd, strict=False)
del sd; torch.cuda.empty_cache()
print(f"loaded: missing={len(missing)} unexpected={len(unexpected)}")
model.eval()

# deterministic inputs
g = torch.Generator(device=DEV).manual_seed(42)
B, R, Cin = 1, cfg["resolution"], cfg["in_channels"]
Lc = 1029
x    = torch.randn(B, Cin, R, R, R, generator=g, device=DEV, dtype=DTYPE)
t    = torch.tensor([float(os.environ.get("TVAL","500"))], device=DEV, dtype=DTYPE)
cond = torch.randn(B, Lc, cfg["cond_channels"], generator=g, device=DEV, dtype=DTYPE) * 0.5
if os.environ.get("ZEROCOND"): cond = torch.zeros_like(cond)

print("\ninputs:")
save("input_x", x); save("input_t", t); save("input_cond", cond)
ph = model.rope_phases                          # [4096,64] complex
save("rope_cos", torch.view_as_real(ph)[..., 0])
save("rope_sin", torch.view_as_real(ph)[..., 1])

caps = {}
hk = lambda n: (lambda m, i, o: caps.__setitem__(n, o.detach()))
hkin = lambda n: (lambda m, i: caps.__setitem__(n, i[0].detach()))
model.input_layer.register_forward_hook(hk("after_input_layer"))
model.adaLN_modulation.register_forward_hook(hk("t_emb_mod"))
model.blocks[0].register_forward_hook(hk("after_block0"))
model.blocks[1].register_forward_hook(hk("after_block1"))
model.blocks[-1].register_forward_hook(hk("after_block29"))
model.out_layer.register_forward_pre_hook(hkin("prefinal"))   # after final F.layer_norm

print("\nrunning forward on", DEV, "...")
with torch.no_grad():
    out = model(x, t, cond)

print("\nintermediates:")
for k in ("after_input_layer","t_emb_mod","after_block0","after_block1","after_block29","prefinal"):
    save(k, caps[k])
save("output", out)
print("\nDONE ->", OUT)
