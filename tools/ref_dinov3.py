#!/usr/bin/env python3
"""Golden dump for DINOv3 ViT-L/16 conditioning features (the TRELLIS image cond).

TRELLIS uses transformers DINOv3ViTModel (gated). We use the timm mirror weights
(same backbone) and replicate TRELLIS's extract_features: resize 512 LANCZOS,
ImageNet-normalize, run the ViT, take the PRE-final-norm hidden states, then a
manual non-affine F.layer_norm. Cross-attention over cond is permutation-invariant,
so token order is irrelevant to the pipeline (we just match our own GGML to this dump).

    /media/ilintar/D_SSD/trellis2-venv/bin/python tools/ref_dinov3.py [image.png]
"""
import os, sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
import timm

W = "/media/ilintar/D_SSD/models/trellis2/dinov3/model.safetensors"
OUT = "/media/ilintar/D_SSD/models/trellis2/ref/dinov3"; os.makedirs(OUT, exist_ok=True)
DEV = os.environ.get("REF_DEV", "cuda:1")
IMG = sys.argv[1] if len(sys.argv) > 1 else "/devel/alt/trellis.cpp/assets/goblin.png"
SIZE = 512

m = timm.create_model("vit_large_patch16_dinov3", pretrained=False, num_classes=0, img_size=SIZE)
missing, unexpected = m.load_state_dict(load_file(W), strict=False)
print(f"timm load: missing={len(missing)} unexpected={len(unexpected)} prefix_tokens={getattr(m,'num_prefix_tokens',None)}")
if missing:    print("  missing   :", missing[:8])
if unexpected: print("  unexpected:", unexpected[:8])
m.norm = nn.Identity()           # we apply our own final LN (pre-norm hidden states)
m.eval().to(DEV)

# preprocess exactly like TRELLIS
img = Image.open(IMG).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
x = torch.from_numpy(np.array(img).astype(np.float32) / 255).permute(2, 0, 1)[None].to(DEV)
mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
x = (x - mean) / std

with torch.no_grad():
    tok = m.forward_features(x)                      # [1, N, 1024] pre-final-norm
    cond = F.layer_norm(tok, tok.shape[-1:])         # manual non-affine LN

a = np.ascontiguousarray(cond.float().cpu().numpy())
print("cond", list(a.shape), "mean=%.5f std=%.5f" % (a.mean(), a.std()))
np.save(f"{OUT}/cond.npy", a)
# also dump the preprocessed input [1,3,512,512] for the C++ side to consume identically
np.save(f"{OUT}/input_chw.npy", np.ascontiguousarray(x.float().cpu().numpy()))
print("DONE ->", OUT)
