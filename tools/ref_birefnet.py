#!/usr/bin/env python3
"""BiRefNet PyTorch reference: load the local model, print its true module tree, run on an image,
dump the alpha matte + intermediate tensor shapes/stats for GGML parity validation.
Usage: ref_birefnet.py <image.png> [out_dir]   (REF_DEV env picks cuda device)"""
import os, sys, types, json, importlib.util
import numpy as np

BIREF = "/media/ilintar/D_SSD/models/trellis2/birefnet"

# kornia is only used by the training-time laplacian; stub it so import works for inference
kf = types.ModuleType("kornia.filters"); kf.laplacian = lambda *a, **k: a[0]
kmod = types.ModuleType("kornia"); kmod.filters = kf
sys.modules.setdefault("kornia", kmod); sys.modules.setdefault("kornia.filters", kf)

import torch, torch.nn.functional as F
from safetensors.torch import load_file

# birefnet.py uses a relative import (from .BiRefNet_config) -> load as a synthetic package
_pkg = types.ModuleType("bn_pkg"); _pkg.__path__ = [BIREF]; sys.modules["bn_pkg"] = _pkg
for _name in ("BiRefNet_config", "birefnet"):
    _spec = importlib.util.spec_from_file_location("bn_pkg." + _name, os.path.join(BIREF, _name + ".py"))
    _m = importlib.util.module_from_spec(_spec); sys.modules["bn_pkg." + _name] = _m; _spec.loader.exec_module(_m)
B = sys.modules["bn_pkg.birefnet"]

dev = os.environ.get("REF_DEV", "cuda:0")
img_path = sys.argv[1]
out_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/birefnet_ref"
os.makedirs(out_dir, exist_ok=True)

# --- build model architecture (no backbone pretrained dl), load full weights ---
cfg = B.BiRefNetConfig(bb_pretrained=False)
model = B.BiRefNet(bb_pretrained=False, config=cfg)
sd = load_file(os.path.join(BIREF, "model.safetensors"))
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
if missing[:5]:    print("  missing[:5]:", missing[:5])
if unexpected[:5]: print("  unexpected[:5]:", unexpected[:5])
model = model.to(dev).eval()

# --- dump the true module tree ---
with open(os.path.join(out_dir, "model_tree.txt"), "w") as f:
    f.write(str(model))
print("module tree ->", os.path.join(out_dir, "model_tree.txt"))
c = model.config   # internal Config() with the architecture flags
print("cfg: bb=%r mul_scl_ipt=%r squeeze_block=%r dec_att=%r cxt_num=%r cxt=%r size=%d lateral=%r" % (
    c.bb, c.mul_scl_ipt, c.squeeze_block, c.dec_att, c.cxt_num, c.cxt, c.size,
    c.lateral_channels_in_collection))

# --- preprocess: resize to 1024, ImageNet normalize ---
from PIL import Image
im = Image.open(img_path).convert("RGB").resize((1024, 1024), Image.BILINEAR)
x = torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0).permute(2, 0, 1)[None]
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
x = ((x - mean) / std).to(dev)
print("input", tuple(x.shape), "range", float(x.min()), float(x.max()))

# --- hook intermediate shapes ---
def stat(t): return f"shape={tuple(t.shape)} mean={t.float().mean().item():.4f} std={t.float().std().item():.4f} min={t.float().min().item():.4f} max={t.float().max().item():.4f}"

with torch.no_grad():
    (x1, x2, x3, x4), _ = model.forward_enc(x)
    print("\n[encoder, post-cxt-cat on x4]")
    for n, t in [("x1", x1), ("x2", x2), ("x3", x3), ("x4", x4)]:
        print(f"  {n}: {stat(t)}")
    sq = model.squeeze_module(x4)
    print("[squeeze]:", stat(sq))
    preds = model(x)   # inference path -> scaled_preds list
    if isinstance(preds, (list, tuple)):
        print("\n[decoder scaled_preds]:", len(preds), "outputs")
        for i, p in enumerate(preds):
            print(f"  pred[{i}]: {stat(p)}")
        final = preds[-1]
    else:
        final = preds
    alpha = torch.sigmoid(final)[0, 0].cpu().numpy()

# --- optional: dump backbone (single-scale 1024) intermediates for C++ parity ---
if os.environ.get("REF_DUMP"):
    dd = os.path.join(out_dir, "dump"); os.makedirs(dd, exist_ok=True)
    with torch.no_grad():
        pe = model.bb.patch_embed(x)                       # [1,192,256,256]
        outs = model.bb(x)                                 # single-scale tuple (4 stage outs, post-norm)
    np.save(os.path.join(dd, "patch_embed.npy"), pe[0].float().cpu().numpy())
    for i, o in enumerate(outs):
        np.save(os.path.join(dd, f"bb_out{i}.npy"), o[0].float().cpu().numpy())
        print(f"  dump bb_out{i}: {tuple(o.shape)}")
    # also dump the half-scale (512) outs + the final doubled x1..x4 (post cxt-cat for x4)
    with torch.no_grad():
        outs_h = model.bb(F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=True))
    for i, o in enumerate(outs_h):
        np.save(os.path.join(dd, f"bb_half_out{i}.npy"), o[0].float().cpu().numpy())
    for n, t in [("x1", x1), ("x2", x2), ("x3", x3), ("x4_cxt", x4)]:
        np.save(os.path.join(dd, f"{n}.npy"), t[0].float().cpu().numpy())
    np.save(os.path.join(dd, "squeeze.npy"), sq[0].float().cpu().numpy())
    np.save(os.path.join(dd, "logits.npy"), final[0].float().cpu().numpy())
    # input after preprocess (for the C++ to start from the identical tensor)
    np.save(os.path.join(dd, "input.npy"), x[0].float().cpu().numpy())
    print("dumped backbone intermediates ->", dd)

print("\nalpha:", "shape", alpha.shape, "min", float(alpha.min()), "max", float(alpha.max()), "mean", float(alpha.mean()))
Image.fromarray((alpha * 255).astype(np.uint8)).save(os.path.join(out_dir, "alpha.png"))
# also dump the cutout (premultiplied) for visual check
rgb = np.asarray(im).astype(np.float32) / 255.0
cut = (rgb * alpha[..., None] * 255).astype(np.uint8)
Image.fromarray(cut).save(os.path.join(out_dir, "cutout.png"))
np.save(os.path.join(out_dir, "alpha.npy"), alpha)
print("saved alpha + cutout ->", out_dir)
