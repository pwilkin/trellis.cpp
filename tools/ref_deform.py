#!/usr/bin/env python3
"""Reference for the custom modulated deformable conv: random x/offset/mask/weight, run
torchvision.ops.deform_conv2d, dump inputs + output for the C++ kernel test.
Usage: ref_deform.py <out_dir> [K] [Cin] [Cout] [H]"""
import sys, os
import numpy as np
import torch
from torchvision.ops import deform_conv2d

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
K   = int(sys.argv[2]) if len(sys.argv) > 2 else 3
Cin = int(sys.argv[3]) if len(sys.argv) > 3 else 8
Cout= int(sys.argv[4]) if len(sys.argv) > 4 else 16
H = W = int(sys.argv[5]) if len(sys.argv) > 5 else 12
pad = K // 2
g = torch.Generator().manual_seed(0)
x      = torch.randn(1, Cin, H, W, generator=g)
weight = torch.randn(Cout, Cin, K, K, generator=g) * 0.1
bias   = torch.randn(Cout, generator=g)
offset = torch.randn(1, 2*K*K, H, W, generator=g) * 1.5     # real offsets
mask   = 2.0 * torch.sigmoid(torch.randn(1, K*K, H, W, generator=g))
y = deform_conv2d(x, offset, weight, bias, stride=1, padding=pad, mask=mask)
print("out", tuple(y.shape), "mean", float(y.mean()), "std", float(y.std()))
np.save(f"{out}/x.npy",      x[0].numpy())
np.save(f"{out}/offset.npy", offset[0].numpy())
np.save(f"{out}/mask.npy",   mask[0].numpy())
np.save(f"{out}/weight.npy", weight.numpy())
np.save(f"{out}/bias.npy",   bias.numpy())
np.save(f"{out}/out.npy",    y[0].numpy())
with open(f"{out}/dims.txt", "w") as f: f.write(f"{Cin} {Cout} {K} {H} {W}\n")
print("dumped ->", out)
