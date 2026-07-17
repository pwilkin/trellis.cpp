#!/usr/bin/env python
"""Stitch 4x 2048 view PNGs into the showcase 4096x4096 quad grid."""
import sys
from PIL import Image
MV = "/devel/alt/trellis.cpp/tools/mv_preview"
for name in sys.argv[1:]:
    tiles = [Image.open(f"{MV}/{name}_{v}.png") for v in ["front", "right", "back", "left"]]
    S = tiles[0].size[0]
    assert all(t.size == (S, S) for t in tiles), [t.size for t in tiles]
    grid = Image.new("RGB", (S*2, S*2), (30, 30, 30))
    for i, t in enumerate(tiles):
        grid.paste(t.convert("RGB"), ((i % 2) * S, (i // 2) * S))
    out = f"{MV}/{name}_quad4k.png"
    grid.save(out)
    print(out, grid.size)
