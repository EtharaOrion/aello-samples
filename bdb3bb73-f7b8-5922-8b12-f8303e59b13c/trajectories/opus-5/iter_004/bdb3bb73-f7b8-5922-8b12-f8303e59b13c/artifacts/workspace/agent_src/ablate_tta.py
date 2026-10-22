"""Paired TTA ablation: train once per seed, evaluate the *same* weights at each
TTA level on the held-out train slice. Pairing removes training noise, so the
differences here are attributable to the eval-time transform alone.

TTA level 0 = plain forward, 1 = + horizontal mirror, 2 = + two 1-px diagonal
translations of both the image and its mirror (6 forward passes).
"""

import json

import numpy as np
import torch

import speedrun as S
from sweep import holdout_acc, make_split

CFG = dict(widths=(128, 384, 512), lr=14.0, epochs=24)
SEEDS = [0, 1, 2]

tx, ty, hx, hy = make_split()
rows = {0: [], 1: [], 2: []}
for s in SEEDS:
    net, _ = S.train(dict(CFG, seed=s), tx, ty, log=lambda *a: None)
    for lvl in (0, 1, 2):
        rows[lvl].append(holdout_acc(net, hx, hy, lvl))
    print(f"seed {s}: " + "  ".join(f"tta{l}={rows[l][-1]*100:.2f}%" for l in (0, 1, 2)), flush=True)

print("\nmean over seeds (paired, same weights per row):")
base = np.mean(rows[0])
for lvl in (0, 1, 2):
    m = np.mean(rows[lvl])
    print(f"  tta{lvl}: {m*100:6.2f}%  ({(m-base)*100:+.2f} vs tta0)")
json.dump({str(k): v for k, v in rows.items()}, open("/workspace/ablate_tta.json", "w"), indent=2)
