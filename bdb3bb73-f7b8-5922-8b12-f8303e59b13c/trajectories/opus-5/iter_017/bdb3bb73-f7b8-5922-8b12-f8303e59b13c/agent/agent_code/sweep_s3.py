"""S3: one-at-a-time factor screen around the S2 winner. Held-out 45k/5k only."""
import json, sys
import numpy as np
sys.path.insert(0, "/workspace")
from sweep import run_grid, summarize

BASE = dict(scale=0.03125, epochs=32, lr=11.5, wd=0.0051)

VARIANTS = [
    {},                                  # base, for a within-batch reference
    dict(wd=0.0017), dict(wd=0.0034), dict(wd=0.0077),
    dict(lr_end=0.0), dict(lr_end=0.02),
    dict(cutout=6), dict(cutout=8), dict(cutout=10),
    dict(ema_decay=0.85), dict(ema_decay=0.95),
    dict(whiten_bias_epochs=8),
    dict(label_smoothing=0.1),
    dict(bn_momentum=0.8),
    dict(pad=3),
]

ovs = [dict(BASE, **v) for v in VARIANTS]
rows = run_grid("S3_factors", ovs, seeds=[0, 1], eval_every=8)
print("\n=== S3 summary (2 seeds; noise floor sd=0.35pp single-seed, 0.25pp 2-seed mean) ===")
summarize(rows)
print("DONE_S3", flush=True)
