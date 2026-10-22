"""Stages S0-S2 of the pre-registered protocol. Held-out 45k/5k only; no test reads."""
import json, sys
import numpy as np
sys.path.insert(0, "/workspace")
from sweep import run_grid, summarize

BASE = dict(scale=0.03125)

# ---- S0: noise floor -------------------------------------------------------
print("=== S0 noise floor (epochs=24, 5 seeds) ===", flush=True)
r0 = run_grid("S0_noise", [dict(BASE, epochs=24)], seeds=[0, 1, 2, 3, 4], eval_every=6)
v = np.array([r["final"] for r in r0])
sd = v.std(ddof=1)
print(f"S0: mean={v.mean():.4f} sd={sd:.4f} (pp={sd*100:.2f})  "
      f"sd_of_3seed_mean={sd/np.sqrt(3)*100:.2f}pp  "
      f"min={v.min():.4f} max={v.max():.4f}", flush=True)

# ---- S1: epoch budget ------------------------------------------------------
print("\n=== S1 epochs ===", flush=True)
r1 = run_grid("S1_epochs", [dict(BASE, epochs=e) for e in [16, 24, 32, 40]],
              seeds=[0, 1], eval_every=4)
summarize(r1)

# ---- S2: lr x weight decay at 32 epochs ------------------------------------
print("\n=== S2 lr x wd (epochs=32, 1 seed screen) ===", flush=True)
ovs = [dict(BASE, epochs=32, lr=lr, wd=wd)
       for lr in [5.75, 11.5, 17.0] for wd in [0.0051, 0.0153, 0.0306]]
r2 = run_grid("S2_lr_wd", ovs, seeds=[0], eval_every=8)
summarize(r2)
print("DONE_S2", flush=True)
