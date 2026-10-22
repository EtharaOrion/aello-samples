"""Per-epoch holdout curves. Answers a robustness question the final-accuracy
number cannot: how many *trailing* checkpoints sit above the 94% line.

A ladder whose only crossing checkpoint is the last one is one bad seed away
from scoring zero, so the schedule tail matters as much as the peak.
Holdout only -- the test split is not read here.
"""
import json, sys, time
sys.path.insert(0, "/workspace")
from train import run, DEFAULTS

W160 = (160, 448, 640)
CASES = [
    ("base_E22_end07",  dict(epochs=22)),
    ("W160_E18_end07",  dict(epochs=18, widths=W160)),
    ("W160_E18_end02",  dict(epochs=18, widths=W160, lr_end=0.02)),
    ("W160_E22_end02",  dict(epochs=22, widths=W160, lr_end=0.02)),
    ("base_E26_end02",  dict(epochs=26, lr_end=0.02)),
]
out = []
for name, ov in CASES:
    for seed in (0, 1):
        cfg = dict(DEFAULTS); cfg.update(ov); cfg["seed"] = seed
        _, r = run(cfg, "/tmp/none", holdout=5000, save_ladder=False,
                   verbose=False, per_epoch_eval=True)
        c = r["holdout_curve"]
        rec = dict(name=name, seed=seed, curve=c, final=c[-1],
                   epoch_sec=r["mean_epoch_sec"],
                   n_within_0p3=sum(1 for v in c if v >= c[-1] - 0.003))
        out.append(rec)
        print(f"{name:16s} s{seed} final={c[-1]:.4f} tail={[round(v,4) for v in c[-5:]]} "
              f"{r['mean_epoch_sec']:.3f}s/ep", flush=True)
# Written only on full completion. This session's run was stopped early once
# the tail shape was clear, so the per-epoch curves live in experiments/curve.log.
json.dump(out, open("/workspace/experiments/curve_study.json", "w"), indent=2)
print("DONE", flush=True)
