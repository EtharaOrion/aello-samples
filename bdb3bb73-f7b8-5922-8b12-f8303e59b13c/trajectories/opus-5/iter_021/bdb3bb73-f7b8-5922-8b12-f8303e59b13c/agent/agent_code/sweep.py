"""Sweep driver: runs configs in-process against the fixed 45k/5k train holdout.

Usage: python sweep.py <plan-name>
Every result is appended to /workspace/sweep.jsonl.  The test split is never
loaded here.
"""
import copy
import json
import sys

import train

BASE = train.get_args.__wrapped__ if hasattr(train.get_args, "__wrapped__") else None


def base_cfg(**over):
    import argparse
    a = argparse.Namespace(
        epochs=32, bs=1000, lr=11.5, wd=0.0153, momentum=0.85, bias_scaler=64.0,
        label_smoothing=0.2, widths=[128, 384, 512], scale=1.0 / 9.0,
        bn_momentum=0.1, peak_frac=0.23, lr_start=0.2, lr_end=0.07, translate=2,
        tta_level=2, seed=0, holdout=5000, ckpt_dir="/workspace/scratch_ckpt",
        save_ckpts=0, eval_every=0, tag="base",
        out_json="/workspace/sweep.jsonl")
    for k, v in over.items():
        setattr(a, k, v)
    return a


PLANS = {
    # noise floor + incumbent multi-seed mean
    "floor": [dict(tag="incumbent", seed=s) for s in range(4)],
    # two-sided peak-LR probe (see DEVIATIONS in PREREGISTRATION.md)
    "lr": [dict(tag=f"lr{m}", lr=11.5 * m, seed=s)
           for m in (0.5, 0.7, 1.25) for s in (0, 1)],
    "lrend": [dict(tag="lrend0", lr_end=0.0, seed=s) for s in (0, 1)],
    "ls": [dict(tag="ls0.1", label_smoothing=0.1, seed=s) for s in (0, 1)],
    "wd": [dict(tag="wd1.5x", wd=0.0153 * 1.5, seed=s) for s in (0, 1)],
    "ep": [dict(tag=f"ep{e}", epochs=e, seed=s) for e in (24, 40) for s in (0, 1)],
    # --- round 2: everything below sits on the promoted lr_end=0 baseline ---
    "r2": ([dict(tag="lrend0", lr_end=0.0, seed=2)]
           + [dict(tag=f"E{e}", lr_end=0.0, epochs=e, seed=s)
              for e in (48, 64) for s in (0, 1)]
           + [dict(tag="E_lr0.5", lr_end=0.0, lr=5.75, seed=s) for s in (0, 1)]
           + [dict(tag="E_pf0.10", lr_end=0.0, peak_frac=0.10, seed=s)
              for s in (0, 1)]),
    # --- round 3: regularisation / capacity on the round-2 winner ---
    "r3": ([dict(tag="R_ls0.1", lr_end=0.0, epochs=32, label_smoothing=0.1, seed=s)
            for s in (0, 1)]
           + [dict(tag="R_wd1.5", lr_end=0.0, epochs=32, wd=0.02295, seed=s)
              for s in (0, 1)]
           + [dict(tag="R_tr3", lr_end=0.0, epochs=32, translate=3, seed=s)
              for s in (0, 1)]),
}


def main():
    for plan in sys.argv[1:]:
        for over in PLANS[plan]:
            cfg = base_cfg(**over)
            cfg.tag = f"{cfg.tag}"
            train.run(cfg)


if __name__ == "__main__":
    main()
