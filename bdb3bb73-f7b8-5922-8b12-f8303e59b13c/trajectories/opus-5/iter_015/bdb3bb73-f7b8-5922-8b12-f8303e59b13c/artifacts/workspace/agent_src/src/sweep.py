"""Hyperparameter search on a 5,000-image holdout carved out of the TRAIN split.

Nothing in here ever reads the test split.  Each config is run on `--seeds` seeds and
scored by the mean holdout accuracy, so that a single lucky seed cannot win.
"""
import argparse
import json
import statistics
import sys
import time

import torch

from train import DEFAULT_CFG, run


def eval_cfg(overrides, seeds, holdout=5000, track_val=True, log=None):
    accs, crosses, times = [], [], []
    cfg = dict(DEFAULT_CFG)
    cfg.update(overrides)
    for s in seeds:
        res, _, _ = run(cfg, "/workspace/scratch_ckpt", seed=s, holdout=holdout,
                        save_ckpts=False, quiet=True, track_val=track_val)
        accs.append(res["val_acc"])
        times.append(res["total_time"])
        if track_val:
            crosses.append(res.get("val_cross_epoch"))
        del res
        torch.cuda.empty_cache()
    out = dict(overrides=overrides, seeds=seeds, accs=accs,
               mean=statistics.mean(accs),
               sd=statistics.stdev(accs) if len(accs) > 1 else 0.0,
               cross=crosses, sec=statistics.mean(times))
    print(json.dumps(out), flush=True)
    if log:
        with open(log, "a") as f:
            f.write(json.dumps(out) + "\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help="JSON list of override dicts")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--log", default="/workspace/logs/sweep.jsonl")
    ap.add_argument("--no-track", action="store_true")
    args = ap.parse_args()

    grid = json.loads(args.grid)
    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()
    results = [eval_cfg(o, seeds, track_val=not args.no_track, log=args.log) for o in grid]
    results.sort(key=lambda r: -r["mean"])
    print("\n=== ranked ===", flush=True)
    for r in results:
        print(f"{r['mean']:.4f} +-{r['sd']:.4f}  cross={r['cross']}  "
              f"{r['sec']:.1f}s  {json.dumps(r['overrides'])}", flush=True)
    print(f"sweep wall-clock {time.time()-t0:.0f}s", flush=True)
