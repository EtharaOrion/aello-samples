"""Holdout sweep driver.

Runs many configs inside one process so CUDA init and the inductor cache are
paid once. Every number produced here comes from the 5k held-out slice of the
*train* split; the test split is not read by this file.
"""
import argparse
import itertools
import json
import statistics
import sys
import time

import torch

from train import DEFAULT, train


def run(overrides, seeds, tag, log):
    cfg = dict(DEFAULT)
    cfg.update(overrides)
    accs, walls = [], []
    for s in seeds:
        t0 = time.time()
        res, _ = train(cfg, "holdout", "/workspace/scratch_ckpt", None,
                       eval_val=True, seed=s, save_ladder=False)
        accs.append(res["val_acc"])
        walls.append(res["wall"])
        torch.cuda.empty_cache()
    rec = dict(tag=tag, overrides=overrides, seeds=seeds, accs=accs,
               mean=statistics.mean(accs), wall=statistics.mean(walls),
               sd=(statistics.stdev(accs) if len(accs) > 1 else 0.0),
               epoch_s=statistics.mean(walls) / cfg["epochs"])
    with open(log, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"{tag:34s} acc {rec['mean'] * 100:6.2f}%"
          + (f" +/-{rec['sd'] * 100:4.2f}" if len(accs) > 1 else "     ")
          + f"  {rec['epoch_s'] * 1000:5.0f} ms/ep  {json.dumps(overrides)}",
          flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="json list of {tag, seeds, **overrides}")
    ap.add_argument("--log", default="/workspace/logs/sweep.jsonl")
    args = ap.parse_args()
    plan = json.loads(args.plan)
    out = []
    for item in plan:
        item = dict(item)
        tag = item.pop("tag")
        seeds = item.pop("seeds", [0])
        out.append(run(item, seeds, tag, args.log))
    best = max(out, key=lambda r: r["mean"])
    print(f"\nBEST: {best['tag']} {best['mean'] * 100:.2f}% {json.dumps(best['overrides'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
