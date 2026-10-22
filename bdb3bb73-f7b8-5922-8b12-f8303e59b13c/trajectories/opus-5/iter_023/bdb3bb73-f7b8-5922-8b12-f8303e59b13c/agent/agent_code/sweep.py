"""Hyperparameter sweep on a class-balanced 5k holdout carved from TRAIN.

Every candidate is trained on the same 45k images and scored on the same
5k holdout images.  The test split is never touched here.  Configs are run
inside one process so the (fixed) data-loading and warm-up costs are paid
once and do not distort the comparison.
"""
import argparse
import json
import sys

import torch

import airbench as ab


def run(candidates, epochs, seeds=(0,), holdout=True):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    x, y = ab.load_split_standardized("train")
    tx, ty, hx, hy = ab.stratified_holdout(x, y)
    ab.warmup(dict(ab.DEFAULTS))

    results = []
    for name, over in candidates:
        for seed in seeds:
            cfg = dict(ab.DEFAULTS)
            cfg.update(over)
            cfg["epochs"] = over.get("epochs", epochs)
            cfg["seed"] = seed
            hist, _, total = ab.train(cfg, tx, ty, eval_x=hx, eval_y=hy)
            rec = dict(name=name, seed=seed, acc=hist[-1]["acc"], sec=total,
                       sec_per_epoch=total / cfg["epochs"], epochs=cfg["epochs"],
                       curve=[round(h["acc"], 4) for h in hist], **over)
            results.append(rec)
            print(json.dumps(rec), flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help="JSON list of [name, overrides]")
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--seeds", default="0")
    a = ap.parse_args()
    grid = json.loads(a.grid)
    seeds = tuple(int(s) for s in a.seeds.split(","))
    res = run([(n, o) for n, o in grid], a.epochs, seeds)
    res.sort(key=lambda r: -r["acc"])
    print("\n=== ranked ===")
    for r in res:
        print(f"{r['acc']:.4f}  seed{r['seed']}  {r['name']}  "
              f"({r['sec_per_epoch']:.3f}s/ep)")
