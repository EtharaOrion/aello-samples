"""In-process sweep driver.

Runs a list of config overrides through train.train() inside a single Python
process (saves ~3 s of interpreter + CUDA startup per config) and appends every
result to a JSONL log.  All runs use --split val, so the CIFAR-10 test split is
never touched during hyperparameter selection.
"""
import argparse
import copy
import itertools
import json
import sys
import time
import types

import train as T

BASE = dict(epochs=16, bs=1024, lr=0.4, wd=0.05, momentum=0.85, label_smoothing=0.2,
            warmup=0.15, final_lr_frac=0.02, logit_scale=0.11, widths=[128, 384, 512],
            pad=2, cutout=0, seed=0, split="val", ckpt_dir="", eval_last=1,
            eval_every=False, tag="", verbose=False)


def run(overrides, log_path):
    cfg = types.SimpleNamespace(**{**BASE, **overrides})
    t0 = time.perf_counter()
    _, history, wall = T.train(cfg)
    rec = {"cfg": {k: v for k, v in overrides.items()},
           "val": history[-1].get("val"), "wall": wall,
           "sec_per_epoch": wall / cfg.epochs, "total": time.perf_counter() - t0}
    print(json.dumps(rec), flush=True)
    with open(log_path, "a") as f:
        f.write(json.dumps({**rec, "full_cfg": vars(cfg)}) + "\n")
    return rec


def expand(grid):
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help="JSON dict of lists, or JSON list of dicts")
    ap.add_argument("--log", default="/workspace/sweeps.jsonl")
    a = ap.parse_args()
    spec = json.loads(a.grid)
    configs = list(expand(spec)) if isinstance(spec, dict) else spec
    print(f"# {len(configs)} configs", flush=True)
    for c in configs:
        run(c, a.log)
