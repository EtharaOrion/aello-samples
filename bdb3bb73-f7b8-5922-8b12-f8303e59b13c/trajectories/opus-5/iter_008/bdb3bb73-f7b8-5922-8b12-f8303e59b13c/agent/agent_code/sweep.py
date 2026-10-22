"""Run a list of configs in one process against the 45k/5k held-out split.

The test split plays no part here: every number this script prints comes from the
5,000-image tail of the (fixed-seed shuffled) TRAIN cache, which the 45k training
portion excludes.  Selection decisions are made on these numbers only.
"""
import argparse
import json
import sys
import time

import torch

import train as T


def base(**over):
    a = argparse.Namespace(**{
        "epochs": 18, "bs": 1000, "lr": 0.2, "wd": 2e-4, "bias_scaler": 8.0,
        "momentum": 0.85, "label_smoothing": 0.2, "widths": [128, 384, 512],
        "logit_scale": 1 / 9, "bn_momentum": 0.6, "warmup_frac": 0.23,
        "lr_start": 0.2, "lr_end": 0.07, "whiten_bias_epochs": 3, "seed": 0,
        "holdout": 5000, "ckpt_dir": "/workspace/scratch_ckpt", "no_save": True,
        "eval_every": 100, "eval_test": False, "tag": "", "json_out": "",
    })
    for k, v in over.items():
        setattr(a, k, v)
    return a


def run(cfgs, out_path=None):
    rows = []
    for cfg in cfgs:
        a = base(**cfg)
        t = time.time()
        r = T.main(a, verbose=False)
        row = dict(cfg)
        row["val"] = r["final_val"]
        row["sec"] = round(time.time() - t, 2)
        rows.append(row)
        print(json.dumps(row), flush=True)
    rows.sort(key=lambda r: -r["val"])
    print("BEST:", json.dumps(rows[0]), flush=True)
    if out_path:
        with open(out_path, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return rows


if __name__ == "__main__":
    cfgs = json.loads(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else "/workspace/sweep_log.jsonl"
    run(cfgs, out)
