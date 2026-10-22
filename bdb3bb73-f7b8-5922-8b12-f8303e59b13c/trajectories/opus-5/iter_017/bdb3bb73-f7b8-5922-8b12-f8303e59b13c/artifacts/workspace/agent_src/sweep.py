"""Held-out sweep harness. Trains on the 45k slice, scores on the 5k val slice.

NEVER reads the test split. Results are appended to /workspace/artifacts/<tag>.json.
"""
import argparse, itertools, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, "/workspace")
from train import (DEFAULT_CFG, load_split, standardize, stratified_holdout, train_run)

ART = "/workspace/artifacts"


def get_split(dev="cuda", full=False):
    xtr_u8, ytr = load_split("train", dev)
    if full:
        return xtr_u8, ytr, None, None
    tr, va = stratified_holdout(ytr, 500, seed=0)
    xv = standardize(xtr_u8[va]).contiguous(memory_format=torch.channels_last)
    return xtr_u8[tr], ytr[tr], xv, ytr[va]


def run_grid(name, overrides_list, seeds, eval_every=1, quiet=True):
    xt, yt, xv, yv = get_split()
    rows = []
    for ov in overrides_list:
        for sd in seeds:
            cfg = dict(DEFAULT_CFG); cfg.update(ov)
            t0 = time.time()
            r = train_run(cfg, seed=sd, val_data=(xv, yv), x_tr=xt, y_tr=yt,
                          eval_every=eval_every, quiet=quiet)
            curve = {h["epoch"]: h["val"] for h in r["hist"] if h["val"] is not None}
            row = dict(name=name, ov=ov, seed=sd, final=r["hist"][-1]["val"],
                       epochs=cfg["epochs"], wall=r["wall"], curve=curve,
                       sec_per_epoch=r["wall"] / cfg["epochs"])
            rows.append(row)
            print(f"{name} {ov} seed={sd} final={row['final']:.4f} "
                  f"({r['wall']:.1f}s, {row['sec_per_epoch']:.3f}s/ep)", flush=True)
    os.makedirs(ART, exist_ok=True)
    path = f"{ART}/sweep_{name}.json"
    with open(path, "w") as f:
        json.dump(rows, f, indent=1)
    return rows


def summarize(rows):
    """Group by override dict, report mean/sd of final acc across seeds."""
    byk = {}
    for r in rows:
        byk.setdefault(json.dumps(r["ov"], sort_keys=True), []).append(r["final"])
    out = []
    for k, v in byk.items():
        v = np.array(v)
        out.append((json.loads(k), v.mean(), v.std(ddof=1) if len(v) > 1 else float("nan"), len(v)))
    out.sort(key=lambda t: -t[1])
    for ov, m, s, n in out:
        print(f"  mean={m:.4f} sd={s:.4f} n={n}  {ov}")
    return out
