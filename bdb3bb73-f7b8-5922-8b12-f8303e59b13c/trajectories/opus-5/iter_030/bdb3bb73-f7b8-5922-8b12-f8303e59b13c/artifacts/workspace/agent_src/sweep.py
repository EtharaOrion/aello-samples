#!/usr/bin/env python
"""Sweep driver: runs configs in one process so CUDA/cuDNN warmup is paid once.

Every config is scored on the 5,000-image holdout carved out of the *train*
split (see PREREGISTRATION.md). Results append to a jsonl so a crash mid-sweep
never loses completed runs.
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

import train as T

BASE = dict(
    epochs=20, batch=1000, lr=11.5, wd=0.0153, momentum=0.85, bias_scaler=64.0,
    smooth=0.2, scale=1 / 9, bn_mom=0.6, widths=[128, 384, 512], translate=2,
    cutout=0, tta=2, warm_frac=0.23, hold_frac=0.0, lo_start=0.2, lo_end=0.07,
    whiten_n=5000, freeze_whiten_bias=0, warmup=6, seed=0, holdout=5000,
    eval_each=1, ckpt_dir="/workspace/scratch_ckpt", final_copy="", out="",
    tag="base",
)


def run(over, seed, log):
    cfg = dict(BASE)
    cfg.update(over)
    cfg["seed"] = seed
    a = argparse.Namespace(**cfg)
    t0 = time.time()
    res = T.main(a)
    torch.cuda.empty_cache()
    rec = dict(
        tag=cfg["tag"], seed=seed, over={k: v for k, v in over.items() if k != "tag"},
        acc=res["holdout_acc"], per_epoch=res["per_epoch_sec"],
        curve=[r.get("acc") for r in res["hist"]], proc_sec=time.time() - t0,
    )
    with open(log, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"  [{cfg['tag']} s{seed}] acc={rec['acc']:.4f} "
          f"per_ep={rec['per_epoch']:.3f}s ({rec['proc_sec']:.0f}s)", flush=True)
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--log", default="/workspace/logs/sweep.jsonl")
    p.add_argument("--deadline", type=float, default=1e18)
    a = p.parse_args()

    stages = {
        # Stage A: noise floor. Identical config, different seeds.
        "noise": [dict(tag="base")],
        # Stage B: one-change-at-a-time screening.
        "screen": [
            dict(tag="H1_loend02", lo_end=0.02),
            dict(tag="H2_cutout8", cutout=8),
            dict(tag="H2_cutout12", cutout=12),
            dict(tag="H3_trans4", translate=4),
            dict(tag="H4_hold15", hold_frac=0.15),
            dict(tag="H5_smooth10", smooth=0.1),
            dict(tag="H5_smooth30", smooth=0.3),
            dict(tag="H6_frzwb", freeze_whiten_bias=1),
            dict(tag="H7_wide", widths=[160, 448, 512]),
            dict(tag="H8_ep24", epochs=24),
        ],
    }
    if a.stage in stages:
        jobs = stages[a.stage]
    else:  # ad-hoc: JSON list of overrides on the command line
        jobs = json.loads(a.stage)

    groups = {}
    for job in jobs:
        for s in a.seeds:
            if time.time() > a.deadline:
                print("DEADLINE reached, stopping sweep", flush=True)
                break
            r = run(job, s, a.log)
            groups.setdefault(job["tag"], []).append(r["acc"])

    print("\n=== summary ===", flush=True)
    for tag, accs in groups.items():
        m = statistics.mean(accs)
        sd = statistics.stdev(accs) if len(accs) > 1 else float("nan")
        print(f"{tag:18s} n={len(accs)} mean={m*100:.2f}  sd={sd*100:.2f}", flush=True)


if __name__ == "__main__":
    main()
