"""Graded training run: full 50k CIFAR-10 train split, one scripted checkpoint
per epoch.

    python train.py --out /workspace/periodic_checkpoints [--epochs N ...]

Every epoch is exactly one pass over the 50k train images.  No test data is
read anywhere in this file or in lib.py.
"""
import argparse
import json
import os
import shutil
import time
from dataclasses import replace

import torch

import lib
from lib import Cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/periodic_checkpoints")
    ap.add_argument("--final", default=None,
                    help="also hardlink/copy the last checkpoint here")
    ap.add_argument("--epochs", type=int, default=Cfg.epochs)
    ap.add_argument("--lr", type=float, default=Cfg.lr)
    ap.add_argument("--bias-mult", type=float, default=Cfg.bias_mult)
    ap.add_argument("--wd", type=float, default=Cfg.wd)
    ap.add_argument("--label-smoothing", type=float, default=Cfg.label_smoothing)
    ap.add_argument("--tta-level", type=int, default=Cfg.tta_level)
    ap.add_argument("--seed", type=int, default=Cfg.seed)
    ap.add_argument("--metrics", default=None)
    a = ap.parse_args()

    # t0 = start of the training run, as reported in agent_reported_metrics.json
    t0 = time.time()

    cfg = replace(Cfg(), epochs=a.epochs, lr=a.lr, bias_mult=a.bias_mult, wd=a.wd,
                  label_smoothing=a.label_smoothing, tta_level=a.tta_level,
                  seed=a.seed)
    print("config:", cfg)

    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(a.out):                 # exactly one run's ladder
        if f.startswith("ckpt_epoch_") and f.endswith(".pt"):
            os.remove(os.path.join(a.out, f))

    xpad, y = lib.load_train()
    print(f"train tensor {tuple(xpad.shape)} on {xpad.device}")

    r = lib.train(cfg, xpad, y, ckpt_dir=a.out,
                  eval_fn=None, log=print)
    torch.cuda.synchronize()

    n = cfg.epochs
    last = os.path.join(a.out, f"ckpt_epoch_{n:04d}.pt")
    if a.final:
        if os.path.exists(a.final):
            os.remove(a.final)
        shutil.copy(last, a.final)              # copy: lets save() own the mtimes

    wall = time.time() - t0
    per_epoch = [round(t, 4) for t in r["epoch_end"]]
    print(f"train loop {r['total']:.2f}s | total run wall clock {wall:.2f}s")
    print(f"epoch wall times: {[round(b-a_, 3) for a_, b in zip([0]+per_epoch, per_epoch)]}")

    if a.metrics:
        with open(a.metrics, "w") as f:
            json.dump(dict(wall_clock_sec=wall, epochs=n,
                           epoch_end_offsets=per_epoch), f, indent=1)


if __name__ == "__main__":
    main()
