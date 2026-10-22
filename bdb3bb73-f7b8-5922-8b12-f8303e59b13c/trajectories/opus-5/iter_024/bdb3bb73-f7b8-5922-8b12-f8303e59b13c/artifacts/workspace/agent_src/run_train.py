"""Run one full training job and write a checkpoint ladder.

    python run_train.py --outdir /workspace/scratch_ckpt --epochs 20 --seed 0

Timing reported here is total process wall-clock (data load + whitening +
training + every checkpoint save), which is >= the span the verifier measures
between ckpt_epoch_0001 and the last checkpoint.
"""
import argparse
import json
import os
import shutil
import sys
import time

T_PROC_START = time.time()

import torch

from speedrun.data import TrainData, load_split
from speedrun.train import Config, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--label-smoothing", type=float, default=None)
    ap.add_argument("--lr-final-frac", type=float, default=None)
    ap.add_argument("--precision", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--clean", action="store_true",
                    help="wipe outdir first so the ladder is from exactly one run")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = Config(epochs=args.epochs, seed=args.seed)
    for k in ("lr", "weight_decay", "label_smoothing", "lr_final_frac", "precision"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)

    if args.clean and os.path.isdir(args.outdir):
        shutil.rmtree(args.outdir)
    os.makedirs(args.outdir, exist_ok=True)

    x, y = load_split("train")
    # Warm up cudnn autotuning / kernel JIT so that epoch 1 is not an outlier.
    from speedrun.train import build_model
    warm = build_model(cfg, x)
    wd = TrainData(x[:2000], y[:2000])
    g = torch.Generator(device="cuda"); g.manual_seed(0)
    for _ in range(2):
        xb, yb = wd.batch(torch.arange(1000, device="cuda"), g)
        warm(xb).sum().backward()
    del warm, wd
    torch.cuda.synchronize()

    out = train(cfg, x, y, outdir=args.outdir)
    out["wall_clock_sec"] = time.time() - T_PROC_START
    out["train_only_sec"] = out["records"][-1]["t"]
    print(json.dumps({k: v for k, v in out.items() if k != "records"}, indent=2))
    if args.log:
        with open(args.log, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
