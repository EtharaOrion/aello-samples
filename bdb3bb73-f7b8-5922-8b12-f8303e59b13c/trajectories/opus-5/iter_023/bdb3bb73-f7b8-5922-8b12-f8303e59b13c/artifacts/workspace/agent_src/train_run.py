"""Train one model end-to-end and write a per-epoch scripted-checkpoint ladder.

Usage:
    python train_run.py --out DIR [--epochs N] [--holdout] [--seed S] ...

With --holdout a class-balanced 5k slice of the TRAIN split is held out and
reported each epoch (used for model selection).  The graded run uses all
50k training images and evaluates nothing during training; its checkpoints
are scored afterwards by verify.py.
"""
import argparse
import json
import os
import shutil
import time

import torch

import airbench as ab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=ab.DEFAULTS["epochs"])
    ap.add_argument("--lr", type=float, default=ab.DEFAULTS["lr"])
    ap.add_argument("--wd", type=float, default=ab.DEFAULTS["weight_decay"])
    ap.add_argument("--momentum", type=float, default=ab.DEFAULTS["momentum"])
    ap.add_argument("--bs", type=int, default=ab.DEFAULTS["batch_size"])
    ap.add_argument("--ls", type=float, default=ab.DEFAULTS["label_smoothing"])
    ap.add_argument("--bias-scaler", type=float, default=ab.DEFAULTS["bias_scaler"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = dict(ab.DEFAULTS)
    cfg.update(epochs=args.epochs, lr=args.lr, weight_decay=args.wd,
               momentum=args.momentum, batch_size=args.bs, label_smoothing=args.ls,
               bias_scaler=args.bias_scaler, seed=args.seed)

    x, y = ab.load_split_standardized("train")
    if args.holdout:
        tx, ty, hx, hy = ab.stratified_holdout(x, y)
    else:
        tx, ty, hx, hy = x, y, None, None

    ckpt_dir = None if args.no_ckpt else args.out
    if ckpt_dir:
        if os.path.isdir(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        os.makedirs(ckpt_dir, exist_ok=True)

    # GPU warm-up so that cudnn autotuning is not billed to epoch 1.
    ab.warmup(cfg)

    t_start = time.time()
    hist, model, total = ab.train(cfg, tx, ty, ckpt_dir=ckpt_dir,
                                  eval_x=hx, eval_y=hy,
                                  log=lambda r: print(json.dumps(r), flush=True))
    print(json.dumps(dict(tag=args.tag, cfg={k: v for k, v in cfg.items()
                                             if k != "widths"},
                          total_sec=total, wall_start=t_start,
                          hist=hist)), flush=True)
    if hx is not None:
        print(f"FINAL holdout acc {hist[-1]['acc']:.4f}  total {total:.2f}s "
              f"({total / args.epochs:.3f}s/epoch)", flush=True)


if __name__ == "__main__":
    main()
