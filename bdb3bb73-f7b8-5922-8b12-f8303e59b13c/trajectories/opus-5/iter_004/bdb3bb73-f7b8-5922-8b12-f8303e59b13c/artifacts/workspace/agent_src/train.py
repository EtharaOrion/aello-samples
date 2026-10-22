"""Entry point: run one training job and write a contiguous checkpoint ladder.

Usage:  python train.py --out /workspace/periodic_checkpoints [--epochs N] [--seed S]

The run performs no evaluation; it only trains and saves. Accuracy is measured
afterwards by `verify_ladder.py` (test, verifier-faithful) or `sweep.py`
(held-out train slice, used for every model-selection decision).
"""

import argparse
import ast
import json
import os
import shutil
import time

import torch

import speedrun as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/periodic_checkpoints")
    ap.add_argument("--config", default="{}", help="python dict literal of overrides")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--final", default=None, help="also hardlink/copy last ckpt here")
    ap.add_argument("--metrics", default=None, help="write agent_reported_metrics.json here")
    args = ap.parse_args()

    t_run_start = time.time()  # total wall clock of the training run, incl. setup

    cfg = ast.literal_eval(args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.seed is not None:
        cfg["seed"] = args.seed

    # Fresh ladder: the verifier scores whatever is left in the directory, so a
    # stale checkpoint from an earlier run must not survive.
    if os.path.isdir(args.out):
        for f in os.listdir(args.out):
            if f.startswith("ckpt_epoch_") and f.endswith(".pt"):
                os.remove(os.path.join(args.out, f))
    os.makedirs(args.out, exist_ok=True)

    train_x, train_y = S.load_uint8("train")
    net, stats = S.train(cfg, train_x, train_y, ckpt_dir=args.out, t0=t_run_start)

    saves = stats["save_times"]
    n = len(saves)
    span = saves[-1] - saves[0]
    t_star_if_last = span * n / (n - 1)
    total_wall = saves[-1] - t_run_start
    print(f"[train] ladder={n} span={span:.2f}s mean_epoch={span/(n-1):.3f}s "
          f"t*(if crossing at last epoch)={t_star_if_last:.2f}s total_wall={total_wall:.2f}s")

    if args.final:
        last = os.path.join(args.out, f"ckpt_epoch_{n:04d}.pt")
        if os.path.exists(args.final):
            os.remove(args.final)
        shutil.copy(last, args.final)
        print(f"[train] final checkpoint -> {args.final}")

    if args.metrics:
        with open(args.metrics, "w") as f:
            json.dump({"wall_clock_sec": total_wall, "epochs": n,
                       "mean_epoch_sec": span / (n - 1)}, f, indent=2)

    with open(os.path.join(args.out, "..", "last_run_stats.json"), "w") as f:
        json.dump({"cfg": {k: list(v) if isinstance(v, tuple) else v for k, v in stats["cfg"].items()},
                   "total_sec": stats["total_sec"], "total_wall_sec": total_wall,
                   "epochs": n, "span_sec": span, "mean_epoch_sec": span / (n - 1),
                   "save_times": saves}, f, indent=2)


if __name__ == "__main__":
    main()
