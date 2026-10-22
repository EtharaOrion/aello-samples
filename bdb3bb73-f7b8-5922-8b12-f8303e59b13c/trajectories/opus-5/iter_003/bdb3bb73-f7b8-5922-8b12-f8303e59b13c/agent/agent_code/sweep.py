"""Held-out hyperparameter search.

Every configuration is trained on 45,000 train images and scored on the same
fixed 5,000-image held-out slice of the *train* split (split seed 1234, fixed
across all runs so configurations are compared on identical data).  The test
split is never loaded by this script.

Runs share one process so the dataset is loaded and the GPU warmed exactly once;
each run still re-seeds and rebuilds the model from scratch.  Results stream to a
JSONL file so a partially finished sweep is still usable.
"""

import argparse
import json
import sys
import time

import torch

from core import DEFAULT_CFG
from train_run import make_cfg, train


def run_stage(specs, out_path, holdout, seeds, epochs_override=None):
    with open(out_path, "a") as fh:
        for name, overrides in specs:
            for seed in seeds:
                cfg = make_cfg(overrides)
                if epochs_override:
                    cfg["epochs"] = epochs_override
                t0 = time.time()
                res, _ = train(cfg, ckpt_dir=None, holdout=holdout, seed=seed,
                               log_every=0, quiet=True)
                rec = dict(
                    name=name, seed=seed, val_acc=res["val_acc"],
                    val_acc_no_tta=res["val_acc_no_tta"],
                    train_sec=res["total_sec"], wall_sec=time.time() - t0,
                    overrides=overrides, epochs=cfg["epochs"],
                )
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"{name:28s} seed={seed}  val={res['val_acc']:.4f} "
                      f"(no-tta {res['val_acc_no_tta']:.4f})  {res['total_sec']:.1f}s", flush=True)
                del _
                torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Stage definitions.  Stage 1 is a one-factor-at-a-time scan around the
# airbench-derived defaults; later stages are written after reading stage 1.
# --------------------------------------------------------------------------- #
def stage1():
    specs = [("baseline", {})]
    axes = {
        "lr": [0.10, 0.14, 0.28, 0.40],
        "weight_decay": [0.004, 0.02, 0.04],
        "label_smoothing": [0.10, 0.30],
        "logit_scale": [0.05, 0.25, 0.50],
        "bias_lr_scale": [4.0, 8.0, 32.0],
        "momentum": [0.80, 0.90],
        "cutout": [6, 8],
        "pad": [1, 3],
        "warmup_frac": [0.05, 0.25],
        "final_frac": [0.0, 0.10],
    }
    for key, vals in axes.items():
        for v in vals:
            specs.append((f"{key}={v}", {key: v}))
    specs += [
        ("widths=small", {"widths": (64, 256, 256)}),
        ("widths=mid", {"widths": (96, 320, 448)}),
        ("widths=big", {"widths": (160, 448, 640)}),
        ("bs=512", {"batch_size": 512}),
        ("bs=2048", {"batch_size": 2048, "lr": 0.4}),
    ]
    return specs


STAGES = {"1": stage1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="1")
    ap.add_argument("--out", default="/workspace/sweep_results.jsonl")
    ap.add_argument("--holdout", type=int, default=5000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--specs", default=None, help="JSON list of [name, overrides] pairs")
    args = ap.parse_args()

    specs = json.loads(args.specs) if args.specs else STAGES[args.stage]()
    specs = [(n, o) for n, o in specs]
    print(f"stage={args.stage} n_specs={len(specs)} seeds={args.seeds} "
          f"holdout={args.holdout} defaults={DEFAULT_CFG}", flush=True)
    run_stage(specs, args.out, args.holdout, args.seeds, args.epochs)


if __name__ == "__main__":
    main()
