"""Hyper-parameter sweeps on a held-out slice of the *training* split.

Every configuration is scored on the same deterministic 5 000-image holdout
carved out of the 50 000 training images (see train.py --holdout).  The test
split is never loaded here, so nothing that is tuned can leak from it.

    python sweep.py stage1 --seeds 2                       # a predefined stage
    python sweep.py tta --seeds 2 --extra '[{"--tta":"1"},{"--tta":"2"}]'

With --extra the positional argument is just a label for the log; the override
dicts are applied on top of BASE, and any train.py flag (including --epochs) can
be overridden per configuration.  Results are appended to
/workspace/sweep_results.jsonl.

Caveat learned the hard way: repeats *within* one sweep process differ by only
~0.1%, but the same config measured in two different processes differed by 0.7%.
Only compare configurations that ran inside the same invocation.
"""

import argparse
import json
import time

import train as trainer

BASE = ["--holdout", "5000", "--save", "0", "--compile", "0", "--verbose", "0"]
WIDE = {"--widths": "128 384 512"}

STAGES = {
    # Stage 1: the structural choices, at a fixed 18-epoch budget.
    "stage1": [
        {},
        {"--head-init": "default"},
        {"--lr": "8"},
        {"--lr": "16"},
        {"--wd": "0.008"},
        {"--wd": "0.025"},
        {"--logit-scale": "0.25"},
        {"--logit-scale": "1.0"},
        {"--bn-momentum": "0.4"},
        {"--bn-momentum": "0.85"},
        {"--label-smoothing": "0.1"},
        {"--final-lr-frac": "0.0"},
        {"--warmup-frac": "0.1"},
        {"--whiten-bias-epochs": "100"},
        {"--widths": "128 384 512"},
        {"--cutout": "8"},
    ],
    # Stage 2: combine the stage-1 winners on the wide net.
    "stage2": [
        dict(WIDE),
        dict(WIDE, **{"--logit-scale": "0.25"}),
        dict(WIDE, **{"--logit-scale": "0.4"}),
        dict(WIDE, **{"--wd": "0.008"}),
        dict(WIDE, **{"--lr": "16"}),
        dict(WIDE, **{"--head-init": "default"}),
        dict(WIDE, **{"--logit-scale": "0.25", "--wd": "0.008"}),
        dict(WIDE, **{"--logit-scale": "0.25", "--lr": "16"}),
        dict(WIDE, **{"--logit-scale": "0.25", "--lr": "16", "--wd": "0.008"}),
        dict(WIDE, **{"--logit-scale": "0.25", "--head-init": "default"}),
        {"--widths": "128 512 512", "--logit-scale": "0.25"},
        {"--widths": "192 384 512", "--logit-scale": "0.25"},
    ],
}


def run(overrides, seeds, epochs):
    accs, walls = [], []
    for seed in range(seeds):
        argv = list(BASE) + ["--epochs", str(epochs), "--seed", str(seed)]
        for k, v in overrides.items():
            argv += [k] + str(v).split()
        res = trainer.main(argv)
        accs.append(res["val_acc"])
        walls.append(res["wall_clock_sec"])
    return sum(accs) / len(accs), accs, min(walls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--extra", default="", help="JSON list of override dicts")
    args = ap.parse_args()

    configs = json.loads(args.extra) if args.extra else STAGES[args.stage]
    rows = []
    for cfg in configs:
        t0 = time.time()
        mean, accs, wall = run(cfg, args.seeds, args.epochs)
        rows.append((mean, cfg, accs, wall))
        label = " ".join(f"{k}={v}" for k, v in cfg.items()) or "(base)"
        print(f"{mean*100:6.2f}%  {[round(a*100,2) for a in accs]}  {wall:5.2f}s  {label}",
              flush=True)
        with open("/workspace/sweep_results.jsonl", "a") as f:
            f.write(json.dumps(dict(stage=args.stage, epochs=args.epochs, cfg=cfg,
                                    mean_val=mean, accs=accs, wall=wall,
                                    sweep_sec=time.time() - t0)) + "\n")

    print("\n=== ranked ===")
    for mean, cfg, accs, wall in sorted(rows, key=lambda r: -r[0]):
        print(f"{mean*100:6.2f}%  {' '.join(f'{k}={v}' for k, v in cfg.items()) or '(base)'}")


if __name__ == "__main__":
    main()
