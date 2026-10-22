"""Held-out hyper-parameter sweep.

Every number produced here comes from a 5,000-image slice of the **train**
split; the test split is not touched.  Protocol and candidate list are frozen
in PREREGISTRATION.md.

    python sweep.py --stage r1 --out /workspace/logs/sweep_r1.json
"""
import argparse
import json
import os
import time

import torch

from speedrun.train import Config, train

HOLDOUT_N = 5000
HOLDOUT_SEED = 1234

# --- frozen candidate list (PREREGISTRATION.md section 4) -------------------
CANDIDATES = {
    "A_baseline":        {},
    "B_lr9":             {"lr": 9.0},
    "C_lr14":            {"lr": 14.0},
    "D_wd0110":          {"weight_decay": 0.0110},
    "E_wd0200":          {"weight_decay": 0.0200},
    "F_ls010":           {"label_smoothing": 0.10},
    "G_ls030":           {"label_smoothing": 0.30},
    "H_mom080":          {"momentum": 0.80},
    "I_mom090":          {"momentum": 0.90},
    "J_finalfrac007":    {"lr_final_frac": 0.07},
    "K_warmup013":       {"warmup_frac": 0.13},
    "L_bnmom08":         {"bn_momentum": 0.8},
    "M_scale16":         {"scaling_factor": 1 / 6},
    "N_wider":           {"widths": (160, 448, 512)},
    "O_whitenbias6":     {"whiten_bias_epochs": 6},
}


def split_train_holdout(device="cuda"):
    from speedrun.data import load_split
    x, y = load_split("train", device=device)
    g = torch.Generator().manual_seed(HOLDOUT_SEED)
    perm = torch.randperm(len(x), generator=g).to(device)
    hold, keep = perm[:HOLDOUT_N], perm[HOLDOUT_N:]
    out = (x[keep].contiguous(), y[keep]), (x[hold].contiguous(), y[hold])
    # This GPU is shared; drop the full-split copy so the sweep fits alongside
    # the other tenant.  Sweep-path only - the shipped run does not use this.
    del x
    torch.cuda.empty_cache()
    return out


def run_one(name, overrides, seed, epochs, keep, hold, compile_mode):
    cfg = Config(epochs=epochs, seed=seed, **overrides)
    t0 = time.time()
    out = train(cfg, keep[0], keep[1], outdir=None,
                eval_sets={"holdout": hold}, eval_epochs={epochs},
                log=lambda *_: None, compile_mode=compile_mode)
    acc = out["records"][-1]["holdout"]
    print(f"  {name:18s} seed={seed} epochs={epochs} holdout={acc:.4f} "
          f"({time.time() - t0:.1f}s)", flush=True)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="all")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compile", dest="compile_mode", default=None)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    keep, hold = split_train_holdout()
    names = list(CANDIDATES) if args.names == "all" else args.names.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    results = {}
    if os.path.exists(args.out):
        results = json.load(open(args.out))
    for name in names:
        results.setdefault(name, {})
        for seed in seeds:
            if str(seed) in results[name]:
                continue
            results[name][str(seed)] = run_one(
                name, CANDIDATES[name], seed, args.epochs, keep, hold,
                args.compile_mode)
            torch.cuda.empty_cache()
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)

    print("\n--- summary (held-out, train-split slice) ---")
    for name, per_seed in sorted(results.items(),
                                 key=lambda kv: -sum(kv[1].values()) / len(kv[1])):
        v = list(per_seed.values())
        m = sum(v) / len(v)
        print(f"  {name:18s} n={len(v)} mean={m:.4f}  " +
              " ".join(f"{a:.4f}" for a in v))


if __name__ == "__main__":
    main()
