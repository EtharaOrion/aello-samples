"""Hypothesis testing on a held-out slice of the TRAIN split.

The test split is never loaded here. Every configuration is trained on 45k
images and scored on the same fixed 5k held-out train images, over several
seeds, so that config-to-config differences can be compared against the
seed-to-seed noise floor.
"""
import argparse
import json
import statistics
import sys
import time

import torch

from train import evaluate, get_cfg, load_split, train

HOLDOUT = 5000


def make_splits(holdout=HOLDOUT):
    tx, ty = load_split("train", "cuda")
    g = torch.Generator().manual_seed(1234)
    perm = torch.randperm(50000, generator=g).cuda()
    tr, va = perm[:-holdout], perm[-holdout:]
    return tx[tr], ty[tr], tx[va], ty[va]


def run(name, seeds, splits, **over):
    trx, try_, vax, vay = splits
    accs = []
    for s in seeds:
        cfg = get_cfg(seed=s, **over)
        t0 = time.time()
        net, _ = train(cfg, trx, try_, ckpt_dir=None, log=lambda *a: None)
        acc = evaluate(net, vax, vay, cfg.tta_weight)
        accs.append(acc)
        print(f"  [{name}] seed={s} val={acc * 100:.2f} "
              f"({time.time() - t0:.1f}s)", flush=True)
        del net
        torch.cuda.empty_cache()
    m = statistics.mean(accs)
    sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
    print(f"==> {name}: mean={m * 100:.2f} sd={sd * 100:.2f} "
          f"n={len(accs)}", flush=True)
    return dict(name=name, mean=m, sd=sd, accs=accs, cfg=over)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="python file defining PLAN")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    splits = make_splits()
    ns = {}
    exec(open(args.plan).read(), ns)
    results = [run(name, seeds, splits, **over)
               for name, seeds, over in ns["PLAN"]]
    results.sort(key=lambda r: -r["mean"])
    print("\n=== ranking (held-out train slice) ===")
    for r in results:
        print(f"{r['mean'] * 100:6.2f} +- {r['sd'] * 100:4.2f}  {r['name']}")
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)
