"""In-process sweep harness.

All model-selection decisions are made on a FIXED 5,000-image holdout carved out
of the TRAIN split (split seed 1234, independent of the training seed). The test
split is never loaded here.
"""
import argparse
import itertools
import json
import sys
import time
from dataclasses import replace

import torch
from train import Config, train

BASE = Config(epochs=16, holdout=5000)


def run(cfg, seeds):
    accs = []
    for s in seeds:
        r = train(replace(cfg, seed=s))
        accs.append(r["acc"])
    a = torch.tensor(accs)
    return a.mean().item(), (a.std().item() if len(accs) > 1 else 0.0), accs, r["wall"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=str, required=True)  # JSON: {"lr":[...], ...}
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--base", type=str, default="{}")
    ap.add_argument("--tag", type=str, default="sweep")
    a = ap.parse_args()

    base = replace(BASE, **json.loads(a.base))
    grid = json.loads(a.grid)
    keys = list(grid)
    rows = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        cfg = replace(base, **dict(zip(keys, combo)))
        t = time.time()
        m, sd, accs, wall = run(cfg, a.seeds)
        rows.append((dict(zip(keys, combo)), m, sd, accs, wall))
        print(f"{dict(zip(keys, combo))}  acc={m*100:.2f} +-{sd*100:.2f} "
              f"{[round(x*100,2) for x in accs]} wall={wall:.1f}s ({time.time()-t:.0f}s)",
              flush=True)

    rows.sort(key=lambda r: -r[1])
    print("\n=== ranked ===")
    for c, m, sd, accs, wall in rows:
        print(f"  {m*100:6.2f} +-{sd*100:.2f}  {c}")
    with open(f"/workspace/results_{a.tag}.json", "w") as f:
        json.dump([{"cfg": c, "mean": m, "std": sd, "accs": accs, "wall": wall}
                   for c, m, sd, accs, wall in rows], f, indent=1)


if __name__ == "__main__":
    main()
