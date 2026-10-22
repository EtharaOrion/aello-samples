"""Summarise sweep JSONL files, with the sampling noise made explicit.

A 5,000-image held-out slice gives a binomial standard error of
sqrt(p(1-p)/n) ~= 0.32% at p = 0.945, so single-seed differences below roughly
0.6% are not distinguishable from noise.  Everything is printed with that error
bar attached and averaged over seeds where more than one was run.
"""

import argparse
import json
import math
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--n-val", type=int, default=5000)
    args = ap.parse_args()

    runs = defaultdict(list)
    for f in args.files:
        for line in open(f):
            r = json.loads(line)
            runs[r["name"]].append(r)

    rows = []
    for name, rs in runs.items():
        accs = [r["val_acc"] for r in rs]
        mean = sum(accs) / len(accs)
        spread = (max(accs) - min(accs)) if len(accs) > 1 else float("nan")
        se = math.sqrt(mean * (1 - mean) / (args.n_val * len(accs)))
        rows.append((mean, name, len(accs), spread, se,
                     sum(r["val_acc_no_tta"] for r in rs) / len(rs)))
    rows.sort(reverse=True)

    base = next((r for r in rows if r[1] == "baseline"), None)
    print(f"{'config':30s} {'seeds':>5s} {'val_acc':>9s} {'+/-SE':>7s} "
          f"{'range':>7s} {'no-TTA':>8s} {'vs base':>8s}")
    for mean, name, n, spread, se, nt in rows:
        delta = f"{(mean - base[0]) * 100:+.2f}" if base else ""
        rng = f"{spread * 100:.2f}" if spread == spread else "   -"
        print(f"{name:30s} {n:5d} {mean:9.4f} {se * 100:7.2f} {rng:>7s} {nt:8.4f} {delta:>8s}")


if __name__ == "__main__":
    main()
