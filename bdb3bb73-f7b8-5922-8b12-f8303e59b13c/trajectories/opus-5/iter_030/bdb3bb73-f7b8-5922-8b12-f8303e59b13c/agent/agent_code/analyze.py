#!/usr/bin/env python
"""Summarise sweep results against the measured noise floor.

Comparisons are made on a *fixed* holdout, so the binomial sampling error of
that holdout is common-mode and cancels; the relevant noise is the seed-to-seed
spread of repeated identical runs (Stage A). A config is only called a win when
    mean(cfg) - mean(base) > 2 * sd_seed * sqrt(1/n_cfg + 1/n_base).
"""
import argparse
import json
import math
import statistics
from collections import defaultdict


def load(path):
    g = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            g[r["tag"]].append(r)
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="/workspace/logs/sweep.jsonl")
    p.add_argument("--base", default="base")
    p.add_argument("--cross", type=float, default=0.94)
    a = p.parse_args()
    g = load(a.log)

    base = [r["acc"] for r in g[a.base]]
    sd = statistics.stdev(base) if len(base) > 1 else 0.0013
    mb, nb = statistics.mean(base), len(base)
    print(f"noise floor: sd_seed = {sd*100:.3f} pp  (n={nb} identical runs)")
    print(f"baseline    : {mb*100:.2f} pp\n")
    print(f"{'tag':18s} {'n':>2s} {'mean%':>7s} {'sd':>5s} {'delta':>7s} "
          f"{'2sig':>6s} {'verdict':>9s} {'s/ep':>5s} {'x94':>4s}")

    rows = []
    for tag, rs in g.items():
        accs = [r["acc"] for r in rs]
        m, n = statistics.mean(accs), len(accs)
        s = statistics.stdev(accs) if n > 1 else float("nan")
        d = (m - mb) * 100
        thr = 2 * sd * math.sqrt(1 / n + 1 / nb) * 100
        verdict = "=" if abs(d) <= thr else ("WIN" if d > 0 else "LOSS")
        if tag == a.base:
            verdict = "base"
        # first epoch whose holdout accuracy crosses the bar, averaged over seeds
        xs = []
        for r in rs:
            c = next((i + 1 for i, v in enumerate(r["curve"] or []) if v and v >= a.cross), None)
            if c:
                xs.append(c)
        x94 = f"{statistics.mean(xs):.1f}" if xs else "-"
        pe = statistics.mean([r["per_epoch"] for r in rs])
        rows.append((d, tag, n, m, s, thr, verdict, pe, x94))

    for d, tag, n, m, s, thr, v, pe, x94 in sorted(rows, key=lambda r: -r[0]):
        print(f"{tag:18s} {n:2d} {m*100:7.2f} {s*100:5.2f} {d:+7.2f} "
              f"{thr:6.2f} {v:>9s} {pe:5.3f} {x94:>4s}")


if __name__ == "__main__":
    main()
