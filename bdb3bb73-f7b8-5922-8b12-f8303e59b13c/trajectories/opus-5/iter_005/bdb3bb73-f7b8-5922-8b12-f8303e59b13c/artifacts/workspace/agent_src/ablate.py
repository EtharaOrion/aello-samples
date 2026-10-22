"""One-factor-at-a-time ablation around a base config, on the 5k train holdout.

Every variant is run with the same seed set as the base, so comparisons are
paired. A difference is only called real if it clears 2 standard errors of the
paired difference (reported per row as `sig`).
"""
import argparse
import json
import math
import time
from dataclasses import replace

import torch
from train import Config, train

ap = argparse.ArgumentParser()
ap.add_argument("--base", type=str, default="{}")
ap.add_argument("--variants", type=str, required=True)  # JSON list of dicts
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--tag", type=str, default="abl")
a = ap.parse_args()

base = replace(Config(epochs=32, holdout=5000), **json.loads(a.base))
variants = json.loads(a.variants)


def run(cfg):
    return [train(replace(cfg, seed=s))["acc"] for s in a.seeds]


t0 = time.time()
b_accs = run(base)
bm = sum(b_accs) / len(b_accs)
print(f"BASE {json.loads(a.base)}  acc={bm*100:.2f}  "
      f"{[round(x*100,2) for x in b_accs]}  ({time.time()-t0:.0f}s)", flush=True)

rows = []
for v in variants:
    t = time.time()
    accs = run(replace(base, **v))
    m = sum(accs) / len(accs)
    d = [x - y for x, y in zip(accs, b_accs)]                 # paired differences
    dm = sum(d) / len(d)
    if len(d) > 1:
        sd = math.sqrt(sum((x - dm) ** 2 for x in d) / (len(d) - 1))
        se = sd / math.sqrt(len(d))
    else:
        se = float("nan")
    sig = "SIG" if (se == se and abs(dm) > 2 * se) else "ns"
    rows.append((v, m, dm, se, accs, sig))
    print(f"{str(v):46s} acc={m*100:6.2f}  d={dm*100:+5.2f} +-{se*100:4.2f} {sig:3s} "
          f"{[round(x*100,2) for x in accs]}  ({time.time()-t:.0f}s)", flush=True)

rows.sort(key=lambda r: -r[1])
print(f"\n=== ranked (base {bm*100:.2f}) ===")
for v, m, dm, se, accs, sig in rows:
    print(f"  {m*100:6.2f}  d={dm*100:+5.2f} +-{se*100:4.2f} {sig:3s}  {v}")
with open(f"/workspace/results_{a.tag}.json", "w") as f:
    json.dump({"base": {"cfg": json.loads(a.base), "accs": b_accs, "mean": bm},
               "variants": [{"cfg": v, "mean": m, "delta": dm, "se": se,
                             "accs": accs, "sig": sig}
                            for v, m, dm, se, accs, sig in rows]}, f, indent=1)
print(f"total {time.time()-t0:.0f}s")
