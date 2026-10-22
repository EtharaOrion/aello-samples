"""Round 2 of the pre-registered sweep (PREREGISTRATION.md sec 5b).

Walks past the edges round 1 hit (momentum low end, lr high end), tests whether
they stack, and re-estimates sigma by pooling every >=3-seed group.
"""
import json
import math
import statistics
import time

import torch

import airbench as A

torch.backends.cudnn.benchmark = True

HOLDOUT_N = 5000
xtr_all, ytr_all = A.load_split("train")
g = torch.Generator().manual_seed(1234)
perm = torch.randperm(len(ytr_all), generator=g)
hidx, kidx = perm[:HOLDOUT_N], perm[HOLDOUT_N:]
xh, yh = xtr_all[hidx], ytr_all[hidx]
xk, yk = xtr_all[kidx], ytr_all[kidx]
data = A.GpuTrainData(xk, yk)
whiten_images = A.standardize(xk[:10000])
eval_sets = {"holdout": (xh, yh)}

BASE = dict(A.DEFAULT_CFG)
A.warmup_gpu(BASE)

CANDIDATES = [
    ("mom0.80+lr1.8", dict(momentum=0.80, lr=1.8)),
    ("mom0.80+wd1.4e-3", dict(momentum=0.80, wd=1.4e-3)),
    ("mom=0.75", dict(momentum=0.75)),
    ("lr=2.1", dict(lr=2.1)),
]

results = []
for name, ov in CANDIDATES:
    for seed in (0, 1, 2):
        cfg = dict(BASE); cfg.update(ov)
        t = time.perf_counter()
        recs, net, wall = A.train(cfg, data, whiten_images, seed=seed,
                                  eval_sets=eval_sets, log=lambda s: None)
        accs = [r["holdout"] for r in recs]
        acc3 = sum(accs[-3:]) / 3
        cross = next((r["epoch"] for r in recs if r["holdout"] >= 0.94), None)
        results.append(dict(name=name, overrides={k: str(v) for k, v in ov.items()},
                            seed=seed, acc3=acc3, final=accs[-1], cross=cross,
                            accs=[round(a, 4) for a in accs]))
        print(f"{name:20s} seed={seed}  acc3={acc3:.4f}  final={accs[-1]:.4f}  "
              f"cross={cross}  ({time.perf_counter()-t:.1f}s)", flush=True)
    with open("/workspace/round2_results.json", "w") as f:
        json.dump(results, f, indent=2)

# ---- pooled sigma over every >=3-seed group from both rounds ----
groups, crosses = {}, {}
for r in json.load(open("/workspace/sweep_results.json"))["runs"] + results:
    groups.setdefault(r["name"], []).append(r["acc3"])
    crosses.setdefault(r["name"], []).append(r["cross"])
multi = {n: a for n, a in groups.items() if len(a) >= 3}
ss = sum(statistics.variance(a) * (len(a) - 1) for a in multi.values())
df = sum(len(a) - 1 for a in multi.values())
sigma_pooled = math.sqrt(ss / df)
print(f"\npooled sigma = {sigma_pooled*100:.3f} pp over {df} df ({len(multi)} groups) "
      f"vs round-1 baseline-only sigma 0.079 pp", flush=True)

base = statistics.mean(multi["baseline"])
nb = len(multi["baseline"])
print(f"\nincumbent baseline mean = {base:.4f} (n={nb})")
rows = []
for n, a in sorted(multi.items(), key=lambda kv: -statistics.mean(kv[1])):
    if n == "baseline":
        continue
    m, k = statistics.mean(a), len(a)
    thresh = 2 * sigma_pooled * math.sqrt(1 / k + 1 / nb)
    delta = m - base
    rows.append(dict(name=n, mean=m, n=k, delta=delta, threshold=thresh,
                     promote=bool(delta > thresh), crosses=crosses[n]))
    print(f"  {n:20s} mean={m:.4f} (n={k})  delta={delta*100:+.3f} pp  "
          f"bar={thresh*100:.3f} pp  cross={crosses[n]}  "
          f"{'PROMOTE' if delta > thresh else 'not significant'}")

winners = [r for r in rows if r["promote"]]
if winners:
    best = max(winners, key=lambda r: (r["mean"], -max(c or 99 for c in r["crosses"])))
    print(f"\nDECISION: promote {best['name']}")
else:
    best = None
    print("\nDECISION: keep baseline -- no candidate clears 2 sigma_pooled")

with open("/workspace/round2_results.json", "w") as f:
    json.dump(dict(runs=results, sigma_pooled=sigma_pooled, baseline_mean=base,
                   comparisons=rows, decision=best["name"] if best else "baseline"),
              f, indent=2)
