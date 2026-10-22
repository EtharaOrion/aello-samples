"""Execute the pre-registered sweep (see PREREGISTRATION.md).

Stage 0: noise floor, baseline x 4 seeds.
Stage 1: screening, one factor at a time, 1 seed.
Stage 2: confirmation, top-4 survivors x 3 seeds.
Nothing here ever touches the test split.
"""
import json
import math
import statistics
import sys
import time

import torch

import airbench as A

torch.backends.cudnn.benchmark = True

HOLDOUT_N = 5000
OUT = "/workspace/sweep_results.json"

xtr_all, ytr_all = A.load_split("train")
g = torch.Generator().manual_seed(1234)
perm = torch.randperm(len(ytr_all), generator=g)
hidx, kidx = perm[:HOLDOUT_N], perm[HOLDOUT_N:]
xh, yh = xtr_all[hidx], ytr_all[hidx]
xk, yk = xtr_all[kidx], ytr_all[kidx]
print(f"holdout {len(yh)} (per-class {yh.bincount().tolist()})   train {len(yk)}", flush=True)

data = A.GpuTrainData(xk, yk)
whiten_images = A.standardize(xk[:10000])
eval_sets = {"holdout": (xh, yh)}

BASE = dict(A.DEFAULT_CFG)
A.warmup_gpu(BASE)

results = []


def run(name, overrides, seed):
    cfg = dict(BASE)
    cfg.update(overrides)
    t = time.perf_counter()
    recs, net, wall = A.train(cfg, data, whiten_images, seed=seed,
                              eval_sets=eval_sets, log=lambda s: None)
    accs = [r["holdout"] for r in recs]
    acc3 = sum(accs[-3:]) / 3
    cross = next((r["epoch"] for r in recs if r["holdout"] >= 0.94), None)
    rec = dict(name=name, overrides={k: str(v) for k, v in overrides.items()}, seed=seed,
               acc3=acc3, final=accs[-1], cross=cross, epochs=cfg["epochs"],
               train_wall=wall, accs=[round(a, 4) for a in accs])
    results.append(rec)
    print(f"{name:26s} seed={seed}  acc3={acc3:.4f}  final={accs[-1]:.4f}  "
          f"cross={cross}  ({time.perf_counter()-t:.1f}s)", flush=True)
    return rec


def save():
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)


# --------------------------------------------------------------- stage 0
print("\n== stage 0: noise floor (baseline x 4 seeds) ==", flush=True)
base_runs = [run("baseline", {}, s) for s in range(4)]
base_acc3 = [r["acc3"] for r in base_runs]
sigma = statistics.stdev(base_acc3)
base_mean = statistics.mean(base_acc3)
print(f"\nbaseline acc3 mean={base_mean:.4f}  sigma_run={sigma*100:.3f} pp  "
      f"crossings={[r['cross'] for r in base_runs]}", flush=True)
save()

# --------------------------------------------------------------- stage 1
CANDIDATES = [
    ("lr=1.2", dict(lr=1.2)), ("lr=1.8", dict(lr=1.8)),
    ("wd=1.4e-3", dict(wd=1.4e-3)), ("wd=2.8e-3", dict(wd=2.8e-3)),
    ("ls=0.1", dict(label_smoothing=0.1)), ("ls=0.3", dict(label_smoothing=0.3)),
    ("epochs=16", dict(epochs=16)), ("epochs=24", dict(epochs=24)),
    ("warm=0.15", dict(warm_frac=0.15)), ("warm=0.35", dict(warm_frac=0.35)),
    ("bn_mom=0.25", dict(bn_mom=0.25)), ("bn_mom=0.6", dict(bn_mom=0.6)),
    ("mom=0.80", dict(momentum=0.80)), ("mom=0.90", dict(momentum=0.90)),
    ("lr_start=0.05", dict(lr_start=0.05)),
    ("wbias=0", dict(whiten_bias_epochs=0)), ("wbias=6", dict(whiten_bias_epochs=6)),
    ("bs=500", dict(batch_size=500)),
]
print("\n== stage 1: screening (1 seed each) ==", flush=True)
screen = [run(n, o, 0) for n, o in CANDIDATES]
save()

cut = base_runs[0]["acc3"] - sigma
survivors = sorted([r for r in screen if r["acc3"] >= cut], key=lambda r: -r["acc3"])
print(f"\ncut = {cut:.4f} (baseline seed0 {base_runs[0]['acc3']:.4f} - 1 sigma)")
print("survivors:", [(r["name"], round(r["acc3"], 4)) for r in survivors], flush=True)

# --------------------------------------------------------------- stage 2
print("\n== stage 2: confirmation (top 4 x 3 seeds) ==", flush=True)
finalists = survivors[:4]
conf = {}
for r in finalists:
    ov = dict(CANDIDATES[[n for n, _ in CANDIDATES].index(r["name"])][1])
    runs = [r] + [run(r["name"], ov, s) for s in (1, 2)]
    conf[r["name"]] = [x["acc3"] for x in runs]
save()

thresh = 2 * math.sqrt(sigma ** 2 / 3 + sigma ** 2 / 4)
print(f"\npromotion threshold: +{thresh*100:.3f} pp over baseline mean {base_mean:.4f}")
winner, best = None, base_mean
for name, accs in conf.items():
    m = statistics.mean(accs)
    delta = m - base_mean
    ok = delta > thresh
    print(f"  {name:26s} mean3={m:.4f}  delta={delta*100:+.3f} pp  "
          f"{'PROMOTE' if ok else 'not significant'}")
    if ok and m > best:
        winner, best = name, m
print(f"\nDECISION: {'promote ' + winner if winner else 'keep baseline'}", flush=True)

with open(OUT, "w") as f:
    json.dump(dict(runs=results, sigma_run=sigma, base_mean=base_mean,
                   threshold=thresh, confirmation=conf, decision=winner), f, indent=2)
