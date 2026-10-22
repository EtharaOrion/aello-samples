"""Holdout sweep driver: noise floor -> screen -> confirm -> ablate.

Runs entirely on the 45k/5k split of the CIFAR-10 *train* set, per
PREREGISTRATION.md.  Results stream to JSONL so a partial sweep is still usable.
"""
import itertools, json, statistics, sys, time
import torch
from train import get_cfg, run

LOG = open("/workspace/logs/sweep.jsonl", "a", buffering=1)
BASE = dict(epochs=26, bs=1024, lr=0.5, momentum=0.85, wd=0.002, bias_scaler=8.0,
            ls=0.2, warm_frac=0.15, pad=2, cutout=0, w1=128, w2=384, w3=512,
            scale=0.125, bn_momentum=0.6, tta=1, split="holdout", save=0)
_seen = {}


def trial(stage, seed=0, **over):
    cfg_d = {**BASE, **over, "seed": seed}
    key = (stage.split(":")[0], json.dumps(cfg_d, sort_keys=True))
    if key in _seen:
        return _seen[key]
    t = time.time()
    _, res = run(get_cfg(**cfg_d))
    rec = {"stage": stage, "acc": res["val_acc"], "wall": res["wall"],
           "seed": seed, "over": over, "elapsed": time.time() - t}
    LOG.write(json.dumps(rec) + "\n")
    print(f"{stage:26s} seed={seed} acc={res['val_acc']:.4f} "
          f"wall={res['wall']:.2f}s {over}", flush=True)
    _seen[key] = res["val_acc"]
    return res["val_acc"]


def mean_over_seeds(stage, seeds=(0, 1, 2), **over):
    a = [trial(stage, seed=s, **over) for s in seeds]
    m = statistics.mean(a)
    print(f"  -> {stage} mean={m:.4f} sd={statistics.stdev(a):.4f} {over}", flush=True)
    return m, a


# ------------------------------------------------------------ 1. noise floor --
def noise_floor():
    accs = [trial("noise:seed", seed=s) for s in range(5)]
    sigma = statistics.stdev(accs)
    # same seed twice in the same process: isolates cuDNN/atomics non-determinism
    _seen.clear()
    rep = trial("noise:repeat", seed=0)
    out = {"accs": accs, "sigma": sigma, "mean": statistics.mean(accs),
           "same_seed_repeat": rep, "same_seed_delta": abs(rep - accs[0])}
    print(json.dumps(out), flush=True)
    LOG.write(json.dumps({"stage": "noise:summary", **out}) + "\n")
    return sigma


# ----------------------------------------------------------------- 2. screen --
AXES = [
    ("lr", [0.3, 0.4, 0.6, 0.8]),
    ("wd", [0.001, 0.004, 0.008]),
    ("ls", [0.0, 0.1, 0.3]),
    ("momentum", [0.8, 0.9]),
    ("bias_scaler", [1.0, 4.0, 16.0]),
    ("cutout", [6, 8, 10]),
    ("bn_momentum", [0.4, 0.8]),
    ("warm_frac", [0.05, 0.25]),
    ("scale", [0.0625, 0.25]),
    ("epochs", [20, 32]),
]


def screen():
    out = {}
    for name, vals in AXES:
        for v in vals:
            out[(name, v)] = trial(f"screen:{name}", **{name: v})
    return out


# ---------------------------------------------------------------- 4. ablate --
def ablate(best):
    for tag, over in [("full", {}), ("no_tta", {"tta": 0}), ("no_translate", {"pad": 0}),
                      ("no_zca_freeze", {}), ("narrow", {"w1": 64, "w2": 256, "w3": 256}),
                      ("no_labelsmooth", {"ls": 0.0}), ("no_warmup_sched", {"warm_frac": 0.0})]:
        if tag == "no_zca_freeze":
            continue
        trial(f"ablate:{tag}", **{**best, **over})


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if which in ("all", "noise"):
        noise_floor()
    if which in ("all", "screen"):
        screen()
    print(f"sweep phase '{which}' done in {time.time()-t0:.0f}s", flush=True)
