"""Held-out experiment driver.

ALL model-selection decisions are made here, on a 5,000-image slice carved out
of the *train* split.  The test split is never read here.  Data is loaded once
and reused across configs so a sweep costs little more than the runs it
contains.

Usage:  python holdout.py <experiment-name>
"""
import json
import sys
import time
from dataclasses import replace

import numpy as np
import torch

import lib
from lib import Cfg

HOLDOUT = 5000


def split_data(seed=1234):
    """Deterministic, class-stratified 45k/5k split of the train set."""
    xpad, y = lib.load_train()
    yc = y.cpu().numpy()
    rng = np.random.RandomState(seed)
    val_idx = np.concatenate([rng.permutation(np.where(yc == c)[0])[:HOLDOUT // 10]
                              for c in range(10)])
    mask = np.ones(len(yc), dtype=bool)
    mask[val_idx] = False
    tr = torch.from_numpy(np.where(mask)[0]).to(lib.DEV)
    va = torch.from_numpy(np.sort(val_idx)).to(lib.DEV)
    # validation images are un-padded and fp32, i.e. exactly the tensor the
    # verifier would hand the model.
    xval = xpad[va][:, :, 2:-2, 2:-2].float().contiguous()
    return xpad[tr].contiguous(), y[tr], xval, y[va]


def run(cfg, data, quiet=False):
    xtr, ytr, xva, yva = data
    ev = (lambda m: lib.evaluate(m, xva, yva))
    log = (lambda *a: None) if quiet else print
    t = time.time()
    r = lib.train(cfg, xtr, ytr, ckpt_dir=None, eval_fn=ev, log=log)
    return dict(accs=r["accs"], best=max(r["accs"]), final=r["accs"][-1],
                per_epoch=r["epoch_end"][-1] / cfg.epochs, wall=time.time() - t)


def first_cross(accs, thr=0.94):
    for i, a in enumerate(accs):
        if a >= thr:
            return i + 1
    return None


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    data = split_data()
    print(f"train {tuple(data[0].shape)}  val {tuple(data[2].shape)}")
    out = {}

    GRIDS = {
        # round 1: the two knobs the bug affected, at a short schedule
        "lr": dict(epochs=[18], lr=[0.2, 0.4, 0.8, 1.6], bias_mult=[1.0, 8.0]),
        # round 2: regularization + schedule length around the round-1 optimum
        "reg": dict(wd=[2.5e-4, 5e-4, 1e-3], label_smoothing=[0.1, 0.2]),
        "len": dict(epochs=[16, 20, 24, 28]),
        "tta": dict(tta_level=[0, 1, 2]),
        "seed": dict(seed=[0, 1, 2, 3]),
    }
    base = Cfg()
    if len(sys.argv) > 2:                      # e.g. `holdout.py reg lr=0.8 epochs=24`
        kw = {}
        for tok in sys.argv[2:]:
            k, v = tok.split("=")
            kw[k] = (int(v) if k in ("epochs", "seed", "tta_level") else float(v))
        base = replace(base, **kw)
        print("base overrides:", kw)

    # Handpicked probes: cheaper than a full grid when the axes are known to
    # interact (logit scale <-> LR <-> bias LR must move together).
    COMBOS = {
        "combo": [
            dict(epochs=28),                                            # control
            dict(epochs=28, wd=1.5e-3),                                 # more decay
            dict(epochs=28, label_smoothing=0.1),
            dict(epochs=28, logit_scale=1 / 9, lr=1.5, bias_mult=64.),  # airbench-style
            dict(epochs=40),                                            # longer
        ],
        # round 3: stack the two independent winners (airbench scaling +
        # stronger decay) and extend the schedule.  0.465 s/epoch means even a
        # 55-epoch crossing still lands under the 28 s S_outcome ceiling, so
        # schedule length is effectively free here.
        "combo2": [
            dict(epochs=28, logit_scale=1 / 9, lr=1.5, bias_mult=64., wd=1.5e-3),
            dict(epochs=40, logit_scale=1 / 9, lr=1.5, bias_mult=64.),
            dict(epochs=40, logit_scale=1 / 9, lr=1.5, bias_mult=64., wd=1.5e-3),
        ],
    }
    if name in COMBOS:
        combos = COMBOS[name]
    else:
        grid = GRIDS[name]
        combos = [{}]
        for k in list(grid):
            combos = [dict(c, **{k: v}) for c in combos for v in grid[k]]

    for c in combos:
        cfg = replace(base, **c)
        r = run(cfg, data, quiet=True)
        tag = " ".join(f"{k}={v}" for k, v in c.items())
        cross = first_cross(r["accs"])
        print(f"{tag:34s} final={r['final']*100:6.2f}  best={r['best']*100:6.2f} "
              f" cross@{cross}  {r['per_epoch']*1000:.0f} ms/ep")
        out[tag] = r

    best = max(out, key=lambda k: out[k]["final"])
    print(f"\nBEST: {best}  final={out[best]['final']*100:.2f}")
    with open(f"/workspace/results_{name}.json", "w") as f:
        json.dump(out, f, indent=1)
