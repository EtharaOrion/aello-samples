"""Model selection on a held-out slice of the TRAIN split. The test split is never read.

Protocol
--------
* 5,000 images (500 per class, deterministic stride-10 pick) are held out; the
  remaining 45,000 are the sweep's training set. Every configuration sees the
  identical split, so comparisons are paired.
* Each configuration is run on >=2 seeds and we report mean and spread. The
  binomial standard error of a 5,000-image estimate near 94% is
  sqrt(.94*.06/5000) = 0.34%, so differences below ~0.7% (2 SE) are noise --
  the harness prints that reminder next to every table.
* Holdout accuracy is measured through the *deployed* path (scripted DeployNet
  semantics: TTA + fp32) so the sweep optimises the thing that is scored.
"""

import argparse
import ast
import copy
import itertools
import json
import time

import numpy as np
import torch

import speedrun as S

HOLDOUT_PER_CLASS = 500
SE_5K = (0.94 * 0.06 / 5000) ** 0.5


def make_split(seed=0):
    x, y = S.load_uint8("train")
    yc = y.cpu().numpy()
    hold = np.concatenate([np.where(yc == c)[0][::10][:HOLDOUT_PER_CLASS] for c in range(10)])
    mask = np.ones(len(yc), bool)
    mask[hold] = False
    tr = np.where(mask)[0]
    dev = x.device
    return (x[torch.from_numpy(tr).to(dev)], y[torch.from_numpy(tr).to(dev)],
            x[torch.from_numpy(hold).to(dev)], y[torch.from_numpy(hold).to(dev)])


@torch.no_grad()
def holdout_acc(net, hx, hy, tta):
    """Accuracy through the deployment wrapper (eager, fp32) -- same math as the ScriptModule."""
    dep = S.DeployNet(copy.deepcopy(net).float().eval(), tta=tta).cuda().eval()
    correct = 0
    for i in range(0, hx.size(0), 500):
        xb = S.standardize(hx[i : i + 500].clone(), torch.float32)
        correct += (dep(xb).argmax(1) == hy[i : i + 500]).sum().item()
    return correct / hx.size(0)


def run(cfg, tx, ty, hx, hy, probe_last=4):
    """Train one config; return {epoch: holdout_acc} for the last `probe_last` epochs."""
    ep_total = cfg.get("epochs", S.DEFAULTS["epochs"])
    probes = set(range(max(1, ep_total - probe_last + 1), ep_total + 1))
    out = {}
    t0 = time.time()

    def hook(ep, net):
        if ep in probes:
            out[ep] = holdout_acc(net, hx, hy, cfg.get("tta", S.DEFAULTS["tta"]))

    S.train(cfg, tx, ty, log=lambda *a: None, eval_hook=hook)
    out["_sec"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help="python dict literal: name -> list of cfg overrides")
    ap.add_argument("--base", default="{}")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--probe", type=int, default=4)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    base = ast.literal_eval(args.base)
    grid = ast.literal_eval(args.grid)
    tx, ty, hx, hy = make_split()
    print(f"[sweep] train={tx.size(0)} holdout={hx.size(0)}  1 SE = {SE_5K*100:.2f}% "
          f"-> treat gaps < {2*SE_5K*100:.2f}% as noise")

    results = {}
    for name, override in grid.items():
        cfg = dict(base)
        cfg.update(override)
        per_seed = []
        for s in args.seeds:
            c = dict(cfg)
            c["seed"] = s
            r = run(c, tx, ty, hx, hy, args.probe)
            per_seed.append(r)
        finals = [r[max(k for k in r if isinstance(k, int))] for r in per_seed]
        best_any = [max(v for k, v in r.items() if isinstance(k, int)) for r in per_seed]
        results[name] = dict(override=override, finals=finals, mean=float(np.mean(finals)),
                             std=float(np.std(finals)), best_mean=float(np.mean(best_any)),
                             sec=float(np.mean([r["_sec"] for r in per_seed])),
                             curves=[{str(k): v for k, v in r.items() if isinstance(k, int)} for r in per_seed])
        print(f"  {name:28s} final={np.mean(finals)*100:6.2f}% +-{np.std(finals)*100:.2f} "
              f"(seeds {[f'{f*100:.2f}' for f in finals]})  {np.mean([r['_sec'] for r in per_seed]):.1f}s")

    order = sorted(results.items(), key=lambda kv: -kv[1]["mean"])
    print("\n[sweep] ranking by mean holdout accuracy")
    for k, v in order:
        print(f"  {v['mean']*100:6.2f}%  {k}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
