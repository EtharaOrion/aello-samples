"""Holdout-only hyperparameter screen, executed exactly as PREREGISTRATION.md commits.

Every run trains on the 45,000-image train remainder and is scored on the fixed
5,000-image class-balanced holdout.  The test split is never loaded here.

Phase 1 establishes the seed-to-seed noise floor of the incumbent; Phase 2 runs
the pre-registered candidate list on the SAME seeds (paired comparison).  The
TTA mixing weight (C2) is swept for free by re-scoring each trained net, so it
costs no extra training runs.
"""

import json
import sys
import time

import torch

sys.path.insert(0, "/workspace/src")
import train as T

TTA_GRID = [0.0, 0.3, 0.5, 0.7]          # 0.0 == mirror-only (no translate branch)
OUT = "/workspace/scratch_ckpt/sweep.jsonl"

BASE = ["--val", "holdout", "--quiet", "--epochs", "20"]

CONFIGS = [
    ("base_e20", [], [0, 1, 2, 3]),          # incumbent + noise floor
    ("e24", ["--epochs", "24"], [0, 1, 2]),
    ("e28", ["--epochs", "28"], [0, 1, 2]),
    ("lr9", ["--lr", "9.0"], [0, 1, 2]),
    ("lr14", ["--lr", "14.0"], [0, 1, 2]),
    ("ls10", ["--label-smoothing", "0.1"], [0, 1, 2]),
]


def main():
    f = open(OUT, "w")
    t0 = time.perf_counter()
    for name, extra, seeds in CONFIGS:
        for s in seeds:
            argv = BASE + extra + ["--seed", str(s)]
            res = T.main(argv)
            net, (xv, yv) = res["_net"], res["_val"]
            accs = {}
            net.tta = False
            accs["notta"] = T.evaluate(net, xv, yv)
            net.tta = True
            for w in TTA_GRID:
                net.tta_w = w
                accs[f"tta{w}"] = T.evaluate(net, xv, yv)
            rec = dict(cfg=name, seed=s, wall=res["train_wall"], accs=accs,
                       elapsed=time.perf_counter() - t0)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{name:10s} seed{s}  wall {res['train_wall']:5.2f}s  " +
                  "  ".join(f"{k}={v*100:.2f}" for k, v in accs.items()), flush=True)
            del net, res
            torch.cuda.empty_cache()
    f.close()


if __name__ == "__main__":
    main()
