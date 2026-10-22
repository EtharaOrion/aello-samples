"""In-process hyperparameter search on a held-out slice of the TRAIN split.

Every configuration is trained with ``--holdout 5000``: the model never sees the
5,000 validation images during training, and the CIFAR-10 test split is not read
at all.  Configurations are compared by mean accuracy over `--seeds` random
seeds, because run-to-run noise on this setup is large enough (~0.4 pp) that
single-run comparisons are not informative.

    python sweep.py NAME --seeds 2 --epochs 16
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch

import train as T

RESULTS = "/workspace/sweep_results"


def run(name, overrides, seeds, base_epochs, out_root="/workspace/scratch_ckpt/sweep"):
    parser = T.get_parser()
    rows = []
    for label, ov in overrides:
        accs = []
        spe = []
        for s in range(seeds):
            argv = ["--out", os.path.join(out_root, label.replace("/", "_")),
                    "--holdout", "5000", "--epochs", str(ov.get("epochs", base_epochs)),
                    "--seed", str(s)]
            for k, v in ov.items():
                if k == "epochs":
                    continue
                argv += [f"--{k}", str(v)]
            cfg = parser.parse_args(argv)
            summ = T.train(cfg)
            accs.append(summ["acc"])
            spe.append(summ["sec_per_epoch"])
            torch.cuda.empty_cache()
        rows.append({"label": label, "overrides": ov, "accs": accs,
                     "mean": statistics.mean(accs),
                     "std": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
                     "sec_per_epoch": statistics.mean(spe)})
        r = rows[-1]
        print(f"  {label:<34} mean={r['mean']*100:6.2f}  "
              f"({' '.join(f'{a*100:.2f}' for a in accs)})  spe={r['sec_per_epoch']:.3f}",
              flush=True)
    rows.sort(key=lambda r: -r["mean"])
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"{name}.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"BEST[{name}] {rows[0]['label']} mean={rows[0]['mean']*100:.2f}", flush=True)
    return rows
