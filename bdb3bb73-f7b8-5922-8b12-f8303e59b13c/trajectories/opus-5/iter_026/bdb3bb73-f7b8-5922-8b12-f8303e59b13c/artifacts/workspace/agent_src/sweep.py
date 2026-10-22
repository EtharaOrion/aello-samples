"""Runs the pre-registered protocol in experiments/PREREGISTRATION.md.

Everything here trains on the 45,000-image train subset and scores on the
5,000-image train holdout. The test split is never opened.
"""
import argparse, itertools, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, "/workspace")
from train import run, DEFAULTS

LOG = "/workspace/experiments/sweep_log.jsonl"
EPOCHS = 18
HOLDOUT = 5000


def trial(tag, overrides, seed):
    cfg = dict(DEFAULTS)
    cfg.update(epochs=EPOCHS, seed=seed)
    cfg.update(overrides)
    t0 = time.time()
    _, res = run(cfg, "/tmp/none", holdout=HOLDOUT, save_ladder=False, verbose=False)
    rec = dict(tag=tag, seed=seed, overrides={k: (list(v) if isinstance(v, tuple) else v)
                                              for k, v in overrides.items()},
               acc=res["holdout_acc_tta"], acc_plain=res["holdout_acc_plain"],
               epoch_sec=res["mean_epoch_sec"], secs=time.time() - t0)
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{tag:34s} seed={seed} acc={rec['acc']:.4f} ({rec['epoch_sec']:.3f}s/ep)", flush=True)
    return rec


STAGE1 = [
    ("lr",             "lr",               [8.0, 16.0, 22.0]),
    ("weight_decay",   "weight_decay",     [0.0100, 0.0230]),
    ("momentum",       "momentum",         [0.80, 0.90]),
    ("label_smoothing","label_smoothing",  [0.10, 0.30]),
    ("bias_scaler",    "bias_scaler",      [32.0, 128.0]),
    ("bn_momentum",    "bn_momentum",      [0.4, 0.8]),
    ("peak_frac",      "peak_frac",        [0.15, 0.35]),
    ("lr_end",         "lr_end",           [0.02, 0.15]),
    ("pad",            "pad",              [3]),
    ("widths",         "widths",           [(96, 320, 448), (160, 448, 640)]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    a = ap.parse_args()

    if a.stage == "0":
        # noise floor: 6 seeds of the baseline at the screening budget
        for s in range(6):
            trial("base", {}, s)
    elif a.stage == "1":
        for name, key, levels in STAGE1:
            for v in levels:
                trial(f"{name}={v}", {key: v}, 0)
        # epochs is scored separately (it trades against t*)
        for e in (14, 22):
            trial(f"epochs={e}", {"epochs": e}, 0)
    elif a.stage.startswith("cand:"):
        # cand:<json list of {name, overrides}>:<nseeds>
        payload = json.loads(a.stage[len("cand:"):])
        for c in payload["cands"]:
            for s in range(payload["seeds"]):
                ov = {k: (tuple(v) if isinstance(v, list) else v)
                      for k, v in c["overrides"].items()}
                trial(c["name"], ov, s)
    print("STAGE DONE", a.stage, flush=True)


if __name__ == "__main__":
    main()
