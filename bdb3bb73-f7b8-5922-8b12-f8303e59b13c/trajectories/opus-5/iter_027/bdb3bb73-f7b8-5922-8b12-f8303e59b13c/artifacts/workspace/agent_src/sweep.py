"""Held-out sweep driver.

Loads the (train-only) data once per `pad` value and runs many configurations
in-process, so a single configuration costs ~0.4 s/epoch rather than paying
python + CUDA startup each time.  Nothing here ever reads the test split.

  python sweep.py noise                     # step 0: seed noise floor
  python sweep.py stage1                    # H1..H8 single-seed screening
  python sweep.py confirm cfgA cfgB ...     # 3-seed confirmation of finalists
"""
import json
import os
import sys
import time

import torch

from train import get_parser, load_train, train

OUT = "/workspace/experiments/sweep.jsonl"
_DATA = {}


def get_data(holdout, pad):
    key = (holdout, pad)
    if key not in _DATA:
        _DATA.clear()  # keep at most one padded copy resident
        torch.cuda.empty_cache()
        _DATA[key] = load_train(holdout, 0, pad)
    return _DATA[key]


def run(tag, seed=0, holdout=5000, epochs=16, **kw):
    cfg = get_parser().parse_args([])
    cfg.tag, cfg.seed, cfg.holdout, cfg.epochs = tag, seed, holdout, epochs
    cfg.no_ckpt, cfg.eval_curve, cfg.warmup = True, True, 1
    for k, v in kw.items():
        setattr(cfg, k, v)
    res = train(cfg, data=get_data(holdout, cfg.pad))
    with open(OUT, "a") as f:
        f.write(json.dumps(res) + "\n")
    curve = [round(100 * r["val"], 2) for r in res["epoch_log"] if "val" in r]
    print(f"  {tag:28s} seed{seed} val={100 * res['val_acc']:.2f}  curve={curve[-6:]}",
          flush=True)
    return res


# --------------------------------------------------------------------------
# Baseline = the config that produced `safety_v0`.
BASE = dict(epochs=16)


def step_noise():
    print("== step 0: seed noise floor (baseline config, 5 seeds) ==", flush=True)
    accs = [run("noise_base", seed=s, **BASE)["val_acc"] for s in range(5)]
    a = torch.tensor(accs) * 100
    print(f"  mean={a.mean():.3f}  sd={a.std():.3f}  min={a.min():.2f} max={a.max():.2f}")
    print(f"  => 3-seed SEM = {a.std() / 3 ** 0.5:.3f} pp", flush=True)


STAGE1 = {
    # H1 schedule length
    "h1_ep20": dict(epochs=20), "h1_ep24": dict(epochs=24), "h1_ep30": dict(epochs=30),
    # H2 flat low-LR tail (on the best-guess longer schedule)
    "h2_tail2": dict(epochs=24, tail_epochs=2), "h2_tail3": dict(epochs=24, tail_epochs=3),
    # H3 final LR
    "h3_end02": dict(epochs=24, end_mult=0.02),
    # H4 peak LR
    "h4_lr9": dict(epochs=24, lr=9.0), "h4_lr14": dict(epochs=24, lr=14.0),
    # H5 weight decay
    "h5_wd020": dict(epochs=24, wd=0.020), "h5_wd025": dict(epochs=24, wd=0.025),
    # H6 translation strength
    "h6_pad3": dict(epochs=24, pad=3),
    # H7 label smoothing
    "h7_ls15": dict(epochs=24, label_smoothing=0.15),
    # H8 whitening-bias window
    "h8_wb5": dict(epochs=24, whiten_bias_epochs=5.0),
}


def step_stage1():
    print("== step 1: single-seed screening (H1..H8) ==", flush=True)
    run("base_ep16", seed=0, **BASE)
    # group by pad so the padded dataset is materialised at most twice
    for name in sorted(STAGE1, key=lambda k: STAGE1[k].get("pad", 2)):
        run(name, seed=0, **STAGE1[name])


STAGE2 = {
    # Stage-1 winners composed: low final LR + flat tail, then longer schedules
    # with the stronger augmentation that a longer schedule can absorb.
    "s2_e24_end02_t3":      dict(epochs=24, end_mult=0.02, tail_epochs=3),
    "s2_e30_end02_t4":      dict(epochs=30, end_mult=0.02, tail_epochs=4),
    "s2_e30_pad4":          dict(epochs=30, end_mult=0.02, tail_epochs=4, pad=4),
    "s2_e30_pad4_cut12":    dict(epochs=30, end_mult=0.02, tail_epochs=4, pad=4,
                                 cutout=12, lr=9.0, wd=0.012),
    "s2_e40_pad4_cut12":    dict(epochs=40, end_mult=0.02, tail_epochs=5, pad=4,
                                 cutout=12, lr=9.0, wd=0.012),
    "s2_e40_pad4_cut8":     dict(epochs=40, end_mult=0.02, tail_epochs=5, pad=4,
                                 cutout=8, lr=9.0, wd=0.012),
    # stack the other (weaker, 1-seed) stage-1 signals on top of the winner
    "s2_e24_stack":         dict(epochs=24, end_mult=0.02, tail_epochs=3, lr=14.0,
                                 label_smoothing=0.15, whiten_bias_epochs=5.0, pad=3),
    "s2_e30_pad4_cut12_lr14": dict(epochs=30, end_mult=0.02, tail_epochs=4, pad=4,
                                 cutout=12, lr=14.0, wd=0.012),
}


def step_stage2():
    print("== step 2: compose stage-1 winners + stronger aug ==", flush=True)
    for name in sorted(STAGE2, key=lambda k: STAGE2[k].get("pad", 2)):
        run(name, seed=0, **STAGE2[name])


if __name__ == "__main__":
    t = time.time()
    cmd = sys.argv[1]
    if cmd == "noise":
        step_noise()
    elif cmd == "stage1":
        step_stage1()
    elif cmd == "stage2":
        step_stage2()
    elif cmd == "confirm":
        # `confirm name=json_kwargs ...`
        for spec in sys.argv[2:]:
            name, kwjson = spec.split("=", 1)
            kw = json.loads(kwjson)
            for s in range(3):
                run("cf_" + name, seed=s, **kw)
    print(f"[{cmd} done in {time.time() - t:.0f}s]")
