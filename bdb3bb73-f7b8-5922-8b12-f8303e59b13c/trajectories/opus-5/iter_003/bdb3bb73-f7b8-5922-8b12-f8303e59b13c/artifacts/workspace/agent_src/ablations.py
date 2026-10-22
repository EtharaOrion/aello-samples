"""Ablations that isolate the contribution of individual design choices.

All of these are measured on the same 5,000-image held-out slice of the *train*
split used by the sweep; the test split is never loaded.

  A. test-time augmentation -- one trained model, three forward-pass variants, so
     the comparison is exactly the TTA effect with zero training noise
  B. patch whitening -- frozen covariance-derived 2x2 filter vs. the same layer
     left randomly initialised and trainable
  C. augmentation -- random translate and horizontal flip switched off
"""

import json
import sys

import torch

from core import evaluate, load_split, standardise
from train_run import make_cfg, train

FINAL_CFG = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
HOLDOUT = 5000
SEEDS = [0, 1]


def holdout_slice():
    raw_x, raw_y = load_split("train")
    images = standardise(raw_x, dtype=torch.float16)
    g = torch.Generator(device="cuda").manual_seed(1234)
    perm = torch.randperm(images.shape[0], device="cuda", generator=g)
    va = perm[-HOLDOUT:]
    return images[va].contiguous(memory_format=torch.channels_last), raw_y[va]


def main():
    val_x, val_y = holdout_slice()
    out = {}

    # -- A: TTA variants of one fixed set of weights ------------------------ #
    tta_rows = []
    for seed in SEEDS:
        _, model = train(make_cfg(FINAL_CFG), holdout=HOLDOUT, seed=seed, log_every=0, quiet=True)
        model.eval()
        row = {}
        for label, (flip, shift) in {
            "none": (False, False),
            "flip": (True, False),
            "flip+shift": (True, True),
        }.items():
            model.tta_flip, model.tta_shift = flip, shift
            row[label] = evaluate(model, val_x, val_y)
        tta_rows.append(row)
        print(f"A tta seed={seed}: {row}", flush=True)
        del model
        torch.cuda.empty_cache()
    out["A_tta"] = {k: sum(r[k] for r in tta_rows) / len(tta_rows) for k in tta_rows[0]}

    # -- B: frozen whitening vs trainable random init ----------------------- #
    import core

    orig_init = core.init_whitening

    def no_whiten_init(model, images, eps=5e-4):
        model.whiten.weight.requires_grad_(True)
        model.whiten.bias.requires_grad_(True)

    b_rows = {}
    for label, fn in [("frozen_whitening", orig_init), ("random_trainable", no_whiten_init)]:
        core.init_whitening = fn
        accs = []
        for seed in SEEDS:
            res, model = train(make_cfg(FINAL_CFG), holdout=HOLDOUT, seed=seed, log_every=0, quiet=True)
            accs.append(res["val_acc"])
            del model
            torch.cuda.empty_cache()
        b_rows[label] = sum(accs) / len(accs)
        print(f"B {label}: {b_rows[label]:.4f}", flush=True)
    core.init_whitening = orig_init
    out["B_whitening"] = b_rows

    # -- C: training augmentation off --------------------------------------- #
    c_rows = {}
    for label, ov in [("full_aug", {}), ("no_translate", {"pad": 0}), ("no_cutout", {"cutout": 0})]:
        accs = []
        for seed in SEEDS:
            cfg = make_cfg({**FINAL_CFG, **ov})
            res, model = train(cfg, holdout=HOLDOUT, seed=seed, log_every=0, quiet=True)
            accs.append(res["val_acc"])
            del model
            torch.cuda.empty_cache()
        c_rows[label] = sum(accs) / len(accs)
        print(f"C {label}: {c_rows[label]:.4f}", flush=True)
    out["C_augmentation"] = c_rows

    out["seeds"] = SEEDS
    out["cfg"] = FINAL_CFG
    with open("/workspace/ablations.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
