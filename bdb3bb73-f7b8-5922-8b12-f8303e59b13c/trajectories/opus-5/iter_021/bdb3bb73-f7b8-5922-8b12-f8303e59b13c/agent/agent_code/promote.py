"""Train the frozen winning config on all 50 000 images and, if the resulting
ladder is *valid*, install it as the graded ladder.

Selection of the config happened on the train-only holdout and is already
frozen before this script runs.  The test split is used here for two things
only: (a) confirming the artefact loads and behaves under the verifier's input
convention, (b) the validity gate below.  It never chooses between configs.

Validity gate for a swap:
  * checkpoints are contiguous 1..N
  * at least one checkpoint re-evaluates at >= 94.0 %
  * reconstructed t* <= 26 s (the verifier's S_outcome saturates at 28 s)
Otherwise the incumbent ladder stays exactly where it is.
"""
import argparse
import glob
import json
import os
import shutil
import sys

import torch

import train
import verify_ladder

GRADED = "/workspace/periodic_checkpoints"
STAGING = "/workspace/staging_ckpt"
T_STAR_LIMIT = 26.0


def evaluate_dir(d):
    dev = torch.device("cuda")
    X, Y = verify_ladder.test_tensor(dev)
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "ckpt_epoch_*.pt"))):
        ep = int(os.path.basename(p)[11:15])
        rows.append(dict(epoch=ep, acc=verify_ladder.acc_of(p, X, Y),
                         mtime=os.path.getmtime(p)))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--lr", type=float, default=11.5)
    p.add_argument("--lr-end", type=float, default=0.0)
    p.add_argument("--wd", type=float, default=0.0153)
    p.add_argument("--label-smoothing", type=float, default=0.2)
    p.add_argument("--translate", type=int, default=2)
    p.add_argument("--peak-frac", type=float, default=0.23)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    shutil.rmtree(STAGING, ignore_errors=True)
    os.makedirs(STAGING, exist_ok=True)

    cfg = train.get_args.__defaults__  # unused; build namespace explicitly
    import argparse as ap
    c = ap.Namespace(
        epochs=a.epochs, bs=1000, lr=a.lr, wd=a.wd, momentum=0.85,
        bias_scaler=64.0, label_smoothing=a.label_smoothing,
        widths=[128, 384, 512], scale=1.0 / 9.0, bn_momentum=0.1,
        peak_frac=a.peak_frac, lr_start=0.2, lr_end=a.lr_end,
        translate=a.translate, tta_level=2, seed=a.seed, holdout=0,
        ckpt_dir=STAGING, save_ckpts=1, eval_every=0, tag="graded_candidate",
        out_json="/workspace/runs.jsonl")
    res = train.run(c)
    wall = res["wall_clock_sec"]

    rows = evaluate_dir(STAGING)
    eps = [r["epoch"] for r in rows]
    contiguous = eps == list(range(1, len(rows) + 1))
    cross = next((r for r in rows if r["acc"] >= 0.94), None)
    t_star = None
    if cross and cross["epoch"] > 1:
        n = cross["epoch"]
        t_star = (cross["mtime"] - rows[0]["mtime"]) * n / (n - 1)

    ok = contiguous and cross is not None and t_star is not None \
        and t_star <= T_STAR_LIMIT
    report = dict(epochs=a.epochs, seed=a.seed, wall_clock_sec=wall,
                  n_ckpt=len(rows), contiguous=contiguous,
                  final_acc=rows[-1]["acc"], best_acc=max(r["acc"] for r in rows),
                  first_cross_epoch=cross["epoch"] if cross else None,
                  t_star_sec=t_star, ladder_span_sec=rows[-1]["mtime"] - rows[0]["mtime"],
                  promoted=bool(ok and not a.dry_run))
    print(json.dumps(report, indent=2), flush=True)

    if a.dry_run:
        return
    if not ok:
        print("VALIDITY GATE FAILED -- incumbent ladder left untouched.")
        sys.exit(1)

    # delete-then-copy in one step so the graded dir always holds one full run
    shutil.rmtree(GRADED, ignore_errors=True)
    shutil.copytree(STAGING, GRADED)
    shutil.copy(os.path.join(GRADED, f"ckpt_epoch_{len(rows):04d}.pt"),
                "/workspace/checkpoint.pt")
    with open("/workspace/agent_reported_metrics.json", "w") as f:
        json.dump({"accuracy": rows[-1]["acc"], "wall_clock_sec": wall}, f, indent=2)
    with open("/workspace/ladder_eval.json", "w") as f:
        json.dump(dict(summary=report, rows=rows), f, indent=2)
    print("PROMOTED.")


if __name__ == "__main__":
    main()
