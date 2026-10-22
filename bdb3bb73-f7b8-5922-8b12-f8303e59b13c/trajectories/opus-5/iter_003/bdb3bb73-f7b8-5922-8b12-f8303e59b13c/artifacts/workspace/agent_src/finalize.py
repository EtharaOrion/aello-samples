"""Produce the graded checkpoint ladder and the advisory metrics file.

The ladder is trained *directly into* ``/workspace/periodic_checkpoints`` rather
than trained elsewhere and copied in.  That matters: ``shutil.copy`` does not
preserve mtimes, so a copied ladder would carry copy-times (a sub-second span)
instead of real per-epoch training times, and the verifier reconstructs its clock
from exactly those mtimes.  Training in place keeps every reported timestamp a
genuine training timestamp.

The old ladder is removed immediately before the new run starts, so the directory
only ever holds checkpoints from a single run.
"""

import argparse
import glob
import json
import os
import shutil
import time

# Process start, captured before torch is imported, so `wall_clock_sec` covers
# the entire training run including imports, data loading and warmup.
T_PROC0 = time.time()

import torch  # noqa: E402

from train_run import make_cfg, train  # noqa: E402

CKPT_DIR = "/workspace/periodic_checkpoints"
FINAL = "/workspace/checkpoint.pt"
METRICS = "/workspace/agent_reported_metrics.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    ap.add_argument("--metrics", default=None, help="write agent_reported_metrics.json here")
    ap.add_argument("--accuracy", type=float, default=None,
                    help="test accuracy of the final checkpoint, measured separately")
    args = ap.parse_args()

    cfg = make_cfg(json.loads(args.cfg))
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # One complete run per directory: clear any previous ladder first.
    stale = sorted(glob.glob(os.path.join(args.ckpt_dir, "ckpt_epoch_*.pt")))
    for p in stale:
        os.remove(p)
    print(f"removed {len(stale)} stale checkpoints from {args.ckpt_dir}", flush=True)

    result, _ = train(cfg, ckpt_dir=args.ckpt_dir, seed=args.seed, log_every=0, quiet=True)

    paths = sorted(glob.glob(os.path.join(args.ckpt_dir, "ckpt_epoch_*.pt")))
    last = paths[-1]
    # Only the graded directory owns /workspace/checkpoint.pt; scratch runs must
    # not leave it pointing at a ladder that is not the graded one.
    if os.path.abspath(args.ckpt_dir) == os.path.abspath(CKPT_DIR):
        if os.path.exists(FINAL):
            os.remove(FINAL)
        shutil.copy(last, FINAL)

    span = os.path.getmtime(paths[-1]) - os.path.getmtime(paths[0])
    wall = os.path.getmtime(paths[-1]) - T_PROC0
    summary = dict(
        n_ckpt=len(paths), epochs=cfg["epochs"], seed=args.seed,
        ckpt_span_sec=span, wall_clock_sec=wall,
        mean_epoch_sec=span / max(1, len(paths) - 1),
        train_loop_sec=result["total_sec"], warmup_sec=result["warmup_sec"],
        cfg=result["cfg"],
    )
    print(json.dumps(summary, indent=2), flush=True)
    with open("/workspace/final_run.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    if args.metrics:
        assert wall >= span, "wall_clock_sec must cover the checkpoint span"
        payload = {"wall_clock_sec": wall}
        if args.accuracy is not None:
            payload["accuracy"] = args.accuracy
        with open(args.metrics, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.metrics}: {payload}", flush=True)


if __name__ == "__main__":
    main()
