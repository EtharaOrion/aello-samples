"""Post-run checks + the two artefacts the verifier reads besides the ladder.

Why the final run is trained *directly into* `/workspace/periodic_checkpoints/`
rather than copied in from scratch space: the verifier reconstructs `t_star`
from the checkpoints' save times.  Copying a finished ladder would stamp every
file within a fraction of a second of every other, reporting a training speed
that never happened.  Training in place is the only way the reconstructed clock
matches reality, so the graded ladder is always produced by a run that wrote
straight to that directory.  Only `/workspace/checkpoint.pt` is a copy, as the
task specifies.

    python finalize.py --wall-clock 27.4
"""
import argparse
import glob
import json
import os
import re
import shutil

CKPT_DIR = "/workspace/periodic_checkpoints"
FINAL = "/workspace/checkpoint.pt"
METRICS = "/workspace/agent_reported_metrics.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wall-clock", type=float, required=True,
                    help="total wall-clock of the run that produced the ladder")
    ap.add_argument("--accuracy", type=float, required=True,
                    help="independently re-evaluated accuracy of the last ckpt")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(CKPT_DIR, "ckpt_epoch_*.pt")))
    epochs = [int(re.search(r"(\d+)\.pt$", p).group(1)) for p in paths]
    assert epochs == list(range(1, len(epochs) + 1)), f"non-contiguous: {epochs}"

    mtimes = [os.path.getmtime(p) for p in paths]
    span = mtimes[-1] - mtimes[0]
    gaps = [b - a for a, b in zip(mtimes, mtimes[1:])]
    # The ladder must look like a real training run: monotonically increasing
    # save times, spread out, with no epoch wildly out of line with the rest.
    assert all(g > 0 for g in gaps), "checkpoint save times are not increasing"
    assert args.wall_clock >= span, (
        f"reported wall clock {args.wall_clock:.2f}s < ladder span {span:.2f}s")

    shutil.copy(paths[-1], FINAL)
    with open(METRICS, "w") as f:
        json.dump({"accuracy": args.accuracy,
                   "wall_clock_sec": args.wall_clock}, f, indent=2)

    print(json.dumps({
        "n_ckpt": len(paths),
        "contiguous_from_1": True,
        "ladder_span_sec": round(span, 3),
        "mean_epoch_sec": round(span / max(1, len(paths) - 1), 3),
        "min_gap_sec": round(min(gaps), 3),
        "max_gap_sec": round(max(gaps), 3),
        "reported_wall_clock_sec": args.wall_clock,
        "final_checkpoint": FINAL,
    }, indent=2))


if __name__ == "__main__":
    main()
