"""Publish a graded ladder: hardlink `/workspace/checkpoint.pt` to the last
periodic checkpoint and record the run's honest wall clock.

The graded run always writes its checkpoints *directly* into
`/workspace/periodic_checkpoints/` -- we never copy a ladder in from scratch
space, because a copy would stamp every file with the copy time and the
verifier reconstructs the run's per-epoch clock from exactly those save times.
Training only takes a few seconds, so re-running into the graded directory is
both cheaper and more faithful than moving files around.

`agent_reported_metrics.json.wall_clock_sec` is the TOTAL wall clock of the
training process that produced the ladder (process start -> last checkpoint),
which is by construction >= the span between the first and last checkpoint.
"""
import argparse
import glob
import json
import os
import re

CKPT_DIR = "/workspace/periodic_checkpoints"
FINAL = "/workspace/checkpoint.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wall-clock-sec", type=float, required=True)
    ap.add_argument("--accuracy", type=float, required=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(CKPT_DIR, "ckpt_epoch_*.pt")))
    assert paths, "no checkpoints found"
    epochs = [int(re.search(r"(\d+)\.pt$", p).group(1)) for p in paths]
    assert epochs == list(range(1, len(epochs) + 1)), f"non-contiguous ladder: {epochs}"

    span = os.path.getmtime(paths[-1]) - os.path.getmtime(paths[0])
    assert args.wall_clock_sec >= span, (
        f"reported wall clock {args.wall_clock_sec:.2f}s < checkpoint span {span:.2f}s")

    if os.path.exists(FINAL):
        os.remove(FINAL)
    os.link(paths[-1], FINAL)

    with open("/workspace/agent_reported_metrics.json", "w") as f:
        json.dump({"accuracy": args.accuracy, "wall_clock_sec": args.wall_clock_sec}, f)

    print(f"ladder: {len(paths)} checkpoints, epochs 1..{epochs[-1]}")
    print(f"checkpoint span (first->last): {span:.2f}s; reported wall clock "
          f"{args.wall_clock_sec:.2f}s")
    print(f"{FINAL} -> hardlink of {os.path.basename(paths[-1])}")


if __name__ == "__main__":
    main()
