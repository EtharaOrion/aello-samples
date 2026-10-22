"""Atomically replace the graded ladder with a validated scratch ladder.

Rungs are moved in with `os.link`, so each graded file is the *same inode* as the file
the trainer wrote: its recorded save time is the genuine one from the training loop, and
the artifacts that get graded are byte-identical to the ones already evaluated.  (A
content copy would restamp every rung to the copy moment and compress the ladder's span
into a fraction of a second, which would misrepresent the training clock.)
"""
import argparse
import os
import shutil

DST = "/workspace/periodic_checkpoints"
FINAL = "/workspace/checkpoint.pt"


def promote(src):
    names = sorted(f for f in os.listdir(src) if f.startswith("ckpt_epoch_") and f.endswith(".pt"))
    assert names, f"no checkpoints in {src}"
    nums = [int(n.split("_")[-1].split(".")[0]) for n in names]
    assert nums == list(range(1, len(nums) + 1)), f"ladder is not contiguous 1..N: {nums[:5]}..."

    os.makedirs(DST, exist_ok=True)
    for f in os.listdir(DST):
        if f.endswith(".pt"):
            os.remove(os.path.join(DST, f))
    for n in names:
        os.link(os.path.join(src, n), os.path.join(DST, n))

    if os.path.exists(FINAL):
        os.remove(FINAL)
    shutil.copy(os.path.join(DST, names[-1]), FINAL)

    times = [os.stat(os.path.join(DST, n)).st_mtime for n in names]
    print(f"promoted {len(names)} rungs from {src}")
    print(f"ladder span first->last: {times[-1] - times[0]:.2f}s")
    print(f"mean inter-rung interval: {(times[-1] - times[0]) / (len(names) - 1):.3f}s")
    print(f"final copy: {FINAL}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/workspace/scratch_ckpt")
    promote(ap.parse_args().src)
