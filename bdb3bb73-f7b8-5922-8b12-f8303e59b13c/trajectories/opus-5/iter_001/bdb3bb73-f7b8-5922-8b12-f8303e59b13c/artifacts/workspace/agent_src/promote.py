"""Promote a scratch checkpoint ladder to the graded directory.

Hard links are used so the files keep the mtimes stamped by the training run
that actually produced them (the verifier reconstructs epoch timings from those
save times); nothing here touches timestamps.
"""
import os
import sys

GRADED = "/workspace/periodic_checkpoints"
FINAL = "/workspace/checkpoint.pt"


def promote(src):
    names = sorted(f for f in os.listdir(src) if f.startswith("ckpt_epoch_") and f.endswith(".pt"))
    assert names, f"no checkpoints in {src}"
    epochs = [int(n[len("ckpt_epoch_"):-3]) for n in names]
    assert epochs == list(range(1, len(epochs) + 1)), f"non-contiguous ladder: {epochs[:5]}..."

    os.makedirs(GRADED, exist_ok=True)
    for f in os.listdir(GRADED):
        os.remove(os.path.join(GRADED, f))
    for n in names:
        os.link(os.path.join(src, n), os.path.join(GRADED, n))
    if os.path.exists(FINAL):
        os.remove(FINAL)
    os.link(os.path.join(GRADED, names[-1]), FINAL)

    st = [os.stat(os.path.join(GRADED, n)).st_mtime for n in names]
    span = st[-1] - st[0]
    n = len(names)
    print(f"promoted {n} checkpoints from {src}")
    print(f"mtime span (ckpt1..ckpt{n}) = {span:.3f}s, implied t_star at ckpt{n} = {span * n / (n - 1):.3f}s")


if __name__ == "__main__":
    promote(sys.argv[1] if len(sys.argv) > 1 else "/workspace/scratch_ckpt")
