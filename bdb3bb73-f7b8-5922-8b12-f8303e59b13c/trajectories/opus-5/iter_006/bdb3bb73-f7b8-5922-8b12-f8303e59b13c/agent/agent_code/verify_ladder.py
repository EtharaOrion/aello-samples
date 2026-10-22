"""Structural self-check of the deliverable, run before declaring the run done.

Checks, in order:
  1. Checkpoints are numbered contiguously 1..N (the verifier needs epoch 0001 and
     an unbroken run up to the crossing epoch, else S_outcome = 0).
  2. Save times are strictly increasing and evenly spaced -- i.e. the ladder comes
     from one training run, not from files copied into place after the fact.
  3. Every checkpoint loads via `torch.jit.load(path).eval()` and answers a
     float32 NCHW batch of 512, which is exactly how the verifier calls it.
  4. Consecutive checkpoints hold DIFFERENT weights.  This is the check that the
     script-once / save-every-epoch pattern really snapshots live parameters
     rather than re-serializing one frozen set.
  5. /workspace/checkpoint.pt is byte-identical to the last periodic checkpoint.
"""
import glob
import hashlib
import os
import re
import sys

import torch

CKPT_DIR = "/workspace/periodic_checkpoints"
FINAL = "/workspace/checkpoint.pt"


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    paths = sorted(glob.glob(os.path.join(CKPT_DIR, "ckpt_epoch_*.pt")))
    eps = [int(re.search(r"(\d+)\.pt$", p).group(1)) for p in paths]
    ok = True

    # 1. contiguity
    want = list(range(1, len(paths) + 1))
    print(f"[1] {len(paths)} checkpoints, epochs {eps[0]}..{eps[-1]}  "
          f"contiguous-from-1: {eps == want}")
    ok &= eps == want

    # 2. monotone, evenly spaced save times
    mt = [os.path.getmtime(p) for p in paths]
    d = [b - a for a, b in zip(mt, mt[1:])]
    span = mt[-1] - mt[0]
    print(f"[2] span {span:.2f}s  per-epoch min/mean/max = "
          f"{min(d):.3f}/{span / (len(d)):.3f}/{max(d):.3f}s  "
          f"monotone: {all(x > 0 for x in d)}")
    ok &= all(x > 0 for x in d)
    # a copied-in ladder would show ~0s gaps; a genuine one shows real epoch times
    ok &= min(d) > 0.02

    # 3 + 4. loadable under the verifier's calling convention, and evolving
    x = torch.randn(512, 3, 32, 32, device="cuda")  # float32 NCHW, verifier batch size
    prev = None
    n_same = 0
    for p in paths:
        m = torch.jit.load(p).eval().cuda()
        with torch.no_grad():
            out = m(x)
        assert out.shape == (512, 10), f"{p}: bad output shape {out.shape}"
        assert out.dtype == torch.float32, f"{p}: bad dtype {out.dtype}"
        assert torch.isfinite(out).all(), f"{p}: non-finite logits"
        w = m.head.weight.float()
        if prev is not None and torch.equal(prev, w):
            n_same += 1
        prev = w
    print(f"[3] all {len(paths)} load, return float32 (512,10), finite")
    print(f"[4] consecutive-identical weight pairs: {n_same} (must be 0)")
    ok &= n_same == 0

    # 5. final copy
    same = sha(FINAL) == sha(paths[-1])
    print(f"[5] checkpoint.pt == {os.path.basename(paths[-1])}: {same}")
    ok &= same

    print("\nLADDER OK" if ok else "\nLADDER FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
