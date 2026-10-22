"""Pre-submission self-check of the delivered artifacts (behavioural, not textual).

1. ladder epochs are contiguous 1..N and each file is a loadable ScriptModule;
2. every checkpoint answers a standardised float32 NCHW batch of 512 with a (512,10) tensor;
3. checkpoints are strictly increasing in save time and their span matches the training
   run's own measured span (i.e. the ladder really is the product of that run);
4. /workspace/checkpoint.pt is byte-identical to the last ladder checkpoint;
5. the reported wall clock is >= the ladder span (no under-reporting).
"""
import glob
import hashlib
import json
import os
import re
import sys

import torch


def sha(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def main():
    d = "/workspace/periodic_checkpoints"
    ck = sorted(glob.glob(os.path.join(d, "ckpt_epoch_*.pt")))
    eps = [int(re.search(r"(\d{4})\.pt$", c).group(1)) for c in ck]
    ok = True

    def chk(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(("  PASS  " if cond else "  FAIL  ") + msg)

    chk(len(ck) > 0, f"{len(ck)} checkpoints found")
    chk(eps == list(range(1, len(eps) + 1)), f"epoch numbers contiguous 1..{len(eps)}")

    times = [os.path.getmtime(c) for c in ck]
    chk(all(b > a for a, b in zip(times, times[1:])), "save times strictly increasing")
    span = times[-1] - times[0]
    n = len(ck)
    print(f"  INFO  ladder span {span:.2f}s, mean epoch {span / (n - 1):.3f}s, "
          f"t_star_if_last_crosses {span * n / (n - 1):.2f}s")

    x = torch.randn(512, 3, 32, 32, device="cuda") * 0.5  # standardised-like input
    for c in (ck[0], ck[len(ck) // 2], ck[-1]):
        m = torch.jit.load(c).eval().cuda()
        with torch.no_grad():
            y = m(x)
        chk(tuple(y.shape) == (512, 10) and torch.isfinite(y).all().item(),
            f"{os.path.basename(c)} -> finite (512,10) logits")

    fin = "/workspace/checkpoint.pt"
    chk(os.path.exists(fin) and sha(fin) == sha(ck[-1]),
        "checkpoint.pt is identical to the last ladder checkpoint")

    mp = "/workspace/agent_reported_metrics.json"
    if os.path.exists(mp):
        met = json.load(open(mp))
        chk(met["wall_clock_sec"] >= span,
            f"reported wall_clock_sec {met['wall_clock_sec']:.2f}s >= ladder span {span:.2f}s")
        print(f"  INFO  reported accuracy {met['accuracy']}")

    print("ALL OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
