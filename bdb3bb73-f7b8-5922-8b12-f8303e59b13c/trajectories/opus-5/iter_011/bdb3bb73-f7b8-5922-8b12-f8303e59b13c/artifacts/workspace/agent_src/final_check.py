"""Pre-submission validation of the graded artefacts.

Asserts the properties the grader depends on, without assuming anything the
training script told us:

 1. the ladder is contiguous 1..N and every file is a loadable ScriptModule
 2. checkpoints differ from each other (i.e. each epoch really was saved, not
    the same weights written 40 times)
 3. checkpoint.pt is byte-identical to the last periodic checkpoint
 4. checkpoint modification times increase monotonically and the first-to-last
    span is consistent with a real ~19 s training run rather than a copy
 5. the wall_clock_sec we report is >= that span
 6. the model consumes the verifier's input convention: standardised float32
    NCHW, batch 512, and reaches >= 94% on the test PNGs at the crossing epoch
"""
import glob
import hashlib
import json
import os
import re

import torch

CKPT_DIR = "/workspace/periodic_checkpoints"
FINAL = "/workspace/checkpoint.pt"
METRICS = "/workspace/agent_reported_metrics.json"


def sha(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read() if limit is None else f.read(limit))
    return h.hexdigest()


def main():
    paths = sorted(glob.glob(os.path.join(CKPT_DIR, "ckpt_epoch_*.pt")))
    epochs = [int(re.search(r"ckpt_epoch_(\d+)\.pt", p).group(1)) for p in paths]
    n = len(paths)
    assert epochs == list(range(1, n + 1)), f"ladder not contiguous 1..N: {epochs}"
    print(f"[1] ladder contiguous 1..{n}  OK")

    # 2. distinct weights per epoch
    digests = [sha(p) for p in paths]
    assert len(set(digests)) == n, "duplicate checkpoint contents"
    print(f"[2] all {n} checkpoints have distinct contents  OK")

    # 3. checkpoint.pt == last periodic checkpoint
    assert sha(FINAL) == digests[-1], "checkpoint.pt != last periodic checkpoint"
    print("[3] checkpoint.pt is a copy of ckpt_epoch_%04d.pt  OK" % n)

    # 4. monotonic save times, plausible span
    ts = [os.path.getmtime(p) for p in paths]
    assert all(b > a for a, b in zip(ts, ts[1:])), "save times not increasing"
    span = ts[-1] - ts[0]
    per_epoch = span / (n - 1)
    assert span > 1.0, "ladder span collapsed -- checkpoints were copied, not saved"
    print(f"[4] save times strictly increasing; span {span:.2f}s "
          f"({per_epoch*1000:.0f} ms/epoch)  OK")

    # 5. reported wall-clock covers the span
    m = json.load(open(METRICS))
    assert m["wall_clock_sec"] >= span, (
        f"reported wall_clock_sec {m['wall_clock_sec']} < ladder span {span}")
    print(f"[5] reported wall_clock_sec {m['wall_clock_sec']:.2f}s >= span "
          f"{span:.2f}s  OK")

    # 6. verifier input convention, end to end
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                  (0.2470, 0.2435, 0.2616))])
    ds = datasets.ImageFolder("/workspace/data/cifar10/test", transform=tf)
    xs, ys = zip(*[(x, y) for x, y in DataLoader(ds, batch_size=512, num_workers=0)])
    x, y = torch.cat(xs).cuda(), torch.cat(ys).cuda()
    assert x.dtype == torch.float32 and x.shape[1:] == (3, 32, 32)
    print(f"[6] test input: {tuple(x.shape)} float32, range "
          f"[{x.min():.2f}, {x.max():.2f}] (standardised)")

    cross = None
    for ep, p in zip(epochs, paths):
        model = torch.jit.load(p).eval().cuda()
        correct = 0
        with torch.no_grad():
            for i in range(0, x.shape[0], 512):
                correct += (model(x[i:i + 512]).argmax(1) == y[i:i + 512]).sum().item()
        acc = correct / x.shape[0]
        if acc >= 0.94:
            cross = (ep, acc)
            break
    assert cross is not None, "no checkpoint reaches 94%"
    ep, acc = cross
    t_star = (ts[ep - 1] - ts[0]) * ep / (ep - 1)
    print(f"    first epoch >= 94%: {ep} at {acc*100:.2f}%  ->  t* = {t_star:.2f}s  OK")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
