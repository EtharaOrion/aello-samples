"""Re-evaluate a checkpoint ladder with the *exact* pipeline the grader uses.

The grader loads each rung with `torch.jit.load(path).eval()` on the GPU and calls
`model(x)` on float32 NCHW batches of 512 test PNGs pushed through

    transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.4914,0.4822,0.4465),
                                             (0.2470,0.2435,0.2616))])

then takes `argmax(1)`.  This script reproduces that literally -- reading the PNGs off
disk with torchvision's ImageFolder rather than reusing the training cache -- so that a
mismatch in class ordering, channel order or normalization cannot hide.
"""
import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TEST_DIR = "/workspace/data/cifar10/test"


def load_test(batch_size=512):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    ds = datasets.ImageFolder(TEST_DIR, transform=tf)
    # num_workers=0: the container's /dev/shm is small and worker processes die on it.
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    xs, ys = [], []
    for xb, yb in dl:
        xs.append(xb)
        ys.append(yb)
    return torch.cat(xs).cuda(), torch.cat(ys).cuda(), ds.classes


@torch.no_grad()
def accuracy(path, x, y, batch_size=512):
    model = torch.jit.load(path).eval().cuda()
    correct = 0
    for i in range(0, len(x), batch_size):
        pred = model(x[i : i + batch_size]).argmax(1)
        correct += (pred == y[i : i + batch_size]).sum().item()
    return correct / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/periodic_checkpoints")
    ap.add_argument("--out", default=None)
    ap.add_argument("--every", type=int, default=1)
    args = ap.parse_args()

    x, y, classes = load_test()
    print(f"test set: {tuple(x.shape)} range=[{x.min():.3f},{x.max():.3f}] classes={classes}",
          flush=True)

    files = sorted(f for f in os.listdir(args.dir) if f.startswith("ckpt_epoch_"))
    rows = []
    first_cross = None
    for i, f in enumerate(files):
        if i % args.every and f != files[-1]:
            continue
        p = os.path.join(args.dir, f)
        acc = accuracy(p, x, y)
        mtime = os.stat(p).st_mtime
        rows.append(dict(file=f, acc=acc, mtime=mtime))
        if first_cross is None and acc >= 0.94:
            first_cross = (f, acc)
        print(f"{f}  acc={acc:.4f}", flush=True)

    if rows:
        # Reproduce the grader's clock: t* = (mtime[N] - mtime[1]) * N / (N-1).
        m = {r["file"]: r["mtime"] for r in rows}
        base = min(r["mtime"] for r in rows)
        span = max(r["mtime"] for r in rows) - base
        print(f"\ncheckpoint span (first->last graded rung): {span:.2f}s", flush=True)
        if first_cross:
            n = int(first_cross[0].split("_")[-1].split(".")[0])
            first_file = files[0]
            t_star = (m[first_cross[0]] - os.stat(os.path.join(args.dir, first_file)).st_mtime)
            if n > 1:
                t_star = t_star * n / (n - 1)
            print(f"first crossing: {first_cross[0]} acc={first_cross[1]:.4f} "
                  f"-> t* ~= {t_star:.2f}s", flush=True)
        else:
            print("NO CHECKPOINT REACHED 94%", flush=True)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
