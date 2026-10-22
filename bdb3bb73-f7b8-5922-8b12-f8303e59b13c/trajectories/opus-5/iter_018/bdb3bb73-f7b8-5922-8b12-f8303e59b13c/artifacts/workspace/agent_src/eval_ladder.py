"""Re-evaluate a checkpoint ladder with the verifier's exact input pipeline.

This deliberately does NOT reuse the training code's data path.  It reads the
test PNGs through ``ImageFolder`` and ``ToTensor() + Normalize(...)``, loads
each checkpoint with ``torch.jit.load``, and takes ``argmax(1)`` -- i.e. the
same contract the verifier uses -- so that a checkpoint which scores well here
cannot be one that only works under my own normalization.
"""

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TEST_DIR = "/workspace/data/cifar10/test"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def load_test():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ds = datasets.ImageFolder(TEST_DIR, transform=tf)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs).cuda(), torch.cat(ys).cuda()


@torch.no_grad()
def accuracy(path, x, y, bs=512):
    model = torch.jit.load(path).eval().cuda()
    correct = 0
    for i in range(0, len(x), bs):
        correct += (model(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
    return correct / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/periodic_checkpoints")
    ap.add_argument("--out", default="")
    ap.add_argument("--first", type=int, default=1, help="only evaluate every k-th epoch")
    args = ap.parse_args()

    x, y = load_test()
    files = sorted(f for f in os.listdir(args.dir) if f.startswith("ckpt_epoch_"))
    if not files:
        sys.exit(f"no checkpoints in {args.dir}")
    rows, crossed = [], None
    for i, f in enumerate(files):
        ep = int(f[len("ckpt_epoch_"):-3])
        if args.first > 1 and ep % args.first and ep != len(files):
            continue
        p = os.path.join(args.dir, f)
        acc = accuracy(p, x, y)
        rows.append({"epoch": ep, "acc": acc, "mtime": os.path.getmtime(p)})
        if crossed is None and acc >= 0.94:
            crossed = rows[-1]
        print(f"{f}  acc={acc:.4f}", flush=True)

    span = rows[-1]["mtime"] - rows[0]["mtime"]
    n_last = rows[-1]["epoch"]
    summary = {"dir": args.dir, "n_ckpt": len(files), "best": max(r["acc"] for r in rows),
               "last": rows[-1]["acc"], "cross_epoch": crossed["epoch"] if crossed else None,
               "cross_acc": crossed["acc"] if crossed else None,
               "ladder_span_sec": span, "sec_per_epoch": span / max(1, n_last - 1)}
    if crossed:
        # Mirror of the verifier's clock: span from epoch 1 to the crossing
        # epoch, extrapolated back over the mean epoch duration.
        t = (crossed["mtime"] - rows[0]["mtime"]) * crossed["epoch"] / (crossed["epoch"] - 1) \
            if crossed["epoch"] > 1 else float("nan")
        summary["t_star_est"] = t
        summary["S_outcome_est"] = max(0.0, min(1.0, (70.0 - t) / (70.0 - 28.0)))
    print("LADDER " + json.dumps(summary), flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"summary": summary, "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
