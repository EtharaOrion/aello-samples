"""Replicate the grader's evaluation pipeline exactly, on saved checkpoints.

The grader loads each checkpoint with torch.jit.load(path).eval() on the GPU
and calls model(x) with float32 NCHW batches of size 512 built from the 32x32
test PNGs via
    transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.4914,0.4822,0.4465),
                                             (0.2470,0.2435,0.2616))])
then takes argmax(1). Nothing here may differ from that.
"""
import argparse
import glob
import os
import re

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TEST_DIR = "/workspace/data/cifar10/test"


def make_loader(batch_size=512, workers=0):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    ds = datasets.ImageFolder(TEST_DIR, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=True)


def cache_batches(loader, device="cuda"):
    return [(x.to(device, non_blocking=True), y.to(device, non_blocking=True))
            for x, y in loader]


@torch.no_grad()
def score(path, batches):
    model = torch.jit.load(path).eval().cuda()
    correct = total = 0
    for x, y in batches:
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/periodic_checkpoints")
    ap.add_argument("--last", type=int, default=0,
                    help="only score the final N checkpoints (0 = all)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "ckpt_epoch_*.pt")))
    if args.last:
        paths = paths[-args.last:]
    batches = cache_batches(make_loader())
    mtimes = {}
    first = None
    for p in paths:
        acc = score(p, batches)
        ep = int(re.search(r"(\d+)\.pt$", p).group(1))
        mtimes[ep] = os.path.getmtime(p)
        flag = ""
        if acc >= 0.94 and first is None:
            first, flag = ep, "  <-- first >= 94%"
        print(f"{os.path.basename(p)}  acc={acc * 100:.2f}{flag}")

    if first is not None and 1 in mtimes and first > 1:
        span = mtimes[first] - mtimes[1]
        t_star = span * first / (first - 1)
        print(f"\nfirst crossing epoch {first};  t* = {t_star:.2f}s")
        s = max(0.0, min(1.0, (70.0 - t_star) / (70.0 - 28.0)))
        print(f"implied S_outcome = {s:.4f}")
    elif first is None:
        print("\nNo checkpoint reached 94%.")
