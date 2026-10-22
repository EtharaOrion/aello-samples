"""
Independent re-evaluation of a checkpoint ladder, using the *exact* pipeline the
verifier uses:

    torch.jit.load(path).eval()  on cuda
    ImageFolder(test) -> transforms.ToTensor() -> transforms.Normalize(MEAN, STD)
    batch size 512, prediction = out.argmax(1)

It also reconstructs the verifier's clock from the checkpoint mtimes:
    t_star = (mtime[ckpt_N] - mtime[ckpt_1]) * N / (N - 1)
for the first N whose accuracy >= threshold.

Usage: python evaluate_ladder.py DIR [--from 1] [--threshold 0.94]
"""

import argparse
import glob
import os
import re

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def build_test_batches(root="/workspace/data/cifar10/test", bs=512, device="cuda"):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ds = datasets.ImageFolder(root, transform=tf)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    return [(x.to(device, non_blocking=True), y.to(device, non_blocking=True)) for x, y in dl]


def accuracy(path, batches):
    model = torch.jit.load(path).eval().cuda()
    correct = total = 0
    with torch.no_grad():
        for x, y in batches:
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    del model
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--threshold", type=float, default=0.94)
    ap.add_argument("--start", type=int, default=1)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "ckpt_epoch_*.pt")))
    assert paths, f"no checkpoints in {args.dir}"
    epochs = [int(re.search(r"(\d{4})\.pt$", p).group(1)) for p in paths]
    assert epochs == list(range(1, len(epochs) + 1)), f"non-contiguous ladder: {epochs}"

    mt = {e: os.path.getmtime(p) for e, p in zip(epochs, paths)}
    batches = build_test_batches()

    crossing = None
    for e, p in zip(epochs, paths):
        if e < args.start:
            continue
        a = accuracy(p, batches)
        span = mt[e] - mt[1]
        tstar = span * e / (e - 1) if e > 1 else float("nan")
        mark = ""
        if crossing is None and a >= args.threshold:
            crossing = (e, a, tstar)
            mark = "  <== FIRST CROSSING"
        print(f"epoch {e:3d}  acc={a*100:6.2f}%  span={span:7.3f}s  t*={tstar:7.3f}s{mark}")

    if crossing:
        e, a, t = crossing
        s = max(0.0, min(1.0, (70.0 - t) / (70.0 - 28.0)))
        print(f"\nFIRST >= {args.threshold:.2%}: epoch {e}  acc={a*100:.2f}%  "
              f"t*={t:.2f}s  ->  S_outcome={s:.4f}")
    else:
        print(f"\nNo checkpoint reached {args.threshold:.2%}  ->  S_outcome=0")
    print(f"ladder wall-clock span (ckpt 1 -> {epochs[-1]}): {mt[epochs[-1]] - mt[1]:.2f}s")


if __name__ == "__main__":
    main()
