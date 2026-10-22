"""Re-evaluate a checkpoint ladder exactly the way the verifier does.

The point of this script is to reproduce the grader's I/O path end-to-end rather
than to reuse anything from training:

  * images are read from the original test PNGs with ``ImageFolder``,
  * transformed with ``ToTensor() + Normalize(CIFAR_MEAN, CIFAR_STD)``,
  * batched at 512,
  * each checkpoint is loaded with ``torch.jit.load(path).eval()``,
  * the prediction is ``argmax(1)`` of the raw model output.

It also reconstructs the verifier's clock,
``t* = (mtime[ckpt_N] - mtime[ckpt_1]) * N / (N - 1)`` for the first checkpoint
at or above the target accuracy, and checks that epoch numbers are contiguous
from 1.

    python verify_ladder.py /workspace/periodic_checkpoints [--first-only]
"""

import argparse
import os
import re
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TEST_DIR = "/workspace/data/cifar10/test"
TARGET = 0.94


def load_test_batches(bs=512):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    ds = datasets.ImageFolder(TEST_DIR, transform=tf)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    return [(x.cuda(), y.cuda()) for x, y in dl]


@torch.no_grad()
def accuracy(path, batches):
    model = torch.jit.load(path).eval().cuda()
    correct = total = 0
    for x, y in batches:
        correct += (model(x).argmax(1) == y).sum().item()
        total += len(y)
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ladder", nargs="?", default="/workspace/periodic_checkpoints")
    ap.add_argument("--first-only", action="store_true",
                    help="stop at the first checkpoint that reaches the target")
    ap.add_argument("--target", type=float, default=TARGET)
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.ladder) if re.fullmatch(r"ckpt_epoch_\d{4}\.pt", f))
    epochs = [int(f[11:15]) for f in files]
    if epochs != list(range(1, len(epochs) + 1)):
        print(f"FAIL: epoch numbers are not contiguous 1..N: {epochs[:5]}...{epochs[-3:]}")
        return 1
    mt = {e: os.path.getmtime(os.path.join(args.ladder, f)) for e, f in zip(epochs, files)}

    batches = load_test_batches()
    print(f"test images: {sum(len(y) for _, y in batches)}")
    crossing = None
    for e, f in zip(epochs, files):
        acc = accuracy(os.path.join(args.ladder, f), batches)
        star = (mt[e] - mt[1]) * e / (e - 1) if e > 1 else float("nan")
        mark = ""
        if crossing is None and acc >= args.target:
            crossing = (e, acc, star)
            mark = "  <-- first crossing"
        print(f"  epoch {e:4d}  acc={acc*100:6.2f}  t*={star:6.2f}s{mark}", flush=True)
        if crossing and args.first_only:
            break

    span = mt[epochs[-1]] - mt[1]
    print(f"\nladder: {len(epochs)} checkpoints, span(ckpt1..ckptN) = {span:.2f}s")
    if crossing:
        e, acc, star = crossing
        s_out = max(0.0, min(1.0, (70.0 - star) / (70.0 - 28.0)))
        print(f"first crossing: epoch {e} acc={acc*100:.2f}  t*={star:.2f}s  "
              f"=> S_outcome={s_out:.4f}")
    else:
        print("NO CHECKPOINT REACHED THE TARGET -> S_outcome = 0")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
