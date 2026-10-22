"""Re-evaluate a checkpoint ladder through the verifier's exact pipeline.

Loads each `ckpt_epoch_*.pt` with `torch.jit.load(...).eval()`, feeds it the
32x32 test PNGs read from disk and transformed by
    Compose([ToTensor(), Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
in batches of 512, and takes argmax(1).  No shortcuts through the cached
tensors: this reads the same PNG files the verifier reads.
"""
import argparse, json, os, sys
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TFM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


def get_loader(root="/workspace/data/cifar10/test", bs=512):
    ds = datasets.ImageFolder(root, transform=TFM)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)


@torch.no_grad()
def accuracy(path, batches):
    m = torch.jit.load(path).eval().cuda()
    correct = total = 0
    for x, y in batches:
        pred = m(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    loader = get_loader()
    batches = [(x.cuda(non_blocking=True), y.cuda(non_blocking=True)) for x, y in loader]
    print(f"test batches: {len(batches)}, total {sum(y.numel() for _, y in batches)}", file=sys.stderr)

    report = {}
    for d in args.dirs:
        files = sorted(f for f in os.listdir(d) if f.startswith("ckpt_epoch_") and f.endswith(".pt"))
        rows = []
        first = None
        for f in files:
            acc = accuracy(os.path.join(d, f), batches)
            rows.append((f, acc))
            if first is None and acc >= 0.94:
                first = f
            print(f"{d} {f} {acc:.4f}", flush=True)
        report[d] = {"rows": rows, "first_cross": first,
                     "final": rows[-1][1] if rows else None}
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
    print(json.dumps({k: {"first_cross": v["first_cross"], "final": v["final"]}
                      for k, v in report.items()}, indent=2))


if __name__ == "__main__":
    main()
