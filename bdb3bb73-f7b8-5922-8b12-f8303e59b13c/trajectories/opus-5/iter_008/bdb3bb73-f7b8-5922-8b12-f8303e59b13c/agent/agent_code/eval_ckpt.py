"""Re-evaluate scripted checkpoints exactly the way the verifier does.

Deliberately bypasses the training cache: images come from the CIFAR-10 test PNGs via
`torchvision.datasets.ImageFolder` with `Compose([ToTensor(), Normalize(mean, std)])`,
batch size 512, `torch.jit.load(path).eval()` on the GPU, prediction = `argmax(1)`.
Going through the real folder pipeline is what catches a channel-order or class-order
mismatch between my cache and the verifier's view of the data.
"""
import argparse
import glob
import json
import os
import re

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
_cached = None


def test_batches():
    """Materialise the verifier's exact input tensors once, on the GPU."""
    global _cached
    if _cached is None:
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        ds = datasets.ImageFolder("/workspace/data/cifar10/test", transform=tf)
        dl = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
        _cached = [(x.cuda(non_blocking=True), y.cuda(non_blocking=True)) for x, y in dl]
        print(f"# classes={ds.classes} n={len(ds)}", flush=True)
    return _cached


@torch.no_grad()
def accuracy(path):
    model = torch.jit.load(path).eval()
    correct = total = 0
    for x, y in test_batches():
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--out", default="")
    ap.add_argument("--first-only", action="store_true",
                    help="stop at the first checkpoint that reaches --target")
    ap.add_argument("--target", type=float, default=0.94)
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.ckpt_dir, "ckpt_epoch_*.pt")))
    rows, crossing = [], None
    for p in paths:
        ep = int(re.search(r"(\d+)\.pt$", p).group(1))
        acc = accuracy(p)
        mtime = os.path.getmtime(p)
        rows.append({"epoch": ep, "acc": acc, "mtime": mtime})
        print(f"epoch {ep:4d}  acc {acc:.4f}", flush=True)
        if crossing is None and acc >= a.target:
            crossing = rows[-1]
            if a.first_only:
                break

    summary = {"n_ckpt": len(paths), "best": max(r["acc"] for r in rows) if rows else None,
               "final": rows[-1]["acc"] if rows else None, "crossing": crossing}
    if crossing and len(rows) > 1:
        t0 = rows[0]["mtime"]
        n = crossing["epoch"]
        if n > 1:
            summary["t_star_sec"] = round((crossing["mtime"] - t0) * n / (n - 1), 3)
        summary["ladder_span_sec"] = round(rows[-1]["mtime"] - t0, 3)
    print(json.dumps(summary), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"rows": rows, "summary": summary}, f, indent=1)


if __name__ == "__main__":
    main()
