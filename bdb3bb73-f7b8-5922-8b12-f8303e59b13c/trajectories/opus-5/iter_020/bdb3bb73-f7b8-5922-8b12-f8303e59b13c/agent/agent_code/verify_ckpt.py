"""Re-evaluate saved TorchScript checkpoints through the *exact* evaluation convention.

The evaluator loads each checkpoint with `torch.jit.load(path).eval()` on the GPU and
calls `model(x)` on float32 NCHW batches of size 512 built with

    transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                             (0.2470, 0.2435, 0.2616))])

so inputs are ALREADY per-channel standardized.  This script reproduces that byte for
byte (torchvision ImageFolder over the PNG directories, same transform, same batch size)
and reports argmax accuracy per checkpoint.

Model *selection* in this project is done on `--split holdout`, a class-balanced 5k slice
carved out of the TRAIN split.  `--split test` is used only for the mandated pre-promotion
convention check and for the final honest accuracy report; it never gates a design choice.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
TF = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def load_split(split):
    """Returns (x, y) with x float32 NCHW standardized exactly like the evaluator."""
    if split == "test":
        ds = datasets.ImageFolder("/workspace/data/cifar10/test", transform=TF)
        dl = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
        xs, ys = zip(*[(x, y) for x, y in dl])
        return torch.cat(xs), torch.cat(ys)
    # holdout: same class-balanced slice train.py carves out (split_seed=1234)
    ds = datasets.ImageFolder("/workspace/data/cifar10/train", transform=TF)
    y_all = np.array(ds.targets)
    g = np.random.default_rng(1234)
    idx = np.concatenate([g.permutation(np.where(y_all == c)[0])[: 5000 // 10] for c in range(10)])
    sub = torch.utils.data.Subset(ds, idx.tolist())
    dl = DataLoader(sub, batch_size=512, shuffle=False, num_workers=0)
    xs, ys = zip(*[(x, y) for x, y in dl])
    return torch.cat(xs), torch.cat(ys)


def evaluate(path, x, y):
    model = torch.jit.load(path).eval().cuda()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), 512):
            xb = x[i : i + 512].cuda()
            correct += (model(xb).argmax(1).cpu() == y[i : i + 512]).sum().item()
    del model
    return correct / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="/workspace/periodic_checkpoints")
    ap.add_argument("--split", default="test", choices=["test", "holdout"])
    ap.add_argument("--from-epoch", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    x, y = load_split(args.split)
    print(f"{args.split}: {tuple(x.shape)} range=[{x.min():.3f},{x.max():.3f}] mean={x.mean():.4f}", flush=True)

    paths = sorted(glob.glob(os.path.join(args.ckpt_dir, "ckpt_epoch_*.pt")))
    rows, first = [], None
    for p in paths:
        ep = int(os.path.basename(p).split("_")[-1].split(".")[0])
        if ep < args.from_epoch:
            continue
        acc = evaluate(p, x, y)
        rows.append({"epoch": ep, "acc": acc, "mtime": os.path.getmtime(p)})
        flag = ""
        if acc >= 0.94 and first is None:
            first = ep
            flag = "  <-- first >= 94%"
        print(f"  epoch {ep:4d}  acc={acc:.4f}{flag}", flush=True)

    if rows and first is not None:
        t1 = min(r["mtime"] for r in rows if r["epoch"] == 1) if any(r["epoch"] == 1 for r in rows) else None
        m = {r["epoch"]: r["mtime"] for r in rows}
        if t1 is not None and first > 1:
            t_star = (m[first] - t1) * first / (first - 1)
            print(f"\nfirst crossing epoch {first};  reconstructed t* = {t_star:.2f}s")
    print(f"\nbest acc = {max(r['acc'] for r in rows):.4f}" if rows else "no checkpoints")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"split": args.split, "rows": rows, "first_cross": first}, f, indent=2)


if __name__ == "__main__":
    main()
