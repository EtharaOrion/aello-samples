#!/usr/bin/env python
"""Re-evaluate a checkpoint ladder through the *exact* verifier pipeline.

The verifier loads each file with `torch.jit.load(path).eval()`, feeds float32
NCHW batches of 512 built with
    transforms.Compose([ToTensor(), Normalize((0.4914,0.4822,0.4465),
                                              (0.2470,0.2435,0.2616))])
and takes `argmax(1)`. This script reproduces that end to end from the PNGs on
disk (ImageFolder, alphabetical class order) so that "94%" here means the same
thing it will mean when the ladder is graded.

`--split holdout` instead scores a deterministic slice of the *train* split and
is what every modelling decision in this session was made on; `--split test` is
used only to confirm a finished ladder, never to choose between configs.
"""
import argparse
import json
import os

import torch
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def verifier_tensors(split, dev):
    # No DataLoader workers: /dev/shm is 64 MB in this container and worker
    # collation dies on it. Single-process iteration over the same ImageFolder
    # + Compose is the identical transform, just slower to build (~2 s).
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ds = datasets.ImageFolder(f"/workspace/data/cifar10/{split}", transform=tf)
    xs = torch.empty(len(ds), 3, 32, 32)
    ys = torch.empty(len(ds), dtype=torch.long)
    for i in range(len(ds)):
        xs[i], ys[i] = ds[i]
    return xs.to(dev), ys.to(dev), ds.classes


def score(path, x, y, bs=512):
    m = torch.jit.load(path).eval().cuda()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), bs):
            correct += (m(x[i : i + bs]).argmax(1) == y[i : i + bs]).sum().item()
    return correct / len(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="/workspace/periodic_checkpoints")
    p.add_argument("--split", default="test", choices=["test", "holdout"])
    p.add_argument("--holdout", type=int, default=5000)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--from-epoch", type=int, default=1)
    p.add_argument("--out", default="")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    dev = torch.device("cuda")

    if a.split == "test":
        x, y, classes = verifier_tensors("test", dev)
    else:
        # Same deterministic carve-out train.py uses for --holdout.
        x, y, classes = verifier_tensors("train", dev)
        g = torch.Generator().manual_seed(1234)
        perm = torch.randperm(len(x), generator=g).to(dev)[: a.holdout]
        x, y = x[perm], y[perm]

    files = sorted(f for f in os.listdir(a.dir) if f.startswith("ckpt_epoch_"))
    rows = []
    for f in files:
        ep = int(f[len("ckpt_epoch_") : -3])
        if ep < a.from_epoch or (ep - a.from_epoch) % a.every:
            continue
        acc = score(os.path.join(a.dir, f), x, y)
        mt = os.path.getmtime(os.path.join(a.dir, f))
        rows.append(dict(epoch=ep, acc=acc, mtime=mt))
        print(f"{f}  acc={acc:.4f}")

    span = rows[-1]["mtime"] - rows[0]["mtime"] if len(rows) > 1 else 0.0
    cross = next((r for r in rows if r["acc"] >= 0.94), None)
    summary = dict(
        tag=a.tag, dir=a.dir, split=a.split, n_eval=len(x), classes=classes,
        best_acc=max(r["acc"] for r in rows), final_acc=rows[-1]["acc"],
        cross_epoch=cross["epoch"] if cross else None,
        cross_acc=cross["acc"] if cross else None,
        ladder_span_sec=span, rows=rows,
    )
    if cross and len(rows) > 1 and a.every == 1 and a.from_epoch == 1:
        n = cross["epoch"]
        per_ep = (rows[-1]["mtime"] - rows[0]["mtime"]) / (len(rows) - 1)
        summary["t_star_est_sec"] = per_ep * n
        summary["S_outcome_est"] = max(0.0, min(1.0, (70 - per_ep * n) / (70 - 28)))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    if a.out:
        with open(a.out, "a") as fh:
            fh.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
