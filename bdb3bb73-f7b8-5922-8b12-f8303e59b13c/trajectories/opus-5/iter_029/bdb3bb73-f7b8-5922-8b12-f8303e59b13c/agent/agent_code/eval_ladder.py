#!/usr/bin/env python
"""Re-evaluate a checkpoint ladder with the EXACT pipeline the verifier uses.

  torchvision ImageFolder over the test PNGs
  -> transforms.Compose([ToTensor(), Normalize((0.4914,0.4822,0.4465),
                                               (0.2470,0.2435,0.2616))])
  -> batch size 512 -> torch.jit.load(path).eval() -> argmax(1)

This is a verification tool only: it runs AFTER training, on saved artifacts, and its
output is never fed back into hyperparameter selection (those decisions come from the
held-out TRAIN slice).  Run it before promoting any ladder.
"""
import argparse
import glob
import json
import os
import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TEST_DIR = "/workspace/data/cifar10/test"


def get_loader(bs=512, workers=0):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    ds = datasets.ImageFolder(TEST_DIR, transform=tf)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers), ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/periodic_checkpoints")
    ap.add_argument("--out", default="")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--last", type=int, default=0, help="only evaluate the last K checkpoints")
    args = ap.parse_args()

    loader, ds = get_loader()
    print(f"classes: {ds.classes}", flush=True)
    # cache the whole standardized test set on GPU once; identical tensors, far less I/O
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    X = torch.cat(xs).cuda()
    Y = torch.cat(ys).cuda()
    print(f"test tensor {tuple(X.shape)} dtype={X.dtype} "
          f"range=[{X.min():.3f},{X.max():.3f}]", flush=True)

    paths = sorted(glob.glob(os.path.join(args.dir, "ckpt_epoch_*.pt")))
    if args.last:
        paths = paths[-args.last:]
    paths = paths[::args.every] if args.every > 1 else paths

    rows = []
    for p in paths:
        m = torch.jit.load(p).eval().cuda()
        correct = 0
        with torch.no_grad():
            for i in range(0, X.size(0), 512):
                correct += (m(X[i:i + 512]).argmax(1) == Y[i:i + 512]).sum().item()
        acc = correct / X.size(0)
        mt = os.path.getmtime(p)
        rows.append(dict(ckpt=os.path.basename(p), acc=acc, mtime=mt))
        flag = "  <== >=94%" if acc >= 0.94 else ""
        print(f"{os.path.basename(p)}  acc {acc:.4f}{flag}", flush=True)

    cross = [r for r in rows if r["acc"] >= 0.94]
    if cross and len(rows) > 1:
        first = cross[0]
        n = int(first["ckpt"].split("_")[-1].split(".")[0])
        t0 = rows[0]["mtime"]
        if n > 1:
            tstar = (first["mtime"] - t0) * n / (n - 1)
            print(f"\nFIRST CROSSING {first['ckpt']} acc={first['acc']:.4f}  "
                  f"t_star~{tstar:.2f}s (verifier formula, ladder must start at epoch 1)",
                  flush=True)
    else:
        print("\nNO CHECKPOINT >= 94%", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
