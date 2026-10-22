"""Independent re-evaluation of saved checkpoints, mirroring the grader.

Replicates the stated evaluation convention exactly:
    torch.jit.load(path).eval()  on GPU
    ImageFolder over the 32x32 test PNGs
    transforms.Compose([ToTensor(), Normalize((0.4914,0.4822,0.4465),
                                              (0.2470,0.2435,0.2616))])
    batch size 512, prediction = output.argmax(1)

Nothing in this file is used by training; it only reads artifacts that
training already wrote to disk.
"""
import argparse
import glob
import json
import os
import re

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TEST_DIR = "/workspace/data/cifar10/test"
TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

_cached = {}


def get_test_tensors(device="cuda"):
    if "x" not in _cached:
        ds = datasets.ImageFolder(TEST_DIR, transform=TF)
        dl = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
        xs, ys = [], []
        for xb, yb in dl:
            xs.append(xb)
            ys.append(yb)
        _cached["x"] = torch.cat(xs).to(device)
        _cached["y"] = torch.cat(ys).to(device)
        _cached["classes"] = ds.classes
    return _cached["x"], _cached["y"]


@torch.no_grad()
def eval_checkpoint(path, device="cuda"):
    x, y = get_test_tensors(device)
    model = torch.jit.load(path, map_location=device).eval()
    correct = 0
    for i in range(0, x.shape[0], 512):
        pred = model(x[i:i + 512]).argmax(1)
        correct += (pred == y[i:i + 512]).sum().item()
    return correct / x.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/periodic_checkpoints")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-epoch", type=int, default=1,
                    help="skip epochs below this (timing still uses ckpt 1)")
    ap.add_argument("--first-only", action="store_true",
                    help="stop after the first checkpoint reaching >= 0.94")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "ckpt_epoch_*.pt")))
    rows, cross = [], None
    for p in paths:
        ep = int(re.search(r"ckpt_epoch_(\d+)", p).group(1))
        mt = os.path.getmtime(p)
        if ep < args.min_epoch:
            rows.append(dict(epoch=ep, acc=None, mtime=mt, path=p))
            continue
        acc = eval_checkpoint(p)
        rows.append(dict(epoch=ep, acc=acc, mtime=mt, path=p))
        print(f"epoch {ep:4d}  acc {acc:.4f}", flush=True)
        if cross is None and acc is not None and acc >= 0.94:
            cross = rows[-1]
            if args.first_only:
                break

    if rows:
        t1 = rows[0]["mtime"]
        span = rows[-1]["mtime"] - t1
        print(f"\ncheckpoints: {len(rows)}  wall span(ckpt1..ckptN): {span:.2f}s")
        if cross is not None:
            n = cross["epoch"]
            tstar = (cross["mtime"] - t1) * n / max(1, n - 1)
            print(f"first >=94%: epoch {n} acc {cross['acc']:.4f}  "
                  f"t_star ~= {tstar:.2f}s  -> S_outcome "
                  f"{max(0.0, min(1.0, (70.0 - tstar) / (70.0 - 28.0))):.4f}")
        else:
            print("NO checkpoint reached 94%")
        ev=[r["acc"] for r in rows if r["acc"] is not None]
        if ev: print(f"best acc: {max(ev):.4f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
