"""Replica of the verifier's evaluation, run against a checkpoint ladder.

This deliberately reproduces the grading path *exactly* rather than reusing
anything from ``common.py``:

  * images are read from the test PNG folder with ``ImageFolder``
  * transform = ``Compose([ToTensor(), Normalize(CIFAR_MEAN, CIFAR_STD)])``
  * batch size 512, ``torch.jit.load(path).eval()`` on the GPU
  * prediction = ``output.argmax(1)``

It also reconstructs the outcome clock the verifier uses,

    t_star = (save_time[ckpt_N] - save_time[ckpt_0001]) * N / (N - 1)

for the FIRST epoch N whose accuracy reaches the target, using the checkpoint
files' own modification times (written by ``save()``; never altered).
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


def load_test(batch=512, workers=0):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ds = datasets.ImageFolder("/workspace/data/cifar10/test", transform=tf)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)
    xs, ys = [], []
    for x, y in dl:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs), torch.cat(ys), ds.classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/periodic_checkpoints")
    ap.add_argument("--target", type=float, default=0.94)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--last-only", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    x, y, classes = load_test(args.batch)
    print(f"test set: {tuple(x.shape)}  range [{x.min():.3f}, {x.max():.3f}]  classes={classes[:3]}...")
    x, y = x.cuda(), y.cuda()

    paths = sorted(glob.glob(os.path.join(args.dir, "ckpt_epoch_*.pt")))
    epochs = [int(re.search(r"ckpt_epoch_(\d+)\.pt", p).group(1)) for p in paths]
    assert epochs == list(range(1, len(epochs) + 1)), f"non-contiguous ladder: {epochs[:5]}..."
    if args.last_only:  # slice AFTER the contiguity check, not before
        paths, epochs = paths[-1:], epochs[-1:]

    rows = []
    for ep, p in zip(epochs, paths):
        m = torch.jit.load(p).eval().cuda()
        correct = 0
        with torch.no_grad():
            for i in range(0, x.shape[0], args.batch):
                correct += (m(x[i:i + args.batch]).argmax(1) == y[i:i + args.batch]).sum().item()
        acc = correct / x.shape[0]
        rows.append({"epoch": ep, "acc": acc, "mtime": os.path.getmtime(p)})
        print(f"  epoch {ep:4d}  acc {acc*100:6.2f}  ({os.path.basename(p)})")
        del m

    t_star = None
    cross = None
    if len(rows) > 1:
        t1 = rows[0]["mtime"]
        for r in rows[1:]:
            if r["acc"] >= args.target:
                n = r["epoch"]
                t_star = (r["mtime"] - t1) * n / (n - 1)
                cross = r
                break

    span = rows[-1]["mtime"] - rows[0]["mtime"] if len(rows) > 1 else 0.0
    best = max(rows, key=lambda r: r["acc"])
    summary = {
        "dir": args.dir, "n_ckpt": len(rows),
        "final_acc": rows[-1]["acc"], "best_acc": best["acc"], "best_epoch": best["epoch"],
        "crossing_epoch": cross["epoch"] if cross else None,
        "crossing_acc": cross["acc"] if cross else None,
        "t_star_sec": t_star, "ladder_span_sec": span,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
