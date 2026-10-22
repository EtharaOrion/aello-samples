"""Re-evaluate a checkpoint ladder exactly the way the verifier does.

For every ckpt_epoch_*.pt in a directory: `torch.jit.load(path).eval()` on the
GPU, feed float32 NCHW batches of 512 built by
`transforms.Compose([ToTensor(), Normalize(MEAN, STD)])` over the 32x32 PNGs,
and take argmax(1).

It also reconstructs the verifier's timing formula from checkpoint save times,
   t_star = (save[N] - save[0001]) * N / (N - 1)
for the first rung that reaches the target accuracy.
"""
import argparse
import glob
import json
import os
import re

import torch
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def build_loader(split, holdout_n=0):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ds = datasets.ImageFolder(f"/workspace/data/cifar10/{split}", transform=tf)
    idx = None
    if holdout_n:
        g = torch.Generator().manual_seed(1234)
        idx = torch.randperm(len(ds), generator=g)[:holdout_n].tolist()
        ds = torch.utils.data.Subset(ds, idx)
    return torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)


def cache_split(loader, dev):
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.to(dev, non_blocking=True))
        ys.append(y.to(dev, non_blocking=True))
    return xs, ys


@torch.no_grad()
def score(path, xs, ys, dev):
    model = torch.jit.load(path, map_location=dev).eval()
    correct = total = 0
    for x, y in zip(xs, ys):
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="/workspace/periodic_checkpoints")
    p.add_argument("--split", default="test", choices=["test", "train"])
    p.add_argument("--holdout", type=int, default=0)
    p.add_argument("--target", type=float, default=0.94)
    p.add_argument("--out", default="")
    args = p.parse_args()

    dev = torch.device("cuda")
    xs, ys = cache_split(build_loader(args.split, args.holdout), dev)
    print(f"scoring on {args.split} split, {sum(y.numel() for y in ys)} images")

    paths = sorted(glob.glob(os.path.join(args.dir, "ckpt_epoch_*.pt")))
    rows = []
    for path in paths:
        ep = int(re.search(r"ckpt_epoch_(\d+)\.pt", path).group(1))
        rows.append({"epoch": ep, "acc": score(path, xs, ys, dev),
                     "mtime": os.path.getmtime(path)})

    epochs = [r["epoch"] for r in rows]
    contiguous = epochs == list(range(1, len(epochs) + 1))
    t0 = rows[0]["mtime"] if rows else 0.0
    for r in rows:
        r["t_rel"] = r["mtime"] - t0
        n = r["epoch"]
        r["t_star_if_first"] = r["t_rel"] * n / (n - 1) if n > 1 else float("nan")
        flag = " <-- first >= target" if r["acc"] >= args.target and not any(
            q["acc"] >= args.target for q in rows[:rows.index(r)]) else ""
        print(f"  epoch {r['epoch']:4d}  acc {r['acc']:.4f}  "
              f"t_rel {r['t_rel']:6.2f}s  t* {r['t_star_if_first']:6.2f}s{flag}")

    first = next((r for r in rows if r["acc"] >= args.target), None)
    summary = {
        "dir": args.dir, "split": args.split, "n_ckpts": len(rows),
        "contiguous_from_1": contiguous,
        "best_acc": max((r["acc"] for r in rows), default=0.0),
        "final_acc": rows[-1]["acc"] if rows else 0.0,
        "ladder_span_sec": rows[-1]["t_rel"] if rows else 0.0,
        "first_cross_epoch": first["epoch"] if first else None,
        "t_star_sec": first["t_star_if_first"] if first else None,
        "rows": rows,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
