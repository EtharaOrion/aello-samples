"""Replica of the grader's evaluation, run against a saved checkpoint ladder.

Deliberately mirrors the stated convention exactly:
  * test PNGs read through torchvision ImageFolder
  * transforms.Compose([ToTensor(), Normalize((0.4914,0.4822,0.4465),
                                              (0.2470,0.2435,0.2616))])
  * torch.jit.load(path).eval() on the GPU, batch size 512, argmax(1)

It also reconstructs the time-to-accuracy the grader will compute from the
checkpoint save times:
    t_star = (save[N] - save[1]) * N / (N - 1)
for the first epoch N whose accuracy reaches the target.
"""
import argparse
import glob
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

_cache = {}


def get_test():
    if "xy" not in _cache:
        ds = datasets.ImageFolder(TEST_DIR, transform=TF)
        dl = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)
        xs, ys = [], []
        for xb, yb in dl:
            xs.append(xb)
            ys.append(yb)
        _cache["xy"] = (torch.cat(xs).cuda(), torch.cat(ys).cuda())
        _cache["classes"] = ds.classes
    return _cache["xy"]


@torch.no_grad()
def accuracy(path):
    x, y = get_test()
    model = torch.jit.load(path).eval().cuda()
    correct = 0
    for i in range(0, x.shape[0], 512):
        correct += (model(x[i:i + 512]).argmax(1) == y[i:i + 512]).sum().item()
    return correct / x.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/periodic_checkpoints")
    ap.add_argument("--target", type=float, default=0.94)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "ckpt_epoch_*.pt")))
    if not paths:
        print("no checkpoints in", args.dir)
        return
    eps = [int(re.search(r"(\d{4})", os.path.basename(p)).group(1)) for p in paths]
    assert eps == list(range(1, len(eps) + 1)), f"non-contiguous epochs: {eps[:5]}..."
    mt = {e: os.path.getmtime(p) for e, p in zip(eps, paths)}

    get_test()
    print(f"classes: {_cache['classes']}")
    first = None
    accs = {}
    for e, p in zip(eps, paths):
        a = accuracy(p)
        accs[e] = a
        if a >= args.target and first is None:
            first = e
        if not args.quiet:
            span = mt[e] - mt[1]
            t = span * e / (e - 1) if e > 1 else float("nan")
            print(f"  epoch {e:3d}  acc {a * 100:6.2f}%   t*_if_crossing {t:6.2f}s"
                  + ("   <-- FIRST >= target" if e == first else ""))

    total_span = mt[eps[-1]] - mt[1]
    print(f"\nladder: {len(eps)} epochs, span(ckpt1..ckptN) = {total_span:.2f}s, "
          f"mean epoch {total_span / (len(eps) - 1):.3f}s")
    print(f"final acc {accs[eps[-1]] * 100:.2f}%   best acc {max(accs.values()) * 100:.2f}%")
    if first is None:
        print(f"*** NO checkpoint reaches {args.target:.2%} -> S_outcome would be 0 ***")
    else:
        t_star = (mt[first] - mt[1]) * first / (first - 1)
        s = min(1.0, max(0.0, (70.0 - t_star) / (70.0 - 28.0)))
        print(f"first crossing: epoch {first} acc {accs[first] * 100:.2f}%  "
              f"t* = {t_star:.2f}s  ->  S_outcome = {s:.4f}")


if __name__ == "__main__":
    main()
