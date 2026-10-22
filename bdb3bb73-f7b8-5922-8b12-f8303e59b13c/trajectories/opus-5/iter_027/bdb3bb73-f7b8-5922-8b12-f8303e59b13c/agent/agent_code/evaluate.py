"""Re-evaluate a checkpoint ladder with the verifier's exact input convention.

The verifier does, per checkpoint:

    model = torch.jit.load(path).eval()            # on GPU
    x     = Compose([ToTensor(), Normalize(MEAN, STD)])(png)   # float32, standardized
    pred  = model(x_batch_of_512).argmax(1)

so this script reproduces that byte-for-byte.  `--mode imagefolder` walks the
PNGs through torchvision exactly as above; `--mode cache` uses the cached uint8
tensors and applies the same arithmetic (they agree to <1e-6, checked by
`--mode both`).  Cache mode is ~40x faster, so it is the default for sweeping a
whole ladder.

  python evaluate.py --dir /workspace/periodic_checkpoints
  python evaluate.py --dir /workspace/periodic_checkpoints --mode imagefolder --only 16
"""
import argparse
import glob
import json
import os
import re
import time

import torch
from torchvision import datasets, transforms

from model import CIFAR_MEAN, CIFAR_STD

CACHE = "/workspace/cache"
ROOT = "/workspace/data/cifar10"
BS = 512


def load_test_imagefolder():
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    ds = datasets.ImageFolder(os.path.join(ROOT, "test"), transform=tf)
    # num_workers=0: the container's /dev/shm is small and worker processes die
    # with a bus error; single-process decode of 10k 32x32 PNGs takes ~5 s.
    dl = torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=False, num_workers=0)
    xs, ys = [], []
    for xb, yb in dl:
        xs.append(xb)
        ys.append(yb)
    return torch.cat(xs), torch.cat(ys)


def load_test_cache():
    d = torch.load(os.path.join(CACHE, "test_u8.pt"))
    x = d["x"].float() / 255.0
    mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
    return (x - mean) / std, d["y"]


@torch.no_grad()
def acc_of(path, x, y):
    model = torch.jit.load(path).eval().cuda()
    correct = 0
    for i in range(0, x.shape[0], BS):
        xb = x[i:i + BS].cuda(non_blocking=True)
        correct += (model(xb).argmax(1).cpu() == y[i:i + BS]).sum().item()
    return correct / x.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/periodic_checkpoints")
    ap.add_argument("--mode", default="cache", choices=["cache", "imagefolder", "both"])
    ap.add_argument("--only", type=int, default=0, help="evaluate a single epoch number")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.mode == "both":
        xa, ya = load_test_imagefolder()
        xb, yb = load_test_cache()
        print(f"imagefolder vs cache: max|dx|={(xa - xb).abs().max():.3e} "
              f"labels_equal={bool((ya == yb).all())}")
        x, y = xa, ya
    elif args.mode == "imagefolder":
        x, y = load_test_imagefolder()
    else:
        x, y = load_test_cache()
    print(f"test batch: {tuple(x.shape)} {x.dtype} range=[{x.min():.3f}, {x.max():.3f}]")

    paths = sorted(glob.glob(os.path.join(args.dir, "ckpt_epoch_*.pt")))
    if args.only:
        paths = [p for p in paths if int(re.search(r"(\d+)\.pt$", p).group(1)) == args.only]
    rows = []
    for p in paths:
        ep = int(re.search(r"(\d+)\.pt$", p).group(1))
        t = time.time()
        a = acc_of(p, x, y)
        rows.append({"epoch": ep, "acc": a, "mtime": os.path.getmtime(p)})
        print(f"  epoch {ep:4d}  acc {a * 100:6.2f}%   ({time.time() - t:.1f}s)")

    if rows:
        # Mirror the verifier's own clock reconstruction so we can sanity-check
        # the time-to-94% we are about to be scored on.
        first = min(r["mtime"] for r in rows)
        cross = [r for r in sorted(rows, key=lambda r: r["epoch"]) if r["acc"] >= 0.94]
        best = max(rows, key=lambda r: r["acc"])
        print(f"\nbest: epoch {best['epoch']} @ {best['acc'] * 100:.2f}%")
        if cross:
            c = cross[0]
            n = c["epoch"]
            tstar = (c["mtime"] - first) * n / (n - 1) if n > 1 else float("nan")
            print(f"first >=94%: epoch {n} @ {c['acc'] * 100:.2f}%  "
                  f"reconstructed t* = {tstar:.2f}s  "
                  f"S_outcome = {max(0.0, min(1.0, (70 - tstar) / (70 - 28))):.4f}")
            print(f"checkpoints >=94%: {[r['epoch'] for r in cross]}")
        else:
            print("NO checkpoint reached 94% -- S_outcome would be 0")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
