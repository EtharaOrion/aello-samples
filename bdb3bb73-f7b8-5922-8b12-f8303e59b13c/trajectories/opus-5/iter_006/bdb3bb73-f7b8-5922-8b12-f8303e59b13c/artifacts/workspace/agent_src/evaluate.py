"""Score saved checkpoints with the verifier's exact input convention.

The verifier does:

    ds = ImageFolder(test_dir, transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616))]))
    model = torch.jit.load(path).eval()          # on GPU
    pred  = model(x).argmax(1)                   # x: float32 NCHW, batch 512

`--faithful` reproduces that literally, through torchvision's ImageFolder/DataLoader
on the PNG directory.  The default path uses the cached uint8 array with an
arithmetically identical transform, which is ~40x faster; `check_equivalence()`
asserts the two produce bit-comparable inputs before the fast path is trusted.

This file is the ONLY place the test split is read.  It runs after training, on
checkpoints that are already on disk, so it cannot influence any weight or any
hyperparameter choice.
"""
import argparse
import glob
import os
import re

import numpy as np
import torch

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
TEST_DIR = "/workspace/data/cifar10/test"
BATCH = 512


def cached_test():
    """Cached equivalent of ImageFolder + ToTensor + Normalize, on GPU."""
    x = np.load("/workspace/cache/test_x.npy")  # (N,32,32,3) uint8, ImageFolder order
    y = np.load("/workspace/cache/test_y.npy")
    x = torch.from_numpy(x).cuda().permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)
    return x.sub_(mean).div_(std), torch.from_numpy(y).cuda()


def faithful_test():
    """The verifier's own pipeline, verbatim."""
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ds = datasets.ImageFolder(TEST_DIR, transform=tf)
    # num_workers=0: this container's /dev/shm is too small for worker tensor sharing.
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)
    xs, ys = [], []
    for xb, yb in loader:
        xs.append(xb)
        ys.append(yb)
    return torch.cat(xs).cuda(), torch.cat(ys).cuda()


def check_equivalence():
    xa, ya = cached_test()
    xb, yb = faithful_test()
    assert torch.equal(ya, yb), "label order differs from ImageFolder"
    d = (xa - xb).abs().max().item()
    print(f"cached-vs-ImageFolder: max|dx| = {d:.3e}, labels identical, N = {len(ya)}")
    assert d < 1e-4, "cached transform is not equivalent to the verifier's"


@torch.no_grad()
def score(path, x, y):
    m = torch.jit.load(path).eval().cuda()
    correct = 0
    for i in range(0, len(x), BATCH):
        correct += (m(x[i:i + BATCH]).argmax(1) == y[i:i + BATCH]).sum().item()
    return correct / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/periodic_checkpoints")
    ap.add_argument("--faithful", action="store_true",
                    help="load the test split through torchvision ImageFolder")
    ap.add_argument("--check-equivalence", action="store_true")
    ap.add_argument("--only", default=None, help="glob suffix filter, e.g. '00(1|2)[0-9]'")
    args = ap.parse_args()

    if args.check_equivalence:
        check_equivalence()
    x, y = faithful_test() if args.faithful else cached_test()
    print(f"test inputs: shape={tuple(x.shape)} dtype={x.dtype} "
          f"range=[{x.min():.3f}, {x.max():.3f}]")

    paths = sorted(glob.glob(os.path.join(args.dir, "ckpt_epoch_*.pt")))
    if args.only:
        paths = [p for p in paths if re.search(args.only, os.path.basename(p))]
    first = None
    for p in paths:
        acc = score(p, x, y)
        ep = int(re.search(r"(\d+)\.pt$", p).group(1))
        flag = ""
        if acc >= 0.94 and first is None:
            first, flag = ep, "  <-- first >= 94%"
        print(f"epoch {ep:4d}  acc {acc * 100:6.2f}%{flag}", flush=True)
    if first is not None:
        mts = {int(re.search(r"(\d+)\.pt$", p).group(1)): os.path.getmtime(p) for p in paths}
        if 1 in mts and first > 1:
            t = (mts[first] - mts[1]) * first / (first - 1)
            print(f"\nfirst crossing at epoch {first};  "
                  f"verifier-style t* = {t:.2f}s")
    else:
        print("\nNO checkpoint reached 94%")


if __name__ == "__main__":
    main()
