"""Score a checkpoint ladder exactly the way the verifier does.

Replicates the stated grading protocol end to end:
  * load with torch.jit.load(path).eval() on the GPU (no access to lib.py --
    the checkpoint must be self-contained),
  * build inputs as transforms.Compose([ToTensor(),
        Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))]) over the
    32x32 test PNGs,
  * feed float32 NCHW batches of 512 and take argmax(1).

It also reconstructs t* the way the verifier does, from checkpoint save times:
    t_star = (mtime[ckpt_N] - mtime[ckpt_0001]) * N / (N - 1)
for the first N whose accuracy >= 0.94.

    python verify_ladder.py [dir] [--cache]

Test images are decoded straight from the PNG files by default, so this script
depends on nothing that training produced.  `--cache` reads the .npy instead
(regenerate it with cache_data.py); both paths were checked to agree exactly.
"""
import argparse
import glob
import os
import re

import numpy as np
import torch

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
ROOT = "/workspace/data/cifar10/test"


def test_tensor(from_cache=False):
    if not from_cache:
        from PIL import Image
        classes = sorted(os.listdir(ROOT))
        paths, labels = [], []
        for ci, c in enumerate(classes):
            for f in sorted(os.listdir(f"{ROOT}/{c}")):
                paths.append(f"{ROOT}/{c}/{f}")
                labels.append(ci)
        x = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])
        x = x.transpose(0, 3, 1, 2)
        y = np.asarray(labels)
    else:
        x = np.load("/workspace/cache/test.npy")
        y = np.load("/workspace/cache/test_labels.npy")
    # transforms.ToTensor(): uint8 HWC [0,255] -> float32 CHW [0,1]
    xt = torch.from_numpy(x).cuda().float().div_(255.0)
    m = torch.tensor(MEAN).cuda().view(1, 3, 1, 1)
    s = torch.tensor(STD).cuda().view(1, 3, 1, 1)
    return ((xt - m) / s).contiguous(), torch.from_numpy(y).cuda()


@torch.no_grad()
def score(path, x, y, batch=512):
    model = torch.jit.load(path).eval().cuda()
    correct = 0
    for i in range(0, x.shape[0], batch):
        correct += (model(x[i:i + batch]).argmax(1) == y[i:i + batch]).sum().item()
    return correct / x.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirpath", nargs="?", default="/workspace/periodic_checkpoints")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args()

    x, y = test_tensor(a.cache)
    print(f"test inputs {tuple(x.shape)} dtype={x.dtype} "
          f"range=[{x.min():.2f},{x.max():.2f}]  (standardized, as the verifier sends)")

    files = sorted(glob.glob(os.path.join(a.dirpath, "ckpt_epoch_*.pt")))
    nums = [int(re.search(r"ckpt_epoch_(\d+)\.pt", f).group(1)) for f in files]
    if not files:
        print("NO CHECKPOINTS FOUND")
        return
    assert nums == list(range(1, len(nums) + 1)), f"epochs not contiguous 1..N: {nums}"
    print(f"{len(files)} checkpoints, contiguous 1..{nums[-1]}")

    mt = {n: os.path.getmtime(f) for n, f in zip(nums, files)}
    cross = None
    for n, f in zip(nums, files):
        if n % a.every and n != nums[-1]:
            continue
        acc = score(f, x, y)
        flag = ""
        if acc >= 0.94 and cross is None:
            cross = n
            flag = "  <-- first crossing"
        print(f"  epoch {n:4d}  acc={acc*100:6.2f}  (+{mt[n]-mt[1]:6.2f}s){flag}")

    if cross and cross > 1:
        t_star = (mt[cross] - mt[1]) * cross / (cross - 1)
        s_out = min(1.0, max(0.0, (70.0 - t_star) / (70.0 - 28.0)))
        print(f"\nfirst crossing epoch {cross}  ->  t* = {t_star:.2f}s  "
              f"S_outcome = {s_out:.4f}")
        print(f"ladder span (ckpt 1 -> ckpt {nums[-1]}): {mt[nums[-1]]-mt[1]:.2f}s")
    else:
        print("\nno checkpoint reached 94%")

    fin = "/workspace/checkpoint.pt"
    if os.path.exists(fin):
        print(f"/workspace/checkpoint.pt acc = {score(fin, x, y)*100:.2f}")
    else:
        print("WARNING: /workspace/checkpoint.pt missing")


if __name__ == "__main__":
    main()
