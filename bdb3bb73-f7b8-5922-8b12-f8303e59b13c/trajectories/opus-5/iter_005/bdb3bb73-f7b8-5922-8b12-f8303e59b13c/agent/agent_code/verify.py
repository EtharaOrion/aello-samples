"""Independent replica of the verifier's evaluation.

Deliberately shares NO code with train.py: it reads the real test PNGs from disk
through torchvision's ImageFolder with exactly the documented transform
    ToTensor() -> Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
loads each checkpoint with torch.jit.load(...).eval() on the GPU, feeds float32
NCHW batches of 512, and takes argmax(1).

It also reconstructs t* the way the verifier does, from checkpoint mtimes:
    t* = (mtime[ckpt_N] - mtime[ckpt_1]) * N / (N - 1)
"""
import argparse
import hashlib
import os
import sys

import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/workspace/periodic_checkpoints")
ap.add_argument("--target", type=float, default=0.94)
ap.add_argument("--only-last", action="store_true")
ap.add_argument("--bs", type=int, default=512)
a = ap.parse_args()

tf = T.Compose([T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
ds = ImageFolder("/workspace/data/cifar10/test", transform=tf)
print(f"test images: {len(ds)}  classes: {ds.classes}")
dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=0)
X, Y = [], []
for xb, yb in dl:
    X.append(xb)
    Y.append(yb)
X = torch.cat(X).cuda()
Y = torch.cat(Y).cuda()
print(f"input range [{X.min():.3f}, {X.max():.3f}] dtype={X.dtype}")

files = sorted(f for f in os.listdir(a.dir)
               if f.startswith("ckpt_epoch_") and f.endswith(".pt"))
if not files:
    sys.exit("NO CHECKPOINTS")
nums = [int(f[11:15]) for f in files]
contiguous = nums == list(range(1, len(nums) + 1))
print(f"{len(files)} checkpoints, epochs {nums[0]}..{nums[-1]}, contiguous={contiguous}")

mt = {n: os.path.getmtime(os.path.join(a.dir, f)) for n, f in zip(nums, files)}
mono = all(mt[nums[i]] < mt[nums[i + 1]] for i in range(len(nums) - 1))
print(f"mtimes strictly increasing: {mono}")

sel = files[-1:] if a.only_last else files
first_cross = None
for f in sel:
    n = int(f[11:15])
    p = os.path.join(a.dir, f)
    m = torch.jit.load(p, map_location="cuda").eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, X.shape[0], a.bs):
            correct += (m(X[i:i + a.bs]).argmax(1) == Y[i:i + a.bs]).sum().item()
    acc = correct / X.shape[0]
    md5 = hashlib.md5(open(p, "rb").read()).hexdigest()[:8]
    flag = ""
    if acc >= a.target and first_cross is None:
        first_cross = (n, acc)
        flag = "  <-- FIRST CROSSING"
    print(f"  epoch {n:4d}  acc={acc*100:6.2f}  md5={md5}{flag}", flush=True)

if first_cross and not a.only_last:
    n, acc = first_cross
    if n == 1:
        print("epoch-1 crossing: t* undefined by the N/(N-1) formula")
    else:
        tstar = (mt[n] - mt[1]) * n / (n - 1)
        span = mt[nums[-1]] - mt[1]
        print(f"\nFIRST CROSSING epoch {n} acc={acc*100:.2f}")
        print(f"t* = ({mt[n]-mt[1]:.2f}) * {n}/{n-1} = {tstar:.2f}s")
        print(f"S_outcome = {max(0.0, min(1.0, (70.0 - tstar) / (70.0 - 28.0))):.4f}"
              "   (t_baseline=70s, t_reference=28s)")
        print(f"ckpt1->ckptN span = {span:.2f}s")
elif not a.only_last:
    print("\nNO CHECKPOINT REACHED TARGET")
