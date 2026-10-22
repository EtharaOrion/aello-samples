"""Re-implement the verifier's evaluation exactly and report the ladder.

Loads each ckpt_epoch_*.pt with torch.jit.load(...).eval() on the GPU, feeds
batches of 512 test PNGs transformed by
    Compose([ToTensor(), Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
and takes argmax(1).  Also reconstructs t_star the way the verifier clocks it,
from the checkpoint save times.

    python verify.py [ckpt_dir]
"""
import glob
import json
import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CKPT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/workspace/periodic_checkpoints"

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
ds = datasets.ImageFolder("/workspace/data/cifar10/test", transform=tf)
loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
batches = [(x.cuda(non_blocking=True), y.cuda(non_blocking=True)) for x, y in loader]
n = sum(len(y) for _, y in batches)
print(f"test images: {n}  classes: {ds.classes}")

paths = sorted(glob.glob(os.path.join(CKPT_DIR, "ckpt_epoch_*.pt")))
rows, first = [], None
for p in paths:
    ep = int(os.path.basename(p).split("_")[-1].split(".")[0])
    model = torch.jit.load(p).eval().cuda()
    correct = 0
    with torch.no_grad():
        for x, y in batches:
            correct += (model(x).argmax(1) == y).sum().item()
    acc = correct / n
    mt = os.path.getmtime(p)
    rows.append(dict(epoch=ep, acc=acc, mtime=mt))
    if first is None and acc >= 0.94:
        first = rows[-1]
    print(f"  epoch {ep:4d}  acc {acc:.4f}" + ("   <-- first >= 0.94" if first is rows[-1] else ""))

if rows:
    t1 = rows[0]["mtime"]
    span = rows[-1]["mtime"] - t1
    print(f"\nckpt span (epoch {rows[0]['epoch']} -> {rows[-1]['epoch']}): {span:.2f}s")
    if first and rows[0]["epoch"] == 1:
        N = first["epoch"]
        t_star = (first["mtime"] - t1) * N / (N - 1) if N > 1 else 0.0
        s_out = max(0.0, min(1.0, (70.0 - t_star) / (70.0 - 28.0)))
        print(f"first crossing: epoch {N} acc {first['acc']:.4f}")
        print(f"t_star = {t_star:.2f}s   S_outcome = {s_out:.4f}")
    else:
        print("NO CHECKPOINT REACHED 0.94" if not first else "ladder does not start at epoch 1")
    print(f"final epoch acc: {rows[-1]['acc']:.4f}")
    eps = [r["epoch"] for r in rows]
    print(f"contiguous 1..N: {eps == list(range(1, len(eps) + 1))}")

with open("/workspace/verify_report.json", "w") as f:
    json.dump(rows, f, indent=2)
