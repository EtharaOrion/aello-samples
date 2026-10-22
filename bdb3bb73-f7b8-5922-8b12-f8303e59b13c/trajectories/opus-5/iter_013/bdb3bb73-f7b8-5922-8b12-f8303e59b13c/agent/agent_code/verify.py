"""Replica of the verifier's evaluation path, run over a checkpoint ladder.

Loads the *PNG* test split through exactly the documented transform

    transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.4914,0.4822,0.4465),
                                             (0.2470,0.2435,0.2616))])

feeds float32 NCHW batches of 512 to `torch.jit.load(path).eval()`, and takes
argmax(1).  This is the only place the test split is ever read, and it runs
after training/selection is complete.
"""
import argparse
import glob
import json
import os
import re

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TFM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


def load_test(bs=512):
    ds = datasets.ImageFolder("/workspace/data/cifar10/test", transform=TFM)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    xs, ys = [], []
    for x, y in dl:
        xs.append(x)
        ys.append(y)
    return xs, [t.cuda() for t in ys], ds.classes


@torch.no_grad()
def accuracy(path, xs, ys):
    model = torch.jit.load(path).eval().cuda()
    correct = total = 0
    for x, y in zip(xs, ys):
        pred = model(x.cuda()).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/periodic_checkpoints")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    xs, ys, classes = load_test()
    print("classes:", classes)
    paths = sorted(glob.glob(os.path.join(a.dir, "ckpt_epoch_*.pt")))
    mtimes = {p: os.path.getmtime(p) for p in paths}
    t_first = mtimes[paths[0]] if paths else 0.0
    rows, first_cross = [], None
    for p in paths:
        n = int(re.search(r"(\d+)", os.path.basename(p)).group(1))
        acc = accuracy(p, xs, ys)
        # verifier's clock: mean epoch duration extrapolated to epoch n
        tstar = (mtimes[p] - t_first) * n / (n - 1) if n > 1 else 0.0
        rows.append({"epoch": n, "acc": acc, "t_star": round(tstar, 3)})
        if acc >= 0.94 and first_cross is None:
            first_cross = rows[-1]
        print(f"  epoch {n:4d}  acc={acc:.4f}  t*={tstar:7.2f}s")
    span = mtimes[paths[-1]] - t_first if paths else 0.0
    summary = {"n_ckpt": len(paths), "ladder_span_sec": round(span, 3),
               "first_cross": first_cross, "final_acc": rows[-1]["acc"] if rows else None,
               "rows": rows}
    print("SUMMARY " + json.dumps({k: v for k, v in summary.items() if k != "rows"}))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(summary, f, indent=2)
