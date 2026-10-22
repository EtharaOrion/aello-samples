"""Evaluate a checkpoint ladder exactly the way the verifier does.

torch.jit.load(path).eval() on GPU, batch size 512, inputs = test PNGs through
transforms.ToTensor() + transforms.Normalize(CIFAR_MEAN, CIFAR_STD), prediction
= argmax(1).
"""
import argparse
import json
import os
import sys

import torch

from model import CIFAR_MEAN, CIFAR_STD

CACHE = "/workspace/data_cache"


def test_tensors(device):
    d = torch.load(os.path.join(CACHE, "test.pt"))
    x = d["x"].to(device).permute(0, 3, 1, 2).float().div_(255.0)   # ToTensor()
    mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std, d["y"].to(device)                       # Normalize()


@torch.no_grad()
def accuracy(path, x, y, bs=512):
    m = torch.jit.load(path, map_location="cuda").eval()
    correct = 0
    for i in range(0, len(y), bs):
        correct += (m(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
    return correct / len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--last-only", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda")
    x, y = test_tensors(dev)
    files = sorted(f for f in os.listdir(a.ckpt_dir) if f.startswith("ckpt_epoch_") and f.endswith(".pt"))
    if a.last_only:
        files = files[-1:]
    res = {}
    first94 = None
    for f in files:
        acc = accuracy(os.path.join(a.ckpt_dir, f), x, y)
        res[f] = acc
        ep = int(f[len("ckpt_epoch_"):-3])
        if first94 is None and acc >= 0.94:
            first94 = ep
        print(f"{f}  acc={acc:.4f}", flush=True)
    print(json.dumps({"first_epoch_ge_94": first94,
                      "final_acc": res[files[-1]] if files else None}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
