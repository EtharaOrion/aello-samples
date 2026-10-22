"""Decode the CIFAR-10 PNG folders once into uint8 tensors on disk.

Class order is the sorted folder order (airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck) -- identical to what torchvision's ImageFolder
produces, which is what the verifier uses.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from PIL import Image

ROOT = "/workspace/data/cifar10"
OUT = "/workspace/data_cache"
CLASSES = sorted(os.listdir(os.path.join(ROOT, "train")))
assert len(CLASSES) == 10, CLASSES


def load_class(args):
    split, cls = args
    d = os.path.join(ROOT, split, cls)
    names = sorted(os.listdir(d))
    arr = np.empty((len(names), 32, 32, 3), dtype=np.uint8)
    for i, n in enumerate(names):
        arr[i] = np.asarray(Image.open(os.path.join(d, n)).convert("RGB"))
    return arr


def build(split):
    jobs = [(split, c) for c in CLASSES]
    with ProcessPoolExecutor(max_workers=10) as ex:
        chunks = list(ex.map(load_class, jobs))
    xs = np.concatenate(chunks, 0)
    ys = np.concatenate([np.full(len(c), i, dtype=np.int64) for i, c in enumerate(chunks)])
    x = torch.from_numpy(xs).permute(0, 3, 1, 2).contiguous()  # NCHW uint8
    y = torch.from_numpy(ys)
    return x, y


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for split in ("train", "test"):
        x, y = build(split)
        torch.save({"x": x, "y": y, "classes": CLASSES}, os.path.join(OUT, f"{split}.pt"))
        print(split, tuple(x.shape), y.shape, y.bincount().tolist(), flush=True)
    print("done", file=sys.stderr)
