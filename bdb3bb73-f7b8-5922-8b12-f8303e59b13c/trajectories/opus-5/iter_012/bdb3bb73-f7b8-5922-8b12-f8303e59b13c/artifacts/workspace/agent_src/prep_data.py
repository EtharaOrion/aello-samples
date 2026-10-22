"""Cache the CIFAR-10 PNG folders into uint8 tensors so training never reads the filesystem.

Class order is alphabetical (ImageFolder convention), which matches the canonical
CIFAR-10 label order: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from PIL import Image

ROOT = "/workspace/data/cifar10"
OUT = "/workspace/data/cache"
CLASSES = sorted(os.listdir(os.path.join(ROOT, "train")))


def load_one(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def build(split):
    paths, labels = [], []
    for ci, cname in enumerate(CLASSES):
        d = os.path.join(ROOT, split, cname)
        for fn in sorted(os.listdir(d)):
            paths.append(os.path.join(d, fn))
            labels.append(ci)
    with ProcessPoolExecutor(max_workers=32) as ex:
        imgs = list(ex.map(load_one, paths, chunksize=64))
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous()  # N,3,32,32 uint8
    y = torch.tensor(labels, dtype=torch.int64)
    torch.save({"x": x, "y": y, "paths": paths}, os.path.join(OUT, f"{split}.pt"))
    print(split, tuple(x.shape), y.bincount().tolist(), flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("classes:", CLASSES, flush=True)
    build("train")
    build("test")
