"""Decode the CIFAR-10 PNG folders once into uint8 tensors cached on disk.

Class index order is the sorted class-name order used by torchvision.ImageFolder,
which for CIFAR-10 coincides with the canonical label ordering.
"""
import os
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

ROOT = "/workspace/data/cifar10"
OUT = "/workspace/data_cache"
CLASSES = sorted(os.listdir(os.path.join(ROOT, "train")))
assert len(CLASSES) == 10, CLASSES


def load_one(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_split(split):
    paths, labels = [], []
    for ci, c in enumerate(CLASSES):
        d = os.path.join(ROOT, split, c)
        for fn in sorted(os.listdir(d)):
            paths.append(os.path.join(d, fn))
            labels.append(ci)
    with ProcessPoolExecutor(max_workers=26) as ex:
        imgs = list(ex.map(load_one, paths, chunksize=256))
    x = torch.from_numpy(np.stack(imgs))            # N,32,32,3 uint8
    y = torch.tensor(labels, dtype=torch.int64)
    return x, y


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("classes:", CLASSES, flush=True)
    for split in ("train", "test"):
        x, y = load_split(split)
        torch.save({"x": x, "y": y}, os.path.join(OUT, f"{split}.pt"))
        print(split, x.shape, y.shape, y.bincount().tolist(), flush=True)
