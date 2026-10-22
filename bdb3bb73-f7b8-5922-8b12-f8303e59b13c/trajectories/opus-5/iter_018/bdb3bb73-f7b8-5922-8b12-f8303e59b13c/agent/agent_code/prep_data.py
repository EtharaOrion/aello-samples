"""Decode the CIFAR-10 PNG folders once into uint8 .npy caches.

The on-disk dataset is 60k individual PNGs; decoding them costs ~30 s of CPU and
we need the pixels many times over the session, so we pay that cost exactly once
here and let every later script mmap the arrays.

Layout produced in /workspace/cache:
    train_x.npy  (50000, 32, 32, 3) uint8, RGB, HWC
    train_y.npy  (50000,)           int64
    test_x.npy   (10000, 32, 32, 3) uint8
    test_y.npy   (10000,)           int64

Class order is the sorted folder order, which is exactly what
torchvision.datasets.ImageFolder (used by the verifier) assigns.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

ROOT = "/workspace/data/cifar10"
OUT = "/workspace/cache"


def _decode(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def build(split):
    classes = sorted(os.listdir(f"{ROOT}/{split}"))
    assert len(classes) == 10, classes
    paths, labels = [], []
    for ci, cname in enumerate(classes):
        d = f"{ROOT}/{split}/{cname}"
        for fn in sorted(os.listdir(d)):
            paths.append(f"{d}/{fn}")
            labels.append(ci)
    with ProcessPoolExecutor(max_workers=24) as ex:
        imgs = list(ex.map(_decode, paths, chunksize=256))
    x = np.stack(imgs)
    y = np.asarray(labels, dtype=np.int64)
    np.save(f"{OUT}/{split}_x.npy", x)
    np.save(f"{OUT}/{split}_y.npy", y)
    print(split, x.shape, y.shape, "classes:", classes, flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    build("train")
    build("test")
    print("done", flush=True)
