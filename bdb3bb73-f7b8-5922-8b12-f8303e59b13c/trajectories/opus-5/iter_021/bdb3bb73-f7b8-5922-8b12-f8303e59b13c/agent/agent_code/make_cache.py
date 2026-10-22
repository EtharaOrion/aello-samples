"""Decode the CIFAR-10 PNG folders once into uint8 .npy caches.

Everything downstream reads these arrays, so PNG decode cost is paid a single
time instead of on every training run.  Class order is the sorted folder order,
which is exactly what torchvision's ImageFolder (and therefore the verifier)
uses.
"""
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

ROOT = "/workspace/data/cifar10"
OUT = "/workspace/cache"
CLASSES = sorted(os.listdir(os.path.join(ROOT, "train")))


def load_one(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def build(split):
    files, labels = [], []
    for ci, cname in enumerate(CLASSES):
        d = os.path.join(ROOT, split, cname)
        for fn in sorted(os.listdir(d)):
            files.append(os.path.join(d, fn))
            labels.append(ci)
    with ProcessPoolExecutor(max_workers=24) as ex:
        imgs = list(ex.map(load_one, files, chunksize=256))
    x = np.stack(imgs)                      # NHWC uint8
    y = np.asarray(labels, dtype=np.int64)
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, f"{split}_x.npy"), x)
    np.save(os.path.join(OUT, f"{split}_y.npy"), y)
    print(split, x.shape, y.shape, "classes:", CLASSES, flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:] or ["train", "test"]:
        build(s)
