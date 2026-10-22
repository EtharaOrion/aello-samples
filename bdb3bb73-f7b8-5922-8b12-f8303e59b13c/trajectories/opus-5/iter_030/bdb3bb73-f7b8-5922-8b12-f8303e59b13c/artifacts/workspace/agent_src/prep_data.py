"""Decode the CIFAR-10 PNG folders once into uint8 .npy caches.

Class order is the alphabetical folder order used by torchvision's ImageFolder
(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), which is
also the canonical CIFAR-10 label order and the one the verifier will use.
"""
import os
import glob
import time
import numpy as np
from multiprocessing import Pool
from PIL import Image

ROOT = "/workspace/data/cifar10"
OUT = "/workspace/data/cache"
CLASSES = sorted(os.listdir(f"{ROOT}/train"))


def _decode(path):
    return np.asarray(Image.open(path), dtype=np.uint8)


def build(split):
    paths, labels = [], []
    for ci, cname in enumerate(CLASSES):
        fs = sorted(glob.glob(f"{ROOT}/{split}/{cname}/*.png"))
        paths += fs
        labels += [ci] * len(fs)
    with Pool(24) as p:
        imgs = p.map(_decode, paths, chunksize=256)
    x = np.stack(imgs)                      # N,32,32,3 uint8
    y = np.asarray(labels, dtype=np.int64)
    np.save(f"{OUT}/{split}_x.npy", x)
    np.save(f"{OUT}/{split}_y.npy", y)
    return x.shape, np.bincount(y)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("classes:", CLASSES)
    for split in ("train", "test"):
        t = time.time()
        shape, counts = build(split)
        print(f"{split}: {shape} counts={counts.tolist()} ({time.time()-t:.1f}s)")
