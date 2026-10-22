"""Decode the CIFAR-10 PNG folders once into uint8 .npy arrays.

Label convention: class directories are sorted alphabetically, which for
CIFAR-10 gives the canonical label order (airplane=0, automobile=1, bird=2,
cat=3, deer=4, dog=5, frog=6, horse=7, ship=8, truck=9) -- the same order
torchvision's ImageFolder produces, so the head's output columns line up with
the labels the verifier scores against.

Only train.npy / train_labels.npy are consumed by training.  test.npy exists
so that an *already-frozen* checkpoint ladder can be scored locally with the
exact verifier pipeline; lib.py and train.py never read it.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

ROOT = "/workspace/data/cifar10"
OUT = "/workspace/cache"


def decode(paths):
    a = np.empty((len(paths), 32, 32, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        a[i] = np.asarray(Image.open(p).convert("RGB"))
    return a


def dump(split):
    classes = sorted(os.listdir(f"{ROOT}/{split}"))
    assert len(classes) == 10, classes
    paths, labels = [], []
    for ci, c in enumerate(classes):
        d = f"{ROOT}/{split}/{c}"
        for f in sorted(os.listdir(d)):
            paths.append(f"{d}/{f}")
            labels.append(ci)
    chunks = [paths[i:i + 500] for i in range(0, len(paths), 500)]
    with ProcessPoolExecutor(max_workers=16) as ex:
        arrs = list(ex.map(decode, chunks))
    x = np.concatenate(arrs).transpose(0, 3, 1, 2).copy()   # NHWC -> NCHW
    y = np.asarray(labels, dtype=np.int64)
    np.save(f"{OUT}/{split}.npy", x)
    np.save(f"{OUT}/{split}_labels.npy", y)
    print(split, classes)
    print(split, x.shape, x.dtype, "label histogram:", np.bincount(y))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    dump("train")
    dump("test")
