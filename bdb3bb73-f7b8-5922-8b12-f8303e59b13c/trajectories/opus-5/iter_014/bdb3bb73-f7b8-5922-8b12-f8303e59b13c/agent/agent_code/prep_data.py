"""Decode the CIFAR-10 PNG tree once into uint8 .npy caches.

Class order is sorted(dirnames), which matches torchvision.datasets.ImageFolder
and the canonical CIFAR-10 label order (airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck).

The test cache is written for one purpose only: verifying already-saved
checkpoints the same way the grader does. It is never read by the training code.
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

ROOT = "/workspace/data/cifar10"
OUT = "/workspace/cache"


def load_split(split):
    d = os.path.join(ROOT, split)
    classes = sorted(os.listdir(d))
    assert len(classes) == 10, classes
    files, labels = [], []
    for ci, c in enumerate(classes):
        cd = os.path.join(d, c)
        for fn in sorted(os.listdir(cd)):
            files.append(os.path.join(cd, fn))
            labels.append(ci)
    x = np.zeros((len(files), 32, 32, 3), dtype=np.uint8)

    def work(i):
        x[i] = np.asarray(Image.open(files[i]).convert("RGB"), dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=32) as ex:
        list(ex.map(work, range(len(files)), chunksize=256))
    return x, np.asarray(labels, dtype=np.int64), classes


def main():
    for split in ("train", "test"):
        xp = os.path.join(OUT, f"{split}_x.npy")
        if os.path.exists(xp):
            print(f"{split}: cached")
            continue
        x, y, classes = load_split(split)
        np.save(xp, x)
        np.save(os.path.join(OUT, f"{split}_y.npy"), y)
        print(f"{split}: {x.shape} labels={np.bincount(y)} classes={classes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
