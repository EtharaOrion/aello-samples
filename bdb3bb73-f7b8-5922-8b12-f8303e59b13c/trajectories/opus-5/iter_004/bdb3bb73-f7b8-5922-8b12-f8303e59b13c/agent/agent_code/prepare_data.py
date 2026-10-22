"""Decode the CIFAR-10 PNG folders once into uint8 .npy caches.

The verifier builds its test batches with torchvision's ImageFolder ordering, i.e.
classes sorted alphabetically, which for CIFAR-10 coincides with the canonical
label ids (airplane=0 ... truck=9). We reproduce that ordering here so the label
space of our checkpoints matches the verifier's without any remapping.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

ROOT = "/workspace/data/cifar10"
OUT = "/workspace/data_npy"
CLASSES = sorted(os.listdir(f"{ROOT}/train"))


def _decode_class(arg):
    split, label, cls = arg
    d = f"{ROOT}/{split}/{cls}"
    names = sorted(os.listdir(d))
    x = np.empty((len(names), 32, 32, 3), dtype=np.uint8)
    for i, n in enumerate(names):
        x[i] = np.asarray(Image.open(f"{d}/{n}").convert("RGB"), dtype=np.uint8)
    return label, x


def build(split):
    jobs = [(split, i, c) for i, c in enumerate(CLASSES)]
    with ProcessPoolExecutor(10) as ex:
        parts = list(ex.map(_decode_class, jobs))
    parts.sort(key=lambda p: p[0])
    x = np.concatenate([p[1] for p in parts])
    y = np.concatenate([np.full(len(p[1]), p[0], dtype=np.int64) for p in parts])
    np.save(f"{OUT}/{split}_x.npy", x)
    np.save(f"{OUT}/{split}_y.npy", y)
    print(split, x.shape, y.shape, np.bincount(y), flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("classes:", CLASSES, flush=True)
    build("train")
    build("test")
