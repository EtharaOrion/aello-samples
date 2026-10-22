"""Decode the CIFAR-10 PNG folders once into uint8 .npy caches.

The verifier feeds the model `ToTensor()` + `Normalize(CIFAR_MEAN, CIFAR_STD)` of the
32x32 test PNGs, i.e. RGB / 255 then per-channel standardisation.  We therefore cache
raw RGB uint8 NHWC arrays and reproduce that exact transform on the GPU at train time.
"""
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

ROOT = "/workspace/data/cifar10"
OUT = "/workspace/cache"
# torchvision.datasets.ImageFolder orders classes alphabetically; match that so the
# label ids we train with are the same ones the verifier's folder order implies.
CLASSES = sorted(os.listdir(os.path.join(ROOT, "train")))


def _load_one(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _load_class(args):
    split, cls = args
    d = os.path.join(ROOT, split, cls)
    files = sorted(os.listdir(d))
    arr = np.stack([_load_one(os.path.join(d, f)) for f in files])
    return cls, arr


def build(split):
    with ProcessPoolExecutor(16) as ex:
        results = dict(ex.map(_load_class, [(split, c) for c in CLASSES]))
    xs, ys = [], []
    for i, c in enumerate(CLASSES):
        xs.append(results[c])
        ys.append(np.full(len(results[c]), i, dtype=np.int64))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    # Deterministic shuffle so that a contiguous slice is a class-balanced sample.
    rng = np.random.default_rng(1234)
    perm = rng.permutation(len(x))
    x, y = x[perm], y[perm]
    np.save(os.path.join(OUT, f"{split}_x.npy"), x)
    np.save(os.path.join(OUT, f"{split}_y.npy"), y)
    print(split, x.shape, y.shape, np.bincount(y), flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("classes:", CLASSES, flush=True)
    for s in ("train", "test"):
        build(s)
    print("done", flush=True)
