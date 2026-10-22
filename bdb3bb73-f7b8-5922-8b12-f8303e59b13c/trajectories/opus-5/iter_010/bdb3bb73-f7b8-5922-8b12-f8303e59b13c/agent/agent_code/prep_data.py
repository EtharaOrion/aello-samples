"""Decode the CIFAR-10 PNG folders once into uint8 numpy caches.

Class order comes from torchvision's ImageFolder (alphabetical), which is the
same ordering the verifier uses, so label indices match.

/dev/shm is only 64 MB here, so we avoid torch DataLoader workers (they pass
tensors through shared memory) and use a plain process pool over file paths.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from torchvision import datasets

OUT = "/workspace/data_cache"
ROOT = "/workspace/data/cifar10"


def decode(paths):
    out = np.empty((len(paths), 32, 32, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        out[i] = np.asarray(Image.open(p).convert("RGB"))
    return out


def dump(split):
    ds = datasets.ImageFolder(os.path.join(ROOT, split))
    paths = [p for p, _ in ds.samples]
    y = np.array([t for _, t in ds.samples], dtype=np.int64)
    chunks = [paths[i:i + 512] for i in range(0, len(paths), 512)]
    with ProcessPoolExecutor(max_workers=24) as ex:
        parts = list(ex.map(decode, chunks))
    x = np.concatenate(parts).transpose(0, 3, 1, 2).copy()   # (N, 3, 32, 32)
    np.save(f"{OUT}/{split}_x.npy", x)
    np.save(f"{OUT}/{split}_y.npy", y)
    print(split, x.shape, x.dtype, ds.classes, np.bincount(y))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    dump("train")
    dump("test")
