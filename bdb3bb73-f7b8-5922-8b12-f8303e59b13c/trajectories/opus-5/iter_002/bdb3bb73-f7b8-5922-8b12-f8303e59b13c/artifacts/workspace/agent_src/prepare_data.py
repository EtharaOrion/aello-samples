"""Decode the CIFAR-10 PNG folders once into uint8 .npy caches.

File order and integer labels are taken from ``torchvision.datasets.ImageFolder``
so the cached arrays are byte-identical to what the verifier's ImageFolder
pipeline sees (classes sorted alphabetically: airplane=0 ... truck=9).

Run once:  python prepare_data.py
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder

ROOT = "/workspace/data/cifar10"
CACHE = "/workspace/cache"


def build(split: str) -> None:
    ds = ImageFolder(os.path.join(ROOT, split))
    paths = [p for p, _ in ds.samples]
    labels = np.array([t for _, t in ds.samples], dtype=np.int64)
    images = np.zeros((len(paths), 32, 32, 3), dtype=np.uint8)

    def load(i: int) -> None:
        with Image.open(paths[i]) as im:
            images[i] = np.asarray(im.convert("RGB"), dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(load, range(len(paths))))

    np.save(os.path.join(CACHE, f"{split}_x.npy"), images)
    np.save(os.path.join(CACHE, f"{split}_y.npy"), labels)
    print(f"{split}: images={images.shape} labels={labels.shape} classes={ds.classes}")


if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    for split in ("train", "test"):
        build(split)
