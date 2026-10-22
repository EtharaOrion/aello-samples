"""Decode the CIFAR-10 PNG folders once into uint8 .npy caches.

The verifier feeds the model `transforms.ToTensor()` + `Normalize(CIFAR_MEAN, CIFAR_STD)`
applied to the 32x32 test PNGs.  `ToTensor()` on a uint8 PIL image is exactly
`uint8_HWC / 255.0 -> float32_CHW`, so caching raw uint8 HWC arrays lets us
reproduce the verifier pipeline bit-for-bit later without re-decoding PNGs.

Class order is `sorted(os.listdir(split_dir))`, which matches
`torchvision.datasets.ImageFolder`'s alphabetical `class_to_idx` and therefore the
canonical CIFAR-10 label ids (airplane=0 ... truck=9).
"""

import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

DATA_ROOT = "/workspace/data/cifar10"
CACHE_DIR = "/workspace/cache"


def _decode_one(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _decode_chunk(paths):
    return np.stack([_decode_one(p) for p in paths])


def build_split(split):
    split_dir = os.path.join(DATA_ROOT, split)
    classes = sorted(os.listdir(split_dir))
    assert len(classes) == 10, classes

    paths, labels = [], []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(split_dir, cls)
        for fname in sorted(os.listdir(cls_dir)):
            paths.append(os.path.join(cls_dir, fname))
            labels.append(idx)

    # Chunk across workers; PNG decode is the whole cost here.
    n_workers = min(24, os.cpu_count() or 8)
    chunks = [paths[i::n_workers] for i in range(n_workers)]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        decoded = list(ex.map(_decode_chunk, chunks))

    images = np.empty((len(paths), 32, 32, 3), dtype=np.uint8)
    for w, arr in enumerate(decoded):
        images[w::n_workers] = arr

    labels = np.asarray(labels, dtype=np.int64)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(os.path.join(CACHE_DIR, f"{split}_x.npy"), images)
    np.save(os.path.join(CACHE_DIR, f"{split}_y.npy"), labels)
    print(f"{split}: {images.shape} labels={np.bincount(labels)} classes={classes}", flush=True)


if __name__ == "__main__":
    for split in sys.argv[1:] or ["train", "test"]:
        build_split(split)
