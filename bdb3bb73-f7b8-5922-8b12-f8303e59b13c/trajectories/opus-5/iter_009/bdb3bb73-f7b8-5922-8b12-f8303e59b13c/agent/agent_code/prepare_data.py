"""Decode the CIFAR-10 PNG folders once into uint8 tensors cached on disk.

This is a one-time preprocessing step; it is deliberately kept *outside* the
timed training run so that every experiment reads the same bytes and the
measured epoch time reflects compute, not PNG decoding.

Class ordering follows ``torchvision.datasets.ImageFolder`` (alphabetical:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), which is
the canonical CIFAR-10 label order and is identical for train and test.
"""

import os
import sys
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

ROOT = "/workspace/data/cifar10"
OUT = "/workspace/cache"

CLASSES = sorted(os.listdir(os.path.join(ROOT, "train")))


def _decode(args):
    """Decode every PNG of one (split, class) pair into a uint8 NHWC array."""
    split, cls = args
    d = os.path.join(ROOT, split, cls)
    files = sorted(os.listdir(d))
    arr = np.empty((len(files), 32, 32, 3), dtype=np.uint8)
    for i, f in enumerate(files):
        arr[i] = np.asarray(Image.open(os.path.join(d, f)).convert("RGB"))
    return split, cls, arr


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = [(s, c) for s in ("train", "test") for c in CLASSES]
    with ProcessPoolExecutor(max_workers=20) as ex:
        results = list(ex.map(_decode, jobs))

    for split in ("train", "test"):
        xs, ys = [], []
        for s, cls, arr in results:
            if s != split:
                continue
            xs.append(arr)
            ys.append(np.full(len(arr), CLASSES.index(cls), dtype=np.int64))
        # Sort by class index so the cache layout is deterministic.
        order = np.argsort([CLASSES.index(c) for s, c, _ in results if s == split], kind="stable")
        x = np.concatenate([xs[i] for i in order])
        y = np.concatenate([ys[i] for i in order])
        # NHWC uint8 -> NCHW uint8 (still uint8: normalization happens on GPU).
        x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
        torch.save({"x": x, "y": torch.from_numpy(y)}, os.path.join(OUT, f"{split}.pt"))
        print(f"{split}: x={tuple(x.shape)} y={tuple(y.shape)} "
              f"class_counts={np.bincount(y).tolist()}", flush=True)

    print("classes:", CLASSES)


if __name__ == "__main__":
    sys.exit(main())
