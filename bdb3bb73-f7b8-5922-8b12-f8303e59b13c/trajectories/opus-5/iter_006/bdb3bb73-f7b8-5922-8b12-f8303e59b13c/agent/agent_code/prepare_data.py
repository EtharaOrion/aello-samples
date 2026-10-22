"""Decode the CIFAR-10 PNG folders once into uint8 .npy caches.

Run:  python prepare_data.py --split train   (writes cache/train_{x,y}.npy)
      python prepare_data.py --split test    (writes cache/test_{x,y}.npy)

The two splits are cached by separate invocations on purpose: `train.py` only ever
reads the train cache, so there is no code path by which test images can reach the
optimizer. The test cache is consumed exclusively by `evaluate.py`, which scores
already-saved checkpoints.

Class order matches `torchvision.datasets.ImageFolder` (alphabetical), which is the
order the verifier's loader uses.
"""
import argparse
import os
import numpy as np
from PIL import Image

DATA_ROOT = "/workspace/data/cifar10"
CACHE_DIR = "/workspace/cache"

# Alphabetical == ImageFolder's class_to_idx ordering.
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def build(split: str):
    xs, ys = [], []
    for label, cls in enumerate(CLASSES):
        d = os.path.join(DATA_ROOT, split, cls)
        files = sorted(os.listdir(d))
        for fn in files:
            if not fn.endswith(".png"):
                continue
            xs.append(np.asarray(Image.open(os.path.join(d, fn)).convert("RGB")))
            ys.append(label)
    x = np.stack(xs).astype(np.uint8)          # (N, 32, 32, 3) HWC
    y = np.asarray(ys, dtype=np.int64)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(os.path.join(CACHE_DIR, f"{split}_x.npy"), x)
    np.save(os.path.join(CACHE_DIR, f"{split}_y.npy"), y)
    print(f"{split}: x={x.shape} {x.dtype}  y={y.shape}  counts={np.bincount(y)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    build(ap.parse_args().split)
