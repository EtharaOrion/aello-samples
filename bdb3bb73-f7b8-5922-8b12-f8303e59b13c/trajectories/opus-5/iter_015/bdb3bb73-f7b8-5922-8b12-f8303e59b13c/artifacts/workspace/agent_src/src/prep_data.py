"""Cache the CIFAR-10 PNG folders into uint8 .npy arrays.

The on-disk layout is `/workspace/data/cifar10/{split}/{class}/{id}.png`.  Decoding
60k PNGs with PIL costs ~30 s single threaded, which we only ever want to pay once,
so the decoded arrays are memoised under /workspace/cache.

Ordering is made deterministic (classes sorted alphabetically, filenames sorted by
integer id) so that every later run sees byte-identical data.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

ROOT = "/workspace/data/cifar10"
CACHE = "/workspace/cache"


def _load_one(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_class(args):
    split, cls_idx, cls_name = args
    d = os.path.join(ROOT, split, cls_name)
    files = sorted(os.listdir(d), key=lambda f: int(os.path.splitext(f)[0]))
    imgs = np.stack([_load_one(os.path.join(d, f)) for f in files])
    return cls_idx, imgs


def build(split):
    xs_path = os.path.join(CACHE, f"{split}_x.npy")
    ys_path = os.path.join(CACHE, f"{split}_y.npy")
    if os.path.exists(xs_path) and os.path.exists(ys_path):
        return np.load(xs_path), np.load(ys_path)

    classes = sorted(os.listdir(os.path.join(ROOT, split)))
    assert len(classes) == 10, classes
    jobs = [(split, i, c) for i, c in enumerate(classes)]
    with ProcessPoolExecutor(max_workers=10) as ex:
        results = sorted(ex.map(_load_class, jobs), key=lambda r: r[0])

    xs = np.concatenate([r[1] for r in results])                       # NHWC uint8
    ys = np.concatenate([np.full(len(r[1]), r[0], np.int64) for r in results])
    np.save(xs_path, xs)
    np.save(ys_path, ys)
    return xs, ys


if __name__ == "__main__":
    for split in ("train", "test"):
        x, y = build(split)
        print(split, x.shape, x.dtype, y.shape, np.bincount(y), flush=True)
    print("classes:", sorted(os.listdir(os.path.join(ROOT, "train"))))
