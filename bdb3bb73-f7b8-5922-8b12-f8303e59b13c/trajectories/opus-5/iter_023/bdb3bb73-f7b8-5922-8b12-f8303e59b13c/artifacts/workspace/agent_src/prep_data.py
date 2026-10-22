"""Cache the CIFAR-10 PNG folders into uint8 numpy arrays (one-time cost).

Train and test are cached separately. The test cache is used ONLY by the
offline verification harness (verify.py), never by training.
"""
import os
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

ROOT = "/workspace/data/cifar10"
OUT = "/workspace/cache"
CLASSES = sorted(os.listdir(os.path.join(ROOT, "train")))


def load_split(split):
    paths, labels = [], []
    for ci, cname in enumerate(CLASSES):
        d = os.path.join(ROOT, split, cname)
        for fn in sorted(os.listdir(d)):
            paths.append(os.path.join(d, fn))
            labels.append(ci)
    n = len(paths)
    arr = np.zeros((n, 32, 32, 3), dtype=np.uint8)

    def work(i):
        arr[i] = np.asarray(Image.open(paths[i]).convert("RGB"), dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=32) as ex:
        list(ex.map(work, range(n), chunksize=256))
    return arr, np.asarray(labels, dtype=np.int64), paths


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for split in ["train", "test"]:
        x, y, paths = load_split(split)
        np.save(os.path.join(OUT, f"{split}_x.npy"), x)
        np.save(os.path.join(OUT, f"{split}_y.npy"), y)
        print(f"{split}: {x.shape} {y.shape} classes={CLASSES}", flush=True)
    print("done", flush=True)
