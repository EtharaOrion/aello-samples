"""Decode CIFAR-10 PNG folders into uint8 .npy caches (train + test).

Test images are cached ONLY for the final verifier-convention check of a
finished ladder; they are never read by the training / tuning code paths.
"""
import os, sys, numpy as np
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

ROOT = "/workspace/data/cifar10"
OUT = "/workspace/cache"
CLASSES = sorted(os.listdir(os.path.join(ROOT, "train")))  # ImageFolder ordering

def load_one(args):
    path, = args
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

def build(split):
    files, labels = [], []
    for ci, c in enumerate(CLASSES):
        d = os.path.join(ROOT, split, c)
        for f in sorted(os.listdir(d)):
            files.append(os.path.join(d, f)); labels.append(ci)
    with ProcessPoolExecutor(max_workers=24) as ex:
        imgs = list(ex.map(load_one, [(f,) for f in files], chunksize=256))
    x = np.stack(imgs).transpose(0, 3, 1, 2).copy()   # NHWC -> NCHW uint8
    y = np.asarray(labels, dtype=np.int64)
    np.save(f"{OUT}/{split}_x.npy", x); np.save(f"{OUT}/{split}_y.npy", y)
    print(split, x.shape, y.shape, "class_counts", np.bincount(y), flush=True)

if __name__ == "__main__":
    print("classes:", CLASSES, flush=True)
    for s in ["train", "test"]:
        build(s)
