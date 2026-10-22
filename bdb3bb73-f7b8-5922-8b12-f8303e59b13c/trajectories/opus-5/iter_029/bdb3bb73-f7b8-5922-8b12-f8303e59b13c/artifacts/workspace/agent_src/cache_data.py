"""Cache CIFAR-10 PNG folders into uint8 tensors on disk.

Class order is sorted(listdir) == torchvision ImageFolder order ==
canonical CIFAR-10 order (airplane, automobile, bird, cat, deer, dog,
frog, horse, ship, truck), so labels match the verifier's ImageFolder.
"""
import os
import numpy as np
from multiprocessing import Pool
from PIL import Image

ROOT = "/workspace/data/cifar10"
OUT = "/workspace/data_cache"


def _load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def build(split):
    d = os.path.join(ROOT, split)
    classes = sorted(os.listdir(d))
    assert len(classes) == 10, classes
    paths, labels = [], []
    for ci, c in enumerate(classes):
        fs = sorted(os.listdir(os.path.join(d, c)))
        for f in fs:
            paths.append(os.path.join(d, c, f))
            labels.append(ci)
    with Pool(26) as p:
        imgs = p.map(_load, paths, chunksize=64)
    x = np.stack(imgs)                      # N,32,32,3 uint8
    y = np.asarray(labels, dtype=np.int64)
    print(split, x.shape, y.shape, classes, flush=True)
    np.save(os.path.join(OUT, f"{split}_x.npy"), x)
    np.save(os.path.join(OUT, f"{split}_y.npy"), y)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    build("train")
    build("test")
    print("done", flush=True)
