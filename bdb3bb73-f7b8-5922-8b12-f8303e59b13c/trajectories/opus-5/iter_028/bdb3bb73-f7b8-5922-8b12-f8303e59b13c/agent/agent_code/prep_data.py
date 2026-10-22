"""One-time cache of the CIFAR-10 PNG folders into uint8 .npy arrays.

Class indices follow torchvision.datasets.ImageFolder convention (sorted class
directory names), which is the same ordering the verifier uses when it scores
the test split.
"""
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

ROOT = "/workspace/data/cifar10"
OUT = "/workspace/data_cache"


def load_class(args):
    split, cls = args
    d = os.path.join(ROOT, split, cls)
    files = sorted(os.listdir(d))
    arr = np.zeros((len(files), 32, 32, 3), dtype=np.uint8)
    for i, f in enumerate(files):
        arr[i] = np.asarray(Image.open(os.path.join(d, f)).convert("RGB"))
    return arr


def build(split):
    classes = sorted(os.listdir(os.path.join(ROOT, split)))
    with ProcessPoolExecutor(max_workers=20) as ex:
        chunks = list(ex.map(load_class, [(split, c) for c in classes]))
    x = np.concatenate(chunks, 0)
    y = np.concatenate([np.full(len(c), i, dtype=np.int64)
                        for i, c in enumerate(chunks)])
    np.save(f"{OUT}/{split}_x.npy", x)
    np.save(f"{OUT}/{split}_y.npy", y)
    print(split, x.shape, y.shape, "classes:", classes)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    build("train")
    build("test")
