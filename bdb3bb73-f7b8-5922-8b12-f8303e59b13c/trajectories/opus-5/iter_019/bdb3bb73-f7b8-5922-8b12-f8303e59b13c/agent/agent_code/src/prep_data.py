"""One-time: decode CIFAR-10 PNG ImageFolder -> uint8 .npy caches.

Class order is alphabetical (airplane, automobile, bird, cat, deer, dog, frog,
horse, ship, truck), which is exactly CIFAR-10's canonical label order and is
what torchvision.datasets.ImageFolder produces for the verifier too.
"""
import os, numpy as np
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

ROOT = "/workspace/data/cifar10"
OUT = "/workspace/cache"
CLASSES = sorted(os.listdir(os.path.join(ROOT, "train")))


def load_one(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def build(split):
    files, labels = [], []
    for ci, c in enumerate(CLASSES):
        d = os.path.join(ROOT, split, c)
        for f in sorted(os.listdir(d)):
            files.append(os.path.join(d, f))
            labels.append(ci)
    with ProcessPoolExecutor(max_workers=26) as ex:
        imgs = list(ex.map(load_one, files, chunksize=256))
    x = np.stack(imgs)                      # N,32,32,3 uint8
    y = np.asarray(labels, dtype=np.int64)
    np.save(f"{OUT}/{split}_x.npy", x)
    np.save(f"{OUT}/{split}_y.npy", y)
    print(split, x.shape, y.shape, np.bincount(y), flush=True)


if __name__ == "__main__":
    print("classes:", CLASSES, flush=True)
    build("train")
    build("test")
