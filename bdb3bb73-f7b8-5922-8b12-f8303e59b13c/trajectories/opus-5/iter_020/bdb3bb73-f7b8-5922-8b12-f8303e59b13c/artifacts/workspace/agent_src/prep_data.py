"""Cache the CIFAR-10 PNG folders into uint8 .npy arrays (one-off, outside the timed run).

Label order = sorted class-directory names, which is the torchvision ImageFolder
convention and matches the canonical CIFAR-10 ordering:
    airplane automobile bird cat deer dog frog horse ship truck
"""
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

ROOT = "/workspace/data/cifar10"
OUT = "/workspace/cache"
os.makedirs(OUT, exist_ok=True)


def load_split(split):
    classes = sorted(os.listdir(os.path.join(ROOT, split)))
    assert len(classes) == 10, classes
    paths, labels = [], []
    for ci, c in enumerate(classes):
        d = os.path.join(ROOT, split, c)
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".png"):
                paths.append(os.path.join(d, fn))
                labels.append(ci)
    imgs = np.zeros((len(paths), 32, 32, 3), dtype=np.uint8)

    def rd(i):
        imgs[i] = np.asarray(Image.open(paths[i]).convert("RGB"), dtype=np.uint8)

    with ThreadPoolExecutor(32) as ex:
        list(ex.map(rd, range(len(paths))))
    return imgs, np.asarray(labels, dtype=np.int64), classes


for split in ("train", "test"):
    x, y, classes = load_split(split)
    np.save(os.path.join(OUT, f"{split}_x.npy"), x)
    np.save(os.path.join(OUT, f"{split}_y.npy"), y)
    print(split, x.shape, y.shape, np.bincount(y), classes)
