"""One-time data prep: decode the CIFAR-10 PNG folders into uint8 tensors on disk.

This runs ONCE, outside any timed training run. Training then memory-maps /
loads the cached tensors so the measured wall-clock contains only optimisation
work, not PNG decoding.

Class order is the sorted directory listing (airplane, automobile, bird, cat,
deer, dog, frog, horse, ship, truck), which is exactly what
torchvision.datasets.ImageFolder produces and matches canonical CIFAR-10 label
ids 0..9.
"""
import os
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

ROOT = "/workspace/data/cifar10"
OUT = "/workspace/data/cache"
os.makedirs(OUT, exist_ok=True)


def load_split(split):
    d = os.path.join(ROOT, split)
    classes = sorted(os.listdir(d))
    assert len(classes) == 10, classes
    paths, labels = [], []
    for ci, c in enumerate(classes):
        cd = os.path.join(d, c)
        for f in sorted(os.listdir(cd)):
            paths.append(os.path.join(cd, f))
            labels.append(ci)

    imgs = np.zeros((len(paths), 32, 32, 3), dtype=np.uint8)

    def work(i):
        imgs[i] = np.asarray(Image.open(paths[i]).convert("RGB"), dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=32) as ex:
        list(ex.map(work, range(len(paths))))

    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()  # NCHW uint8
    y = torch.tensor(labels, dtype=torch.int64)
    print(split, x.shape, y.shape, "classes:", classes)
    print("  label histogram:", torch.bincount(y).tolist())
    return x, y


if __name__ == "__main__":
    for split in ("train", "test"):
        x, y = load_split(split)
        torch.save({"x": x, "y": y}, os.path.join(OUT, f"{split}.pt"))
    print("wrote cache to", OUT)
