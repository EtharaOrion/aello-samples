"""Decode the CIFAR-10 PNG folders once into uint8 tensors cached on disk.

Class indices follow ``torchvision.datasets.ImageFolder`` semantics (sorted
directory names), which is exactly what the verifier uses, so label ids are
guaranteed to line up with the verifier's.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

ROOT = "/workspace/data/cifar10"
OUT = "/workspace/cache"


def load_split(split):
    d = os.path.join(ROOT, split)
    classes = sorted(e.name for e in os.scandir(d) if e.is_dir())
    files, labels = [], []
    for ci, c in enumerate(classes):
        cd = os.path.join(d, c)
        for fn in sorted(os.listdir(cd)):
            files.append(os.path.join(cd, fn))
            labels.append(ci)
    n = len(files)
    images = np.zeros((n, 32, 32, 3), dtype=np.uint8)

    def work(i):
        images[i] = np.asarray(Image.open(files[i]).convert("RGB"))

    with ThreadPoolExecutor(max_workers=32) as ex:
        list(ex.map(work, range(n)))
    return torch.from_numpy(images), torch.tensor(labels, dtype=torch.int64), classes


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for split in ("train", "test"):
        t0 = time.time()
        x, y, classes = load_split(split)
        torch.save({"images": x, "labels": y, "classes": classes},
                   os.path.join(OUT, f"{split}.pt"))
        print(f"{split}: {tuple(x.shape)} labels={y.shape} "
              f"classes={classes} in {time.time() - t0:.1f}s", flush=True)
    sys.exit(0)
