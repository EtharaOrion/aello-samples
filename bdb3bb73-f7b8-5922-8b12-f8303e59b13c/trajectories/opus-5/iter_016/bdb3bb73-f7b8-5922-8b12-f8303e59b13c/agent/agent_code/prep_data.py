"""One-time cache of the CIFAR-10 PNG folders into uint8 NCHW tensors.

Reads /workspace/data/cifar10/{train,test} (torchvision ImageFolder layout) and
writes cache/{train,test}_{x,y}.npy. Class order is ImageFolder's sorted order,
which is exactly what the verifier uses, so label indices match.
"""
import os, numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

ROOT = "/workspace/data/cifar10"
OUT = "/workspace/cache"


def load_split(split):
    d = os.path.join(ROOT, split)
    classes = sorted(e.name for e in os.scandir(d) if e.is_dir())
    files, labels = [], []
    for ci, c in enumerate(classes):
        for f in sorted(os.listdir(os.path.join(d, c))):
            files.append(os.path.join(d, c, f))
            labels.append(ci)
    x = np.zeros((len(files), 32, 32, 3), dtype=np.uint8)

    def rd(i):
        x[i] = np.asarray(Image.open(files[i]).convert("RGB"), dtype=np.uint8)

    with ThreadPoolExecutor(32) as ex:
        list(ex.map(rd, range(len(files))))
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2)), np.array(labels, dtype=np.int64), classes


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for split in ("train", "test"):
        x, y, classes = load_split(split)
        np.save(f"{OUT}/{split}_x.npy", x)
        np.save(f"{OUT}/{split}_y.npy", y)
        print(split, x.shape, y.shape, np.bincount(y), classes, flush=True)
