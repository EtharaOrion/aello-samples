"""Cache CIFAR-10 ImageFolder PNGs into uint8 numpy arrays for fast GPU-resident training."""
import os, numpy as np
from PIL import Image

ROOT = "/workspace/data/cifar10"
OUT = "/workspace/cache"

def dump(split):
    classes = sorted(os.listdir(os.path.join(ROOT, split)))
    assert len(classes) == 10, classes
    xs, ys = [], []
    for ci, c in enumerate(classes):
        d = os.path.join(ROOT, split, c)
        for fn in sorted(os.listdir(d)):
            xs.append(np.asarray(Image.open(os.path.join(d, fn)).convert("RGB"), dtype=np.uint8))
            ys.append(ci)
    x = np.stack(xs)            # N,32,32,3
    y = np.asarray(ys, dtype=np.int64)
    np.save(f"{OUT}/{split}_x.npy", x)
    np.save(f"{OUT}/{split}_y.npy", y)
    print(split, x.shape, y.shape, classes)

for s in ("train", "test"):
    dump(s)
