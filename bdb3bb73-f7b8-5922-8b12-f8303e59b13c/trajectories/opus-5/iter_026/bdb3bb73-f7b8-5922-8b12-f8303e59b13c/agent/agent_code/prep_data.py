"""One-time: decode the CIFAR-10 PNG folders into uint8 .npy caches.

Ordering note: we sort by (class_index, filename) exactly as torchvision's
ImageFolder does, so our label vector matches the verifier's ImageFolder
labels. The arrays are stored as raw uint8 HWC -- identical to what
PIL.Image.open(...).convert('RGB') yields -- so ToTensor()/255 reproduces the
verifier's floats bit-for-bit.
"""
import os, sys, numpy as np
from PIL import Image

ROOT = "/workspace/data/cifar10"
OUT = "/workspace/cache"

def dump(split):
    d = os.path.join(ROOT, split)
    classes = sorted(e.name for e in os.scandir(d) if e.is_dir())
    assert len(classes) == 10, classes
    imgs, labels = [], []
    for ci, c in enumerate(classes):
        cd = os.path.join(d, c)
        for fn in sorted(os.listdir(cd)):
            imgs.append(np.asarray(Image.open(os.path.join(cd, fn)).convert("RGB"), dtype=np.uint8))
            labels.append(ci)
    x = np.stack(imgs)                      # (N,32,32,3) uint8
    y = np.asarray(labels, dtype=np.int64)  # (N,)
    np.save(os.path.join(OUT, f"{split}_x.npy"), x)
    np.save(os.path.join(OUT, f"{split}_y.npy"), y)
    print(split, x.shape, y.shape, "classes:", classes, flush=True)

if __name__ == "__main__":
    for s in ("train", "test"):
        dump(s)
    print("done", flush=True)
