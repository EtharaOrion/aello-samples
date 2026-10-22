"""Cache the CIFAR-10 PNG folders into uint8 .npy arrays.

Layout on disk is torchvision-ImageFolder style: <root>/<split>/<class>/NNNN.png.
Class order follows sorted class-directory names, matching ImageFolder's label
assignment so our label ids agree with the verifier's.

Output: /workspace/cache/{train,test}_{x,y}.npy  (x: uint8 NCHW, y: int64)
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

ROOT = "/workspace/data/cifar10"
OUT = "/workspace/cache"
CLASSES = sorted(os.listdir(os.path.join(ROOT, "train")))
assert len(CLASSES) == 10, CLASSES


def load_split(split):
    paths, labels = [], []
    for label, cls in enumerate(CLASSES):
        d = os.path.join(ROOT, split, cls)
        for fn in sorted(os.listdir(d)):
            paths.append(os.path.join(d, fn))
            labels.append(label)
    x = np.zeros((len(paths), 32, 32, 3), dtype=np.uint8)

    def read(i):
        x[i] = np.asarray(Image.open(paths[i]).convert("RGB"), dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(read, range(len(paths))))
    # NHWC -> NCHW, matching transforms.ToTensor()'s channel-first layout.
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2)), np.asarray(labels, dtype=np.int64)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for split in ("train", "test"):
        x, y = load_split(split)
        np.save(f"{OUT}/{split}_x.npy", x)
        np.save(f"{OUT}/{split}_y.npy", y)
        print(split, x.shape, x.dtype, y.shape, np.bincount(y))
    print("classes:", CLASSES)
