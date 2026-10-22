"""Decode the CIFAR-10 PNG folders once into uint8 tensors cached on disk.

Layout on disk is  /workspace/data/cifar10/{train,test}/<class_name>/<id>.png .
We decode with a process pool (26 cores available) and cache

    /workspace/cache/train_u8.pt  ->  dict(x=uint8[N,3,32,32], y=int64[N])
    /workspace/cache/test_u8.pt   ->  dict(x=uint8[N,3,32,32], y=int64[N])

The *test* cache is only ever consumed by `evaluate.py`, which mirrors the
verifier's own pipeline.  Nothing in `train.py` reads it -- every modelling
decision in this project is made against a held-out slice of the *train*
split (see README.md, "No test-set leakage").
"""
import os
import sys
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

ROOT = "/workspace/data/cifar10"
CACHE = "/workspace/cache"
# torchvision.datasets.ImageFolder orders classes alphabetically; the verifier
# uses that same convention, so we hard-code it to stay in lockstep.
CLASSES = sorted(
    ["airplane", "automobile", "bird", "cat", "deer", "dog",
     "frog", "horse", "ship", "truck"]
)


def _decode_one(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _decode_split(split):
    paths, labels = [], []
    for ci, cname in enumerate(CLASSES):
        d = os.path.join(ROOT, split, cname)
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".png"):
                paths.append(os.path.join(d, fn))
                labels.append(ci)
    with ProcessPoolExecutor(max_workers=24) as ex:
        arrs = list(ex.map(_decode_one, paths, chunksize=256))
    x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).contiguous()  # NCHW uint8
    y = torch.tensor(labels, dtype=torch.int64)
    return x, y


def main():
    os.makedirs(CACHE, exist_ok=True)
    for split in ("train", "test"):
        out = os.path.join(CACHE, f"{split}_u8.pt")
        if os.path.exists(out):
            print(f"{split}: cache already present at {out}")
            continue
        x, y = _decode_split(split)
        torch.save({"x": x, "y": y}, out)
        print(f"{split}: {tuple(x.shape)} {x.dtype}  labels={torch.bincount(y).tolist()}")


if __name__ == "__main__":
    sys.exit(main())
