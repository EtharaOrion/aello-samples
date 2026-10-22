"""Cache CIFAR-10 PNG folders into uint8 tensors for fast GPU-resident training.

Class order is torchvision ImageFolder's: sorted directory names, which for
CIFAR-10 gives the canonical airplane..truck -> 0..9 mapping.
"""
import os
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

ROOT = "/workspace/data/cifar10"
OUT = "/workspace/cache"
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

    def rd(p):
        with Image.open(p) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=32) as ex:
        arrs = list(ex.map(rd, paths))
    x = torch.from_numpy(np.stack(arrs))          # N,H,W,C uint8
    x = x.permute(0, 3, 1, 2).contiguous()        # N,C,H,W uint8
    y = torch.tensor(labels, dtype=torch.int64)
    return x, y, classes


if __name__ == "__main__":
    for split in ["train", "test"]:
        x, y, classes = load_split(split)
        torch.save({"x": x, "y": y, "classes": classes},
                   os.path.join(OUT, f"{split}.pt"))
        print(split, tuple(x.shape), x.dtype, "labels:", torch.bincount(y).tolist())
        print("  classes:", classes)
