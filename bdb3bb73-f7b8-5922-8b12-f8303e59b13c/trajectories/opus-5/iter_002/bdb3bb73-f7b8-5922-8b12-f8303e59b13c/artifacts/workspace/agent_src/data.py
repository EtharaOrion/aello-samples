"""GPU-resident CIFAR-10 data + on-device augmentation.

The whole split lives on the GPU as one tensor, and a full epoch's worth of
augmentation is generated in a handful of kernel launches, so the input
pipeline costs a fraction of a millisecond per step and the H100 never waits
on the host.
"""

import numpy as np
import torch
import torch.nn.functional as F

from model import CIFAR_MEAN, CIFAR_STD

CACHE = "/workspace/cache"
_LOADED = {}


def load_split(split: str, device: str = "cuda"):
    """Return (standardized float32 NCHW images, int64 labels) for a split.

    Memoised so a sweep of many configs in one process pays the decode once.
    """
    if (split, device) in _LOADED:
        return _LOADED[(split, device)]
    x = np.load(f"{CACHE}/{split}_x.npy")  # (N, 32, 32, 3) uint8
    y = np.load(f"{CACHE}/{split}_y.npy")
    images = torch.from_numpy(x).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)
    images = (images - mean) / std
    _LOADED[(split, device)] = (images.contiguous(), torch.from_numpy(y).to(device))
    return _LOADED[(split, device)]


class EpochAugmenter:
    """Pre-pads the training set once, then emits an augmented epoch at a time.

    Augmentations (all applied to the standardized tensor, so a fill value of 0
    is the per-channel mean colour):
      * random translation by up to ``pad`` pixels, from a reflect-padded copy
      * random horizontal flip
      * optional cutout square
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, pad: int = 2,
                 dtype: torch.dtype = torch.bfloat16):
        self.n = images.shape[0]
        self.pad = pad
        self.padded = F.pad(images, (pad,) * 4, mode="reflect").to(dtype).contiguous()
        self.labels = labels
        self.device = images.device
        self.ar = torch.arange(32, device=self.device)

    def epoch(self, gen: torch.Generator, flip: bool = True, cutout: int = 0):
        n, p, dev = self.n, self.pad, self.device
        perm = torch.randperm(n, device=dev, generator=gen)
        x = self.padded[perm]  # (n, 3, 32+2p, 32+2p)
        w = 32 + 2 * p

        # Random crop == random translation, done as two gathers.
        oy = torch.randint(0, 2 * p + 1, (n,), device=dev, generator=gen)
        ox = torch.randint(0, 2 * p + 1, (n,), device=dev, generator=gen)
        idx_h = (self.ar.view(1, 1, 32, 1) + oy.view(n, 1, 1, 1)).expand(n, 3, 32, w)
        x = torch.gather(x, 2, idx_h)
        idx_w = (self.ar.view(1, 1, 1, 32) + ox.view(n, 1, 1, 1)).expand(n, 3, 32, 32)
        x = torch.gather(x, 3, idx_w)

        if flip:
            mask = torch.rand(n, device=dev, generator=gen) < 0.5
            x = torch.where(mask.view(n, 1, 1, 1), torch.flip(x, [3]), x)

        if cutout > 0:
            cy = torch.randint(0, 32, (n, 1, 1), device=dev, generator=gen)
            cx = torch.randint(0, 32, (n, 1, 1), device=dev, generator=gen)
            r = cutout // 2
            hit = ((self.ar.view(1, 32, 1) - cy).abs() < r) & (
                (self.ar.view(1, 1, 32) - cx).abs() < r
            )
            x = x.masked_fill(hit.unsqueeze(1), 0.0)

        x = x.contiguous(memory_format=torch.channels_last)
        return x, self.labels[perm]
