"""GPU-resident CIFAR-10 tensors + on-device augmentation.

All 50k training images live on the GPU as standardized fp16, so a training step
never reaches back to the host.  Augmentation is three tensor ops (two gathers + a where),
which keeps the augmentation cost well under a millisecond per batch.
"""
import numpy as np
import torch

from net import CIFAR_MEAN, CIFAR_STD

CACHE = "/workspace/cache"


def _standardize(x_uint8_nhwc, device, dtype):
    """uint8 NHWC -> standardized float NCHW, matching ToTensor()+Normalize() exactly."""
    x = torch.from_numpy(x_uint8_nhwc).to(device)
    x = x.permute(0, 3, 1, 2).contiguous().float().div_(255.0)  # ToTensor()
    mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)
    x = (x - mean) / std                                        # Normalize()
    return x.to(dtype)


def load_split(split, device="cuda", dtype=torch.float16):
    x = np.load(f"{CACHE}/{split}_x.npy")
    y = np.load(f"{CACHE}/{split}_y.npy")
    return _standardize(x, device, dtype), torch.from_numpy(y).to(device)


def train_val_split(x, y, n_val=5000, seed=0):
    """Deterministic class-balanced-ish holdout carved out of the *train* split.

    Hyperparameter selection uses only this holdout; the test split is never read
    during training or tuning.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(len(x), generator=g).to(x.device)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    return x[tr_idx], y[tr_idx], x[val_idx], y[val_idx]


class Augmenter:
    """Random translate (reflect-pad, +-`pad` px) + horizontal flip + optional cutout."""

    def __init__(self, x, pad=2, flip=True, cutout=0):
        self.pad = pad
        self.flip = flip
        self.cutout = cutout
        self.n, _, self.h, self.w = x.shape
        if pad > 0:
            self.padded = torch.nn.functional.pad(x, (pad,) * 4, mode="reflect")
        else:
            self.padded = x
        self.ar_h = torch.arange(self.h, device=x.device)
        self.ar_w = torch.arange(self.w, device=x.device)

    def __call__(self, idx):
        x = self.padded[idx]
        b = x.size(0)
        dev = x.device
        if self.pad > 0:
            r = self.pad
            dy = torch.randint(0, 2 * r + 1, (b,), device=dev)
            dx = torch.randint(0, 2 * r + 1, (b,), device=dev)
            iy = (dy.view(b, 1) + self.ar_h.view(1, -1)).view(b, 1, self.h, 1)
            x = x.gather(2, iy.expand(b, 3, self.h, x.size(3)))
            ix = (dx.view(b, 1) + self.ar_w.view(1, -1)).view(b, 1, 1, self.w)
            x = x.gather(3, ix.expand(b, 3, self.h, self.w))
        if self.flip:
            m = torch.rand(b, 1, 1, 1, device=dev) < 0.5
            x = torch.where(m, x.flip(3), x)
        if self.cutout > 0:
            c = self.cutout
            cy = torch.randint(0, self.h, (b, 1, 1), device=dev)
            cx = torch.randint(0, self.w, (b, 1, 1), device=dev)
            gy = self.ar_h.view(1, -1, 1)
            gx = self.ar_w.view(1, 1, -1)
            keep = ~(((gy - cy).abs() < c // 2) & ((gx - cx).abs() < c // 2))
            x = x * keep.unsqueeze(1).to(x.dtype)
        return x
