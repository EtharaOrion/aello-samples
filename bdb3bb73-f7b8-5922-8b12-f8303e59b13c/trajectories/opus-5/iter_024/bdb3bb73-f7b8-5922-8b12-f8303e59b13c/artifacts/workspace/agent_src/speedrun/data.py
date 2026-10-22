"""GPU-resident CIFAR-10 tensors + on-device augmentation.

The whole train split (50k x 3 x 36 x 36, reflect-padded once) lives in GPU
memory, so there is no DataLoader, no host-device copy and no CPU work in the
training loop.
"""
import torch
import torch.nn.functional as F

from .net import CIFAR_MEAN, CIFAR_STD

CACHE = "/workspace/cache"
PAD = 2  # translate augmentation radius, in pixels


def _to_normalised(u8_nhwc, device, dtype):
    x = u8_nhwc.to(device, non_blocking=True).permute(0, 3, 1, 2).to(dtype).div_(255.0)
    mean = torch.tensor(CIFAR_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def load_split(split, device="cuda", dtype=torch.float16):
    blob = torch.load(f"{CACHE}/{split}.pt", map_location="cpu")
    return _to_normalised(blob["images"], device, dtype), blob["labels"].to(device)


class TrainData:
    """Holds the (optionally sub-sampled) train split, padded and ready."""

    def __init__(self, images, labels, device="cuda"):
        self.labels = labels
        self.raw = images                                   # N,3,32,32 (unpadded)
        self.padded = F.pad(images, (PAD,) * 4, "reflect").contiguous()
        self.n = images.shape[0]
        self.device = device
        side = 32 + 2 * PAD
        self.side = side
        ar = torch.arange(32, device=device)
        self._rows = ar.view(1, 32, 1)
        self._cols_fwd = ar.view(1, 1, 32)
        self._cols_rev = ar.flip(0).view(1, 1, 32)

    def batch(self, idx, generator):
        """Return an augmented batch: random h-flip + random +-PAD translation.

        Both augmentations are expressed as one index tensor and applied with a
        single ``gather``, which keeps the augmentation cost at ~2% of step time.
        """
        b = idx.shape[0]
        shifts = torch.randint(0, 2 * PAD + 1, (2, b), device=self.device,
                               generator=generator)
        flip = torch.randint(0, 2, (b, 1, 1), device=self.device,
                             generator=generator, dtype=torch.bool)
        rows = self._rows + shifts[0].view(b, 1, 1)
        cols = torch.where(flip, self._cols_rev, self._cols_fwd) + shifts[1].view(b, 1, 1)
        index = (rows * self.side + cols).reshape(b, 1, 32 * 32).expand(b, 3, 32 * 32)
        src = self.padded[idx].reshape(b, 3, self.side * self.side)
        out = torch.gather(src, 2, index).view(b, 3, 32, 32)
        return out.contiguous(memory_format=torch.channels_last), self.labels[idx]
