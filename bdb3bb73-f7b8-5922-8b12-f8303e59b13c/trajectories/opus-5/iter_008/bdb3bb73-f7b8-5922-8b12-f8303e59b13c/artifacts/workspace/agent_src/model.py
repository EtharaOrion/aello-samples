"""CIFAR-10 speedrun net (airbench-lineage), written from scratch for this task.

Design notes
------------
* Input contract: the verifier hands `forward` a float32 NCHW batch that has ALREADY
  been through `ToTensor()` + `Normalize(CIFAR_MEAN, CIFAR_STD)`.  The network is
  trained on exactly that representation, so `forward` does NOT re-normalise.  A cheap
  range probe additionally makes it tolerant of [0,1] and [0,255] inputs, so a harness
  that skips the normalisation cannot silently collapse accuracy.
* Test-time augmentation (mirror + 1px translate) is folded into `forward` and gated on
  `self.training`, so a `torch.jit.script(...).save(...)` checkpoint carries it.
* No pretrained weights anywhere: the only non-random initialisation is the frozen 2x2
  whitening filter bank, computed from the CIFAR-10 *train* split's own patch covariance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class ConvGroup(nn.Module):
    """conv -> maxpool -> BN -> GELU -> conv -> BN -> GELU."""

    def __init__(self, c_in: int, c_out: int, bn_momentum: float):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = nn.BatchNorm2d(c_out, eps=1e-12, momentum=1 - bn_momentum)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(c_out, eps=1e-12, momentum=1 - bn_momentum)
        # BN scale is frozen at 1: the following conv can absorb any per-channel gain,
        # so the extra parameter only adds optimisation noise on a <30 epoch budget.
        for n in (self.norm1, self.norm2):
            n.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.norm1(self.pool(self.conv1(x))))
        x = F.gelu(self.norm2(self.conv2(x)))
        return x


class Net(nn.Module):
    def __init__(self, w1: int = 128, w2: int = 384, w3: int = 512,
                 logit_scale: float = 1.0 / 9.0, bn_momentum: float = 0.6,
                 tta: bool = True):
        super().__init__()
        self.whiten = nn.Conv2d(3, 24, 2, padding=0, bias=True)
        self.whiten.weight.requires_grad_(False)
        self.group1 = ConvGroup(24, w1, bn_momentum)
        self.group2 = ConvGroup(w1, w2, bn_momentum)
        self.group3 = ConvGroup(w2, w3, bn_momentum)
        self.final_pool = nn.MaxPool2d(3)
        self.head = nn.Linear(w3, 10, bias=False)
        self.head.weight.data *= logit_scale
        self.tta = tta
        self.register_buffer("mean", torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR_STD).view(1, 3, 1, 1))

    # --- input canonicalisation -------------------------------------------------
    def canon(self, x: torch.Tensor) -> torch.Tensor:
        """Return a per-channel standardised batch whatever scale we were handed.

        Standardised CIFAR batches live in about [-2.0, 2.1]; [0,1] and [0,255] batches
        are unambiguously separated from that by their min/max, so this never fires on
        the verifier's own (already standardised) input.
        """
        hi = float(x.max().item())
        lo = float(x.min().item())
        if hi > 4.0:
            return (x / 255.0 - self.mean) / self.std
        if lo >= -0.02 and hi <= 1.02:
            return (x - self.mean) / self.std
        return x

    # --- backbone ---------------------------------------------------------------
    def core(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.whiten(x))
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.final_pool(x)
        return self.head(x.flatten(1))

    def mirror(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.core(x) + 0.5 * self.core(x.flip(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.core(x)
        x = self.canon(x)
        if not self.tta:
            return self.core(x)
        logits = self.mirror(x)
        p = F.pad(x, (1, 1, 1, 1), mode="reflect")
        shifted = 0.5 * self.mirror(p[:, :, 0:32, 0:32]) + 0.5 * self.mirror(p[:, :, 2:34, 2:34])
        return 0.6 * logits + 0.4 * shifted


@torch.no_grad()
def init_whitening(conv: nn.Conv2d, images: torch.Tensor, eps: float = 5e-4) -> None:
    """Set the frozen 2x2 filter bank to the ZCA basis of the train split's patches.

    `images` is a standardised float NCHW tensor drawn from the TRAIN split only.
    """
    k = conv.weight.shape[-1]
    patches = images.unfold(2, k, 1).unfold(3, k, 1)          # N,C,H',W',k,k
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3 * k * k).float()
    cov = (patches.T @ patches) / patches.shape[0]
    evals, evecs = torch.linalg.eigh(cov.double())
    evals = evals.flip(0).clamp_min(0).float()
    evecs = evecs.T.flip(0).reshape(3 * k * k, 3, k, k).float()
    scaled = evecs / torch.sqrt(evals.view(-1, 1, 1, 1) + eps)
    conv.weight.data[:] = torch.cat([scaled, -scaled])
    conv.bias.data.zero_()
