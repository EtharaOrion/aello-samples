"""CIFAR-10 speedrun network (airbench lineage) + TorchScript-safe inference.

Design notes
------------
* A frozen 2x2 "whitening" convolution is initialised from the eigenvectors of
  the training-set 2x2 patch covariance.  It gives the network a well
  conditioned first layer for free, which is what makes very short (<25 epoch)
  schedules reach 94%+.
* Each ``ConvGroup`` is conv -> maxpool -> BN -> GELU -> conv -> BN -> GELU.
  BatchNorm scales are frozen at 1.0 (only the shift is learned); this is an
  airbench trick that lets the run tolerate a very large learning rate.
* ``forward`` is what the verifier calls.  It (a) canonicalises the input scale
  so the module is correct for standardized / [0,1] / [0,255] inputs, and
  (b) folds horizontal-flip test-time augmentation into the model itself.
  Training uses ``logits`` directly on already-standardized, already-augmented
  batches, so none of this is in the hot loop.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class ConvGroup(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, bn_momentum: float):
        super().__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, 3, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = _frozen_scale_bn(channels_out, bn_momentum)
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding=1, bias=False)
        self.norm2 = _frozen_scale_bn(channels_out, bn_momentum)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm1(self.pool(self.conv1(x))))
        x = self.act(self.norm2(self.conv2(x)))
        return x


def _frozen_scale_bn(channels: int, momentum: float) -> nn.BatchNorm2d:
    """BatchNorm with a frozen unit scale and a learned shift."""
    bn = nn.BatchNorm2d(channels, eps=1e-12, momentum=momentum)
    bn.weight.data.fill_(1.0)
    bn.bias.data.zero_()
    bn.weight.requires_grad = False
    return bn


class CifarNet(nn.Module):
    def __init__(
        self,
        widths: List[int] = (64, 256, 256),
        bn_momentum: float = 0.6,
        logit_scale: float = 1 / 9,
        tta_level: int = 1,
        head_init: str = "zero",
    ):
        super().__init__()
        whiten_width = 2 * 3 * 2 * 2  # +/- eigenvectors of the 2x2x3 patch covariance
        self.whiten = nn.Conv2d(3, whiten_width, 2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.act = nn.GELU()
        self.group1 = ConvGroup(whiten_width, widths[0], bn_momentum)
        self.group2 = ConvGroup(widths[0], widths[1], bn_momentum)
        self.group3 = ConvGroup(widths[1], widths[2], bn_momentum)
        self.pool = nn.MaxPool2d(3)
        self.head = nn.Linear(widths[2], 10, bias=False)
        if head_init == "zero":
            self.head.weight.data.zero_()
        self.logit_scale = logit_scale
        self.tta_level = tta_level
        self.register_buffer("in_mean", torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1))
        self.register_buffer("in_std", torch.tensor(CIFAR_STD).view(1, 3, 1, 1))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass on an already-standardized batch (the training path)."""
        x = self.act(self.whiten(x))
        x = self.group3(self.group2(self.group1(x)))
        x = self.pool(x).flatten(1)
        return self.head(x) * self.logit_scale

    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        """Accept standardized (the verifier's convention), [0,1] or [0,255] input.

        The verifier hands us ToTensor+Normalize output, whose per-batch minimum
        is ~-2.0 and maximum ~+2.1, so the two cheap scalar tests below identify
        the convention unambiguously for any real batch of images.
        """
        lo = float(x.min().item())
        hi = float(x.max().item())
        if hi > 3.0:  # [0, 255] pixel values
            x = x / 255.0
        elif lo < -0.05:  # already standardized -> leave alone
            return x
        return (x - self.in_mean) / self.in_std

    def mirror_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x) + self.logits(torch.flip(x, [3]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference entry point: input-scale canonicalisation + test-time augmentation.

        tta_level 0 = plain, 1 = horizontal mirror, 2 = mirror + two 1-pixel
        diagonal translations (all folded into the model so the checkpoint is
        self-contained).
        """
        x = self.standardize(x.float())
        if self.tta_level == 0:
            return self.logits(x)
        out = self.mirror_logits(x)
        if self.tta_level >= 2:
            padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
            shifted = 0.5 * (
                self.mirror_logits(padded[:, :, 0:32, 0:32])
                + self.mirror_logits(padded[:, :, 2:34, 2:34])
            )
            out = 0.5 * out + 0.5 * shifted
        return out


@torch.no_grad()
def init_whitening(model: CifarNet, images: torch.Tensor, eps: float = 5e-4) -> None:
    """Set the frozen 2x2 conv to the (scaled) eigenvectors of the patch covariance.

    ``images`` is a standardized float32 NCHW batch drawn from the *training*
    split only.
    """
    patches = (
        images.unfold(2, 2, 1).unfold(3, 2, 1).transpose(1, 3).reshape(-1, 3, 2, 2).float()
    )
    cov = torch.cov(patches.reshape(len(patches), -1).T)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    scaled = eigenvectors.T.reshape(-1, 3, 2, 2).flip(0) / torch.sqrt(
        eigenvalues.view(-1, 1, 1, 1).flip(0) + eps
    )
    model.whiten.weight.data[:] = torch.cat((scaled, -scaled))
    model.whiten.bias.data.zero_()


def param_groups(model: CifarNet):
    """Split parameters into (weights, batchnorm shifts, whitening bias)."""
    weights, biases, whiten_bias = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name == "whiten.bias":
            whiten_bias.append(p)
        elif p.ndim == 1:
            biases.append(p)
        else:
            weights.append(p)
    return weights, biases, whiten_bias
