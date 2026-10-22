"""Model definition for fast CIFAR-10 training (airbench-lineage convnet).

Architecture
------------
    frozen 2x2 ZCA whitening conv (3 -> 24)  + trainable bias + GELU
    ConvGroup(24  -> w1)   : conv3x3 -> maxpool2 -> BN -> GELU -> conv3x3 -> BN -> GELU
    ConvGroup(w1  -> w2)   : same
    ConvGroup(w2  -> w3)   : same
    maxpool(3) -> flatten -> Linear(w3, 10, bias=False) * scaling_factor

BatchNorm affine *scale* is frozen at 1 (only the shift is learned): with GELU
this keeps the effective per-layer gain fixed and lets a single global learning
rate work across depths.

Input convention (IMPORTANT)
----------------------------
``forward`` expects inputs that are ALREADY per-channel standardized with
CIFAR-10 statistics, i.e. exactly what
``transforms.Compose([ToTensor(), Normalize((0.4914,0.4822,0.4465),
(0.2470,0.2435,0.2616))])`` produces.  No normalization happens inside the
model, so feeding it standardized inputs twice is impossible.

Test-time augmentation
----------------------
When the module is in eval mode (``model.eval()``, which is what the verifier
calls), ``forward`` internally evaluates a 6-way TTA ensemble:
horizontal mirror x {identity, -1px diagonal shift, +1px diagonal shift},
weighted 0.6 / 0.4 between the centered and the shifted views.  This is baked
into ``forward`` so that a plain ``torch.jit.load(path).eval()`` + ``model(x)``
call gets the full ensemble with no caller-side changes.
"""

import torch
import torch.nn.functional as F
from torch import nn

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class ConvGroup(nn.Module):
    def __init__(self, c_in: int, c_out: int, bn_momentum: float):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = nn.BatchNorm2d(c_out, eps=1e-12, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(c_out, eps=1e-12, momentum=bn_momentum)
        for norm in (self.norm1, self.norm2):
            norm.weight.data.fill_(1.0)
            norm.bias.data.fill_(0.0)
            norm.weight.requires_grad = False  # frozen unit scale; only the shift is learned

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.norm1(self.pool(self.conv1(x))))
        x = F.gelu(self.norm2(self.conv2(x)))
        return x


class Net(nn.Module):
    def __init__(self, w1: int = 128, w2: int = 384, w3: int = 512,
                 bn_momentum: float = 0.6, scaling_factor: float = 1 / 9,
                 tta: bool = True):
        super().__init__()
        self.whiten = nn.Conv2d(3, 24, kernel_size=2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False  # set by init_whitening() and kept frozen
        self.group1 = ConvGroup(24, w1, bn_momentum)
        self.group2 = ConvGroup(w1, w2, bn_momentum)
        self.group3 = ConvGroup(w2, w3, bn_momentum)
        self.pool = nn.MaxPool2d(3)
        self.head = nn.Linear(w3, 10, bias=False)
        self.head.weight.data *= scaling_factor
        self.tta = tta

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.whiten(x))
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.pool(x)
        return self.head(x.flatten(1))

    def _mirror(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.features(x) + 0.5 * self.features(x.flip(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.whiten.weight.dtype).contiguous(memory_format=torch.channels_last)
        if self.training or not self.tta:
            return self.features(x)
        centered = self._mirror(x)
        padded = F.pad(x, (1, 1, 1, 1), "reflect")
        shifted = 0.5 * self._mirror(padded[:, :, 0:32, 0:32]) \
            + 0.5 * self._mirror(padded[:, :, 2:34, 2:34])
        return 0.6 * centered + 0.4 * shifted


@torch.no_grad()
def init_whitening(model: Net, images: torch.Tensor, eps: float = 5e-4) -> None:
    """Set the frozen first layer to a ZCA whitening transform of 2x2 patches.

    ``images`` must be standardized training images (NCHW float).  The layer is
    a fixed, data-dependent linear map -- it carries no label information and is
    computed from the training split only.
    """
    k = model.whiten.weight.shape[-1]
    patches = (images.unfold(2, k, 1).unfold(3, k, 1)
               .permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, k, k).float())
    flat = patches.reshape(len(patches), -1)
    flat = flat - flat.mean(0, keepdim=True)
    cov = (flat.T @ flat) / len(flat)
    eigval, eigvec = torch.linalg.eigh(cov.double())
    eigval = eigval.flip(0).clamp_min(0).float().view(-1, 1, 1, 1)
    eigvec = eigvec.T.reshape(-1, 3, k, k).flip(0).float()
    scaled = eigvec / (eigval + eps).sqrt()
    # +/- pairs so that the following GELU sees both signs of every projection.
    model.whiten.weight.data[:] = torch.cat((scaled, -scaled)).to(model.whiten.weight.dtype)
    model.whiten.bias.data.zero_()
