"""CIFAR-10 network: frozen ZCA-whitening stem + 3 conv groups + bias-free head.

IMPORTANT input convention: `forward` expects inputs that are ALREADY
per-channel standardized with the CIFAR-10 mean/std, exactly as the verifier
produces them (ToTensor -> Normalize). No normalization happens inside the
module.

Test-time augmentation (mirror + 1-px horizontal translate, 6 views) is folded
into `forward` and gated on `self.training`, so a `torch.jit.script`-ed export
does plain forward while training and TTA once the loader calls `.eval()`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvGroup(nn.Module):
    def __init__(self, cin: int, cout: int, bn_mom: float):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = nn.BatchNorm2d(cout, eps=1e-5, momentum=bn_mom)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(cout, eps=1e-5, momentum=bn_mom)

    def forward(self, x: Tensor) -> Tensor:
        x = F.gelu(self.norm1(self.pool(self.conv1(x))))
        x = F.gelu(self.norm2(self.conv2(x)))
        return x


class CifarNet(nn.Module):
    def __init__(self, w1: int = 128, w2: int = 384, w3: int = 512,
                 bn_mom: float = 0.15, logit_scale: float = 1.0,
                 tta: bool = True):
        super().__init__()
        self.whiten = nn.Conv2d(3, 24, 2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.whiten.bias.requires_grad = False
        self.group1 = ConvGroup(24, w1, bn_mom)
        self.group2 = ConvGroup(w1, w2, bn_mom)
        self.group3 = ConvGroup(w2, w3, bn_mom)
        self.head = nn.Linear(w3, 10, bias=False)
        self.logit_scale = logit_scale
        self.tta = tta

    def _logits(self, x: Tensor) -> Tensor:
        x = F.gelu(self.whiten(x))
        x = self.group3(self.group2(self.group1(x)))
        x = F.max_pool2d(x, x.shape[-1]).flatten(1)
        return self.head(x) * self.logit_scale

    def forward(self, x: Tensor) -> Tensor:
        if self.training or not self.tta:
            return self._logits(x)
        # 6-way TTA: {identity, shift-left-1px, shift-right-1px} x {id, mirror}
        # Accumulated one view at a time to keep peak activation memory low.
        shl = torch.cat([x[:, :, :, 1:], x[:, :, :, -1:]], dim=3)
        shr = torch.cat([x[:, :, :, :1], x[:, :, :, :-1]], dim=3)
        out = self._logits(x) + self._logits(torch.flip(x, [3]))
        out = out + self._logits(shl) + self._logits(torch.flip(shl, [3]))
        out = out + self._logits(shr) + self._logits(torch.flip(shr, [3]))
        return out / 6.0


@torch.no_grad()
def init_whitening(model: CifarNet, images: Tensor, eps: float = 5e-4) -> None:
    """Freeze the stem to a ZCA/PCA whitening transform of 2x2 train patches.

    `images` must be standardized train-split images only -- the test split is
    never touched here.
    """
    patches = (images.unfold(2, 2, 1).unfold(3, 2, 1)
               .permute(0, 2, 3, 1, 4, 5).reshape(-1, 12).float())
    patches = patches - patches.mean(0, keepdim=True)
    cov = (patches.T @ patches) / (patches.shape[0] - 1)
    eigval, eigvec = torch.linalg.eigh(cov.double())
    # descending order, scaled by 1/sqrt(lambda) -> unit-variance responses
    w = (eigvec.T.flip(0) / (eigval.flip(0) + eps).sqrt().unsqueeze(1)).float()
    w = w.reshape(12, 3, 2, 2)
    model.whiten.weight.copy_(torch.cat([w, -w]))   # +/- pairs: GELU is asymmetric
    model.whiten.bias.zero_()
