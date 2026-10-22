"""Shared components for the CIFAR-10 speedrun.

Design summary
--------------
* Architecture: a whitening-convnet in the style of Keller Jordan's CIFAR-10
  airbench (https://github.com/KellerJordan/cifar10-airbench, MIT).  Trained
  from scratch -- no pretrained weights, no external data.
    - layer 0 is a FROZEN 2x2 ZCA-whitening convolution whose filters are the
      eigenvectors of the 2x2 patch covariance of the *training* split
      (+/- pairs -> 24 output channels), scaled by 1/sqrt(eig + eps).
      Only its bias is learned, and only for the first few epochs.
    - three conv groups (conv-pool-BN-GELU-conv-BN-GELU), widths 128/384/512.
    - BatchNorm gammas are frozen at 1.0 (only the betas train), which is an
      airbench trick that removes a redundant degree of freedom.
    - convs use a partial-Dirac (identity) init on the first in_channels
      filters.
    - bias-free linear head followed by a constant logit scale (`Mul`).
* Precision: fp16 weights/activations with fp32 BatchNorm statistics, plus
  channels_last.  The loss uses reduction='sum', which scales gradients by the
  batch size and is what keeps fp16 gradients out of the underflow region
  without a GradScaler (same trick airbench uses).
* Data: the whole train split lives on the GPU as one reflect-padded fp16
  tensor; augmentation (random 32x32 crop out of 36x36 + random mirror) is two
  `gather`s and a `where`, so there is no host<->device traffic and no
  DataLoader in the training loop.
* Test-time augmentation (mirror x {identity, -1px, +1px} translate, 6 views)
  is baked into `forward`, so the *scripted checkpoint itself* performs TTA.

Input convention (matches the verifier exactly)
-----------------------------------------------
`forward` expects float32 NCHW batches that are ALREADY per-channel
standardized with CIFAR_MEAN/CIFAR_STD.  We therefore standardize the training
tensors with the identical constants and never normalize inside `forward`.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CACHE = "/workspace/cache"

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #
def load_split(split, device="cuda"):
    """Return (x_uint8_NCHW, y) on `device` in torchvision ImageFolder order."""
    x = np.load(os.path.join(CACHE, f"{split}_x.npy"))          # N,32,32,3 uint8
    y = np.load(os.path.join(CACHE, f"{split}_y.npy"))
    xt = torch.from_numpy(x).to(device).permute(0, 3, 1, 2).contiguous()
    return xt, torch.from_numpy(y).to(device)


def standardize(x_uint8, dtype=torch.float32):
    """uint8 NCHW -> standardized float, bit-identical to ToTensor+Normalize."""
    mean = torch.tensor(CIFAR_MEAN, device=x_uint8.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=x_uint8.device).view(1, 3, 1, 1)
    x = x_uint8.float().div_(255.0)
    x = (x - mean) / std
    return x.to(dtype)


# --------------------------------------------------------------------------- #
#  Model
# --------------------------------------------------------------------------- #
class Conv(nn.Conv2d):
    def __init__(self, cin, cout, k=3, padding=1, bias=False):
        super().__init__(cin, cout, k, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[: w.size(1)])


class BatchNorm(nn.BatchNorm2d):
    """gamma frozen at 1, beta learned; `momentum` is the retention factor."""

    def __init__(self, n, momentum, eps=1e-12):
        super().__init__(n, eps=eps, momentum=1.0 - momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = False


class ConvGroup(nn.Module):
    def __init__(self, cin, cout, bn_mom):
        super().__init__()
        self.conv1 = Conv(cin, cout)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(cout, bn_mom)
        self.conv2 = Conv(cout, cout)
        self.norm2 = BatchNorm(cout, bn_mom)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x


class Net(nn.Module):
    """Whitening convnet with optional 6-way TTA folded into `forward`."""

    def __init__(self, widths=(128, 384, 512), bn_mom=0.6, scale=1.0 / 9, tta=True):
        super().__init__()
        ww = 2 * 3 * 2 * 2  # +/- eigenvector pairs of a 3x2x2 patch
        self.whiten = nn.Conv2d(3, ww, 2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.activ = nn.GELU()
        self.group1 = ConvGroup(ww, widths[0], bn_mom)
        self.group2 = ConvGroup(widths[0], widths[1], bn_mom)
        self.group3 = ConvGroup(widths[1], widths[2], bn_mom)
        self.pool = nn.MaxPool2d(3)
        self.head = nn.Linear(widths[2], 10, bias=False)
        self.scale = float(scale)
        self.tta = bool(tta)

    def logits(self, x):
        x = self.activ(self.whiten(x))
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x) * self.scale

    def forward(self, x):
        # Inputs are already per-channel standardized -- do NOT normalize here.
        x = x.to(dtype=self.head.weight.dtype)
        x = x.contiguous(memory_format=torch.channels_last)
        if not self.tta:
            return self.logits(x)
        h = x.size(2)
        w = x.size(3)
        direct = self.logits(x) + self.logits(torch.flip(x, [-1]))
        p = F.pad(x, [1, 1, 1, 1], mode="reflect")
        a = p[:, :, 0:h, 0:w]
        b = p[:, :, 2 : 2 + h, 2 : 2 + w]
        shift = (
            self.logits(a)
            + self.logits(torch.flip(a, [-1]))
            + self.logits(b)
            + self.logits(torch.flip(b, [-1]))
        )
        return (0.25 * direct + 0.125 * shift).float()


# --------------------------------------------------------------------------- #
#  ZCA whitening initialisation (train split only)
# --------------------------------------------------------------------------- #
def init_whitening(layer, train_images, eps=5e-4):
    """train_images: standardized fp32 NCHW tensor from the TRAIN split only."""
    k = layer.weight.shape[-1]
    patches = (
        train_images.unfold(2, k, 1)
        .unfold(3, k, 1)
        .transpose(1, 3)
        .reshape(-1, train_images.size(1), k, k)
        .float()
    )
    n = patches.shape[0]
    flat = patches.reshape(n, -1)
    cov = (flat.T @ flat) / n
    eigvals, eigvecs = torch.linalg.eigh(cov, UPLO="U")
    eigvals = eigvals.flip(0).view(-1, 1, 1, 1)
    eigvecs = eigvecs.T.reshape(-1, train_images.size(1), k, k).flip(0)
    scaled = eigvecs / torch.sqrt(eigvals + eps)
    layer.weight.data[:] = torch.cat((scaled, -scaled))
    layer.bias.data.zero_()


# --------------------------------------------------------------------------- #
#  GPU augmentation
# --------------------------------------------------------------------------- #
def rand_crop_flip(padded, out, gen):
    """padded: (B,3,P,P). Random out x out crop + random horizontal mirror."""
    b = padded.size(0)
    p = padded.size(-1)
    r = p - out
    dev = padded.device
    ar = torch.arange(out, device=dev)
    oy = torch.randint(0, r + 1, (b,), device=dev, generator=gen)
    ox = torch.randint(0, r + 1, (b,), device=dev, generator=gen)
    ih = (oy.view(b, 1) + ar.view(1, out)).view(b, 1, out, 1).expand(b, 3, out, p)
    x = padded.gather(2, ih)
    iw = (ox.view(b, 1) + ar.view(1, out)).view(b, 1, 1, out).expand(b, 3, out, out)
    x = x.gather(3, iw)
    flip = (torch.rand(b, 1, 1, 1, device=dev, generator=gen) < 0.5)
    return torch.where(flip, x.flip(-1), x)


# --------------------------------------------------------------------------- #
#  Fast evaluation used for *our own* measurements (fp16, TTA optional)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, x_std, y, batch=1000, tta=True):
    was_training = model.training
    model.eval()
    prev = model.tta
    model.tta = tta
    correct = 0
    for i in range(0, x_std.size(0), batch):
        out = model(x_std[i : i + batch])
        correct += (out.argmax(1) == y[i : i + batch]).sum().item()
    model.tta = prev
    if was_training:
        model.train()
    return correct / x_std.size(0)
