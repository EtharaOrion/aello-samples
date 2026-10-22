"""
CIFAR-10 -> 94% top-1 from scratch on one H100.

Architecture / recipe lineage: a compact "airbench"-style convnet (Keller Jordan's
CIFAR-10 speedrun family) reimplemented here from scratch -- no pretrained weights,
no external data, CIFAR-10 train split only.

Key structural choices:
  * A frozen 2x2 ZCA-whitening convolution (eigenbasis of 2x2x3 patch covariance,
    estimated on the *train* split only) as layer 0, with a learnable bias.
  * Three conv/pool/BN/GELU groups at widths 128 / 384 / 512.
  * Bias-free linear head with a small init scale and a fixed logit scale.
  * BatchNorm weights frozen at 1 (only the biases are learned).

Everything lives on the GPU: the train split is cached as a uint8 tensor, augmentation
(reflect-pad random crop + horizontal flip) is done with two gathers and a `where`.

The saved checkpoint is a TorchScript module whose `forward` consumes *already
standardized* NCHW float32 (the verifier's ToTensor+Normalize convention) and performs
6-way test-time augmentation internally (2 mirrors x 3 translations).
"""

import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------

def load_split(split, device="cuda"):
    """Return (uint8 NCHW on device, int64 labels on device)."""
    x = np.load(f"/workspace/cache/{split}_x.npy")           # N,32,32,3 uint8
    y = np.load(f"/workspace/cache/{split}_y.npy")
    x = torch.from_numpy(x).to(device).permute(0, 3, 1, 2).contiguous()
    y = torch.from_numpy(y).to(device)
    return x, y


def standardize(x_uint8, dtype=torch.float32):
    """uint8 NCHW -> standardized float NCHW, matching ToTensor()+Normalize()."""
    mean = torch.tensor(CIFAR_MEAN, device=x_uint8.device).view(1, 3, 1, 1) * 255.0
    std = torch.tensor(CIFAR_STD, device=x_uint8.device).view(1, 3, 1, 1) * 255.0
    return ((x_uint8.float() - mean) / std).to(dtype)


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------

class BatchNorm(nn.BatchNorm2d):
    """BN with the scale frozen at 1; only the shift is learned (airbench convention)."""

    def __init__(self, num_features, momentum, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1.0 - momentum)
        self.weight.requires_grad = False


def conv(cin, cout):
    return nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False)


class ConvGroup(nn.Module):
    def __init__(self, cin, cout, bn_momentum):
        super().__init__()
        self.conv1 = conv(cin, cout)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(cout, bn_momentum)
        self.conv2 = conv(cout, cout)
        self.norm2 = BatchNorm(cout, bn_momentum)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.activ(self.norm1(self.pool(self.conv1(x))))
        x = self.activ(self.norm2(self.conv2(x)))
        return x


class Net(nn.Module):
    """Core classifier. Input: standardized NCHW float. Output: logits."""

    def __init__(self, widths=(128, 384, 512), bn_momentum=0.6, scaling_factor=1 / 9,
                 logit_scale=1.0):
        super().__init__()
        whiten_width = 2 * 3 * 2 * 2  # 24 = +/- the 12 eigenvectors of a 2x2x3 patch
        self.whiten = nn.Conv2d(3, whiten_width, kernel_size=2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.whiten_activ = nn.GELU()
        self.group1 = ConvGroup(whiten_width, widths[0], bn_momentum)
        self.group2 = ConvGroup(widths[0], widths[1], bn_momentum)
        self.group3 = ConvGroup(widths[1], widths[2], bn_momentum)
        self.pool = nn.MaxPool2d(3)
        self.head = nn.Linear(widths[2], 10, bias=False)
        # Logits are damped by `scaling_factor` as an output *layer* (not just an init
        # rescale): this keeps logits small for the whole run, which is what makes the
        # aggressive LR + label-smoothing 0.2 combination stable.
        self.out_scale = scaling_factor * logit_scale

    def forward(self, x):
        x = self.whiten_activ(self.whiten(x))
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.pool(x).flatten(1)
        return self.head(x) * self.out_scale


@torch.no_grad()
def init_whitening(layer, images_std, eps=5e-4, n_images=10000):
    """ZCA-whitening init for the 2x2 stem, estimated on train images only."""
    x = images_std[:n_images].float()
    patches = x.unfold(2, 2, 1).unfold(3, 2, 1)          # N,C,H',W',2,2
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3 * 2 * 2)
    cov = (patches.T @ patches) / patches.shape[0]
    evals, evecs = torch.linalg.eigh(cov.double())
    evals = evals.flip(0).float().view(-1, 1, 1, 1)
    evecs = evecs.T.reshape(12, 3, 2, 2).flip(0).float()
    scaled = evecs / torch.sqrt(evals + eps)
    layer.weight.data[:] = torch.cat((scaled, -scaled))
    layer.bias.data[:] = 0.0


class TTAModel(nn.Module):
    """
    Deployment wrapper -- this is what gets TorchScript-ed and graded.

    `forward` expects float32 NCHW already standardized with CIFAR_MEAN/CIFAR_STD
    (the verifier's ToTensor+Normalize output). It does NOT re-normalize.

    6-way TTA = {identity, horizontal mirror} x {center, shift(-1,-1), shift(+1,+1)},
    with the two shifted views produced from a 1-px reflect pad.
    """

    def __init__(self, net: Net):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        logits = self.net(x) + self.net(x.flip(-1))
        p = F.pad(x, (1, 1, 1, 1), mode="reflect")
        a = p[:, :, 0:32, 0:32]
        b = p[:, :, 2:34, 2:34]
        shifted = (self.net(a) + self.net(a.flip(-1))
                   + self.net(b) + self.net(b.flip(-1))) * 0.5
        return 0.5 * logits + 0.5 * shifted


# --------------------------------------------------------------------------------------
# augmentation
# --------------------------------------------------------------------------------------

class Augmenter:
    """Random 32x32 crop out of a 2-px reflect-padded image + random horizontal flip."""

    def __init__(self, x_std, pad=2, cutout=0):
        self.pad = pad
        self.cutout = cutout
        self.padded = F.pad(x_std, (pad,) * 4, mode="reflect")
        self.n, self.c, self.hp, self.wp = self.padded.shape
        self.ar = torch.arange(32, device=x_std.device)

    def __call__(self, idx):
        b = idx.shape[0]
        dev = idx.device
        x = self.padded[idx]
        span = 2 * self.pad + 1
        oy = torch.randint(0, span, (b, 1), device=dev)
        ox = torch.randint(0, span, (b, 1), device=dev)
        rows = (oy + self.ar).view(b, 1, 32, 1).expand(b, self.c, 32, self.wp)
        x = x.gather(2, rows)
        cols = (ox + self.ar).view(b, 1, 1, 32).expand(b, self.c, 32, 32)
        x = x.gather(3, cols)
        flip = torch.rand(b, 1, 1, 1, device=dev) < 0.5
        x = torch.where(flip, x.flip(3), x)
        if self.cutout:
            # Zero a random square. Inputs are standardized, so 0 == per-channel mean,
            # which is the usual cutout fill value.
            h = self.cutout // 2
            cy = torch.randint(0, 32, (b, 1, 1), device=dev)
            cx = torch.randint(0, 32, (b, 1, 1), device=dev)
            m = (((self.ar.view(1, 32, 1) - cy).abs() < h)
                 & ((self.ar.view(1, 1, 32) - cx).abs() < h))
            x = x.masked_fill(m.unsqueeze(1), 0.0)
        return x


# --------------------------------------------------------------------------------------
# optimiser plumbing
# --------------------------------------------------------------------------------------

def make_optimizer(net, cfg, batch_size):
    """
    airbench-style parameterisation: the loss is *summed* over the batch and the LR is
    expressed per-1024-examples, then divided by the Nesterov momentum gain so that the
    effective step size is invariant to the momentum coefficient.
    """
    m = cfg["momentum"]
    kilostep = 1024 * (1 + 1 / (1 - m))
    lr = cfg["lr"] / kilostep
    wd = cfg["weight_decay"] * batch_size / kilostep
    lr_bias = lr * cfg["bias_scaler"]

    norm_biases = [p for k, p in net.named_parameters()
                   if "norm" in k and p.requires_grad]
    whiten_bias = [net.whiten.bias]
    others = [p for k, p in net.named_parameters()
              if "norm" not in k and "whiten.bias" not in k and p.requires_grad]

    groups = [
        dict(params=norm_biases, lr=lr_bias, weight_decay=wd / lr_bias),
        dict(params=whiten_bias, lr=lr_bias, weight_decay=wd / lr_bias),
        dict(params=others, lr=lr, weight_decay=wd / lr),
    ]
    return torch.optim.SGD(groups, momentum=m, nesterov=True)


def lr_lambda_factory(total_steps, whiten_bias_steps, cfg):
    warmup = max(1, int(total_steps * cfg["warmup_frac"]))
    lo, hi, end = cfg["lr_start"], 1.0, cfg["lr_end"]

    def base(step):
        if step < warmup:
            f = step / warmup
            return lo * (1 - f) + hi * f
        f = (step - warmup) / max(1, total_steps - warmup)
        return hi * (1 - f) + end * f

    def whiten(step):
        return base(step) if step < whiten_bias_steps else 0.0

    return [base, whiten, base]


DEFAULT_CFG = dict(
    epochs=16,
    batch_size=512,
    lr=11.5,
    momentum=0.85,
    weight_decay=0.0153,
    bias_scaler=64.0,
    label_smoothing=0.2,
    whiten_bias_epochs=3,
    warmup_frac=0.23,
    lr_start=0.2,
    lr_end=0.07,
    widths=(128, 384, 512),
    bn_momentum=0.6,
    scaling_factor=1 / 9,
    logit_scale=1.0,
    pad=2,
    cutout=0,
)
