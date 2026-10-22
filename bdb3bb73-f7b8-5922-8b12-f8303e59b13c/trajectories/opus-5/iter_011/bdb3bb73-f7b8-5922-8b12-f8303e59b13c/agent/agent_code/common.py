"""Shared model / data / training utilities for the CIFAR-10 speedrun.

Design notes
------------
* Everything lives on the GPU: the train split is uploaded once as uint8 and
  augmented in-place with tensor ops, so there is no DataLoader / host-device
  traffic inside the timed loop.
* The network is a compact 3-group VGG-style convnet preceded by a *frozen*
  2x2 ZCA whitening convolution.  Whitening is estimated from the TRAIN split
  only (never the test split) and folded into a conv layer so it costs one
  cheap op at inference.
* Test-time augmentation (mirror x 3 horizontal shifts) is folded into
  ``forward`` behind ``self.training`` so that the scripted checkpoint the
  verifier loads gets TTA for free with no extra harness code.

Input convention
----------------
``forward`` expects float32 NCHW inputs that are ALREADY per-channel
standardised with CIFAR_MEAN / CIFAR_STD (this is exactly what the verifier
feeds: ``ToTensor()`` then ``Normalize(mean, std)``).  Training feeds the same
representation, so there is a single normalisation convention end to end and
no chance of double-normalising.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CACHE = "/workspace/data/cache"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_split(split, device="cuda"):
    d = torch.load(os.path.join(CACHE, f"{split}.pt"))
    return d["x"].to(device), d["y"].to(device)


def standardize(x_uint8):
    """uint8 NCHW -> standardised float32 NCHW (verifier's convention)."""
    mean = torch.tensor(CIFAR_MEAN, device=x_uint8.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=x_uint8.device).view(1, 3, 1, 1)
    return (x_uint8.float() / 255.0 - mean) / std


def reflect_pad_uint8(x_uint8, pad):
    """Pad once up-front so per-batch cropping is two cheap gathers."""
    if pad == 0:
        return x_uint8
    out = F.pad(x_uint8.float(), (pad,) * 4, mode="reflect")
    return out.round_().clamp_(0, 255).to(torch.uint8)


class GpuTrainData:
    """Holds the padded train split on-device and emits augmented batches."""

    def __init__(self, x_uint8, y, pad=4, device="cuda"):
        self.pad = pad
        self.padded = reflect_pad_uint8(x_uint8, pad).contiguous()
        self.y = y
        self.n = x_uint8.shape[0]
        self.device = device
        self.mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)
        self.ar = torch.arange(32, device=device)

    def batch(self, idx, translate, flip, cutout, gen):
        """Return (standardised augmented float batch, labels)."""
        b = idx.numel()
        raw = self.padded[idx]  # (b,3,32+2p,32+2p) uint8

        # --- random translation: two gathers on the pre-padded tensor -------
        p = self.pad
        if translate > 0:
            lo, hi = p - translate, p + translate + 1
            oy = torch.randint(lo, hi, (b,), device=self.device, generator=gen)
            ox = torch.randint(lo, hi, (b,), device=self.device, generator=gen)
        else:
            oy = torch.full((b,), p, device=self.device, dtype=torch.long)
            ox = oy
        rows = (oy[:, None] + self.ar[None, :])  # (b,32)
        cols = (ox[:, None] + self.ar[None, :])
        w = raw.shape[-1]
        x = raw.gather(2, rows[:, None, :, None].expand(b, 3, 32, w))
        x = x.gather(3, cols[:, None, None, :].expand(b, 3, 32, 32))

        x = (x.float() / 255.0 - self.mean) / self.std

        # --- random horizontal mirror --------------------------------------
        if flip:
            m = torch.rand(b, 1, 1, 1, device=self.device, generator=gen) < 0.5
            x = torch.where(m, x.flip(3), x)

        # --- cutout (holes are set to 0 == per-channel dataset mean) --------
        if cutout > 0:
            cy = torch.randint(0, 32, (b, 1), device=self.device, generator=gen)
            cx = torch.randint(0, 32, (b, 1), device=self.device, generator=gen)
            h = self.ar[None, :]
            my = (h >= cy - cutout // 2) & (h < cy + (cutout + 1) // 2)
            mx = (h >= cx - cutout // 2) & (h < cx + (cutout + 1) // 2)
            hole = my[:, None, :, None] & mx[:, None, None, :]
            x = x.masked_fill(hole, 0.0)

        return x.contiguous(memory_format=torch.channels_last), self.y[idx]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def zca_patch_filters(x_std, eps=5e-4, max_images=10000):
    """ZCA-whitening filters for 2x2x3 patches, estimated on TRAIN data only."""
    x = x_std[:max_images]
    patches = F.unfold(x, 2)                       # (N, 12, L)
    patches = patches.permute(0, 2, 1).reshape(-1, 12).double()
    patches = patches - patches.mean(0, keepdim=True)
    cov = (patches.T @ patches) / patches.shape[0]
    evals, evecs = torch.linalg.eigh(cov)
    # Row i of evecs.T is eigenvector i and must be scaled by 1/sqrt(eval_i),
    # so the divisor needs an explicit trailing axis -- plain broadcasting
    # would divide *column* j by sqrt(eval_j), which is a different (wrong)
    # matrix and leaves the whitened patches badly conditioned.
    W = (evecs.T / torch.sqrt(evals + eps)[:, None]).float()  # (12,12)
    W = W.reshape(12, 3, 2, 2)
    # Duplicate with opposite sign so the following GELU sees both half-planes.
    return torch.cat([W, -W], dim=0)                 # (24,3,2,2)


def make_conv(cin, cout, dirac=False):
    c = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
    if dirac:
        # Partial identity init (airbench trick): the first `cin` output
        # channels start as a pass-through, so a deep plain convnet behaves
        # like a shallow one at step 0 and trains usefully much sooner.
        with torch.no_grad():
            nn.init.dirac_(c.weight[:cin])
    return c


def make_bn(c, momentum=0.6, freeze_scale=True):
    bn = nn.BatchNorm2d(c, eps=1e-12, momentum=momentum)
    if freeze_scale:
        # The following GELU is not scale-invariant, but the *next* conv can
        # absorb any per-channel gain, so a learnable BN scale is redundant
        # and only adds gradient noise on a short schedule.
        bn.weight.requires_grad_(False)
    return bn


class ConvGroup(nn.Module):
    def __init__(self, cin, cout, dirac=False, bn_momentum=0.6, bn_freeze_scale=True):
        super().__init__()
        self.conv1 = make_conv(cin, cout, dirac)
        self.pool = nn.MaxPool2d(2)
        self.bn1 = make_bn(cout, bn_momentum, bn_freeze_scale)
        self.conv2 = make_conv(cout, cout, dirac)
        self.bn2 = make_bn(cout, bn_momentum, bn_freeze_scale)

    def forward(self, x):
        x = F.gelu(self.bn1(self.pool(self.conv1(x))))
        x = F.gelu(self.bn2(self.conv2(x)))
        return x


class Net(nn.Module):
    """Whiten -> 3 conv groups -> global max-pool -> bias-free linear head.

    In eval mode ``forward`` averages the logits over a 6-way test-time
    augmentation (mirror x {-1,0,+1} horizontal pixel shift).
    """

    def __init__(self, widths=(128, 384, 512), logit_scale=1 / 9.0, tta=True,
                 dirac=False, bn_momentum=0.6, bn_freeze_scale=True):
        super().__init__()
        self.whiten = nn.Conv2d(3, 24, 2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        kw = dict(dirac=dirac, bn_momentum=bn_momentum, bn_freeze_scale=bn_freeze_scale)
        self.group1 = ConvGroup(24, widths[0], **kw)
        self.group2 = ConvGroup(widths[0], widths[1], **kw)
        self.group3 = ConvGroup(widths[1], widths[2], **kw)
        self.head = nn.Linear(widths[2], 10, bias=False)
        self.head.weight.data.zero_()
        self.logit_scale = logit_scale
        self.tta = tta

    def init_whiten(self, x_std):
        with torch.no_grad():
            self.whiten.weight.copy_(zca_patch_filters(x_std).to(self.whiten.weight.device))
            self.whiten.bias.zero_()

    def core(self, x):
        x = F.gelu(self.whiten(x))
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = F.max_pool2d(x, x.shape[-1]).flatten(1)
        return self.head(x) * self.logit_scale

    def forward(self, x):
        x = x.to(self.head.weight.dtype)
        if self.training or not self.tta:
            return self.core(x)
        # 6-way TTA: {identity, mirror} x {shift -1, 0, +1} along width.
        xp = F.pad(x, (1, 1, 0, 0), mode="replicate")
        xm = x.flip(3)
        xmp = xp.flip(3)
        out = self.core(x) + self.core(xm)
        out = out + self.core(xp[:, :, :, 0:32]) + self.core(xmp[:, :, :, 0:32])
        out = out + self.core(xp[:, :, :, 2:34]) + self.core(xmp[:, :, :, 2:34])
        return (out / 6.0).float()


# --------------------------------------------------------------------------- #
# Optimisation
# --------------------------------------------------------------------------- #
def make_optimizer(model, lr, wd, momentum=0.85):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    return torch.optim.SGD(
        [{"params": decay, "weight_decay": wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, momentum=momentum, nesterov=True,
    )


def lr_at(step, total, peak, warmup_frac=0.2, final_frac=0.02):
    """Linear warm-up then linear decay -- one triangular cycle."""
    w = max(1, int(total * warmup_frac))
    if step < w:
        return peak * (step + 1) / w
    t = (step - w) / max(1, total - w)
    return peak * ((1 - t) + t * final_frac)


@torch.no_grad()
def evaluate(model, x_uint8, y, bs=1000):
    """Accuracy on an already-cached uint8 split (standardised on the fly)."""
    model.eval()
    correct = 0
    for i in range(0, x_uint8.shape[0], bs):
        xb = standardize(x_uint8[i:i + bs]).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", torch.bfloat16):
            out = model(xb)
        correct += (out.float().argmax(1) == y[i:i + bs]).sum().item()
    model.train()
    return correct / x_uint8.shape[0]
