"""Network definition for the CIFAR-10 speed run.

Topology (a widened variant of the well-known "airbench" CIFAR speedrun net,
re-implemented from scratch here -- no pretrained weights are ever loaded):

    input (standardized NCHW float)
      -> frozen 2x2 ZCA whitening conv, 3 -> 24 ch, valid padding   (32 -> 31)
      -> GELU
      -> ConvGroup(24  -> W1)   conv3 / maxpool2 / BN / GELU / conv3 / BN / GELU
      -> ConvGroup(W1  -> W2)
      -> ConvGroup(W2  -> W3)
      -> MaxPool2d(3)   (3x3 -> 1x1)
      -> Linear(W3, 10, bias=False)
      -> Mul(1/9)                       # logit temperature, applied in forward

Design notes that matter and are easy to get wrong:

* The whitening conv weights are computed once from the *training* images
  (ZCA of all 2x2x3 patches) and frozen.  Its *bias* is trainable for the
  first few epochs only.  This is a data-dependent initialization, not a
  pretrained weight: it is derived solely from the CIFAR-10 train split.
* BatchNorm affine *weight* is frozen at 1.0; only the bias is learned.
* The 1/9 logit scale lives in `forward` as its own layer rather than being
  folded into the linear layer's initialization.  Folding it into init makes
  the effective learning rate on the head 9x wrong and costs ~0.5 pp.
* `Deploy` is the module that gets scripted and saved.  It (a) makes the
  forward pass robust to the three plausible input conventions and (b) folds
  6-way test-time augmentation into `forward`, so the verifier gets TTA for
  free just by calling `model(x)`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class Mul(nn.Module):
    """Constant scalar multiply -- used for the logit temperature."""

    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BatchNorm(nn.BatchNorm2d):
    """BatchNorm with a frozen unit scale and a learnable bias."""

    def __init__(self, num_features, momentum, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1.0 - momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = False
        self.bias.requires_grad = True


class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out, bn_momentum):
        super().__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, 3, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out, bn_momentum)
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding=1, bias=False)
        self.norm2 = BatchNorm(channels_out, bn_momentum)
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


def make_net(widths=(128, 384, 512), bn_momentum=0.6, scaling_factor=1 / 9,
             whiten_width=2 * 3 * 2 * 2):
    whiten = nn.Conv2d(3, whiten_width, kernel_size=2, padding=0, bias=True)
    whiten.weight.requires_grad = False
    net = nn.Sequential(
        whiten,
        nn.GELU(),
        ConvGroup(whiten_width, widths[0], bn_momentum),
        ConvGroup(widths[0], widths[1], bn_momentum),
        ConvGroup(widths[1], widths[2], bn_momentum),
        nn.MaxPool2d(3),
        Flatten(),
        nn.Linear(widths[2], 10, bias=False),
        Mul(scaling_factor),
    )
    net = net.half().cuda().to(memory_format=torch.channels_last)
    for mod in net.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.float()
    return net


# --------------------------------------------------------------------------
# ZCA whitening initialization (computed from the train split only)
# --------------------------------------------------------------------------
def _get_patches(x, patch_shape):
    c = x.shape[1]
    h, w = patch_shape
    return x.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).float()


def init_whitening_conv(layer, train_images, eps=5e-4):
    """Set `layer.weight` to the (scaled) ZCA eigenbasis of 2x2 image patches."""
    patches = _get_patches(train_images, layer.weight.shape[2:])
    n = patches.shape[0]
    flat = patches.view(n, -1)
    cov = (flat.T @ flat) / n
    eigenvalues, eigenvectors = torch.linalg.eigh(cov, UPLO="U")
    eigenvalues = eigenvalues.flip(0).view(-1, 1, 1, 1)
    eigenvectors = eigenvectors.T.reshape(-1, *patches.shape[1:]).flip(0)
    scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    layer.weight.data[:] = torch.cat((scaled, -scaled)).to(layer.weight.dtype)
    layer.bias.data.zero_()


# --------------------------------------------------------------------------
# Deployment wrapper: input-convention robustness + 6-way TTA
# --------------------------------------------------------------------------
class Deploy(nn.Module):
    """What we script and save.

    Input robustness
    ----------------
    The verifier hands us a float32 NCHW batch that has ALREADY been
    standardized with CIFAR_MEAN/CIFAR_STD (range ~[-2, 2.1]).  Re-normalizing
    such a batch would collapse accuracy, so `forward` detects the convention
    from the batch statistics -- branchlessly, so TorchScript stays happy and
    no host sync is forced:

        max > 20            -> [0, 255] uint8-ish  -> /255 then standardize
        min >= -0.02        -> [0, 1]              -> standardize
        otherwise           -> already standardized -> pass through

    Test-time augmentation
    ----------------------
    6 views: {identity, +1px diagonal shift, -1px diagonal shift} x {no flip,
    horizontal flip}, combined as 0.5 * mirror-only + 0.5 * mean(translates),
    matching the "tta_level=2" scheme.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR_STD).view(1, 3, 1, 1))

    def _standardize(self, x):
        mx = x.amax()
        x = torch.where(mx > 20.0, x / 255.0, x)
        mn = x.amin()
        return torch.where(mn >= -0.02, (x - self.mean) / self.std, x)

    def _basic(self, x):
        return self.net(x).float() + self.net(x.flip(-1)).float()

    def forward(self, x):
        x = self._standardize(x.float())
        x = x.half().contiguous(memory_format=torch.channels_last)
        logits = self._basic(x)
        padded = F.pad(x, (1, 1, 1, 1), "reflect")
        t1 = self._basic(padded[:, :, 0:32, 0:32].contiguous(memory_format=torch.channels_last))
        t2 = self._basic(padded[:, :, 2:34, 2:34].contiguous(memory_format=torch.channels_last))
        return 0.5 * logits + 0.25 * (t1 + t2)
