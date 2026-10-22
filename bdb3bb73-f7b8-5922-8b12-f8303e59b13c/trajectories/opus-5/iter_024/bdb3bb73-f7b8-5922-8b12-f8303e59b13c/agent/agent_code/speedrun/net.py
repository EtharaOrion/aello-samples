"""Model definition for the CIFAR-10 speedrun.

Architecture (an "airbench"-style compact convnet, ~6.1M params):

    frozen 2x2 patch-whitening conv (24 filters, trainable bias)  -> GELU
    ConvGroup(24  -> 128)   conv3x3, maxpool2, BN, GELU, conv3x3, BN, GELU
    ConvGroup(128 -> 384)
    ConvGroup(384 -> 512)
    MaxPool2d(3) -> Flatten -> Linear(512, 10, bias=False) -> * scale

Design notes
------------
* The first layer is a *frozen* ZCA/patch-whitening transform estimated from
  2x2 patches of the **training split only**.  It is data-derived, not a
  pretrained weight: it is recomputed from scratch at the start of every run
  and costs ~40 ms.
* BatchNorm uses a trainable bias but a frozen unit weight - the following
  GELU is not scale-equivariant, so the scale degree of freedom is redundant
  with the next conv and only adds optimizer noise.
* Convs are bias-free (BN supplies the shift) and use the "dirac / identity"
  partial initialisation, which behaves like a residual shortcut at step 0.

The inference wrapper folds 6-way test-time augmentation into ``forward`` so
that the artefact the verifier loads is self-contained.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class Conv(nn.Conv2d):
    def __init__(self, c_in, c_out, kernel_size=3, padding="same", bias=False):
        super().__init__(c_in, c_out, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[: w.size(1)])


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1.0 - momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.zero_()
        self.weight.requires_grad = False


class ConvGroup(nn.Module):
    def __init__(self, c_in, c_out, bn_momentum):
        super().__init__()
        self.conv1 = Conv(c_in, c_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(c_out, bn_momentum)
        self.conv2 = Conv(c_out, c_out)
        self.norm2 = BatchNorm(c_out, bn_momentum)
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


class Mul(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def make_net(widths=(128, 384, 512), bn_momentum=0.6, scaling_factor=1 / 9,
             whiten_kernel=2):
    whiten_width = 2 * 3 * whiten_kernel ** 2
    return nn.Sequential(
        Conv(3, whiten_width, whiten_kernel, padding=0, bias=True),
        nn.GELU(),
        ConvGroup(whiten_width, widths[0], bn_momentum),
        ConvGroup(widths[0], widths[1], bn_momentum),
        ConvGroup(widths[1], widths[2], bn_momentum),
        nn.MaxPool2d(3),
        Flatten(),
        nn.Linear(widths[2], 10, bias=False),
        Mul(scaling_factor),
    )


# --------------------------------------------------------------------------
# frozen patch-whitening initialisation (train split only)
# --------------------------------------------------------------------------
def init_whitening_conv(layer, train_images, eps=5e-4):
    """Set ``layer.weight`` to the ZCA whitening basis of KxK image patches."""
    k = layer.weight.shape[-1]
    with torch.no_grad():
        patches = train_images.unfold(2, k, 1).unfold(3, k, 1)          # N,C,H',W',k,k
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, k, k).float()
        flat = patches.reshape(len(patches), -1)
        cov = (flat.T @ flat) / len(flat)
        eigvals, eigvecs = torch.linalg.eigh(cov.double(), UPLO="U")
        eigvals, eigvecs = eigvals.float(), eigvecs.float()
        scaled = eigvecs.T.reshape(-1, 3, k, k) / torch.sqrt(eigvals.view(-1, 1, 1, 1) + eps)
        layer.weight.data[:] = torch.cat((scaled, -scaled))
        layer.weight.requires_grad = False


# --------------------------------------------------------------------------
# inference wrapper: 6-way TTA + input-convention guard, TorchScript friendly
# --------------------------------------------------------------------------
class TTAModel(nn.Module):
    """Wraps the trained net with mirror + 1-px-translate TTA (6 passes).

    ``forward`` expects the verifier's convention: float32 NCHW images already
    standardised with ``Normalize(CIFAR_MEAN, CIFAR_STD)``.  As a defensive
    measure it also accepts raw [0,1] or [0,255] tensors - the three ranges are
    disjoint (standardised data has min < -0.5 and max < 2.2), so the guard can
    never misfire on correctly-preprocessed input.  It never *re*-normalises
    already-standardised input.
    """

    def __init__(self, net: nn.Module, tta: bool = True):
        super().__init__()
        self.net = net
        self.tta = tta
        self.register_buffer("mean", torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR_STD).view(1, 3, 1, 1))
        # dtype/device carrier so ``forward`` can match the net without
        # indexing into the Sequential (not TorchScript friendly).
        self.register_buffer("dtype_probe", torch.zeros(1))

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        hi = float(x.max())
        lo = float(x.min())
        if hi > 8.0:                      # [0, 255]
            x = x / 255.0
            x = (x - self.mean) / self.std
        elif lo >= -0.01 and hi <= 1.01:  # [0, 1]
            x = (x - self.mean) / self.std
        return x                          # already standardised -> untouched

    def _mirror(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.net(x.flip(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._standardize(x.float())
        x = x.to(memory_format=torch.channels_last).to(self.dtype_probe.dtype)
        logits = self._mirror(x)
        if not self.tta:
            return logits.float()
        padded = F.pad(x, (1, 1, 1, 1), "reflect")
        t1 = self._mirror(padded[:, :, 0:32, 0:32])
        t2 = self._mirror(padded[:, :, 2:34, 2:34])
        return (0.5 * logits + 0.25 * t1 + 0.25 * t2).float()
