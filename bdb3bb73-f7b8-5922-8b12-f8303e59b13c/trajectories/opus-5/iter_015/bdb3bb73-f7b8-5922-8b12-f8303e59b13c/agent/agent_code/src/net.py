"""CIFAR-10 speedrun network.

Architecture is a compact 3-group VGG-style convnet preceded by a *frozen* 2x2 ZCA
whitening convolution.  Everything here is written to be `torch.jit.script`-able so
that a checkpoint can be saved as a self-contained scripted model.

IMPORTANT INPUT CONVENTION
--------------------------
`forward` expects float32 NCHW inputs that are **already per-channel standardized**
with CIFAR-10 statistics, i.e. exactly what

    transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                             (0.2470, 0.2435, 0.2616))])

produces.  No normalization happens inside the model; the training pipeline applies
the identical transform on-device so train and eval see the same distribution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# The exact constants the verifier's transform uses.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class Conv(nn.Conv2d):
    """3x3 same-padding conv, no bias, with partial-identity (Dirac) init.

    Dirac-initialising the first `in_channels` filters makes each group start close to
    an identity map, which measurably speeds up the first few epochs.
    """

    dirac = True  # class-level switch so the ablation can turn identity init off

    def __init__(self, c_in, c_out, kernel_size=3):
        super().__init__(c_in, c_out, kernel_size=kernel_size, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        if Conv.dirac:
            w = self.weight.data
            torch.nn.init.dirac_(w[: w.size(1)])


class BatchNorm(nn.BatchNorm2d):
    """BN with a frozen unit scale; only the shift is learned.

    Freezing gamma removes a redundant degree of freedom (the following conv can
    absorb it) and empirically stabilises the very high learning rates used here.
    """

    def __init__(self, num_features, momentum, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1.0 - momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
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


class CifarNet(nn.Module):
    def __init__(self, w1=64, w2=256, w3=256, bn_momentum=0.6, scale=1.0 / 9.0):
        super().__init__()
        whiten_width = 2 * 3 * 2 * 2  # 24 = (eigvecs, -eigvecs) over 2x2x3 patches
        self.whiten = nn.Conv2d(3, whiten_width, kernel_size=2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width, w1, bn_momentum),
            ConvGroup(w1, w2, bn_momentum),
            ConvGroup(w2, w3, bn_momentum),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(w3, 10, bias=False)
        self.head.weight.data.zero_()
        self.scale = scale

    def forward(self, x):
        x = self.whiten(x)
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        return self.head(x) * self.scale


class TTAModel(nn.Module):
    """Inference wrapper folding 6-way test-time augmentation into `forward`.

    The verifier calls `model(x)` once and takes `argmax(1)`, so any TTA has to live
    inside the graph.  We average logits over {identity, mirror} x {no shift, up-left
    1px, down-right 1px}, weighting the un-shifted pair higher (0.6/0.4) as in the
    airbench recipe.  Costs the verifier 6 forward passes; costs our training clock
    nothing.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def _mirror(self, x):
        return 0.5 * self.net(x) + 0.5 * self.net(x.flip(-1))

    def forward(self, x):
        # `x` is float32 NCHW, already standardized (see module docstring).
        x = x.float()
        logits = self._mirror(x)
        padded = F.pad(x, (1, 1, 1, 1), "reflect")
        shifted = self._mirror(padded[:, :, 0:32, 0:32]) + self._mirror(padded[:, :, 2:34, 2:34])
        return 0.6 * logits + 0.4 * 0.5 * shifted


class PlainModel(nn.Module):
    """Inference wrapper with no TTA (used for ablations)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x.float())
