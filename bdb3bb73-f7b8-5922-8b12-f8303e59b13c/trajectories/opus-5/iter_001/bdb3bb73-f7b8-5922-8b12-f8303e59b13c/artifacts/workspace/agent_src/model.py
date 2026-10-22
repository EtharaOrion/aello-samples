"""CIFAR-10 speedrun network (airbench-style) + TorchScript-friendly inference wrapper.

The wrapper's `forward` is what the verifier calls. It expects inputs already
standardized with CIFAR-10 mean/std (the documented verifier convention) but
also tolerates raw [0,1] or [0,255] inputs so a mis-scaled evaluation pipeline
degrades gracefully instead of collapsing.
"""
import torch
import torch.nn as nn

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class Mul(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.4, eps=1e-12, weight=False, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias


class Conv(nn.Conv2d):
    def __init__(self, c_in, c_out, kernel_size=3):
        super().__init__(c_in, c_out, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        # Partial identity ("dirac") init: speeds up early optimization a lot.
        torch.nn.init.dirac_(w[:w.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, c_in, c_out, bn_momentum=0.4):
        super().__init__()
        self.conv1 = Conv(c_in, c_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(c_out, momentum=bn_momentum)
        self.conv2 = Conv(c_out, c_out)
        self.norm2 = BatchNorm(c_out, momentum=bn_momentum)
        self.activ = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x


def make_body(widths=(64, 256, 256), scaling_factor=1 / 9, bn_momentum=0.4, whiten_k=2):
    whiten_width = 2 * 3 * whiten_k ** 2
    return nn.Sequential(
        nn.Conv2d(3, whiten_width, whiten_k, padding=0, bias=True),
        nn.GELU(),
        ConvGroup(whiten_width, widths[0], bn_momentum),
        ConvGroup(widths[0], widths[1], bn_momentum),
        ConvGroup(widths[1], widths[2], bn_momentum),
        nn.MaxPool2d(3),
        nn.Flatten(),
        nn.Linear(widths[2], 10, bias=False),
        Mul(scaling_factor),
    )


class Net(nn.Module):
    """Inference-facing module. TTA (horizontal flip) is applied in eval mode only."""

    def __init__(self, widths=(64, 256, 256), scaling_factor=1 / 9, bn_momentum=0.4,
                 whiten_k=2, tta: bool = True, robust_input: bool = True):
        super().__init__()
        self.body = make_body(widths, scaling_factor, bn_momentum, whiten_k)
        self.tta = tta
        self.robust_input = robust_input
        self.register_buffer("mean", torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.robust_input and not self.training:
            # Inputs are expected pre-standardized; rescale only if they clearly are not.
            hi = float(torch.max(x))
            lo = float(torch.min(x))
            if hi > 3.0:                      # looks like [0, 255]
                x = (x / 255.0 - self.mean) / self.std
            elif lo >= -0.05 and hi <= 1.5:   # looks like [0, 1]
                x = (x - self.mean) / self.std
        x = x.to(memory_format=torch.channels_last)
        out = self.body(x)
        if self.tta and not self.training:
            out = out + self.body(torch.flip(x, [3]))
        return out


@torch.no_grad()
def init_whitening_conv(layer, train_images, eps=5e-4):
    """Set the first conv to (+/-) whitening filters of 2x2 image patches, then freeze it."""
    k = layer.weight.shape[-1]
    x = train_images.float()
    patches = x.unfold(2, k, 1).unfold(3, k, 1).transpose(1, 3).reshape(-1, x.shape[1], k, k)
    n = patches.shape[0]
    flat = patches.reshape(n, -1)
    cov = (flat.T @ flat) / n
    eigval, eigvec = torch.linalg.eigh(cov.double(), UPLO="U")
    eigval = eigval.flip(0).float().view(-1, 1, 1, 1)
    eigvec = eigvec.T.reshape(-1, x.shape[1], k, k).flip(0).float()
    scaled = eigvec / torch.sqrt(eigval.clamp_min(0) + eps)
    layer.weight.data[:] = torch.cat((scaled, -scaled))
    layer.bias.data.zero_()
    layer.weight.requires_grad = False
