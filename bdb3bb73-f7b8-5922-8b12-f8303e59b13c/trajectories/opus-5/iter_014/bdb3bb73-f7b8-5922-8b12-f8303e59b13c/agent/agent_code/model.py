"""Model definition for the CIFAR-10 speedrun.

Input convention (matches the grader exactly): forward() receives a float32
NCHW batch that has ALREADY been standardized with
    mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)
so no normalization happens inside the network. The first layer is a frozen
2x2 ZCA whitening convolution whose statistics are estimated from the *training*
split only.

Test-time augmentation (mirror x 3 horizontal shifts) is folded into forward()
so the grader gets it for free from `argmax(model(x))`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGroup(nn.Module):
    def __init__(self, cin: int, cout: int, bn_momentum: float):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = nn.BatchNorm2d(cout, momentum=bn_momentum, eps=1e-5)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(cout, momentum=bn_momentum, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.gelu(x)
        return x


class Net(nn.Module):
    """widths: (w1, w2, w3). tta: 1 = none, 2 = mirror, 6 = mirror x 3 shifts."""

    def __init__(self, widths=(128, 384, 512), bn_momentum: float = 0.6,
                 logit_scale: float = 1 / 9, tta: int = 6):
        super().__init__()
        self.tta = tta
        self.logit_scale = logit_scale
        self.whiten = nn.Conv2d(3, 24, 2, padding=0, bias=False)
        self.whiten.weight.requires_grad = False
        w1, w2, w3 = widths
        self.group1 = ConvGroup(24, w1, bn_momentum)
        self.group2 = ConvGroup(w1, w2, bn_momentum)
        self.group3 = ConvGroup(w2, w3, bn_momentum)
        self.head_pool = nn.MaxPool2d(3)
        self.fc = nn.Linear(w3, 10, bias=False)
        self.fc.weight.data.zero_()

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.whiten(x)
        x = F.gelu(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.head_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x) * self.logit_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is already standardized; never re-normalize here.
        if self.tta == 1:
            return self.features(x)
        if self.tta == 2:
            return self.features(x) + self.features(torch.flip(x, [3]))
        # 6-way: {identity, mirror} x {shift -1, 0, +1} horizontally.
        out = self.features(x) + self.features(torch.flip(x, [3]))
        xp = F.pad(x, [1, 1, 0, 0], mode="replicate")
        left = xp[:, :, :, 0:32]
        right = xp[:, :, :, 2:34]
        out = out + 0.5 * (self.features(left) + self.features(torch.flip(left, [3])))
        out = out + 0.5 * (self.features(right) + self.features(torch.flip(right, [3])))
        return out


@torch.no_grad()
def init_whitening(conv: nn.Conv2d, train_images: torch.Tensor, eps: float = 5e-4):
    """ZCA-whitening init for a 2x2 conv, estimated from standardized train images.

    train_images: (N, 3, 32, 32) float32, already standardized.
    The 24 output filters are +/- pairs of the 12 scaled eigenvectors, so that a
    GELU afterwards keeps both signs of every whitened coordinate.
    """
    kh, kw = conv.weight.shape[2], conv.weight.shape[3]
    patches = (train_images.unfold(2, kh, 1).unfold(3, kw, 1)
               .permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, kh, kw).float())
    n = patches.shape[0]
    flat = patches.reshape(n, -1)
    cov = (flat.T @ flat) / n
    eigvals, eigvecs = torch.linalg.eigh(cov)
    # eigh returns ascending; flip to descending so filter order is stable.
    eigvals = eigvals.flip(0)
    eigvecs = eigvecs.T.reshape(3 * kh * kw, 3, kh, kw).flip(0)
    scaled = eigvecs / torch.sqrt(eigvals + eps).view(-1, 1, 1, 1)
    conv.weight.data[:] = torch.cat((scaled, -scaled))
    return patches, scaled
