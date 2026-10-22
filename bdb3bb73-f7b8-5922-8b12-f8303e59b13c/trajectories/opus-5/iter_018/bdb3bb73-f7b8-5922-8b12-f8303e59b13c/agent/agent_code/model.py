"""Model definition for the CIFAR-10 speedrun.

A compact three-group convnet in the airbench lineage:

    frozen 2x2 patch-ZCA whitening conv (24 ch, +/- eigenvector pairs)
      -> GELU
      -> ConvGroup(w1) -> ConvGroup(w2) -> ConvGroup(w3)
      -> global 3x3 max-pool -> bias-free linear head -> scale

Two things are deliberate and load-bearing for the verifier contract:

*   ``forward`` expects inputs that are ALREADY per-channel standardized with
    the CIFAR-10 train statistics, because that is exactly what the verifier
    feeds (``ToTensor`` + ``Normalize``).  There is no normalization inside the
    module, so training and verification see identical tensors.
*   Test-time augmentation lives inside ``forward`` behind the ``tta`` flag, so
    a plain ``torch.jit.load(path)(x)`` call gets the augmented prediction with
    no cooperation required from the caller.
"""

import torch
import torch.nn.functional as F
from torch import nn

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def dirac_conv(c_in, c_out):
    """3x3 conv whose first ``c_in`` filters start as an identity map.

    The partial-Dirac init makes the network close to a (whitened) identity at
    step 0, which is what lets the very aggressive LR schedule below be stable
    from the first step instead of needing a long warmup.
    """
    conv = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
    with torch.no_grad():
        nn.init.dirac_(conv.weight[:c_in])
    return conv


def frozen_bn(c, momentum):
    """BatchNorm with gamma pinned at 1; only beta is learned.

    Gamma is redundant with the next conv's scale and, at the 64x bias LR used
    here, is the main source of instability -- so it stays frozen.
    """
    bn = nn.BatchNorm2d(c, eps=1e-12, momentum=momentum)
    bn.weight.requires_grad = False
    return bn


class ConvGroup(nn.Module):
    """conv -> pool -> norm -> act -> conv -> norm -> act (halves resolution)."""

    def __init__(self, c_in, c_out, bn_momentum, ablate=""):
        super().__init__()
        conv = (lambda a, b: nn.Conv2d(a, b, 3, padding=1, bias=False)) \
            if ablate == "dirac" else dirac_conv
        norm = (lambda c, m: nn.BatchNorm2d(c, eps=1e-12, momentum=m)) \
            if ablate == "gamma" else frozen_bn
        self.conv1 = conv(c_in, c_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = norm(c_out, bn_momentum)
        self.conv2 = conv(c_out, c_out)
        self.norm2 = norm(c_out, bn_momentum)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.activ(self.norm1(self.pool(self.conv1(x))))
        x = self.activ(self.norm2(self.conv2(x)))
        return x


class Net(nn.Module):
    def __init__(self, widths=(128, 384, 512), bn_momentum=0.4, scale=1.0 / 9,
                 whiten_k=2, tta=False, ablate=""):
        super().__init__()
        whiten_width = 2 * 3 * whiten_k ** 2
        self.whiten = nn.Conv2d(3, whiten_width, whiten_k, padding=0, bias=True)
        self.whiten.weight.requires_grad = False  # frozen ZCA basis
        self.activ = nn.GELU()
        self.group1 = ConvGroup(whiten_width, widths[0], bn_momentum, ablate)
        self.group2 = ConvGroup(widths[0], widths[1], bn_momentum, ablate)
        self.group3 = ConvGroup(widths[1], widths[2], bn_momentum, ablate)
        self.pool = nn.MaxPool2d(3)
        self.head = nn.Linear(widths[2], 10, bias=False)
        self.scale = scale
        self.tta = tta

    def logits(self, x):
        x = self.activ(self.whiten(x))
        x = self.group3(self.group2(self.group1(x)))
        x = self.pool(x)
        x = x.flatten(1)
        return self.head(x) * self.scale

    def _mirror(self, x):
        # Average the upright and left-right-mirrored views.
        return self.logits(x) + self.logits(torch.flip(x, [3]))

    def forward(self, x):
        if not self.tta:
            return self.logits(x)
        # 6-way TTA: {upright, mirrored} x {no shift, -1px diagonal, +1px diagonal}.
        base = self._mirror(x)
        padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
        shift_a = self._mirror(padded[:, :, 0:32, 0:32])
        shift_b = self._mirror(padded[:, :, 2:34, 2:34])
        return 0.5 * base + 0.25 * (shift_a + shift_b)


@torch.no_grad()
def init_whitening(model, images, eps=5e-4):
    """Set the frozen whitening conv to the patch-ZCA basis of ``images``.

    ``images`` is a standardized float NCHW batch from the *training* split.
    We take every 2x2 patch, eigendecompose the 12-dim patch covariance, and
    write ``eigvec / sqrt(eigval)`` into the filters -- once with each sign, so
    the following GELU sees both halves of every direction.
    """
    k = model.whiten.kernel_size[0]
    patches = images.unfold(2, k, 1).unfold(3, k, 1)  # N,C,H',W',k,k
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3 * k * k).float()
    patches = patches - patches.mean(0, keepdim=True)
    cov = (patches.T @ patches) / (patches.shape[0] - 1)
    eigval, eigvec = torch.linalg.eigh(cov)
    filters = (eigvec.T / torch.sqrt(eigval + eps).view(-1, 1)).reshape(-1, 3, k, k)
    model.whiten.weight.copy_(torch.cat([filters, -filters]).to(model.whiten.weight.dtype))
    model.whiten.bias.zero_()
