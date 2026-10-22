"""Model definition for the CIFAR-10 speedrun.

Architecture is a compact three-stage VGG-style convnet preceded by a frozen
2x2 ZCA-whitening convolution (the "airbench" family of nets).  Everything the
verifier needs lives inside `forward`:

  * input canonicalisation (see `Net.forward` docstring),
  * test-time augmentation (mirror x 3 one-pixel translations),
  * logit scaling.

so that `torch.jit.script(model).save(...)` produces a checkpoint that is
correct under the verifier's exact call convention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# The verifier standardises its inputs with these constants; we use the same
# ones for training so that train- and test-time distributions match exactly.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class ConvGroup(nn.Module):
    def __init__(self, cin: int, cout: int, bn_momentum: float):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = nn.BatchNorm2d(cout, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(cout, momentum=bn_momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.norm1(self.pool(self.conv1(x))))
        x = F.gelu(self.norm2(self.conv2(x)))
        return x


class Net(nn.Module):
    def __init__(self, w1: int = 128, w2: int = 384, w3: int = 512,
                 scale: float = 0.125, bn_momentum: float = 0.6,
                 tta: bool = True):
        super().__init__()
        self.whiten = nn.Conv2d(3, 24, 2, padding=0, bias=True)
        self.g1 = ConvGroup(24, w1, bn_momentum)
        self.g2 = ConvGroup(w1, w2, bn_momentum)
        self.g3 = ConvGroup(w2, w3, bn_momentum)
        self.head = nn.Linear(w3, 10, bias=False)
        self.scale = scale
        self.tta = tta
        self.register_buffer("mean", torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR_STD).view(1, 3, 1, 1))
        # A frozen whitening layer: no gradient ever flows to the weight.
        self.whiten.weight.requires_grad = False

    # ---------------------------------------------------------------- body --
    def body(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.whiten(x))
        x = self.g3(self.g2(self.g1(x)))
        x = F.max_pool2d(x, x.shape[-1]).flatten(1)
        return self.head(x) * self.scale

    # ------------------------------------------------------------- helpers --
    def canonicalize(self, x: torch.Tensor) -> torch.Tensor:
        """Accept [0,255], [0,1] or already-standardised NCHW input.

        The verifier hands us tensors that have *already* been through
        `Normalize(CIFAR_MEAN, CIFAR_STD)`, so the common case must be the
        identity -- double normalisation is the documented failure mode.  Real
        standardised CIFAR batches live in roughly [-1.99, 2.13]: every batch
        contains near-black pixels, so `min < -0.05` identifies them
        unambiguously, and `max > 4` identifies raw uint8-valued input.
        """
        hi = torch.max(x).item()
        lo = torch.min(x).item()
        if hi > 4.0:                       # [0,255]
            return (x / 255.0 - self.mean) / self.std
        if lo >= -0.05:                    # [0,1]
            return (x - self.mean) / self.std
        return x                           # already standardised -> identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.canonicalize(x)
        if not self.tta or self.training:
            return self.body(x)
        # 6-way TTA: {no shift, 1px left, 1px right} x {identity, mirror}.
        # Shifts replicate the edge column rather than zero-padding.
        xp = F.pad(x, (1, 1, 0, 0), mode="replicate")
        views = [x, xp[:, :, :, 0:32], xp[:, :, :, 2:34]]
        out = torch.zeros([x.shape[0], 10], device=x.device, dtype=x.dtype)
        for v in views:
            out = out + self.body(v) + self.body(torch.flip(v, [3]))
        return out / 6.0


@torch.no_grad()
def init_whitening(model: Net, images: torch.Tensor, eps: float = 5e-4) -> None:
    """Set the frozen 2x2 conv to a ZCA-whitening basis of train-split patches.

    `images` must be standardised float NCHW data from the *training* split
    only.  The 12-dimensional patch covariance is eigendecomposed and each
    eigenvector is emitted with both signs, giving 24 output channels whose
    responses are decorrelated and unit-variance.

    The covariance is accumulated over *all* supplied images in chunks: a
    subsample would have to be drawn at random because the cached arrays are
    ordered by class, and getting that wrong silently biases the basis.
    """
    n = 0
    s = torch.zeros(12, dtype=torch.float64, device=images.device)
    ss = torch.zeros(12, 12, dtype=torch.float64, device=images.device)
    for i in range(0, images.shape[0], 2048):
        p = images[i:i + 2048].unfold(2, 2, 1).unfold(3, 2, 1)       # N,C,H',W',2,2
        p = p.permute(0, 2, 3, 1, 4, 5).reshape(-1, 12).double()
        n += p.shape[0]
        s += p.sum(0)
        ss += p.T @ p
    mu = s / n
    cov = (ss - n * torch.outer(mu, mu)) / (n - 1)
    cov = cov.float()
    eigval, eigvec = torch.linalg.eigh(cov)
    w = (eigvec / torch.sqrt(eigval + eps).unsqueeze(0)).T.reshape(12, 3, 2, 2)
    model.whiten.weight.data[:] = torch.cat([w, -w]).to(model.whiten.weight.dtype)
    model.whiten.bias.data.zero_()
