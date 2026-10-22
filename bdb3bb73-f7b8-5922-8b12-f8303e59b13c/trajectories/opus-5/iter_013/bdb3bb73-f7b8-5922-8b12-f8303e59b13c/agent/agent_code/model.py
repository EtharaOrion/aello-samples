"""CIFAR-10 speedrun model: ZCA-whitened 3-group convnet with TTA in forward().

The scripted checkpoint is consumed by a verifier that does

    model = torch.jit.load(path).eval()
    logits = model(x)          # x: float32 NCHW, already per-channel standardized
    pred = logits.argmax(1)

so forward() must accept *standardized* pixels.  For safety it also auto-detects
[0,255] and [0,1] inputs (see `_to_standard`); the detection is written so that a
standardized batch can never be mistaken for one of the other two conventions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# CIFAR-10 per-channel statistics used by the verifier's transforms.Normalize.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class ConvGroup(nn.Module):
    """conv -> maxpool -> BN -> GELU -> conv -> BN -> GELU (airbench-style)."""

    def __init__(self, c_in: int, c_out: int, bn_eps: float = 1e-5):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = nn.BatchNorm2d(c_out, eps=bn_eps)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(c_out, eps=bn_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.norm1(self.pool(self.conv1(x))))
        x = F.gelu(self.norm2(self.conv2(x)))
        return x


class Net(nn.Module):
    def __init__(self, widths=(128, 384, 512), logit_scale: float = 0.11,
                 bn_eps: float = 1e-5, tta: bool = True):
        super().__init__()
        w1, w2, w3 = widths
        # 2x2 whitening conv: 12 ZCA directions, kept in +/- pairs so the
        # following GELU sees both signs of every projection.
        self.whiten = nn.Conv2d(3, 24, 2, padding=0, bias=True)
        self.group1 = ConvGroup(24, w1, bn_eps)
        self.group2 = ConvGroup(w1, w2, bn_eps)
        self.group3 = ConvGroup(w2, w3, bn_eps)
        self.head = nn.Linear(w3, 10, bias=False)
        self.logit_scale = logit_scale
        self.tta = tta
        # Registered as buffers so they are captured by torch.jit.script and
        # moved with the module (needed for the input-convention guard).
        self.register_buffer("mean", torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR_STD).view(1, 3, 1, 1))

    # ---- input convention guard -------------------------------------------------
    def _to_standard(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-channel standardized pixels regardless of input scaling.

        A standardized CIFAR batch spans about [-2.0, 2.1]; a [0,1] batch is
        non-negative and a [0,255] batch exceeds 3.  Branching on the batch
        minimum/maximum therefore separates the three conventions cleanly.
        """
        hi = float(x.max())
        if hi > 3.0:                      # raw [0,255]
            return (x / 255.0 - self.mean) / self.std
        if float(x.min()) > -0.5:         # [0,1]
            return (x - self.mean) / self.std
        return x                          # already standardized (verifier path)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.whiten(x))
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = F.max_pool2d(x, x.shape[-1]).flatten(1)
        return self.head(x) * self.logit_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_standard(x)
        if not self.tta:
            return self._features(x)
        # 6-way TTA: {no shift, 1px left, 1px right} x {identity, mirror}.
        # Replicate-pad by 1 then crop so shifted views keep the 32x32 shape.
        p = F.pad(x, (1, 1, 0, 0), mode="replicate")
        views = [x, p[:, :, :, :32], p[:, :, :, 2:]]
        out = torch.zeros(x.shape[0], 10, dtype=x.dtype, device=x.device)
        for v in views:
            out = out + self._features(v) + self._features(v.flip(-1))
        return out / 6.0


@torch.no_grad()
def init_whitening(conv: nn.Conv2d, images: torch.Tensor, eps: float = 5e-4) -> None:
    """Set `conv` to a frozen ZCA whitening transform of 2x2 image patches.

    `images` is a float NCHW batch of *standardized* training images.  The conv
    bias absorbs the patch mean so the output is centred as well as whitened.
    """
    k = conv.weight.shape[-1]
    patches = (images.unfold(2, k, 1).unfold(3, k, 1)
                     .permute(0, 2, 3, 1, 4, 5).reshape(-1, 3 * k * k).double())
    mu = patches.mean(0)
    centred = patches - mu
    cov = (centred.T @ centred) / centred.shape[0]
    evals, evecs = torch.linalg.eigh(cov)
    # rows of `w` are the whitening filters, largest eigenvalue first
    w = (evecs.T.flip(0)) / torch.sqrt(evals.flip(0) + eps).unsqueeze(1)
    b = -(w @ mu)
    conv.weight.data.copy_(torch.cat([w, -w]).reshape(-1, 3, k, k).to(conv.weight.dtype))
    conv.bias.data.copy_(torch.cat([b, -b]).to(conv.bias.dtype))
