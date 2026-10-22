import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.io import read_image


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device="cuda").view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616), device="cuda").view(1, 3, 1, 1)


def load_split(root: str):
    class_names = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
    paths, labels = [], []
    for label, name in enumerate(class_names):
        files = sorted(Path(root, name).glob("*.png"))
        paths.extend(files)
        labels.extend([label] * len(files))
    # Decode on CPU once; all epoch work happens from a resident GPU tensor.
    images = torch.stack([read_image(str(p)) for p in paths])
    images = images.cuda().half().div_(255).to(memory_format=torch.channels_last)
    images = images.sub_(MEAN).div_(STD)
    return images, torch.tensor(labels, device="cuda")


def batch_flip_lr(x):
    mask = (torch.rand(len(x), device=x.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(mask, x.flip(-1), x)


def batch_crop(x, size=32):
    radius = (x.size(-1) - size) // 2
    shifts = torch.randint(-radius, radius + 1, (len(x), 2), device=x.device)
    out = torch.empty((len(x), 3, size, size), device=x.device, dtype=x.dtype,
                      memory_format=torch.channels_last)
    for sy in range(-radius, radius + 1):
        for sx in range(-radius, radius + 1):
            mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
            out[mask] = x[mask, :, radius + sy:radius + sy + size,
                          radius + sx:radius + sx + size]
    return out


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, n):
        super().__init__(n, eps=1e-12, momentum=0.4)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, cin, cout):
        super().__init__(cin, cout, 3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        torch.nn.init.dirac_(self.weight.data[:self.weight.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = Conv(cin, cout)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(cout)
        self.conv2 = Conv(cout, cout)
        self.norm2 = BatchNorm(cout)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.activ(self.norm1(self.pool(self.conv1(x))))
        return self.activ(self.norm2(self.conv2(x)))


class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.whiten = nn.Conv2d(3, 24, 2, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(nn.GELU(), ConvGroup(24, 64),
                                    ConvGroup(64, 256), ConvGroup(256, 256),
                                    nn.MaxPool2d(3))
        self.head = nn.Linear(256, 10, bias=False)
        for module in self.modules():
            if isinstance(module, BatchNorm):
                module.float()
            else:
                module.half()

    def reset(self):
        for module in self.modules():
            if type(module) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                module.reset_parameters()
        self.head.weight.data.mul_(1 / self.head.weight.data.std())

    def init_whiten(self, images, eps=5e-4):
        patches = images.unfold(2, 2, 1).unfold(3, 2, 1)
        patches = patches.transpose(1, 3).reshape(-1, 3, 2, 2).float()
        flat = patches.reshape(len(patches), -1)
        covariance = flat.T @ flat / len(flat)
        values, vectors = torch.linalg.eigh(covariance, UPLO="U")
        filters = vectors.T.reshape(-1, 3, 2, 2)
        filters = filters / torch.sqrt(values.view(-1, 1, 1, 1) + eps)
        self.whiten.weight.data.copy_(torch.cat((filters, -filters)).half())

    def forward(self, x, whiten_bias_grad: bool = True):
        raw_bias = self.whiten.bias
        assert raw_bias is not None
        bias = raw_bias if whiten_bias_grad else raw_bias.detach()
        x = F.conv2d(x, self.whiten.weight, bias)
        x = self.layers(x).reshape(len(x), -1)
        return self.head(x) / x.size(-1)


class InferenceModel(nn.Module):
    """Verifier-facing model: standardized float input and deterministic 6-view TTA."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def mirrored(self, x):
        return (self.net(x, False) + self.net(x.flip(-1), False)) * 0.5

    def forward(self, x):
        x = x.half().contiguous(memory_format=torch.channels_last)
        center = self.mirrored(x)
        padded = F.pad(x, (1, 1, 1, 1), "reflect")
        shifted = (self.mirrored(padded[:, :, :32, :32]) +
                   self.mirrored(padded[:, :, 2:34, 2:34])) * 0.5
        return (center + shifted) * 0.5


@torch.compile
def zeropower(g, steps: int = 3):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    x = x / (x.norm() + 1e-7)
    transposed = g.size(0) > g.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        aa = x @ x.T
        x = a * x + (b * aa + c * aa @ aa) @ x
    return x.T if transposed else x


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.24, momentum=0.6):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(p.grad)
                g = p.grad.add(buf, alpha=group["momentum"])
                p.mul_(len(p) ** 0.5 / p.norm())
                update = zeropower(g.reshape(len(g), -1)).reshape_as(g)
                p.add_(update, alpha=-group["lr"])


def make_optimizers(model, batch_size):
    wd = 2e-6 * batch_size
    convs = [p for p in model.parameters() if p.ndim == 4 and p.requires_grad]
    biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    groups = [dict(params=[model.whiten.bias], lr=0.053, weight_decay=wd / 0.053),
              dict(params=biases, lr=0.053, weight_decay=wd / 0.053),
              dict(params=[model.head.weight], lr=0.67, weight_decay=wd / 0.67)]
    sgd = torch.optim.SGD(groups, momentum=0.85, nesterov=True, fused=True)
    muon = Muon(convs)
    for opt in (sgd, muon):
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return sgd, muon


def train_run(images, labels, out_dir, epochs=8, seed=1):
    torch.manual_seed(seed)
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    model.reset()
    model.init_whiten(images[:5000])
    batch_size = 2000
    steps_per_epoch = len(images) // batch_size
    total_steps = epochs * steps_per_epoch
    bias_steps = 3 * steps_per_epoch
    sgd, muon = make_optimizers(model, batch_size)
    compiled = torch.compile(model, mode="max-autotune")
    preflipped = batch_flip_lr(images)
    padded = F.pad(preflipped, (2, 2, 2, 2), "reflect")
    step = 0
    run_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        augmented = batch_crop(padded)
        if epoch % 2 == 0:
            augmented = augmented.flip(-1)
        order = torch.randperm(len(images), device="cuda")
        for offset in range(0, steps_per_epoch * batch_size, batch_size):
            idx = order[offset:offset + batch_size]
            logits = compiled(augmented[idx], step < bias_steps)
            F.cross_entropy(logits, labels[idx], label_smoothing=0.2,
                            reduction="sum").backward()
            sgd.param_groups[0]["lr"] = sgd.param_groups[0]["initial_lr"] * max(0, 1 - step / bias_steps)
            for group in sgd.param_groups[1:] + muon.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_steps)
            sgd.step()
            muon.step()
            model.zero_grad(set_to_none=True)
            step += 1
        torch.cuda.synchronize()
        wrapper = InferenceModel(model).eval()
        scripted = torch.jit.script(wrapper)
        path = Path(out_dir, f"ckpt_epoch_{epoch:04d}.pt")
        scripted.save(str(path))
        print(f"saved epoch {epoch} elapsed={time.time() - run_start:.3f}s", flush=True)
    final = Path("/workspace/checkpoint.pt")
    if final.exists():
        final.unlink()
    os.link(path, final)
    return time.time() - run_start


if __name__ == "__main__":
    train_images, train_labels = load_split("/workspace/data/cifar10/train")
    destination = Path("/workspace/periodic_checkpoints")
    destination.mkdir(exist_ok=True)
    for old in destination.glob("ckpt_epoch_*.pt"):
        old.unlink()
    elapsed = train_run(train_images, train_labels, destination)
    print(f"training wall time: {elapsed:.3f}s")
