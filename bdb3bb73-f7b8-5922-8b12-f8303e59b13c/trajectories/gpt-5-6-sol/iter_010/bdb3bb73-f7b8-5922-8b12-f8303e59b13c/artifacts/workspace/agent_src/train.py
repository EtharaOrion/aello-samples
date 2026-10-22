import math
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.io import read_image


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

ROOT = Path("/workspace")
DATA = ROOT / "data" / "cifar10" / "train"
OUT = ROOT / "periodic_checkpoints"
MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device="cuda").view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616), device="cuda").view(1, 3, 1, 1)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def load_train():
    cache = ROOT / "train_data.pt"
    if cache.exists():
        obj = torch.load(cache, map_location="cpu", weights_only=True)
        return obj["images"], obj["labels"]
    images, labels = [], []
    for label, name in enumerate(CLASSES):
        paths = sorted((DATA / name).glob("*.png"))
        images.extend(read_image(str(p)) for p in paths)
        labels.extend([label] * len(paths))
    images = torch.stack(images).permute(0, 2, 3, 1).contiguous()
    labels = torch.tensor(labels, dtype=torch.long)
    torch.save({"images": images, "labels": labels}, cache)
    return images, labels


@torch.compile
def zeropower(g: torch.Tensor, steps: int = 3):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    x = x / (x.norm() + 1e-7)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        aa = x @ x.T
        x = a * x + (b * aa + c * aa @ aa) @ x
    if transposed:
        x = x.T
    return x


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
                p.add_(zeropower(g.reshape(len(g), -1)).reshape_as(g), alpha=-group["lr"])


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, channels):
        super().__init__(channels, eps=1e-12, momentum=0.4)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, cin, cout):
        super().__init__(cin, cout, 3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        torch.nn.init.dirac_(self.weight.data[: self.weight.size(1)])


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
        self.layers = nn.Sequential(
            nn.GELU(), ConvGroup(24, 64), ConvGroup(64, 256),
            ConvGroup(256, 256), nn.MaxPool2d(3),
        )
        self.head = nn.Linear(256, 10, bias=False)
        for mod in self.modules():
            if isinstance(mod, BatchNorm):
                mod.float()
            else:
                mod.half()

    def reset(self):
        for mod in self.modules():
            if type(mod) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                mod.reset_parameters()
        self.head.weight.data.mul_(1 / self.head.weight.data.std())

    def init_whiten(self, images):
        patches = images.unfold(2, 2, 1).unfold(3, 2, 1).transpose(1, 3).reshape(-1, 3, 2, 2).float()
        flat = patches.reshape(len(patches), -1)
        cov = flat.T @ flat / len(flat)
        vals, vecs = torch.linalg.eigh(cov, UPLO="U")
        kernels = vecs.T.reshape(-1, 3, 2, 2) / torch.sqrt(vals.reshape(-1, 1, 1, 1) + 5e-4)
        self.whiten.weight.data.copy_(torch.cat((kernels, -kernels)).half())

    def forward(self, x):
        x = self.whiten(x)
        x = self.layers(x).reshape(len(x), -1)
        return self.head(x) / x.size(-1)


class VerifierModel(nn.Module):
    """Consumes already standardized float32 tensors and embeds the reference 6-view TTA."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def mirrored(self, x):
        return 0.5 * self.net(x) + 0.5 * self.net(x.flip(-1))

    def forward(self, x):
        x = x.half()
        base = self.mirrored(x)
        padded = F.pad(x, (1, 1, 1, 1), "reflect")
        shifted = 0.5 * (self.mirrored(padded[:, :, :32, :32]) + self.mirrored(padded[:, :, 2:34, 2:34]))
        return 0.5 * (base + shifted)


def crop(images):
    shifts = torch.randint(-2, 3, (len(images), 2), device="cuda")
    out = torch.empty((len(images), 3, 32, 32), device="cuda", dtype=images.dtype)
    for sy in range(-2, 3):
        for sx in range(-2, 3):
            mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
            out[mask] = images[mask, :, 2 + sy:34 + sy, 2 + sx:34 + sx]
    return out


def export(net, epoch, directory):
    clean = CifarNet().cuda().to(memory_format=torch.channels_last)
    clean.load_state_dict(net.state_dict())
    clean.eval()
    scripted = torch.jit.script(VerifierModel(clean))
    path = directory / f"ckpt_epoch_{epoch:04d}.pt"
    scripted.save(str(path))


def run(net, images, labels, directory, save):
    batch_size = 2000
    batches = len(images) // batch_size
    total_steps = 8 * batches
    whiten_steps = 3 * batches
    net.reset()
    net.init_whiten(images[:5000])
    filters = [p for p in net.parameters() if p.ndim == 4 and p.requires_grad]
    norm_bias = [p for n, p in net.named_parameters() if "norm" in n and p.requires_grad]
    wd = 2e-6 * batch_size
    configs = [
        dict(params=[net.whiten.bias], lr=0.053, weight_decay=wd / 0.053),
        dict(params=norm_bias, lr=0.053, weight_decay=wd / 0.053),
        dict(params=[net.head.weight], lr=0.67, weight_decay=wd / 0.67),
    ]
    sgd = torch.optim.SGD(configs, momentum=0.85, nesterov=True, fused=True)
    muon = Muon(filters)
    initial = [g["lr"] for g in sgd.param_groups]
    padded = F.pad(images, (2, 2, 2, 2), "reflect")
    fixed_flip = torch.where((torch.rand(len(images), device="cuda") < 0.5).view(-1, 1, 1, 1), padded.flip(-1), padded)
    step = 0
    for epoch in range(1, 9):
        net.train()
        augmented = crop(fixed_flip if epoch % 2 else fixed_flip.flip(-1))
        order = torch.randperm(len(images), device="cuda")
        for i in range(batches):
            idx = order[i * batch_size:(i + 1) * batch_size]
            logits = net(augmented[idx])
            F.cross_entropy(logits, labels[idx], label_smoothing=0.2, reduction="sum").backward()
            sgd.param_groups[0]["lr"] = initial[0] * max(0.0, 1 - step / whiten_steps)
            for j in (1, 2):
                sgd.param_groups[j]["lr"] = initial[j] * (1 - step / total_steps)
            muon.param_groups[0]["lr"] = 0.24 * (1 - step / total_steps)
            sgd.step(); muon.step(); net.zero_grad(set_to_none=True)
            step += 1
        if save:
            export(net, epoch, directory)
            print(f"saved epoch {epoch}", flush=True)


def main():
    torch.manual_seed(0)
    cpu_images, cpu_labels = load_train()
    images = cpu_images.cuda().half().permute(0, 3, 1, 2) / 255
    images = ((images - MEAN) / STD).half().to(memory_format=torch.channels_last)
    labels = cpu_labels.cuda()
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    model.compile(mode="max-autotune")

    # Compile all training kernels before the checkpoint clock begins.
    run(model, images, torch.randint(0, 10, labels.shape, device="cuda"), ROOT / "scratch_ckpt", False)
    torch.manual_seed(17)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    start = time.time()
    run(model, images, labels, OUT, True)
    shutil.copy(OUT / "ckpt_epoch_0008.pt", ROOT / "checkpoint.pt")
    elapsed = time.time() - start
    print(f"final run wall clock {elapsed:.3f}s", flush=True)


if __name__ == "__main__":
    main()
