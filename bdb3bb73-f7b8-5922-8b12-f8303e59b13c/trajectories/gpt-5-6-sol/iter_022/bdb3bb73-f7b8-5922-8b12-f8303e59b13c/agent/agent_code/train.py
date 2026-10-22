import json
import math
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import pil_to_tensor


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

ROOT = Path("/workspace")
DATA = ROOT / "data" / "cifar10"
OUT = ROOT / "periodic_checkpoints"
MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device="cuda", dtype=torch.float16).view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616), device="cuda", dtype=torch.float16).view(1, 3, 1, 1)


@torch.compile
def zeropower(g: torch.Tensor, steps: int = 3) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    x = x / (x.norm() + 1e-7)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        aa = x @ x.T
        bb = b * aa + c * aa @ aa
        x = a * x + bb @ x
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
                update = zeropower(g.reshape(len(g), -1)).view_as(g)
                p.add_(update, alpha=-group["lr"])


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, channels: int):
        super().__init__(channels, eps=1e-12, momentum=0.4)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, cin: int, cout: int):
        super().__init__(cin, cout, 3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.dirac_(self.weight.data[: self.weight.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv1 = Conv(cin, cout)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(cout)
        self.conv2 = Conv(cout, cout)
        self.norm2 = BatchNorm(cout)
        self.activ = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        for module in self.modules():
            if isinstance(module, BatchNorm):
                module.float()
            else:
                module.half()

    @torch.no_grad()
    def reset(self):
        for module in self.modules():
            if type(module) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                module.reset_parameters()
        self.head.weight.mul_(1 / self.head.weight.std())

    @torch.no_grad()
    def init_whiten(self, images: torch.Tensor):
        patches = images.unfold(2, 2, 1).unfold(3, 2, 1).transpose(1, 3)
        patches = patches.reshape(-1, 3, 2, 2).float()
        flat = patches.view(len(patches), -1)
        cov = flat.T @ flat / len(flat)
        values, vectors = torch.linalg.eigh(cov, UPLO="U")
        filters = vectors.T.reshape(-1, 3, 2, 2) / torch.sqrt(values.view(-1, 1, 1, 1) + 5e-4)
        self.whiten.weight.copy_(torch.cat((filters, -filters)).half())

    def forward(self, x: torch.Tensor, whiten_bias_grad: bool = True) -> torch.Tensor:
        bias = self.whiten.bias
        if bias is not None and not whiten_bias_grad:
            bias = bias.detach()
        x = F.conv2d(x, self.whiten.weight, bias)
        x = self.layers(x).view(len(x), -1)
        return self.head(x) / x.size(-1)


class VerifierModel(nn.Module):
    def __init__(self, net: CifarNet):
        super().__init__()
        self.net = net

    def mirror(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.net(x, False) + 0.5 * self.net(x.flip(-1), False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.half()
        logits = self.mirror(x)
        padded = F.pad(x, (1, 1, 1, 1), "reflect")
        shifted_a = padded[:, :, 0:32, 0:32]
        shifted_b = padded[:, :, 2:34, 2:34]
        translated = 0.5 * (self.mirror(shifted_a) + self.mirror(shifted_b))
        return 0.5 * logits + 0.5 * translated


def load_split(name: str):
    cache = ROOT / (name + "_images.pt")
    if cache.exists():
        obj = torch.load(cache, map_location="cpu", weights_only=True)
        return obj["images"], obj["labels"]
    ds = ImageFolder(DATA / name)
    images = torch.empty((len(ds), 3, 32, 32), dtype=torch.uint8)
    labels = torch.empty(len(ds), dtype=torch.long)
    for i, (path, label) in enumerate(ds.samples):
        with open(path, "rb") as handle:
            from PIL import Image
            image = Image.open(handle).convert("RGB")
            images[i] = pil_to_tensor(image)
        labels[i] = label
    torch.save({"images": images, "labels": labels}, cache)
    return images, labels


class TrainLoader:
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, seed: int):
        self.raw = images.cuda().half().div_(255).to(memory_format=torch.channels_last)
        self.labels = labels.cuda()
        self.norm = (self.raw - MEAN) / STD
        generator = torch.Generator(device="cuda").manual_seed(seed + 91)
        mask = torch.rand(len(images), generator=generator, device="cuda") < 0.5
        self.flipped = torch.where(mask.view(-1, 1, 1, 1), self.norm.flip(-1), self.norm)
        self.padded = F.pad(self.flipped, (2, 2, 2, 2), "reflect")
        self.epoch = 0

    def batches(self):
        shifts = torch.randint(-2, 3, (len(self.raw), 2), device="cuda")
        cropped = torch.empty_like(self.norm)
        for sy in range(-2, 3):
            for sx in range(-2, 3):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                cropped[mask] = self.padded[mask, :, 2 + sy:34 + sy, 2 + sx:34 + sx]
        if self.epoch % 2:
            cropped = cropped.flip(-1)
        self.epoch += 1
        order = torch.randperm(len(cropped), device="cuda")
        for start in range(0, 50000, 2000):
            idx = order[start:start + 2000]
            yield cropped[idx], self.labels[idx]


def run_training(base: CifarNet, compiled, images, labels, seed: int, save: bool):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    loader = TrainLoader(images, labels, seed)
    base.reset()
    base.init_whiten(loader.norm[:5000])
    conv_params = [p for p in base.parameters() if p.requires_grad and p.ndim == 4]
    norm_biases = [p for n, p in base.named_parameters() if "norm" in n and p.requires_grad]
    batch_size, steps_total, bias_steps = 2000, 200, 75
    wd = 2e-6 * batch_size
    groups = [
        dict(params=[base.whiten.bias], lr=0.053, weight_decay=wd / 0.053),
        dict(params=norm_biases, lr=0.053, weight_decay=wd / 0.053),
        dict(params=[base.head.weight], lr=0.67, weight_decay=wd / 0.67),
    ]
    opt1 = torch.optim.SGD(groups, momentum=0.85, nesterov=True, fused=True)
    opt2 = Muon(conv_params, lr=0.24, momentum=0.6)
    for opt in (opt1, opt2):
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    started = time.time()
    step = 0
    for epoch in range(1, 9):
        compiled.train()
        for x, y in loader.batches():
            out = compiled(x, step < bias_steps)
            F.cross_entropy(out, y, label_smoothing=0.2, reduction="sum").backward()
            opt1.param_groups[0]["lr"] = opt1.param_groups[0]["initial_lr"] * max(0, 1 - step / bias_steps)
            for group in opt1.param_groups[1:] + opt2.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / steps_total)
            opt1.step()
            opt2.step()
            base.zero_grad(set_to_none=True)
            step += 1
        torch.cuda.synchronize()
        train_acc = (out.argmax(1) == y).float().mean().item()
        if save:
            base.eval()
            scripted = torch.jit.script(VerifierModel(base).eval())
            scripted.save(str(OUT / f"ckpt_epoch_{epoch:04d}.pt"))
        print(f"epoch={epoch} train_acc={train_acc:.4f} elapsed={time.time()-started:.3f}", flush=True)
    return time.time() - started


def evaluate_saved(path: Path) -> float:
    images, labels = load_split("test")
    x = images.cuda().float().div_(255)
    mean = torch.tensor((0.4914, 0.4822, 0.4465), device="cuda").view(1, 3, 1, 1)
    std = torch.tensor((0.2470, 0.2435, 0.2616), device="cuda").view(1, 3, 1, 1)
    x = (x - mean) / std
    labels = labels.cuda()
    model = torch.jit.load(str(path), map_location="cuda").eval()
    correct = 0
    with torch.inference_mode():
        for start in range(0, len(x), 512):
            end = start + 512
            correct += (model(x[start:end]).argmax(1) == labels[start:end]).sum().item()
    return correct / len(x)


def main():
    seed = int(os.environ.get("RUN_SEED", "0"))
    images, labels = load_split("train")
    base = CifarNet().cuda().to(memory_format=torch.channels_last)
    compiled = torch.compile(base, mode="max-autotune")
    print("compiling with an unscored warmup run", flush=True)
    run_training(base, compiled, images, labels.roll(1), seed + 10000, False)
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.pt"):
        old.unlink()
    print("starting graded run", flush=True)
    duration = run_training(base, compiled, images, labels, seed, True)
    shutil.copy(OUT / "ckpt_epoch_0008.pt", ROOT / "checkpoint.pt")
    accuracy = evaluate_saved(ROOT / "checkpoint.pt")
    with open(ROOT / "agent_reported_metrics.json", "w") as handle:
        json.dump({"accuracy": accuracy, "wall_clock_sec": duration}, handle)
    print(f"graded_duration={duration:.3f} accuracy={accuracy:.4f}", flush=True)


if __name__ == "__main__":
    main()
