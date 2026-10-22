import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device="cuda").view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616), device="cuda").view(1, 3, 1, 1)


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
                p.mul_(math.sqrt(len(p)) / p.norm())
                update = zeropower(g.reshape(len(g), -1)).reshape_as(g)
                p.add_(update, alpha=-group["lr"])


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, channels):
        super().__init__(channels, eps=1e-12, momentum=0.4)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, cin, cout):
        super().__init__(cin, cout, 3, padding=1, bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.dirac_(self.weight.data[: self.weight.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = Conv(cin, cout)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(cout)
        self.conv2 = Conv(cout, cout)
        self.norm2 = BatchNorm(cout)

    def forward(self, x):
        x = F.gelu(self.norm1(self.pool(self.conv1(x))))
        return F.gelu(self.norm2(self.conv2(x)))


class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.whiten = nn.Conv2d(3, 24, 2, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(), ConvGroup(24, 64), ConvGroup(64, 256),
            ConvGroup(256, 256), nn.MaxPool2d(3)
        )
        self.head = nn.Linear(256, 10, bias=False)
        for mod in self.modules():
            if isinstance(mod, BatchNorm):
                mod.float()
            else:
                mod.half()

    @torch.no_grad()
    def reset(self):
        for mod in self.modules():
            if type(mod) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                mod.reset_parameters()
        self.head.weight.mul_(1 / self.head.weight.std())

    @torch.no_grad()
    def init_whiten(self, images, eps=5e-4):
        patches = images.unfold(2, 2, 1).unfold(3, 2, 1)
        patches = patches.transpose(1, 3).reshape(-1, 3, 2, 2).float()
        flat = patches.view(len(patches), -1)
        cov = flat.T @ flat / len(flat)
        vals, vecs = torch.linalg.eigh(cov, UPLO="U")
        kernels = vecs.T.reshape(-1, 3, 2, 2) / torch.sqrt(vals.view(-1, 1, 1, 1) + eps)
        self.whiten.weight.copy_(torch.cat((kernels, -kernels)).half())

    def forward(self, x: torch.Tensor, whiten_bias_grad: bool = True) -> torch.Tensor:
        bias = self.whiten.bias
        assert bias is not None
        bias = bias if whiten_bias_grad else bias.detach()
        x = F.conv2d(x, self.whiten.weight, bias)
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        return self.head(x) / x.size(-1)


class VerifierModel(nn.Module):
    def __init__(self, net: CifarNet):
        super().__init__()
        self.net = net

    def mirror(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.net(x, False) + 0.5 * self.net(x.flip(-1), False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.half()
        base = self.mirror(x)
        padded = F.pad(x, (1, 1, 1, 1), "reflect")
        shifted = 0.5 * (self.mirror(padded[:, :, :32, :32]) +
                         self.mirror(padded[:, :, 2:34, 2:34]))
        return 0.5 * base + 0.5 * shifted


def load_folder(path: str):
    dataset = ImageFolder(path, transform=ToTensor())
    images = torch.empty((len(dataset), 3, 32, 32), dtype=torch.uint8)
    labels = torch.empty(len(dataset), dtype=torch.long)
    for i, (image, label) in enumerate(dataset):
        images[i] = (image * 255).round().to(torch.uint8)
        labels[i] = label
    return images, labels


def crop_batch(images: torch.Tensor, pad: int = 2):
    shifts = torch.randint(-pad, pad + 1, (len(images), 2), device=images.device)
    out = torch.empty((len(images), 3, 32, 32), device=images.device,
                      dtype=images.dtype, memory_format=torch.channels_last)
    for sy in range(-pad, pad + 1):
        for sx in range(-pad, pad + 1):
            mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
            out[mask] = images[mask, :, pad + sy:pad + sy + 32, pad + sx:pad + sx + 32]
    return out


def save_model(net: CifarNet, path: Path):
    net.eval()
    wrapped = VerifierModel(net)
    scripted = torch.jit.script(wrapped)
    scripted.save(str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="/workspace/periodic_checkpoints")
    parser.add_argument("--final", default="/workspace/checkpoint.pt")
    parser.add_argument("--report", default="/workspace/agent_reported_metrics.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cache_path = Path("/workspace/train_tensor_cache.pt")
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        raw, labels_cpu = cached["images"], cached["labels"]
    else:
        raw, labels_cpu = load_folder("/workspace/data/cifar10/train")
        torch.save({"images": raw, "labels": labels_cpu}, cache_path)
    images = raw.cuda().half().div_(255).to(memory_format=torch.channels_last)
    labels = labels_cpu.cuda()
    images = (images - MEAN.half()) / STD.half()
    flip_mask = torch.rand(len(images), device="cuda") < 0.5
    flipped = torch.where(flip_mask[:, None, None, None], images.flip(-1), images)
    padded = F.pad(flipped, (2, 2, 2, 2), "reflect")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("ckpt_epoch_*.pt"):
        old.unlink()
    final_path = Path(args.final)
    if final_path.exists():
        final_path.unlink()

    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    model.reset()
    model.init_whiten(images[:5000])
    model.compile(mode="max-autotune")

    batch_size = 2000
    batches = len(images) // batch_size
    total_steps = args.epochs * batches
    bias_steps = 3 * batches
    wd = 2e-6 * batch_size
    filters = [p for p in model.parameters() if p.ndim == 4 and p.requires_grad]
    norm_biases = [p for name, p in model.named_parameters() if "norm" in name and p.requires_grad]
    configs = [
        dict(params=[model.whiten.bias], lr=0.053, weight_decay=wd / 0.053),
        dict(params=norm_biases, lr=0.053, weight_decay=wd / 0.053),
        dict(params=[model.head.weight], lr=0.67, weight_decay=wd / 0.67),
    ]
    sgd = torch.optim.SGD(configs, momentum=0.85, nesterov=True, fused=True)
    muon = Muon(filters, lr=0.24, momentum=0.6)
    for opt in (sgd, muon):
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    run_start = time.time()
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_images = crop_batch(padded)
        if epoch % 2 == 0:
            epoch_images = epoch_images.flip(-1)
        order = torch.randperm(len(images), device="cuda")
        for batch_idx in range(batches):
            idx = order[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            inputs, targets = epoch_images[idx], labels[idx]
            outputs = model(inputs, step < bias_steps)
            F.cross_entropy(outputs, targets, label_smoothing=0.2, reduction="sum").backward()
            for group in sgd.param_groups[:1]:
                group["lr"] = group["initial_lr"] * max(0.0, 1 - step / bias_steps)
            for group in sgd.param_groups[1:] + muon.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_steps)
            sgd.step()
            muon.step()
            model.zero_grad(set_to_none=True)
            step += 1
        torch.cuda.synchronize()
        path = out_dir / f"ckpt_epoch_{epoch:04d}.pt"
        save_model(model, path)
        print(f"epoch={epoch} train_acc={(outputs.argmax(1) == targets).float().mean().item():.4f} path={path}", flush=True)

    run_wall = time.time() - run_start
    os.link(out_dir / f"ckpt_epoch_{args.epochs:04d}.pt", final_path)
    with open(args.report, "w") as handle:
        json.dump({"accuracy": 0.0, "wall_clock_sec": run_wall}, handle)
    print(f"run_wall={run_wall:.3f}", flush=True)


if __name__ == "__main__":
    main()
