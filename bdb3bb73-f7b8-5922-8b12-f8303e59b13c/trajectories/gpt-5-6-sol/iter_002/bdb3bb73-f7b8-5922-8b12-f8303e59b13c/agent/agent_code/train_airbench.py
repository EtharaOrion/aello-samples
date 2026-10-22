"""Train AirBench94 from scratch on the local CIFAR-10 train ImageFolder.

The test split is intentionally not opened here.  Each epoch is one complete
pass over all 50,000 training examples.  Saved TorchScript modules consume
already-standardized float32 NCHW tensors and include the recipe's test-time
augmentation in forward().
"""
import argparse
import json
import math
import os
import shutil
import time

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

torch.backends.cudnn.benchmark = True

MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device="cuda").view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616), device="cuda").view(1, 3, 1, 1)


@torch.compile
def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 3) -> torch.Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.bfloat16()
    x /= x.norm() + 1e-7
    transposed = g.size(0) > g.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        aa = x @ x.T
        bb = b * aa + c * aa @ aa
        x = a * x + bb @ x
    return x.T if transposed else x


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.24, momentum=0.6, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=group["momentum"])
                else:
                    g = buf
                p.mul_(len(p) ** 0.5 / p.norm())
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view_as(g)
                p.add_(update, alpha=-group["lr"])


def batch_flip_lr(images: torch.Tensor) -> torch.Tensor:
    mask = (torch.rand(len(images), device=images.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(mask, images.flip(-1), images)


def batch_crop(images: torch.Tensor, crop_size: int) -> torch.Tensor:
    r = (images.size(-1) - crop_size) // 2
    shifts = torch.randint(-r, r + 1, (len(images), 2), device=images.device)
    out = torch.empty((len(images), 3, crop_size, crop_size), device="cuda",
                      dtype=images.dtype, memory_format=torch.channels_last)
    for sy in range(-r, r + 1):
        for sx in range(-r, r + 1):
            mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
            out[mask] = images[mask, :, r+sy:r+sy+crop_size, r+sx:r+sx+crop_size]
    return out


class CifarLoader:
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, batch_size=2000):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.epoch = 0
        self.norm = (images.half() / 255 - MEAN.half()) / STD.half()
        self.norm = self.norm.contiguous(memory_format=torch.channels_last)
        self.flipped = batch_flip_lr(self.norm)
        self.padded = F.pad(self.flipped, (2, 2, 2, 2), "reflect")

    def __len__(self):
        return len(self.images) // self.batch_size

    def __iter__(self):
        images = batch_crop(self.padded, 32)
        if self.epoch % 2 == 1:
            images = images.flip(-1)
        self.epoch += 1
        indices = torch.randperm(len(images), device="cuda")
        for i in range(len(self)):
            idx = indices[i*self.batch_size:(i+1)*self.batch_size]
            yield images[idx], self.labels[idx]


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, features: int, momentum=0.6, eps=1e-12):
        super().__init__(features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, cin: int, cout: int):
        super().__init__(cin, cout, 3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        torch.nn.init.dirac_(self.weight.data[:self.weight.size(1)])


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
            ConvGroup(256, 256), nn.MaxPool2d(3))
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
    def init_whiten(self, train_images: torch.Tensor, eps=5e-4):
        c = train_images.shape[1]
        patches = train_images.unfold(2, 2, 1).unfold(3, 2, 1)
        patches = patches.transpose(1, 3).reshape(-1, c, 2, 2).float()
        flat = patches.view(len(patches), -1)
        covariance = (flat.T @ flat) / len(flat)
        values, vectors = torch.linalg.eigh(covariance, UPLO="U")
        scaled = vectors.T.reshape(-1, c, 2, 2) / torch.sqrt(values.view(-1, 1, 1, 1) + eps)
        self.whiten.weight.copy_(torch.cat((scaled, -scaled)).half())

    def forward(self, x: torch.Tensor, whiten_bias_grad: bool = True) -> torch.Tensor:
        base_bias = self.whiten.bias
        assert base_bias is not None
        bias = base_bias if whiten_bias_grad else base_bias.detach()
        x = F.conv2d(x, self.whiten.weight, bias)
        x = self.layers(x)
        x = x.view(len(x), -1)
        return self.head(x) / x.size(-1)


class ScriptedTTA(nn.Module):
    """Verifier-facing model: float32 standardized input -> class logits."""
    def __init__(self, net: CifarNet):
        super().__init__()
        self.net = net

    def basic(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x, False)

    def mirror(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.basic(x) + 0.5 * self.basic(x.flip(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.half().contiguous(memory_format=torch.channels_last)
        logits = self.mirror(x)
        padded = F.pad(x, (1, 1, 1, 1), "reflect")
        translated = 0.5 * (self.mirror(padded[:, :, :32, :32]) +
                            self.mirror(padded[:, :, 2:34, 2:34]))
        return 0.5 * logits + 0.5 * translated


def load_train(root: str):
    cache = os.path.join(root, "train_tensor.pt")
    if os.path.exists(cache):
        obj = torch.load(cache, map_location="cpu", weights_only=True)
        return obj["images"].cuda(), obj["labels"].cuda()
    ds = torchvision.datasets.ImageFolder(os.path.join(root, "train"), transform=T.PILToTensor())
    # Keep collation in the main process: the execution container has a small
    # /dev/shm, while this one-time preprocessing is outside the graded run.
    loader = torch.utils.data.DataLoader(ds, batch_size=1000, shuffle=False, num_workers=0)
    images, labels = [], []
    for x, y in loader:
        images.append(x)
        labels.append(y)
    obj = {"images": torch.cat(images), "labels": torch.cat(labels)}
    torch.save(obj, cache)
    return obj["images"].cuda(), obj["labels"].cuda()


def train_run(model: CifarNet, compiled, images, labels, out_dir=None, warmup=False):
    loader = CifarLoader(images, labels, batch_size=2000)
    if warmup:
        loader.labels = torch.randint(0, 10, labels.shape, device="cuda")
    total_steps = 8 * len(loader)
    bias_steps = 3 * len(loader)
    bias_lr, head_lr, wd = 0.053, 0.67, 2e-6 * 2000
    filters = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norms = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    configs = [dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd/bias_lr),
               dict(params=norms, lr=bias_lr, weight_decay=wd/bias_lr),
               dict(params=[model.head.weight], lr=head_lr, weight_decay=wd/head_lr)]
    opt1 = torch.optim.SGD(configs, momentum=0.85, nesterov=True, fused=True)
    opt2 = Muon(filters, lr=0.24, momentum=0.6, nesterov=True)
    opts = (opt1, opt2)
    for opt in opts:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    model.reset()
    normalized = (images[:5000].half() / 255 - MEAN.half()) / STD.half()
    model.init_whiten(normalized)
    step = 0
    start = time.perf_counter()
    for epoch in range(1, 9):
        model.train()
        for inputs, targets in loader:
            outputs = compiled(inputs, step < bias_steps)
            F.cross_entropy(outputs, targets, label_smoothing=0.2, reduction="sum").backward()
            opt1.param_groups[0]["lr"] = bias_lr * (1 - step / bias_steps)
            for group in opt1.param_groups[1:] + opt2.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_steps)
            for opt in opts:
                opt.step()
            model.zero_grad(set_to_none=True)
            step += 1
        if out_dir is not None:
            torch.cuda.synchronize()
            model.eval()
            scripted = torch.jit.script(ScriptedTTA(model))
            scripted.save(os.path.join(out_dir, f"ckpt_epoch_{epoch:04d}.pt"))
            print(f"epoch {epoch}: {time.perf_counter()-start:.3f}s", flush=True)
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/workspace/data/cifar10")
    parser.add_argument("--out", default="/workspace/periodic_checkpoints")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--final", default="/workspace/checkpoint.pt")
    parser.add_argument("--metrics", default="/workspace/agent_reported_metrics.json")
    args = parser.parse_args()
    torch.manual_seed(0)
    images, labels = load_train(args.data)
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    compiled = torch.compile(model, mode="max-autotune")
    if not args.skip_warmup:
        print("compilation warmup", flush=True)
        train_run(model, compiled, images, labels, warmup=True)
    os.makedirs(args.out, exist_ok=True)
    for name in os.listdir(args.out):
        if name.startswith("ckpt_epoch_") and name.endswith(".pt"):
            os.remove(os.path.join(args.out, name))
    torch.manual_seed(0)
    elapsed = train_run(model, compiled, images, labels, out_dir=args.out)
    final_src = os.path.join(args.out, "ckpt_epoch_0008.pt")
    final_dst = args.final
    if os.path.exists(final_dst):
        os.remove(final_dst)
    os.link(final_src, final_dst)
    with open(args.metrics, "w") as f:
        json.dump({"accuracy": 0.0, "wall_clock_sec": elapsed}, f)
    print(f"training complete: {elapsed:.3f}s", flush=True)


if __name__ == "__main__":
    main()
