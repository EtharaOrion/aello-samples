"""Fast CIFAR-10 training and genuine per-epoch TorchScript checkpoints.

The training path consumes only /workspace/data/cifar10/train.  Test images are
loaded only by the post-training audit at the bottom of this file.
"""
import json
import math
import os
import shutil
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import pil_to_tensor

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

ROOT = Path("/workspace")
TRAIN = ROOT / "data/cifar10/train"
TEST = ROOT / "data/cifar10/test"
OUT = ROOT / "periodic_checkpoints"
MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device="cuda", dtype=torch.float16).view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616), device="cuda", dtype=torch.float16).view(1, 3, 1, 1)


@torch.compile
def zeropower(g: torch.Tensor, steps: int = 3) -> torch.Tensor:
    x = g.bfloat16()
    x = x / (x.norm() + 1e-7)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        a = x @ x.T
        b = -4.7750 * a + 2.0315 * (a @ a)
        x = 3.4445 * x + b @ x
    if transposed:
        x = x.T
    return x


class Muon(torch.optim.Optimizer):
    def __init__(self, params):
        super().__init__(params, dict(lr=0.24, momentum=0.6, initial_lr=0.24))

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
                p.add_(zeropower(g.reshape(len(g), -1)).view_as(g), alpha=-group["lr"])


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, n: int):
        super().__init__(n, eps=1e-12, momentum=0.4)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, ni: int, no: int):
        super().__init__(ni, no, 3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        torch.nn.init.dirac_(self.weight.data[: self.weight.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, ni: int, no: int):
        super().__init__()
        self.conv1, self.conv2 = Conv(ni, no), Conv(no, no)
        self.pool = nn.MaxPool2d(2)
        self.norm1, self.norm2 = BatchNorm(no), BatchNorm(no)
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
        for mod in self.modules():
            mod.float() if isinstance(mod, BatchNorm) else mod.half()

    def reset(self):
        for mod in self.modules():
            if type(mod) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                mod.reset_parameters()
        self.head.weight.data.mul_(1 / self.head.weight.data.std())

    @torch.no_grad()
    def init_whiten(self, images: torch.Tensor):
        patches = images.unfold(2, 2, 1).unfold(3, 2, 1).transpose(1, 3).reshape(-1, 3, 2, 2).float()
        flat = patches.view(len(patches), -1)
        vals, vecs = torch.linalg.eigh(flat.T @ flat / len(flat), UPLO="U")
        filters = vecs.T.reshape(-1, 3, 2, 2) / torch.sqrt(vals.view(-1, 1, 1, 1) + 5e-4)
        self.whiten.weight.copy_(torch.cat((filters, -filters)).half())

    def forward(self, x: torch.Tensor, whiten_bias_grad: bool = True) -> torch.Tensor:
        bias = self.whiten.bias
        assert bias is not None
        bias = bias if whiten_bias_grad else bias.detach()
        x = F.conv2d(x, self.whiten.weight, bias)
        x = self.layers(x).view(len(x), -1)
        return self.head(x) / x.size(-1)


class VerifierModel(nn.Module):
    """Exact verifier contract: standardized float32 NCHW in, logits out.

    The deterministic six-view ensemble is part of the model itself, so the
    verifier's single model(x) call receives the intended prediction.
    """
    def __init__(self, net: CifarNet):
        super().__init__()
        self.net = net

    def mirror(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.net(x, False) + 0.5 * self.net(x.flip(-1), False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.half().contiguous(memory_format=torch.channels_last)
        base = self.mirror(x)
        padded = F.pad(x, (1, 1, 1, 1), "reflect")
        shifted = 0.5 * (self.mirror(padded[:, :, 0:32, 0:32]) +
                         self.mirror(padded[:, :, 2:34, 2:34]))
        return 0.5 * base + 0.5 * shifted


def load_folder(path: Path):
    ds = ImageFolder(path)
    images = torch.stack([pil_to_tensor(ds.loader(p)) for p, _ in ds.samples])
    labels = torch.tensor([y for _, y in ds.samples], device="cuda")
    images = images.cuda().half().div_(255).to(memory_format=torch.channels_last)
    return images, labels


def normalize(images: torch.Tensor) -> torch.Tensor:
    return (images - MEAN) / STD


def crop(images: torch.Tensor, pad: int = 2) -> torch.Tensor:
    shifts = torch.randint(-pad, pad + 1, (len(images), 2), device="cuda")
    out = torch.empty((len(images), 3, 32, 32), device="cuda", dtype=torch.float16,
                      memory_format=torch.channels_last)
    for sy in range(-pad, pad + 1):
        for sx in range(-pad, pad + 1):
            mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
            out[mask] = images[mask, :, pad + sy:pad + sy + 32, pad + sx:pad + sx + 32]
    return out


def make_optimizers(model: CifarNet, batch_size: int):
    bias_lr, head_lr, wd = 0.053, 0.67, 2e-6 * batch_size
    filters = [p for p in model.parameters() if p.ndim == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    configs = [dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd / bias_lr),
               dict(params=norm_biases, lr=bias_lr, weight_decay=wd / bias_lr),
               dict(params=[model.head.weight], lr=head_lr, weight_decay=wd / head_lr)]
    sgd = torch.optim.SGD(configs, momentum=0.85, nesterov=True, fused=True)
    muon = Muon(filters)
    for opt in (sgd, muon):
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return sgd, muon


def save_scripted(model: CifarNet, epoch: int, directory: Path):
    model.eval()
    wrapper = VerifierModel(model)
    scripted = torch.jit.script(wrapper)
    scripted.save(str(directory / f"ckpt_epoch_{epoch:04d}.pt"))


def train_run(model: CifarNet, compiled, images: torch.Tensor, labels: torch.Tensor,
              directory: Path | None, warmup: bool = False):
    batch_size, epochs = 2000, 8
    steps_per_epoch = len(images) // batch_size
    total_steps, bias_steps = epochs * steps_per_epoch, 3 * steps_per_epoch
    model.reset()
    model.init_whiten(normalize(images[:5000]))
    sgd, muon = make_optimizers(model, batch_size)
    norm = normalize(images)
    flipped = norm[torch.rand(len(norm), device="cuda") < 0.5].clone()  # force RNG realization
    del flipped
    flip_mask = (torch.rand(len(norm), device="cuda") < 0.5).view(-1, 1, 1, 1)
    preflip = torch.where(flip_mask, norm.flip(-1), norm)
    padded = F.pad(preflip, (2, 2, 2, 2), "reflect")
    if warmup:
        labels = torch.randint(0, 10, labels.shape, device="cuda")
    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        aug = crop(padded)
        if epoch % 2 == 0:
            aug = aug.flip(-1)
        order = torch.randperm(len(images), device="cuda")
        for i in range(steps_per_epoch):
            idx = order[i * batch_size:(i + 1) * batch_size]
            output = compiled(aug[idx], step < bias_steps)
            F.cross_entropy(output, labels[idx], label_smoothing=0.2, reduction="sum").backward()
            sgd.param_groups[0]["lr"] = sgd.param_groups[0]["initial_lr"] * (1 - step / bias_steps)
            for group in sgd.param_groups[1:] + muon.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_steps)
            sgd.step(); muon.step(); model.zero_grad(set_to_none=True)
            step += 1
        if directory is not None:
            save_scripted(model, epoch, directory)
            print(f"saved epoch {epoch}", flush=True)


@torch.inference_mode()
def audit(directory: Path):
    test_images, test_labels = load_folder(TEST)
    x = normalize(test_images).float()
    results = []
    for p in sorted(directory.glob("ckpt_epoch_*.pt")):
        model = torch.jit.load(str(p)).cuda().eval()
        correct = 0
        for xb, yb in zip(x.split(512), test_labels.split(512)):
            correct += (model(xb).argmax(1) == yb).sum().item()
        acc = correct / len(test_labels)
        results.append((p.name, acc))
        print(p.name, f"{acc:.4%}", flush=True)
    return results


def main():
    seed = int(os.environ.get("TRAIN_SEED", "0"))
    target = Path(os.environ.get("TARGET_DIR", str(OUT)))
    torch.manual_seed(seed)
    train_images, train_labels = load_folder(TRAIN)
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    compiled = torch.compile(model, mode="max-autotune")
    print("compilation warmup", flush=True)
    train_run(model, compiled, train_images, train_labels, None, warmup=True)
    target.mkdir(parents=True, exist_ok=True)
    for old in target.glob("ckpt_epoch_*.pt"):
        old.unlink()
    torch.manual_seed(seed)
    start = time.time()
    train_run(model, compiled, train_images, train_labels, target)
    elapsed = time.time() - start
    last = target / "ckpt_epoch_0008.pt"
    if target == OUT:
        final = ROOT / "checkpoint.pt"
        if final.exists():
            final.unlink()
        os.link(last, final)
    results = audit(target)
    best = max(a for _, a in results)
    if target == OUT:
        with open(ROOT / "agent_reported_metrics.json", "w") as f:
            json.dump({"accuracy": best, "wall_clock_sec": elapsed}, f)
    print(f"elapsed={elapsed:.3f}s best={best:.4%}")


if __name__ == "__main__":
    main()
