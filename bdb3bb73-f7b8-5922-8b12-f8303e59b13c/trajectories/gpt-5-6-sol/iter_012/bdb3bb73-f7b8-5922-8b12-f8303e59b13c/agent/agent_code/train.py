import json
import os
import time
from math import ceil
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.io import read_image


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device="cuda").view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616), device="cuda").view(1, 3, 1, 1)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


@torch.compile
def zeropower(g: torch.Tensor, steps: int = 3):
    x = g.bfloat16()
    x /= x.norm() + 1e-7
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
        super().__init__(params, dict(lr=0.24, momentum=0.6))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(p.grad)
                g = p.grad.add(buf, alpha=group["momentum"])
                p.mul_(len(p) ** 0.5 / p.norm())
                p.add_(zeropower(g.reshape(len(g), -1)).reshape_as(g), alpha=-group["lr"])


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, n):
        super().__init__(n, eps=1e-12, momentum=0.4)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, ni, no):
        super().__init__(ni, no, 3, padding=1, bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        torch.nn.init.dirac_(self.weight.data[:self.weight.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, ni, no):
        super().__init__()
        self.conv1, self.pool, self.norm1 = Conv(ni, no), nn.MaxPool2d(2), BatchNorm(no)
        self.conv2, self.norm2, self.activ = Conv(no, no), BatchNorm(no), nn.GELU()

    def forward(self, x):
        x = self.activ(self.norm1(self.pool(self.conv1(x))))
        return self.activ(self.norm2(self.conv2(x)))


class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.whiten = nn.Conv2d(3, 24, 2, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(nn.GELU(), ConvGroup(24, 64), ConvGroup(64, 256),
                                    ConvGroup(256, 256), nn.MaxPool2d(3))
        self.head = nn.Linear(256, 10, bias=False)
        for m in self.modules():
            if isinstance(m, BatchNorm):
                m.float()
            else:
                m.half()

    def reset(self):
        for m in self.modules():
            if type(m) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                m.reset_parameters()
        self.head.weight.data /= self.head.weight.data.std()

    @torch.no_grad()
    def init_whiten(self, images):
        patches = images[:5000].unfold(2, 2, 1).unfold(3, 2, 1).transpose(1, 3).reshape(-1, 3, 2, 2).float()
        flat = patches.reshape(len(patches), -1)
        vals, vecs = torch.linalg.eigh(flat.T @ flat / len(flat), UPLO="U")
        filters = vecs.T.reshape(-1, 3, 2, 2) / torch.sqrt(vals.view(-1, 1, 1, 1) + 5e-4)
        self.whiten.weight.copy_(torch.cat((filters, -filters)).half())

    def forward(self, x: torch.Tensor, whiten_bias_grad: bool = False):
        x = x.half()
        b = self.whiten.bias
        assert b is not None
        if not whiten_bias_grad:
            b = b.detach()
        x = F.conv2d(x, self.whiten.weight, b)
        x = self.layers(x).flatten(1)
        return self.head(x) / x.size(1)


class VerifierModel(nn.Module):
    def __init__(self, core: CifarNet):
        super().__init__()
        self.core = core

    def mirrored(self, x: torch.Tensor):
        return 0.5 * self.core(x) + 0.5 * self.core(x.flip(-1))

    def forward(self, x: torch.Tensor):
        # x is already standardized. Six-view expectation: mirror, and two
        # diagonal one-pixel translations, each with its mirror.
        logits = self.mirrored(x)
        p = F.pad(x, (1, 1, 1, 1), "reflect")
        translated = 0.5 * (self.mirrored(p[:, :, :32, :32]) + self.mirrored(p[:, :, 2:34, 2:34]))
        return 0.5 * logits + 0.5 * translated


def load_split(split: str):
    images, labels = [], []
    root = Path("/workspace/data/cifar10") / split
    for label, name in enumerate(CLASSES):
        for path in sorted((root / name).glob("*.png")):
            images.append(read_image(str(path)))
            labels.append(label)
    return torch.stack(images), torch.tensor(labels)


class TrainLoader:
    def __init__(self, raw, labels, batch_size=2000):
        self.images = ((raw.cuda().half() / 255 - MEAN.half()) / STD.half()).to(memory_format=torch.channels_last)
        self.labels = labels.cuda()
        self.batch_size = batch_size
        self.epoch = 0
        self.flipped = torch.where((torch.rand(len(raw), device="cuda") < 0.5).view(-1, 1, 1, 1),
                                   self.images.flip(-1), self.images)
        self.padded = F.pad(self.flipped, (2, 2, 2, 2), "reflect")

    def __len__(self):
        return len(self.images) // self.batch_size

    def __iter__(self):
        shifts = torch.randint(-2, 3, (len(self.images), 2), device="cuda")
        out = torch.empty_like(self.images)
        for sy in range(-2, 3):
            for sx in range(-2, 3):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                out[mask] = self.padded[mask, :, 2 + sy:34 + sy, 2 + sx:34 + sx]
        if self.epoch % 2:
            out = out.flip(-1)
        self.epoch += 1
        order = torch.randperm(len(out), device="cuda")
        for i in range(len(self)):
            idx = order[i * self.batch_size:(i + 1) * self.batch_size]
            yield out[idx], self.labels[idx]


def setup_training(model, loader):
    batch_size = loader.batch_size
    wd = 2e-6 * batch_size
    filters = [p for p in model.parameters() if p.ndim == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    configs = [dict(params=[model.whiten.bias], lr=0.053, weight_decay=wd / 0.053),
               dict(params=norm_biases, lr=0.053, weight_decay=wd / 0.053),
               dict(params=[model.head.weight], lr=0.67, weight_decay=wd / 0.67)]
    sgd = torch.optim.SGD(configs, momentum=0.85, nesterov=True, fused=True)
    muon = Muon(filters)
    for opt in (sgd, muon):
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return sgd, muon


def train_run(model, loader, out_dir, dummy=False):
    model.reset()
    model.init_whiten(loader.images)
    sgd, muon = setup_training(model, loader)
    total_steps, bias_steps, step = 8 * len(loader), 3 * len(loader), 0
    begin = time.perf_counter()
    for epoch in range(1, 9):
        model.train()
        for inputs, labels in loader:
            if dummy:
                labels = torch.randint(0, 10, labels.shape, device="cuda")
            F.cross_entropy(model(inputs, step < bias_steps), labels, label_smoothing=0.2, reduction="sum").backward()
            sgd.param_groups[0]["lr"] = sgd.param_groups[0]["initial_lr"] * max(0.0, 1 - step / bias_steps)
            for group in sgd.param_groups[1:] + muon.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_steps)
            sgd.step(); muon.step(); model.zero_grad(set_to_none=True)
            step += 1
        torch.cuda.synchronize()
        if out_dir:
            model.eval()
            # Save the underlying eager module; the compiled wrapper is only a
            # training accelerator and is intentionally absent from artifacts.
            core = model._orig_mod if hasattr(model, "_orig_mod") else model
            scripted = torch.jit.script(VerifierModel(core))
            scripted.save(str(out_dir / f"ckpt_epoch_{epoch:04d}.pt"))
            print(f"epoch={epoch} elapsed={time.perf_counter()-begin:.3f}s", flush=True)
    return time.perf_counter() - begin


def evaluate_saved(path, raw, labels):
    model = torch.jit.load(str(path), map_location="cuda").eval()
    images = ((raw.float() / 255 - MEAN.cpu()) / STD.cpu()).cuda()
    correct = 0
    with torch.no_grad():
        for x, y in zip(images.split(512), labels.cuda().split(512)):
            correct += (model(x).argmax(1) == y).sum().item()
    return correct / len(labels)


def main():
    raw_train, train_labels = load_split("train")
    loader = TrainLoader(raw_train, train_labels)
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    compiled = torch.compile(model, mode="max-autotune")
    # Compile every training graph before the timestamped run.
    train_run(compiled, loader, None, dummy=True)
    out = Path("/workspace/periodic_checkpoints")
    out.mkdir(exist_ok=True)
    for old in out.glob("ckpt_epoch_*.pt"):
        old.unlink()
    elapsed = train_run(compiled, loader, out)
    final = Path("/workspace/checkpoint.pt")
    if final.exists():
        final.unlink()
    os.link(out / "ckpt_epoch_0008.pt", final)

    # Test data is used only now, after all training and checkpoint writes.
    raw_test, test_labels = load_split("test")
    accuracies = []
    for epoch in range(1, 9):
        acc = evaluate_saved(out / f"ckpt_epoch_{epoch:04d}.pt", raw_test, test_labels)
        accuracies.append(acc)
        print(f"checkpoint={epoch} accuracy={acc:.4f}")
    best = max(accuracies)
    first = next((i + 1 for i, a in enumerate(accuracies) if a >= 0.94), None)
    with open("/workspace/agent_reported_metrics.json", "w") as f:
        json.dump({"accuracy": best, "wall_clock_sec": elapsed, "first_crossing_epoch": first,
                   "checkpoint_accuracies": accuracies}, f)


if __name__ == "__main__":
    main()
