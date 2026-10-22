import json
import math
import os
import shutil
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.io import read_image


ROOT = Path("/workspace")
DATA = ROOT / "data" / "cifar10"
OUT = ROOT / "periodic_checkpoints"
MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device="cuda").view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616), device="cuda").view(1, 3, 1, 1)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

torch.backends.cudnn.benchmark = True


@torch.compile
def zeropower(g: torch.Tensor, steps: int = 3, eps: float = 1e-7):
    x = g.bfloat16()
    x = x / (x.norm() + eps)
    transposed = g.size(0) > g.size(1)
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
    def __init__(self, n):
        super().__init__(n, eps=1e-12, momentum=0.4)
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
            ConvGroup(256, 256), nn.MaxPool2d(3)
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

    @torch.no_grad()
    def init_whiten(self, images):
        patches = images.unfold(2, 2, 1).unfold(3, 2, 1).transpose(1, 3).reshape(-1, 3, 2, 2).float()
        flat = patches.view(len(patches), -1)
        vals, vecs = torch.linalg.eigh((flat.T @ flat) / len(flat), UPLO="U")
        filters = vecs.T.reshape(-1, 3, 2, 2) / torch.sqrt(vals.view(-1, 1, 1, 1) + 5e-4)
        self.whiten.weight.data.copy_(torch.cat((filters, -filters)).half())

    def forward(self, x: torch.Tensor, whiten_bias_grad: bool = True):
        stored_bias = self.whiten.bias
        if stored_bias is None:
            raise RuntimeError("whitening convolution requires bias")
        bias = stored_bias if whiten_bias_grad else stored_bias.detach()
        x = F.conv2d(x, self.whiten.weight, bias)
        x = self.layers(x).view(len(x), -1)
        return self.head(x) / x.size(-1)


class VerifierModel(nn.Module):
    def __init__(self, core: CifarNet):
        super().__init__()
        self.core = core

    def mirror(self, x: torch.Tensor):
        return 0.5 * self.core(x, False) + 0.5 * self.core(x.flip(-1), False)

    def forward(self, x: torch.Tensor):
        # x is already standardized by the evaluator.
        x = x.half().contiguous(memory_format=torch.channels_last)
        logits = self.mirror(x)
        padded = F.pad(x, (1, 1, 1, 1), "reflect")
        shifted = 0.5 * (self.mirror(padded[:, :, :32, :32]) + self.mirror(padded[:, :, 2:34, 2:34]))
        return 0.5 * logits + 0.5 * shifted


def load_split(name: str):
    cache = DATA / (name + ".pt")
    if cache.exists():
        obj = torch.load(cache, map_location="cpu", weights_only=True)
        return obj["images"], obj["labels"]
    images, labels = [], []
    for label, cls in enumerate(CLASSES):
        for path in sorted((DATA / name / cls).glob("*.png")):
            images.append(read_image(str(path)))
            labels.append(label)
    result = {"images": torch.stack(images), "labels": torch.tensor(labels)}
    torch.save(result, cache)
    return result["images"], result["labels"]


class TrainLoader:
    def __init__(self, images, labels, batch_size=2000):
        self.images = ((images.cuda().half() / 255) - MEAN.half()) / STD.half()
        self.images = self.images.contiguous(memory_format=torch.channels_last)
        self.labels = labels.cuda()
        self.batch_size = batch_size
        self.epoch = 0
        self.padded = None

    def __len__(self):
        return len(self.images) // self.batch_size

    def __iter__(self):
        if self.padded is None:
            flipped = torch.where(
                (torch.rand(len(self.images), device="cuda") < 0.5).view(-1, 1, 1, 1),
                self.images.flip(-1), self.images
            )
            self.padded = F.pad(flipped, (2, 2, 2, 2), "reflect")
        shifts = torch.randint(-2, 3, (len(self.images), 2), device="cuda")
        cropped = torch.empty_like(self.images)
        for sy in range(-2, 3):
            for sx in range(-2, 3):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                cropped[mask] = self.padded[mask, :, 2 + sy:34 + sy, 2 + sx:34 + sx]
        if self.epoch % 2:
            cropped = cropped.flip(-1)
        self.epoch += 1
        indices = torch.randperm(len(cropped), device="cuda")
        for start in range(0, len(cropped), self.batch_size):
            idx = indices[start:start + self.batch_size]
            yield cropped[idx], self.labels[idx]


def save_model(model, epoch, directory):
    model.eval()
    scripted = torch.jit.script(VerifierModel(model).eval())
    scripted.save(str(directory / f"ckpt_epoch_{epoch:04d}.pt"))


def train_run(model, compiled, images, labels, directory=None, epochs=8, dummy=False):
    loader = TrainLoader(images, labels)
    if dummy:
        loader.labels = torch.randint(0, 10, loader.labels.shape, device="cuda")
    steps_per_epoch = len(loader)
    total_steps = epochs * steps_per_epoch
    bias_steps = 3 * steps_per_epoch
    conv_params = [p for p in model.parameters() if p.requires_grad and p.ndim == 4]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    wd = 2e-6 * loader.batch_size
    groups = [
        dict(params=[model.whiten.bias], lr=0.053, weight_decay=wd / 0.053),
        dict(params=norm_biases, lr=0.053, weight_decay=wd / 0.053),
        dict(params=[model.head.weight], lr=0.67, weight_decay=wd / 0.67),
    ]
    sgd = torch.optim.SGD(groups, momentum=0.85, nesterov=True, fused=True)
    muon = Muon(conv_params)
    for opt in (sgd, muon):
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    model.reset()
    normalized = ((images[:5000].cuda().half() / 255) - MEAN.half()) / STD.half()
    model.init_whiten(normalized)
    step = 0
    model.train()
    for epoch in range(1, epochs + 1):
        for x, y in loader:
            loss = F.cross_entropy(compiled(x, step < bias_steps), y, label_smoothing=0.2, reduction="sum")
            loss.backward()
            sgd.param_groups[0]["lr"] = 0.053 * max(0.0, 1 - step / bias_steps)
            for group in sgd.param_groups[1:]:
                group["lr"] = group["initial_lr"] * (1 - step / total_steps)
            for group in muon.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_steps)
            sgd.step(); muon.step(); model.zero_grad(set_to_none=True)
            step += 1
        torch.cuda.synchronize()
        if directory is not None:
            save_model(model, epoch, directory)
        model.train()
        print(f"epoch={epoch} loss={loss.item() / len(y):.4f}", flush=True)


@torch.no_grad()
def evaluate(path, images, labels):
    model = torch.jit.load(str(path), map_location="cuda").eval()
    correct = 0
    for start in range(0, len(images), 512):
        x = images[start:start + 512].cuda().float() / 255
        x = (x - MEAN) / STD
        correct += (model(x).argmax(1).cpu() == labels[start:start + 512]).sum().item()
    return correct / len(images)


def main():
    torch.manual_seed(0)
    train_images, train_labels = load_split("train")
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    compiled = torch.compile(model, mode="max-autotune")
    print("compilation warm-up", flush=True)
    train_run(model, compiled, train_images, train_labels, dummy=True)
    scratch = ROOT / "scratch_ckpt"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir()
    torch.manual_seed(0)
    run_start = time.time()
    print("graded run", flush=True)
    train_run(model, compiled, train_images, train_labels, directory=scratch)
    for old in OUT.glob("*.pt"):
        old.unlink()
    OUT.mkdir(exist_ok=True)
    for fresh in sorted(scratch.glob("*.pt")):
        os.replace(fresh, OUT / fresh.name)
    scratch.rmdir()
    final_path = OUT / "ckpt_epoch_0008.pt"
    shutil.copy(final_path, ROOT / "checkpoint.pt")
    run_elapsed = time.time() - run_start
    test_images, test_labels = load_split("test")
    accuracy = evaluate(final_path, test_images, test_labels)
    with open(ROOT / "agent_reported_metrics.json", "w") as f:
        json.dump({"accuracy": accuracy, "wall_clock_sec": run_elapsed}, f)
    print(f"final accuracy={accuracy:.4f} run_wall_clock={run_elapsed:.3f}s", flush=True)


if __name__ == "__main__":
    main()
