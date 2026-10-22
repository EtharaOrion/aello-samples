import json
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
DATA = ROOT / "data" / "cifar10"
OUT = Path(os.environ.get("RUN_OUT", str(ROOT / "periodic_checkpoints")))
MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device="cuda").view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616), device="cuda").view(1, 3, 1, 1)


def load_split(name):
    classes = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
    images, labels = [], []
    for label, cls in enumerate(classes):
        files = sorted((DATA / name / cls).glob("*.png"))
        images.extend(read_image(str(p)) for p in files)
        labels.extend([label] * len(files))
    x = torch.stack(images).cuda().half().div_(255).to(memory_format=torch.channels_last)
    y = torch.tensor(labels, device="cuda")
    return x, y


def batch_flip_lr(x):
    mask = (torch.rand(len(x), device=x.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(mask, x.flip(-1), x)


def batch_crop(images, size=32, pad=2):
    shifts = torch.randint(-pad, pad + 1, (len(images), 2), device=images.device)
    out = torch.empty((len(images), 3, size, size), device=images.device,
                      dtype=images.dtype, memory_format=torch.channels_last)
    for sy in range(-pad, pad + 1):
        for sx in range(-pad, pad + 1):
            mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
            out[mask] = images[mask, :, pad + sy:pad + sy + size,
                               pad + sx:pad + sx + size]
    return out


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, channels):
        super().__init__(channels, eps=1e-12, momentum=0.4)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, cin, cout):
        super().__init__(cin, cout, 3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.dirac_(self.weight.data[:self.weight.size(1)])


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
        patches = images.unfold(2, 2, 1).unfold(3, 2, 1).transpose(1, 3)
        patches = patches.reshape(-1, 3, 2, 2).float()
        flat = patches.view(len(patches), -1)
        covariance = flat.T @ flat / len(flat)
        values, vectors = torch.linalg.eigh(covariance, UPLO="U")
        filters = vectors.T.reshape(-1, 3, 2, 2) / torch.sqrt(values.view(-1, 1, 1, 1) + eps)
        self.whiten.weight.data[:] = torch.cat((filters, -filters)).half()

    def forward(self, x, whiten_bias_grad: bool = True):
        raw_bias = self.whiten.bias
        assert raw_bias is not None
        bias = raw_bias if whiten_bias_grad else raw_bias.detach()
        x = F.conv2d(x, self.whiten.weight, bias)
        x = self.layers(x)
        x = x.view(len(x), -1)
        return self.head(x) / x.size(-1)


class ExportModel(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def mirrored(self, x):
        return 0.5 * self.core(x) + 0.5 * self.core(x.flip(-1))

    def forward(self, x):
        x = x.half().contiguous(memory_format=torch.channels_last)
        logits = self.mirrored(x)
        padded = F.pad(x, (1, 1, 1, 1), "reflect")
        shifted = 0.5 * (self.mirrored(padded[:, :, :32, :32]) +
                         self.mirrored(padded[:, :, 2:34, 2:34]))
        return 0.5 * logits + 0.5 * shifted


@torch.compile
def zeropower(g, steps: int = 3, eps: float = 1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    x = x / (x.norm() + eps)
    transposed = g.size(0) > g.size(1)
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
                update = zeropower(g.reshape(len(g), -1)).view_as(g)
                p.add_(update, alpha=-group["lr"])


def save_epoch(core, epoch, out_dir):
    core.eval()
    exported = ExportModel(core)
    scripted = torch.jit.script(exported)
    scripted.save(str(out_dir / f"ckpt_epoch_{epoch:04d}.pt"))


def accuracy(path, test_x, test_y):
    model = torch.jit.load(str(path), map_location="cuda").eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(test_x), 512):
            standardized = (test_x[i:i+512].float() - MEAN) / STD
            correct += (model(standardized).argmax(1) == test_y[i:i+512]).sum().item()
    return correct / len(test_y)


def main():
    torch.manual_seed(int(os.environ.get("RUN_SEED", "0")))
    train_x, train_y = load_split("train")
    test_x, test_y = load_split("test")
    normalized = (train_x - MEAN.half()) / STD.half()
    flipped = batch_flip_lr(normalized)
    padded = F.pad(flipped, (2, 2, 2, 2), "reflect")

    OUT.mkdir(exist_ok=True)
    for p in OUT.glob("*.pt"):
        p.unlink()
    graded = OUT == ROOT / "periodic_checkpoints"
    final_path = ROOT / "checkpoint.pt"
    if graded and final_path.exists():
        final_path.unlink()

    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    model.reset()
    model.init_whiten(normalized[:5000])
    train_model = model

    batch_size = 2000
    steps_per_epoch = len(train_x) // batch_size
    total_steps = 8 * steps_per_epoch
    bias_steps = 3 * steps_per_epoch
    wd = 2e-6 * batch_size
    conv_params = [p for p in model.parameters() if p.ndim == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    configs = [dict(params=[model.whiten.bias], lr=0.053, weight_decay=wd / 0.053),
               dict(params=norm_biases, lr=0.053, weight_decay=wd / 0.053),
               dict(params=[model.head.weight], lr=0.67, weight_decay=wd / 0.67)]
    sgd = torch.optim.SGD(configs, momentum=0.85, nesterov=True, fused=True)
    muon = Muon(conv_params)
    for opt in (sgd, muon):
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    run_start = time.time()
    step = 0
    for epoch in range(1, 9):
        model.train()
        images = batch_crop(padded)
        if epoch % 2 == 0:
            images = images.flip(-1)
        order = torch.randperm(len(train_x), device="cuda")
        for offset in range(0, len(train_x), batch_size):
            idx = order[offset:offset + batch_size]
            outputs = train_model(images[idx], step < bias_steps)
            F.cross_entropy(outputs, train_y[idx], label_smoothing=0.2,
                            reduction="sum").backward()
            sgd.param_groups[0]["lr"] = 0.053 * max(0.0, 1 - step / bias_steps)
            for group in sgd.param_groups[1:]:
                group["lr"] = group["initial_lr"] * (1 - step / total_steps)
            muon.param_groups[0]["lr"] = 0.24 * (1 - step / total_steps)
            sgd.step()
            muon.step()
            model.zero_grad(set_to_none=True)
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise RuntimeError(f"non-finite parameters after step {step}, loss={outputs.float().abs().max().item()}")
            step += 1
        save_epoch(model, epoch, OUT)
        print(f"saved epoch {epoch}", flush=True)

    if graded:
        shutil.copy(OUT / "ckpt_epoch_0008.pt", final_path)
    run_wall = time.time() - run_start
    scores = []
    for epoch in range(1, 9):
        score = accuracy(OUT / f"ckpt_epoch_{epoch:04d}.pt", test_x, test_y)
        scores.append(score)
        print(f"epoch {epoch}: {score:.4%}", flush=True)
    if graded:
        with open(ROOT / "agent_reported_metrics.json", "w") as f:
            json.dump({"accuracy": scores[-1], "wall_clock_sec": run_wall}, f)
    print(f"training wall {run_wall:.3f}s; best {max(scores):.4%}")


if __name__ == "__main__":
    main()
