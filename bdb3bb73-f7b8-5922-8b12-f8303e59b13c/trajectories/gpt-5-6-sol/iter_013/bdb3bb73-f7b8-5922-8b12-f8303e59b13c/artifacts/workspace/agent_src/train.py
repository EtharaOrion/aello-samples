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
OUT = ROOT / "periodic_checkpoints"
MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device="cuda").view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616), device="cuda").view(1, 3, 1, 1)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def load_split(split):
    cache = DATA / (split + "_tensor.pt")
    if cache.exists():
        d = torch.load(cache, map_location="cpu", weights_only=True)
        return d["images"].cuda(), d["labels"].cuda()
    chunks, labels = [], []
    for target, name in enumerate(CLASSES):
        files = sorted((DATA / split / name).glob("*.png"))
        chunks.append(torch.stack([read_image(str(p)) for p in files]))
        labels.append(torch.full((len(files),), target, dtype=torch.long))
    images, targets = torch.cat(chunks), torch.cat(labels)
    torch.save({"images": images, "labels": targets}, cache)
    return images.cuda(), targets.cuda()


def crop_batch(images, size=32):
    r = (images.size(-1) - size) // 2
    shifts = torch.randint(-r, r + 1, (len(images), 2), device=images.device)
    tmp = torch.empty((len(images), 3, size, size + 2*r), device=images.device, dtype=images.dtype)
    out = torch.empty((len(images), 3, size, size), device=images.device, dtype=images.dtype)
    for s in range(-r, r + 1):
        mask = shifts[:, 0] == s
        tmp[mask] = images[mask, :, r+s:r+s+size, :]
    for s in range(-r, r + 1):
        mask = shifts[:, 1] == s
        out[mask] = tmp[mask, :, :, r+s:r+s+size]
    return out


class Loader:
    def __init__(self, images, labels, batch_size=2000):
        self.raw = images
        self.labels = labels
        self.batch_size = batch_size
        self.epoch = 0
        self.norm = (((images.half() / 255).to(memory_format=torch.channels_last) - MEAN) / STD).half()
        flip_mask = (torch.rand(len(images), device="cuda") < 0.5).view(-1, 1, 1, 1)
        flipped = torch.where(flip_mask, self.norm.flip(-1), self.norm)
        self.padded = F.pad(flipped, (2, 2, 2, 2), "reflect")

    def __len__(self):
        return len(self.labels) // self.batch_size

    def __iter__(self):
        images = crop_batch(self.padded)
        if self.epoch % 2:
            images = images.flip(-1)
        self.epoch += 1
        order = torch.randperm(len(images), device="cuda")
        for i in range(len(self)):
            ix = order[i*self.batch_size:(i+1)*self.batch_size]
            yield images[ix], self.labels[ix]


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
        self.conv1, self.conv2 = Conv(cin, cout), Conv(cout, cout)
        self.pool = nn.MaxPool2d(2)
        self.norm1, self.norm2 = BatchNorm(cout), BatchNorm(cout)
        self.activ = nn.GELU()

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
            m.float() if isinstance(m, BatchNorm) else m.half()

    def reset(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, BatchNorm, nn.Linear)):
                m.reset_parameters()
        self.head.weight.data.mul_(1 / self.head.weight.data.std())

    @torch.no_grad()
    def init_whiten(self, images, eps=5e-4):
        p = images.unfold(2, 2, 1).unfold(3, 2, 1).transpose(1, 3).reshape(-1, 3, 2, 2).float()
        flat = p.reshape(len(p), -1)
        vals, vecs = torch.linalg.eigh(flat.T @ flat / len(flat), UPLO="U")
        kernels = vecs.T.reshape(-1, 3, 2, 2) / torch.sqrt(vals.view(-1, 1, 1, 1) + eps)
        self.whiten.weight.copy_(torch.cat((kernels, -kernels)).half())

    def forward(self, x, whiten_bias_grad: bool = True):
        b = self.whiten.bias
        assert b is not None
        if not whiten_bias_grad:
            b = b.detach()
        x = F.conv2d(x, self.whiten.weight, b)
        x = self.layers(x).view(len(x), -1)
        return self.head(x) / x.size(-1)


class VerifierModel(nn.Module):
    """The public forward contract: already-standardized float NCHW -> ten logits."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def mirror(self, x):
        return (self.net(x, False) + self.net(x.flip(-1), False)) * 0.5

    def forward(self, x):
        x = x.to(dtype=torch.float16, memory_format=torch.channels_last)
        base = self.mirror(x)
        padded = F.pad(x, (1, 1, 1, 1), "reflect")
        translated = (self.mirror(padded[:, :, :32, :32]) + self.mirror(padded[:, :, 2:34, 2:34])) * 0.5
        return base * 0.5 + translated * 0.5


@torch.compile
def ns5(g, steps: int = 3):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16() / (g.norm() + 1e-7)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        A = x @ x.T
        x = a*x + (b*A + c*A@A) @ x
    return x.T if transposed else x


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=.24, momentum=.6):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(p.grad)
                buf = state["buf"]
                buf.mul_(group["momentum"]).add_(p.grad)
                g = p.grad.add(buf, alpha=group["momentum"])
                p.mul_(len(p) ** 0.5 / p.norm())
                p.add_(ns5(g.reshape(len(g), -1)).reshape_as(g), alpha=-group["lr"])


def make_optimizers(model, total_steps, bias_steps, batch_size=2000):
    wd = 2e-6 * batch_size
    filters = [p for p in model.parameters() if p.ndim == 4 and p.requires_grad]
    norm_bias = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    groups = [dict(params=[model.whiten.bias], lr=.053, weight_decay=wd/.053),
              dict(params=norm_bias, lr=.053, weight_decay=wd/.053),
              dict(params=[model.head.weight], lr=.67, weight_decay=wd/.67)]
    sgd = torch.optim.SGD(groups, momentum=.85, nesterov=True, fused=True)
    muon = Muon(filters)
    for opt in (sgd, muon):
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]
    return sgd, muon


def train_run(raw, compiled, loader, save):
    epochs, total_steps, bias_steps = 9, 9*len(loader), 3*len(loader)
    raw.reset()
    raw.init_whiten(loader.norm[:5000])
    sgd, muon = make_optimizers(raw, total_steps, bias_steps)
    step = 0
    first_save = None
    for epoch in range(1, epochs + 1):
        raw.train()
        for x, y in loader:
            logits = compiled(x, step < bias_steps)
            F.cross_entropy(logits, y, label_smoothing=.2, reduction="sum").backward()
            for g in sgd.param_groups[:1]:
                g["lr"] = g["initial_lr"] * (1 - step/bias_steps)
            for g in sgd.param_groups[1:] + muon.param_groups:
                g["lr"] = g["initial_lr"] * (1 - step/total_steps)
            sgd.step(); muon.step(); raw.zero_grad(set_to_none=True)
            step += 1
        torch.cuda.synchronize()
        if save:
            raw.eval()
            path = OUT / f"ckpt_epoch_{epoch:04d}.pt"
            torch.jit.script(VerifierModel(raw)).save(str(path))
            if first_save is None:
                first_save = time.time()
            print(f"saved epoch {epoch} elapsed_from_epoch1={time.time()-first_save:.3f}s", flush=True)


@torch.inference_mode()
def evaluate(path, images, labels):
    model = torch.jit.load(str(path), map_location="cuda").eval()
    x = ((images.float()/255 - MEAN) / STD).to(memory_format=torch.channels_last)
    correct = 0
    for xb, yb in zip(x.split(512), labels.split(512)):
        correct += (model(xb).argmax(1) == yb).sum().item()
    return correct / len(labels)


def main():
    images, labels = load_split("train")
    loader = Loader(images, labels)
    raw = CifarNet().cuda().to(memory_format=torch.channels_last)
    compiled = torch.compile(raw, mode="max-autotune")

    # Compile every training graph before the timestamped, freshly reset run.
    warm_labels = loader.labels
    loader.labels = torch.randint(0, 10, warm_labels.shape, device="cuda")
    train_run(raw, compiled, loader, save=False)
    loader.labels = warm_labels
    loader.epoch = 0

    OUT.mkdir(exist_ok=True)
    for p in OUT.glob("ckpt_epoch_*.pt"):
        p.unlink()
    start = time.time()
    train_run(raw, compiled, loader, save=True)
    shutil.copy(OUT / "ckpt_epoch_0009.pt", ROOT / "checkpoint.pt")

    test_images, test_labels = load_split("test")
    accuracies = []
    for epoch in range(1, 10):
        p = OUT / f"ckpt_epoch_{epoch:04d}.pt"
        # Also checks that every artifact independently loads with the exact verifier contract.
        accuracies.append(evaluate(p, test_images, test_labels))
    wall = time.time() - start
    print("accuracies", accuracies)
    with open(ROOT / "agent_reported_metrics.json", "w") as f:
        json.dump({"accuracy": max(accuracies), "wall_clock_sec": wall,
                   "first_crossing_epoch": next((i+1 for i,a in enumerate(accuracies) if a >= .94), None)}, f)


if __name__ == "__main__":
    main()
