import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F


torch.backends.cudnn.benchmark = True
MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device="cuda").view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616), device="cuda").view(1, 3, 1, 1)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def load_split(root: Path, split: str):
    cache = root / (split + "_tensor.pt")
    if cache.exists():
        data = torch.load(cache, map_location="cpu", weights_only=True)
        return data["images"], data["labels"]
    images, labels = [], []
    for label, name in enumerate(CLASSES):
        for path in sorted((root / split / name).glob("*.png")):
            images.append(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8))
            labels.append(label)
    data = {"images": torch.from_numpy(np.stack(images)), "labels": torch.tensor(labels)}
    torch.save(data, cache)
    return data["images"], data["labels"]


def flip_batch(x):
    mask = (torch.rand(len(x), device=x.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(mask, x.flip(-1), x)


def crop_batch(x, size=32):
    r = (x.size(-1) - size) // 2
    shifts = torch.randint(-r, r + 1, (len(x), 2), device=x.device)
    tmp = torch.empty((len(x), 3, size, size + 2 * r), device=x.device, dtype=x.dtype)
    out = torch.empty((len(x), 3, size, size), device=x.device, dtype=x.dtype)
    for s in range(-r, r + 1):
        mask = shifts[:, 0] == s
        tmp[mask] = x[mask, :, r+s:r+s+size, :]
    for s in range(-r, r + 1):
        mask = shifts[:, 1] == s
        out[mask] = tmp[mask, :, :, r+s:r+s+size]
    return out


class Loader:
    def __init__(self, images, labels, batch_size=2000):
        x = images.cuda().half().permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last) / 255
        self.base = ((x - MEAN.half()) / STD.half()).contiguous(memory_format=torch.channels_last)
        self.labels = labels.cuda()
        self.batch_size = batch_size
        self.epoch = 0
        self.flipped = flip_batch(self.base)

    def __len__(self):
        return len(self.base) // self.batch_size

    def __iter__(self):
        x = self.flipped if self.epoch % 2 == 0 else self.flipped.flip(-1)
        x = crop_batch(F.pad(x, (2, 2, 2, 2), mode="reflect"))
        self.epoch += 1
        order = torch.randperm(len(x), device=x.device)
        for start in range(0, len(x), self.batch_size):
            ids = order[start:start+self.batch_size]
            if len(ids) == self.batch_size:
                yield x[ids], self.labels[ids]


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, width):
        super().__init__(width, eps=1e-12, momentum=0.4)
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
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.norm1(self.pool(self.conv1(x))))
        return self.act(self.norm2(self.conv2(x)))


class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.whiten = nn.Conv2d(3, 24, 2, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(nn.GELU(), ConvGroup(24, 64), ConvGroup(64, 256),
                                    ConvGroup(256, 256), nn.MaxPool2d(3))
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

    @torch.no_grad()
    def init_whiten(self, images, eps=5e-4):
        patches = images.unfold(2, 2, 1).unfold(3, 2, 1).transpose(1, 3).reshape(-1, 3, 2, 2).float()
        flat = patches.view(len(patches), -1)
        vals, vecs = torch.linalg.eigh((flat.T @ flat) / len(flat), UPLO="U")
        filters = vecs.T.reshape(-1, 3, 2, 2) / torch.sqrt(vals.view(-1, 1, 1, 1) + eps)
        self.whiten.weight.copy_(torch.cat((filters, -filters)).half())

    def forward(self, x):
        x = x.to(dtype=self.whiten.weight.dtype, memory_format=torch.channels_last)
        x = self.layers(self.whiten(x))
        x = x.reshape(x.size(0), -1)
        return self.head(x) / x.size(-1)


class InferenceModel(nn.Module):
    def __init__(self, core: CifarNet):
        super().__init__()
        self.core = core

    def mirror(self, x):
        return 0.5 * self.core(x) + 0.5 * self.core(x.flip(-1))

    def forward(self, x):
        center = self.mirror(x)
        padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
        upper = self.mirror(padded[:, :, :32, :32])
        lower = self.mirror(padded[:, :, 2:34, 2:34])
        return 0.5 * center + 0.25 * upper + 0.25 * lower


@torch.compile
def zeropower(g, steps: int = 3):
    x = g.bfloat16()
    x = x / (x.norm() + 1e-7)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        a = x @ x.T
        x = 3.4445 * x + (-4.7750 * a + 2.0315 * a @ a) @ x
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
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(p.grad)
                buf = state["buf"]
                buf.mul_(group["momentum"]).add_(p.grad)
                grad = p.grad.add(buf, alpha=group["momentum"])
                p.mul_(len(p) ** 0.5 / p.norm())
                p.add_(zeropower(grad.reshape(len(grad), -1)).reshape_as(p), alpha=-group["lr"])


def optimizers(model, batch_size=2000):
    wd = 2e-6 * batch_size
    convs = [p for p in model.parameters() if p.ndim == 4 and p.requires_grad]
    norms = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    configs = [dict(params=[model.whiten.bias], lr=0.053, weight_decay=wd/0.053),
               dict(params=norms, lr=0.053, weight_decay=wd/0.053),
               dict(params=[model.head.weight], lr=0.67, weight_decay=wd/0.67)]
    sgd = torch.optim.SGD(configs, momentum=0.85, nesterov=True, fused=True)
    muon = Muon(convs)
    for opt in (sgd, muon):
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return sgd, muon


def train_run(model, compiled, train_images, train_labels, out_dir: Path, seed: int, save=True):
    torch.manual_seed(seed)
    model.reset()
    loader = Loader(train_images, train_labels)
    model.init_whiten(loader.base[:5000])
    sgd, muon = optimizers(model)
    total_steps = 8 * len(loader)
    bias_steps = 3 * len(loader)
    step = 0
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, 9):
        model.train()
        for inputs, labels in loader:
            logits = compiled(inputs)
            F.cross_entropy(logits, labels, label_smoothing=0.2, reduction="sum").backward()
            for group in sgd.param_groups[:1]:
                group["lr"] = group["initial_lr"] * max(0.0, 1 - step / bias_steps)
            for group in sgd.param_groups[1:] + muon.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_steps)
            sgd.step(); muon.step()
            model.zero_grad(set_to_none=True)
            step += 1
        if save:
            model.eval()
            scripted = torch.jit.script(InferenceModel(model))
            scripted.save(str(out_dir / f"ckpt_epoch_{epoch:04d}.pt"))
        print(f"epoch={epoch} train_acc={(logits.argmax(1)==labels).float().mean().item():.4f} elapsed={time.time()-started:.3f}", flush=True)
    return time.time() - started


@torch.inference_mode()
def evaluate(path: Path, images, labels):
    model = torch.jit.load(str(path), map_location="cuda").eval()
    x = images.cuda().float().permute(0, 3, 1, 2) / 255
    x = (x - MEAN) / STD
    labels = labels.cuda()
    correct = 0
    for start in range(0, len(x), 512):
        correct += (model(x[start:start+512]).argmax(1) == labels[start:start+512]).sum().item()
    return correct / len(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--out", type=Path, default=Path("/workspace/scratch_ckpt/run"))
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    root = Path("/workspace/data/cifar10")
    train_images, train_labels = load_split(root, "train")
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    compiled = torch.compile(model, mode="max-autotune")
    # Compile kernels with a disposable initialization before starting the measured run.
    train_run(model, compiled, train_images, torch.randint_like(train_labels, high=10), Path("/tmp/air_warm"), args.seed + 10000, save=False)
    if args.out.exists():
        shutil.rmtree(args.out)
    elapsed = train_run(model, compiled, train_images, train_labels, args.out, args.seed, save=True)
    test_images, test_labels = load_split(root, "test")
    scores = [evaluate(args.out / f"ckpt_epoch_{e:04d}.pt", test_images, test_labels) for e in range(1, 9)]
    print("accuracies", scores, "wall", elapsed, flush=True)
    if args.promote:
        graded = Path("/workspace/periodic_checkpoints")
        replacement = Path("/workspace/periodic_checkpoints_new")
        if replacement.exists():
            shutil.rmtree(replacement)
        shutil.copytree(args.out, replacement)
        if graded.exists():
            shutil.rmtree(graded)
        replacement.rename(graded)
        final = Path("/workspace/checkpoint.pt")
        if final.exists():
            final.unlink()
        os.link(graded / "ckpt_epoch_0008.pt", final)
        with open("/workspace/agent_reported_metrics.json", "w") as f:
            json.dump({"accuracy": scores[-1], "wall_clock_sec": elapsed}, f)


if __name__ == "__main__":
    main()
