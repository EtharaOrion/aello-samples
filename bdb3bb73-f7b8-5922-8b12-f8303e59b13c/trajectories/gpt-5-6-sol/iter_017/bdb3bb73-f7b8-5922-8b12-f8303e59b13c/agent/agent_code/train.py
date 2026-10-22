import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

ROOT = Path("/workspace")
TRAIN_DIR = ROOT / "data" / "cifar10" / "train"
OUT_DIR = ROOT / "periodic_checkpoints"
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device="cuda").view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616), device="cuda").view(1, 3, 1, 1)


def load_training_images():
    cache = ROOT / "train_cache.pt"
    if cache.exists():
        item = torch.load(cache, map_location="cpu")
        return item["images"], item["labels"]
    images, labels = [], []
    for label, name in enumerate(CLASSES):
        for path in sorted((TRAIN_DIR / name).glob("*.png")):
            images.append(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8))
            labels.append(label)
    images = torch.from_numpy(np.stack(images))
    labels = torch.tensor(labels, dtype=torch.long)
    torch.save({"images": images, "labels": labels}, cache)
    return images, labels


def batch_crop(images, crop_size=32):
    r = (images.size(-1) - crop_size) // 2
    shifts = torch.randint(-r, r + 1, (len(images), 2), device=images.device)
    out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device,
                      dtype=images.dtype, memory_format=torch.channels_last)
    for sy in range(-r, r + 1):
        for sx in range(-r, r + 1):
            mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
            out[mask] = images[mask, :, r + sy:r + sy + crop_size, r + sx:r + sx + crop_size]
    return out


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
        self.layers = nn.Sequential(nn.GELU(), ConvGroup(24, 64), ConvGroup(64, 256),
                                    ConvGroup(256, 256), nn.MaxPool2d(3))
        self.head = nn.Linear(256, 10, bias=False)
        for mod in self.modules():
            if isinstance(mod, BatchNorm):
                mod.float()
            else:
                mod.half()

    def reset_all(self):
        for mod in self.modules():
            if type(mod) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                mod.reset_parameters()
        self.head.weight.data.mul_(1 / self.head.weight.data.std())

    def init_whiten(self, images):
        patches = images.unfold(2, 2, 1).unfold(3, 2, 1).transpose(1, 3).reshape(-1, 3, 2, 2).float()
        flat = patches.view(len(patches), -1)
        covariance = flat.T @ flat / len(flat)
        values, vectors = torch.linalg.eigh(covariance, UPLO="U")
        kernels = vectors.T.reshape(-1, 3, 2, 2) / torch.sqrt(values.view(-1, 1, 1, 1) + 5e-4)
        self.whiten.weight.data[:] = torch.cat((kernels, -kernels)).half()

    def forward(self, x, whiten_bias_grad: bool = False):
        stored_bias = self.whiten.bias
        assert stored_bias is not None
        bias = stored_bias if whiten_bias_grad else stored_bias.detach()
        x = F.conv2d(x, self.whiten.weight, bias)
        x = self.layers(x)
        x = x.view(len(x), -1)
        return self.head(x) / x.size(-1)


class VerifierModel(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def mirror(self, x):
        return 0.5 * self.core(x, False) + 0.5 * self.core(x.flip(-1), False)

    def forward(self, x):
        x = x.half().contiguous(memory_format=torch.channels_last)
        logits = self.mirror(x)
        padded = F.pad(x, (1, 1, 1, 1), "reflect")
        shifted = 0.5 * (self.mirror(padded[:, :, :32, :32]) +
                         self.mirror(padded[:, :, 2:34, 2:34]))
        return (0.5 * logits + 0.5 * shifted).float()


@torch.compile
def zeropower(g, steps: int = 3):
    x = g.bfloat16()
    x /= x.norm() + 1e-7
    transposed = g.size(0) > g.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        a = x @ x.T
        b = -4.7750 * a + 2.0315 * a @ a
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
                grad = p.grad.add(buf, alpha=group["momentum"])
                p.data.mul_(len(p.data) ** 0.5 / p.data.norm())
                update = zeropower(grad.reshape(len(grad), -1)).view_as(grad)
                p.data.add_(update, alpha=-group["lr"])


def save_checkpoint(model, epoch):
    model.eval()
    wrapper = VerifierModel(model)
    scripted = torch.jit.script(wrapper)
    scripted.save(str(OUT_DIR / f"ckpt_epoch_{epoch:04d}.pt"))


def main():
    torch.manual_seed(int(os.environ.get("TRAIN_SEED", "0")))
    cpu_images, cpu_labels = load_training_images()
    images = cpu_images.cuda().half().div_(255).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
    labels = cpu_labels.cuda()
    images = (images - MEAN.half()) / STD.half()
    flipped = torch.where((torch.rand(len(images), device="cuda") < 0.5).view(-1, 1, 1, 1),
                          images.flip(-1), images)
    padded = F.pad(flipped, (2, 2, 2, 2), "reflect")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.pt"):
        old.unlink()
    final_path = ROOT / "checkpoint.pt"
    if final_path.exists():
        final_path.unlink()

    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    model.reset_all()
    model.init_whiten(images[:5000])
    compiled = torch.compile(model, mode="max-autotune")

    batch_size = 2000
    steps_per_epoch = len(images) // batch_size
    total_steps = 10 * steps_per_epoch
    bias_steps = 3 * steps_per_epoch
    conv_params = [p for p in model.parameters() if p.ndim == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    sgd = torch.optim.SGD([
        {"params": [model.whiten.bias], "lr": 0.053, "weight_decay": 0.004 / 0.053},
        {"params": norm_biases, "lr": 0.053, "weight_decay": 0.004 / 0.053},
        {"params": [model.head.weight], "lr": 0.67, "weight_decay": 0.004 / 0.67},
    ], momentum=0.85, nesterov=True, fused=True)
    muon = Muon(conv_params)
    initial_lrs = [g["lr"] for g in sgd.param_groups]
    step = 0
    run_start = time.perf_counter()

    for epoch in range(1, 11):
        model.train()
        aug = batch_crop(padded)
        if epoch % 2 == 0:
            aug = aug.flip(-1)
        order = torch.randperm(len(images), device="cuda")
        for batch in range(steps_per_epoch):
            idx = order[batch * batch_size:(batch + 1) * batch_size]
            logits = compiled(aug[idx], step < bias_steps)
            F.cross_entropy(logits, labels[idx], label_smoothing=0.2, reduction="sum").backward()
            sgd.param_groups[0]["lr"] = initial_lrs[0] * max(0.0, 1 - step / bias_steps)
            for i in (1, 2):
                sgd.param_groups[i]["lr"] = initial_lrs[i] * (1 - step / total_steps)
            muon.param_groups[0]["lr"] = 0.24 * (1 - step / total_steps)
            sgd.step()
            muon.step()
            model.zero_grad(set_to_none=True)
            step += 1
        torch.cuda.synchronize()
        save_checkpoint(model, epoch)
        print(f"epoch={epoch} elapsed={time.perf_counter() - run_start:.3f}s", flush=True)

    os.link(OUT_DIR / "ckpt_epoch_0010.pt", final_path)


if __name__ == "__main__":
    main()
