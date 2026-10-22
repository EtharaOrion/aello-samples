import json
import math
import os
import shutil
import time

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.transforms import PILToTensor


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
ROOT = "/workspace"
OUT = os.path.join(ROOT, "periodic_checkpoints")
MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device="cuda").view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616), device="cuda").view(1, 3, 1, 1)


@torch.compile
def zeropower(g: torch.Tensor, steps: int = 3):
    x = g.bfloat16()
    x /= x.norm() + 1e-7
    if g.size(0) > g.size(1):
        x = x.T
    for _ in range(steps):
        a = x @ x.T
        b = -4.7750 * a + 2.0315 * (a @ a)
        x = 3.4445 * x + b @ x
    if g.size(0) > g.size(1):
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

    def reset(self):
        for mod in self.modules():
            if type(mod) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                mod.reset_parameters()
        self.head.weight.data /= self.head.weight.data.std()

    @torch.no_grad()
    def init_whiten(self, images):
        patches = images.unfold(2, 2, 1).unfold(3, 2, 1).transpose(1, 3).reshape(-1, 3, 2, 2).float()
        flat = patches.reshape(len(patches), -1)
        cov = flat.T @ flat / len(flat)
        vals, vecs = torch.linalg.eigh(cov, UPLO="U")
        filt = vecs.T.reshape(-1, 3, 2, 2) / torch.sqrt(vals.view(-1, 1, 1, 1) + 5e-4)
        self.whiten.weight[:] = torch.cat((filt, -filt)).half()

    def forward(self, x: torch.Tensor, whiten_bias_grad: bool = True):
        raw_bias = self.whiten.bias
        assert raw_bias is not None
        bias = raw_bias if whiten_bias_grad else raw_bias.detach()
        x = F.conv2d(x, self.whiten.weight, bias)
        x = self.layers(x).reshape(len(x), -1)
        return self.head(x) / x.size(-1)


class VerifierModel(nn.Module):
    def __init__(self, net: CifarNet):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor):
        x = x.half().contiguous(memory_format=torch.channels_last)
        logits = self.net(x, False)
        logits = logits + self.net(x.flip(-1), False)
        p = F.pad(x, (1, 1, 1, 1), "reflect")
        a = p[:, :, 0:32, 0:32]
        b = p[:, :, 2:34, 2:34]
        logits = logits + 0.5 * (self.net(a, False) + self.net(a.flip(-1), False))
        logits = logits + 0.5 * (self.net(b, False) + self.net(b.flip(-1), False))
        return logits.float() * 0.25


def load_split(name):
    ds = ImageFolder(os.path.join(ROOT, "data", "cifar10", name))
    conv = PILToTensor()
    images = torch.empty((len(ds), 3, 32, 32), dtype=torch.uint8)
    labels = torch.empty(len(ds), dtype=torch.long)
    for i, (path, y) in enumerate(ds.samples):
        from PIL import Image
        with Image.open(path) as im:
            images[i] = conv(im.convert("RGB"))
        labels[i] = y
    return images, labels


def crop(images, pad=2):
    padded = F.pad(images, (pad,) * 4, "reflect")
    shifts = torch.randint(0, 2 * pad + 1, (len(images), 2), device=images.device)
    out = torch.empty_like(images)
    for sy in range(2 * pad + 1):
        for sx in range(2 * pad + 1):
            mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
            out[mask] = padded[mask, :, sy:sy + 32, sx:sx + 32]
    return out


def save_model(net, epoch):
    net.eval()
    wrapper = VerifierModel(net)
    scripted = torch.jit.script(wrapper)
    scripted.save(os.path.join(OUT, f"ckpt_epoch_{epoch:04d}.pt"))


def main():
    torch.manual_seed(15)
    raw_images, raw_labels = load_split("train")
    images = raw_images.cuda().half().div_(255).sub_(MEAN.half()).div_(STD.half())
    images = images.contiguous(memory_format=torch.channels_last)
    labels = raw_labels.cuda()
    del raw_images, raw_labels

    os.makedirs(OUT, exist_ok=True)
    for name in os.listdir(OUT):
        if name.endswith(".pt"):
            os.remove(os.path.join(OUT, name))

    net = CifarNet().cuda().to(memory_format=torch.channels_last)
    net.reset()
    net.init_whiten(images[:5000])
    compiled = torch.compile(net, mode="max-autotune")
    batch_size = 2000
    batches = len(images) // batch_size
    total_steps = 8 * batches
    bias_steps = 3 * batches
    conv_params = [p for p in net.parameters() if p.ndim == 4 and p.requires_grad]
    norm_biases = [p for n, p in net.named_parameters() if "norm" in n and p.requires_grad]
    configs = [dict(params=[net.whiten.bias], lr=0.053, weight_decay=0.004 / 0.053),
               dict(params=norm_biases, lr=0.053, weight_decay=0.004 / 0.053),
               dict(params=[net.head.weight], lr=0.67, weight_decay=0.004 / 0.67)]
    opt1 = torch.optim.SGD(configs, momentum=0.85, nesterov=True, fused=True)
    opt2 = Muon(conv_params)
    for opt in (opt1, opt2):
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    flipped = images[torch.rand(len(images), device="cuda") < 0.5]
    # Pre-flip a random half in-place, matching fixed-per-example augmentation.
    mask = torch.rand(len(images), device="cuda") < 0.5
    images[mask] = images[mask].flip(-1)
    del flipped
    step = 0
    run_start = time.time()
    for epoch in range(1, 9):
        net.train()
        aug = crop(images)
        if epoch % 2 == 0:
            aug = aug.flip(-1)
        order = torch.randperm(len(images), device="cuda")
        for j in range(batches):
            idx = order[j * batch_size:(j + 1) * batch_size]
            output = compiled(aug[idx], step < bias_steps)
            F.cross_entropy(output, labels[idx], label_smoothing=0.2, reduction="sum").backward()
            opt1.param_groups[0]["lr"] = 0.053 * max(0.0, 1 - step / bias_steps)
            for group in opt1.param_groups[1:]:
                group["lr"] = group["initial_lr"] * (1 - step / total_steps)
            opt2.param_groups[0]["lr"] = 0.24 * (1 - step / total_steps)
            opt1.step(); opt2.step()
            net.zero_grad(set_to_none=True)
            step += 1
        torch.cuda.synchronize()
        save_model(net, epoch)
        print(f"saved epoch {epoch} elapsed={time.time()-run_start:.3f}", flush=True)

    final_path = os.path.join(ROOT, "checkpoint.pt")
    if os.path.exists(final_path):
        os.remove(final_path)
    shutil.copy(os.path.join(OUT, "ckpt_epoch_0008.pt"), final_path)
    elapsed = time.time() - run_start
    with open(os.path.join(ROOT, "agent_reported_metrics.json"), "w") as f:
        json.dump({"accuracy": 0.0, "wall_clock_sec": elapsed}, f)


if __name__ == "__main__":
    main()
