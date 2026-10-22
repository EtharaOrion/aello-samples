import json
import os
import shutil
import time
from math import ceil

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import PILToTensor


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device="cuda").view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616), device="cuda").view(1, 3, 1, 1)


@torch.compile
def zeropower(g: torch.Tensor, steps: int = 3):
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
                p.mul_(len(p) ** 0.5 / p.norm())
                u = zeropower(g.reshape(len(g), -1)).reshape_as(g)
                p.add_(u, alpha=-group["lr"])


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, n):
        super().__init__(n, eps=1e-12, momentum=0.4)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, cin, cout):
        super().__init__(cin, cout, 3, padding=1, bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        nn.init.dirac_(w[:w.size(1)])


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
        self.head.weight.data.mul_(1 / self.head.weight.data.std())

    @torch.no_grad()
    def init_whiten(self, images):
        patches = images[:5000].unfold(2, 2, 1).unfold(3, 2, 1)
        patches = patches.transpose(1, 3).reshape(-1, 3, 2, 2).float()
        flat = patches.view(len(patches), -1)
        cov = flat.T @ flat / len(flat)
        vals, vecs = torch.linalg.eigh(cov, UPLO="U")
        filters = vecs.T.reshape(-1, 3, 2, 2) / torch.sqrt(vals.view(-1, 1, 1, 1) + 5e-4)
        self.whiten.weight.data.copy_(torch.cat((filters, -filters)).half())

    def forward(self, x: torch.Tensor, whiten_bias_grad: bool = True):
        stored_bias = self.whiten.bias
        assert stored_bias is not None
        bias = stored_bias if whiten_bias_grad else stored_bias.detach()
        x = F.conv2d(x, self.whiten.weight, bias)
        x = self.layers(x).view(x.size(0), -1)
        return self.head(x) / x.size(1)


class VerifierModel(nn.Module):
    """Standardized float32 input; six-view TTA is part of the scripted predictor."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def mirror(self, x: torch.Tensor):
        return 0.5 * self.net(x, False) + 0.5 * self.net(x.flip(-1), False)

    def forward(self, x: torch.Tensor):
        x = x.half()
        base = self.mirror(x)
        p = F.pad(x, (1, 1, 1, 1), "reflect")
        shifted = 0.5 * self.mirror(p[:, :, 0:32, 0:32]) + 0.5 * self.mirror(p[:, :, 2:34, 2:34])
        return 0.5 * base + 0.5 * shifted


def load_train():
    ds = ImageFolder("/workspace/data/cifar10/train", transform=PILToTensor())
    dl = DataLoader(ds, batch_size=1000, shuffle=False, num_workers=16,
                    pin_memory=True, persistent_workers=False)
    xs, ys = [], []
    for x, y in dl:
        xs.append(x)
        ys.append(y)
    x = torch.cat(xs).cuda().half().div_(255).contiguous(memory_format=torch.channels_last)
    x = (x - MEAN.half()) / STD.half()
    return x.contiguous(memory_format=torch.channels_last), torch.cat(ys).cuda()


def crop(images, pad=2):
    padded = F.pad(images, (pad,) * 4, "reflect")
    shifts = torch.randint(0, 2 * pad + 1, (len(images), 2), device="cuda")
    out = torch.empty_like(images)
    for yy in range(2 * pad + 1):
        for xx in range(2 * pad + 1):
            mask = (shifts[:, 0] == yy) & (shifts[:, 1] == xx)
            out[mask] = padded[mask, :, yy:yy + 32, xx:xx + 32]
    return out


def make_opts(model, batch_size=2000):
    wd = 2e-6 * batch_size
    convs = [p for p in model.parameters() if p.ndim == 4 and p.requires_grad]
    norm_bias = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    configs = [dict(params=[model.whiten.bias], lr=0.053, weight_decay=wd / 0.053),
               dict(params=norm_bias, lr=0.053, weight_decay=wd / 0.053),
               dict(params=[model.head.weight], lr=0.67, weight_decay=wd / 0.67)]
    sgd = torch.optim.SGD(configs, momentum=0.85, nesterov=True, fused=True)
    muon = Muon(convs)
    for opt in (sgd, muon):
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]
    return sgd, muon


def train_run(model, compiled, images, labels, save_dir=None, dummy=False):
    model.reset()
    model.init_whiten(images)
    sgd, muon = make_opts(model)
    n_batches, total_steps, whiten_steps = 25, 225, 75
    step = 0
    started = None
    for epoch in range(1, 10):
        model.train()
        epoch_images = crop(images)
        # One random pre-flip, complemented on alternating epochs.
        if epoch == 1:
            flipmask = torch.rand(len(images), device="cuda") < 0.5
            train_run.flipmask = flipmask
        flipmask = train_run.flipmask
        useflip = flipmask if epoch % 2 else ~flipmask
        epoch_images[useflip] = epoch_images[useflip].flip(-1)
        order = torch.randperm(len(images), device="cuda")
        if started is None:
            torch.cuda.synchronize()
            started = time.time()
        for bi in range(n_batches):
            ix = order[bi * 2000:(bi + 1) * 2000]
            target = labels[ix] if not dummy else torch.randint(10, (len(ix),), device="cuda")
            logits = compiled(epoch_images[ix], step < whiten_steps)
            F.cross_entropy(logits, target, label_smoothing=0.2, reduction="sum").backward()
            sgd.param_groups[0]["lr"] = sgd.param_groups[0]["initial_lr"] * max(0, 1 - step / whiten_steps)
            for g in sgd.param_groups[1:] + muon.param_groups:
                g["lr"] = g["initial_lr"] * (1 - step / total_steps)
            sgd.step(); muon.step()
            model.zero_grad(set_to_none=True)
            step += 1
        torch.cuda.synchronize()
        if save_dir is not None:
            model.eval()
            scripted = torch.jit.script(VerifierModel(model))
            scripted.save(os.path.join(save_dir, f"ckpt_epoch_{epoch:04d}.pt"))
            print(f"saved epoch {epoch}, elapsed={time.time()-started:.3f}s", flush=True)
    return time.time() - started


def main():
    images, labels = load_train()
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    compiled = torch.compile(model, mode="max-autotune")
    print("warming compiled training graph", flush=True)
    train_run(model, compiled, images, labels, dummy=True)
    out = "/workspace/periodic_checkpoints"
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(out):
        os.remove(os.path.join(out, name))
    print("starting measured fresh run", flush=True)
    duration = train_run(model, compiled, images, labels, save_dir=out)
    shutil.copy(os.path.join(out, "ckpt_epoch_0009.pt"), "/workspace/checkpoint.pt")
    with open("/workspace/agent_reported_metrics.json", "w") as f:
        json.dump({"accuracy": 0.0, "wall_clock_sec": duration}, f)
    print(f"complete in {duration:.3f}s", flush=True)


if __name__ == "__main__":
    main()
