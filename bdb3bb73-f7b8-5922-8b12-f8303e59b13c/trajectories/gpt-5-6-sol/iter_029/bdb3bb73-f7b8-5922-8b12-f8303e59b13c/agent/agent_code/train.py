import argparse
import os
import random
import shutil
import time
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image


MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616)).view(1, 3, 1, 1)


def zeropower(g, steps=5):
    shape = g.shape
    x = g.reshape(g.shape[0], -1).bfloat16()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        a = x @ x.T
        x = 3.4445 * x + (-4.7750 * a + 2.0315 * (a @ a)) @ x
    if transposed:
        x = x.T
    return x.reshape(shape).to(g.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.24, momentum=0.6, weight_decay=0.01):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

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
                u = zeropower(g)
                u.mul_((max(1.0, p.shape[0] / (p.numel() / p.shape[0]))) ** 0.5)
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(u, alpha=-group["lr"])


def load_split(root):
    classes = sorted(p.name for p in Path(root).iterdir() if p.is_dir())
    items = [(p, y) for y, c in enumerate(classes) for p in sorted((Path(root) / c).glob("*.png"))]
    with ThreadPoolExecutor(32) as ex:
        images = list(ex.map(lambda z: read_image(str(z[0])), items))
    return torch.stack(images), torch.tensor([z[1] for z in items])


class Conv(nn.Module):
    def __init__(self, ci, co, k=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, k, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(co, eps=1e-5, momentum=0.4)

    def forward(self, x):
        return F.gelu(self.bn(self.conv(x)))


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(Conv(channels, channels), Conv(channels, channels))

    def forward(self, x):
        return x + self.block(x)


class AirNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.whiten = nn.Conv2d(3, 24, 2, bias=False)
        self.whiten.weight.requires_grad_(False)
        self.body = nn.Sequential(
            Conv(24, 64),
            nn.MaxPool2d(2),
            Residual(64),
            Conv(64, 256),
            nn.MaxPool2d(2),
            Residual(256),
            Conv(256, 256),
            nn.MaxPool2d(2),
            Residual(256),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(256, 10, bias=False)

    def _logits(self, x):
        return self.head(self.body(self.whiten(x))) / 9.0

    def forward(self, x):
        if self.training:
            return self._logits(x)
        return (self._logits(x) + self._logits(x.flip(-1))) * 0.5


def init_whitening(model, images):
    x = images[:5000].float().div_(255)
    x = (x - MEAN) / STD
    patches = x.unfold(2, 2, 1).unfold(3, 2, 1).permute(0, 2, 3, 1, 4, 5).reshape(-1, 12)
    patches = patches[::20]
    patches = patches - patches.mean(0, keepdim=True)
    cov = patches.T @ patches / patches.shape[0]
    vals, vecs = torch.linalg.eigh(cov)
    filters = (vecs.T / torch.sqrt(vals[:, None] + 1e-2)).reshape(12, 3, 2, 2)
    model.whiten.weight.data[:12].copy_(filters)
    model.whiten.weight.data[12:].copy_(-filters)


@torch.no_grad()
def augment(x):
    n = x.shape[0]
    x = F.pad(x, (2, 2, 2, 2), mode="reflect")
    yy = torch.arange(32, device=x.device).view(1, 32, 1) + torch.randint(0, 5, (n, 1, 1), device=x.device)
    xx = torch.arange(32, device=x.device).view(1, 1, 32) + torch.randint(0, 5, (n, 1, 1), device=x.device)
    ii = torch.arange(n, device=x.device).view(n, 1, 1)
    x = x[ii, :, yy, xx].permute(0, 3, 1, 2)
    flip = torch.rand(n, 1, 1, 1, device=x.device) < 0.5
    x = torch.where(flip, x.flip(-1), x)
    return x


def cutmix(x, y, alpha=1.0):
    n = x.shape[0]
    perm = torch.randperm(n, device=x.device)
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    side = int(32 * (1.0 - lam) ** 0.5)
    cx, cy = random.randrange(32), random.randrange(32)
    x1, x2 = max(0, cx - side // 2), min(32, cx + side // 2)
    y1, y2 = max(0, cy - side // 2), min(32, cy + side // 2)
    z = x.clone()
    z[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    area = (x2 - x1) * (y2 - y1) / 1024
    return z, y, y[perm], 1.0 - area


def save_script(model, path):
    model.eval()
    torch.jit.script(model).save(str(path))
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--out", default="/workspace/periodic_checkpoints")
    ap.add_argument("--muon", action="store_true")
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    train_u8, train_y = load_split("/workspace/data/cifar10/train")
    model = AirNet()
    init_whitening(model, train_u8)
    model.cuda().train()
    ema = copy.deepcopy(model).requires_grad_(False)
    xall = train_u8.cuda().float().div_(255)
    xall.sub_(MEAN.cuda()).div_(STD.cuda())
    yall = train_y.cuda()
    del train_u8, train_y

    body, head, norm = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "head" in name:
            head.append(p)
        elif p.ndim == 1:
            norm.append(p)
        else:
            body.append(p)
    if args.muon:
        muon_params = [p for p in body if p.ndim >= 2]
        other_body = [p for p in body if p.ndim < 2]
        opt = torch.optim.SGD([
        {"params": other_body, "lr": 0.30, "weight_decay": 0.002},
        {"params": head, "lr": 0.55, "weight_decay": 0.002},
        {"params": norm, "lr": 0.08, "weight_decay": 0.0},
        ], momentum=0.85, nesterov=True)
        muon = Muon(muon_params)
    else:
        opt = torch.optim.SGD([
            {"params": body, "lr": 0.30, "weight_decay": 0.002},
            {"params": head, "lr": 0.55, "weight_decay": 0.002},
            {"params": norm, "lr": 0.08, "weight_decay": 0.0},
        ], momentum=0.85, nesterov=True)
        muon = None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.pt"):
        old.unlink()
    bs = 1024
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        frac = (epoch - 1) / args.epochs
        mult = min((epoch / 2), max(0.0, (1.0 - frac) / 0.82))
        for group in opt.param_groups:
            group["lr"] = group.get("initial_lr", group["lr"]) if False else group["lr"]
        # Explicit base rates avoid compounding the schedule.
        bases = (0.30, 0.55, 0.08) if not args.muon else (0.30, 0.55, 0.08)
        for group, base in zip(opt.param_groups, bases):
            group["lr"] = base * mult
        if muon is not None:
            muon.param_groups[0]["lr"] = 0.24 * mult
        order = torch.randperm(50000, device="cuda")
        for j in range(0, 50000, bs):
            ids = order[j:j + bs]
            xb, yb = augment(xall[ids]), yall[ids]
            use_cm = False
            if use_cm:
                xb, ya, yb2, lam = cutmix(xb, yb)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xb)
                if use_cm:
                    loss = lam * F.cross_entropy(logits, ya, label_smoothing=0.1) + (1-lam) * F.cross_entropy(logits, yb2, label_smoothing=0.1)
                else:
                    loss = F.cross_entropy(logits, yb, label_smoothing=0.1)
            loss.backward()
            opt.step()
            if muon is not None:
                muon.step()
            with torch.no_grad():
                ms = model.state_dict()
                for name, value in ema.state_dict().items():
                    if value.is_floating_point():
                        value.lerp_(ms[name], 0.05)
                    else:
                        value.copy_(ms[name])
        torch.cuda.synchronize()
        path = out / f"ckpt_epoch_{epoch:04d}.pt"
        save_script(ema, path)
        print(f"epoch {epoch:02d} loss {loss.item():.4f} elapsed {time.perf_counter()-start:.3f}", flush=True)
    shutil.copy(out / f"ckpt_epoch_{args.epochs:04d}.pt", "/workspace/checkpoint.pt")


if __name__ == "__main__":
    main()
