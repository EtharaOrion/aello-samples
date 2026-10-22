import argparse
import copy
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGroup(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.res1 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.rbn1 = nn.BatchNorm2d(cout)
        self.res2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.rbn2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = F.gelu(self.bn(self.conv(x)))
        x = F.max_pool2d(x, 2)
        return x + self.rbn2(self.res2(F.gelu(self.rbn1(self.res1(x)))))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 24, 2, bias=True)
        self.g1 = ConvGroup(24, 64)
        self.g2 = ConvGroup(64, 256)
        self.g3 = ConvGroup(256, 256)
        self.fc = nn.Linear(256, 10, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.fc.weight)

    def forward(self, x):
        x = F.gelu(self.stem(x))
        x = self.g1(x)
        x = self.g2(x)
        x = self.g3(x)
        x = F.adaptive_max_pool2d(x, 1).flatten(1)
        return self.fc(x) / 9.0


class InferenceNet(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x):
        return (self.base(x) + self.base(x.flip(-1))) * 0.5


@torch.no_grad()
def zero_power(g, steps=5):
    shape = g.shape
    x = g.reshape(shape[0], -1).bfloat16()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.t()
    x = x / (x.norm() + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = x @ x.t()
        x = a * x + (b * A + c * (A @ A)) @ x
    if transposed:
        x = x.t()
    return x.reshape(shape).to(g.dtype)


class Muon:
    def __init__(self, params, lr=0.05, momentum=0.95, wd=0.01):
        self.params = list(params)
        self.lr, self.momentum, self.wd = lr, momentum, wd
        self.buf = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self, lr_scale):
        for p, b in zip(self.params, self.buf):
            if p.grad is None:
                continue
            p.mul_(1 - self.lr * lr_scale * self.wd)
            b.lerp_(p.grad, 1 - self.momentum)
            g = p.grad.lerp_(b, self.momentum)
            u = zero_power(g)
            scale = math.sqrt(max(1.0, p.shape[0] / (p.numel() / p.shape[0])))
            p.add_(u, alpha=-self.lr * lr_scale * scale)


def load_split(root):
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    images, labels = [], []
    for y, name in enumerate(classes):
        for fn in sorted((root / name).glob("*.png")):
            images.append(np.asarray(Image.open(fn).convert("RGB"), dtype=np.uint8))
            labels.append(y)
    x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()
    return x, torch.tensor(labels), classes


def augment(x):
    n = x.shape[0]
    x = F.pad(x, (2, 2, 2, 2), mode="reflect")
    oy = torch.randint(0, 5, (n,), device=x.device)
    ox = torch.randint(0, 5, (n,), device=x.device)
    yy = torch.arange(32, device=x.device)[None, :, None] + oy[:, None, None]
    xx = torch.arange(32, device=x.device)[None, None, :] + ox[:, None, None]
    ii = torch.arange(n, device=x.device)[:, None, None]
    x = x[ii, :, yy, xx].permute(0, 3, 1, 2)
    flip = torch.rand(n, device=x.device) < 0.5
    x[flip] = x[flip].flip(-1)
    cy = torch.randint(0, 32, (n,), device=x.device)
    cx = torch.randint(0, 32, (n,), device=x.device)
    gy = torch.arange(32, device=x.device)[None, :, None]
    gx = torch.arange(32, device=x.device)[None, None, :]
    mask = (gy - cy[:, None, None]).abs().lt(4) & (gx - cx[:, None, None]).abs().lt(4)
    return x.masked_fill_(mask[:, None], 0)


def save_script(model, path):
    was_training = model.training
    model.eval()
    torch.jit.script(InferenceNet(model)).save(str(path))
    model.train(was_training)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/scratch_ckpt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=15)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    raw_x, raw_y, _ = load_split(Path("/workspace/data/cifar10/train"))
    device = torch.device("cuda")
    mean = torch.tensor((0.4914, 0.4822, 0.4465), device=device)[None, :, None, None]
    std = torch.tensor((0.2470, 0.2435, 0.2616), device=device)[None, :, None, None]
    x = (raw_x.to(device, torch.float32).div_(255).sub_(mean).div_(std)).to(torch.bfloat16)
    y = raw_y.to(device)
    model = Net().to(device)
    # Exact PCA color/spatial whitening from train-split 2x2 patches only.
    patches = x[:10000].unfold(2, 2, 2).unfold(3, 2, 2)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, 12).float()
    patches = patches - patches.mean(0)
    cov = patches.t().mm(patches) / patches.shape[0]
    vals, vecs = torch.linalg.eigh(cov)
    filters = (vecs / (vals + 0.10).sqrt()[None]).t().reshape(12, 3, 2, 2)
    with torch.no_grad():
        model.stem.weight.copy_(torch.cat((filters, -filters)))
        model.stem.bias.zero_()
    model.stem.weight.requires_grad_(False)
    hidden = [p for n, p in model.named_parameters() if p.ndim >= 2 and "fc" not in n and "stem" not in n]
    other = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "fc" not in n and "stem" not in n)]
    muon = Muon(hidden, lr=0.24, momentum=0.95, wd=0.01)
    adam = torch.optim.AdamW(other, lr=0.018, betas=(0.9, 0.95), weight_decay=0.01)
    steps_per_epoch = math.ceil(len(x) / args.batch)
    total_steps = args.epochs * steps_per_epoch
    start = time.time()
    step = 0
    ema = None
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(x), device=device)
        model.train()
        correct = seen = 0
        for pos in range(0, len(x), args.batch):
            ids = order[pos:pos + args.batch]
            xb, yb = augment(x[ids]), y[ids]
            progress = step / total_steps
            lr_scale = min((step + 1) / 20, 1.0) * max(0.02, 1 - progress)
            for pg in adam.param_groups:
                pg["lr"] = 0.018 * lr_scale
            adam.zero_grad(set_to_none=True)
            for p in hidden:
                p.grad = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb, label_smoothing=0.2)
            loss.backward()
            muon.step(lr_scale)
            adam.step()
            if epoch == args.epochs // 2 and ema is None:
                ema = copy.deepcopy(model).eval()
                for p in ema.parameters():
                    p.requires_grad_(False)
            elif ema is not None:
                with torch.no_grad():
                    for ep, p in zip(ema.parameters(), model.parameters()):
                        ep.lerp_(p, 0.01)
                    for eb, b in zip(ema.buffers(), model.buffers()):
                        if eb.is_floating_point():
                            eb.lerp_(b, 0.01)
                        else:
                            eb.copy_(b)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += len(yb)
            step += 1
        torch.cuda.synchronize()
        save_script(ema if ema is not None else model, out / f"ckpt_epoch_{epoch:04d}.pt")
        print(json.dumps({"epoch": epoch, "train_acc": correct / seen, "loss": loss.item(), "elapsed": time.time() - start}), flush=True)
    shutil.copy(out / f"ckpt_epoch_{args.epochs:04d}.pt", out / "checkpoint.pt")
    (out / "run.json").write_text(json.dumps({"wall_clock_sec": time.time() - start, "seed": args.seed}))


if __name__ == "__main__":
    main()
