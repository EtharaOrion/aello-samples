"""CIFAR-10 -> 94% top-1 on 1xH100, from scratch.

Architecture / recipe lineage: a from-scratch reimplementation of the
"airbench"-style CIFAR speedrun convnet (frozen ZCA patch-whitening stem, three
conv/pool/BN/GELU groups, bias-free scaled linear head) trained with Nesterov
SGD, label smoothing and a warmup/warmdown LR schedule.  Everything below is
written from scratch: no pretrained weights, no external data, CIFAR-10 train
split only.

Verifier contract (see README.md):
  * checkpoints are ``torch.jit.script(model).save(path)`` -- loadable modules,
    not state_dicts;
  * ``forward`` receives a float32 NCHW batch that is ALREADY per-channel
    standardized with mean (0.4914, 0.4822, 0.4465) / std (0.2470, 0.2435,
    0.2616).  We train on exactly the same standardization, so ``forward``
    performs NO normalization of its own.
  * 6-way test-time augmentation (mirror x {identity, 2 one-pixel diagonal
    translates}) is folded into ``forward``.
"""

import argparse
import json
import math
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# The verifier's normalization constants.  Training uses these verbatim so the
# model sees the identical input distribution at train and verification time.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CACHE = "/workspace/cache"


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class Conv(nn.Conv2d):
    """3x3 conv, no bias, partial-Dirac init (airbench-style)."""

    def __init__(self, c_in, c_out, k=3, padding="same", bias=False):
        super().__init__(c_in, c_out, kernel_size=k, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        # Identity-initialize the first c_in output filters; the rest keep the
        # default Kaiming-uniform draw.  Gives the net a clean residual-like
        # signal path at step 0 without any skip connections.
        torch.nn.init.dirac_(w[: w.size(1)])


class BatchNorm(nn.BatchNorm2d):
    """BN with frozen gamma (==1) and trainable beta."""

    def __init__(self, c, momentum, eps=1e-12):
        super().__init__(c, eps=eps, momentum=1.0 - momentum)
        self.weight.requires_grad = False
        self.bias.requires_grad = True


class ConvGroup(nn.Module):
    def __init__(self, c_in, c_out, bn_mom):
        super().__init__()
        self.conv1 = Conv(c_in, c_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(c_out, bn_mom)
        self.conv2 = Conv(c_out, c_out)
        self.norm2 = BatchNorm(c_out, bn_mom)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.activ(self.norm1(self.pool(self.conv1(x))))
        x = self.activ(self.norm2(self.conv2(x)))
        return x


class Net(nn.Module):
    """32x32 -> (whiten 2x2) 31x31 -> 15x15 -> 7x7 -> 3x3 -> pool3 -> 1x1."""

    def __init__(self, widths=(128, 384, 512), bn_mom=0.6, scale=1.0 / 9,
                 tta=True, tta_w=0.5):
        super().__init__()
        w1, w2, w3 = widths
        self.whiten = nn.Conv2d(3, 24, kernel_size=2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.whiten.bias.data.zero_()
        self.activ = nn.GELU()
        self.group1 = ConvGroup(24, w1, bn_mom)
        self.group2 = ConvGroup(w1, w2, bn_mom)
        self.group3 = ConvGroup(w2, w3, bn_mom)
        self.pool = nn.MaxPool2d(3)
        self.head = nn.Linear(w3, 10, bias=False)
        self.head.weight.data.zero_()
        self.scale = scale
        self.tta = tta
        self.tta_w = tta_w

    def features(self, x):
        x = self.activ(self.whiten(x))
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.pool(x)
        return self.head(x.flatten(1)) * self.scale

    def _mirror(self, x):
        # Average of the image and its horizontal mirror.
        return self.features(x) + self.features(x.flip(3))

    def forward(self, x):
        """x: float32 NCHW, ALREADY standardized by the verifier's transform."""
        if not self.tta:
            return self.features(x)
        base = self._mirror(x)
        p = F.pad(x, (1, 1, 1, 1), mode="reflect")
        shifted = self._mirror(p[:, :, 0:32, 0:32]) + self._mirror(p[:, :, 2:34, 2:34])
        # base is a sum of 2 views, shifted a sum of 4 -> match their scales
        # before blending so tta_w is a genuine 0..1 mixing weight.
        return (1.0 - self.tta_w) * (base / 2.0) + self.tta_w * (shifted / 4.0)


@torch.no_grad()
def init_whitening(net, images, eps=5e-4):
    """Freeze the stem to the ZCA whitening filter of 2x2x3 train patches.

    ``images`` are standardized train images (N,3,32,32) -- the TRAIN split
    only; the test split is never touched by this (or any other) fit.
    """
    c = images.shape[1]
    kh, kw = net.whiten.weight.shape[2:]
    patches = (images.unfold(2, kh, 1).unfold(3, kw, 1)
               .transpose(1, 3).reshape(-1, c, kh, kw).float())
    flat = patches.reshape(len(patches), -1)
    cov = (flat.T @ flat) / len(flat)
    evals, evecs = torch.linalg.eigh(cov, UPLO="U")
    scaled = evecs.T.reshape(-1, c, kh, kw) / torch.sqrt(evals.view(-1, 1, 1, 1) + eps)
    net.whiten.weight.data[:] = torch.cat((scaled, -scaled))


# --------------------------------------------------------------------------
# Data (all resident on GPU; no DataLoader in the training loop)
# --------------------------------------------------------------------------
def load_split(split, device):
    x = np.load(f"{CACHE}/{split}_x.npy")          # N,32,32,3 uint8
    y = np.load(f"{CACHE}/{split}_y.npy")
    x = torch.from_numpy(x).to(device).permute(0, 3, 1, 2).contiguous()
    return x, torch.from_numpy(y).to(device)


def standardize(x_uint8, device):
    mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1) * 255.0
    std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1) * 255.0
    return (x_uint8.float() - mean) / std


def stratified_holdout(y, n_per_class, seed=0):
    """Deterministic class-balanced holdout indices (used for ALL tuning)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    y_cpu = y.cpu()
    val, tr = [], []
    for c in range(10):
        idx = (y_cpu == c).nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(len(idx), generator=g)]
        val.append(perm[:n_per_class])
        tr.append(perm[n_per_class:])
    return torch.cat(val).sort().values, torch.cat(tr).sort().values


class GPUAugmenter:
    """Random 2px-translate + horizontal flip, entirely on device.

    The padded source is materialized once; each batch costs two gathers and
    one ``where``.
    """

    def __init__(self, x_std, pad=2):
        self.pad = pad
        self.src = F.pad(x_std, (pad,) * 4, mode="reflect")   # N,3,32+2p,32+2p
        self.n, self.c, self.hp, self.wp = self.src.shape
        self.dev = x_std.device
        self.ar = torch.arange(32, device=self.dev)

    def batch(self, idx, gen):
        b = idx.numel()
        img = self.src[idx]                                    # B,3,hp,wp
        k = 2 * self.pad + 1
        dy = torch.randint(0, k, (b,), device=self.dev, generator=gen)
        dx = torch.randint(0, k, (b,), device=self.dev, generator=gen)
        rows = (dy[:, None] + self.ar).view(b, 1, 32, 1).expand(b, self.c, 32, self.wp)
        img = img.gather(2, rows)
        cols = (dx[:, None] + self.ar).view(b, 1, 1, 32).expand(b, self.c, 32, 32)
        img = img.gather(3, cols)
        flip = torch.rand(b, 1, 1, 1, device=self.dev, generator=gen) < 0.5
        return torch.where(flip, img.flip(3), img)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def make_lr_fn(total_steps, warm_frac, lr_start, lr_end):
    warm = max(1, int(total_steps * warm_frac))

    def fn(step):
        if step < warm:
            f = step / warm
            return lr_start * (1 - f) + 1.0 * f
        f = (step - warm) / max(1, total_steps - warm)
        return 1.0 * (1 - f) + lr_end * f
    return fn


@torch.no_grad()
def evaluate(net, x_std, y, bs=1000):
    net.eval()
    correct = 0
    for i in range(0, len(x_std), bs):
        out = net(x_std[i:i + bs])
        correct += (out.argmax(1) == y[i:i + bs]).sum().item()
    net.train()
    return correct / len(x_std)


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--bs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=11.5)
    p.add_argument("--momentum", type=float, default=0.85)
    p.add_argument("--wd", type=float, default=0.0153)
    p.add_argument("--bias-scaler", type=float, default=64.0)
    p.add_argument("--label-smoothing", type=float, default=0.2)
    p.add_argument("--whiten-bias-epochs", type=float, default=3)
    p.add_argument("--warm-frac", type=float, default=0.23)
    p.add_argument("--lr-start", type=float, default=0.2)
    p.add_argument("--lr-end", type=float, default=0.07)
    p.add_argument("--widths", type=str, default="128,384,512")
    p.add_argument("--bn-mom", type=float, default=0.6)
    p.add_argument("--scale", type=float, default=1.0 / 9)
    p.add_argument("--pad", type=int, default=2)
    p.add_argument("--tta-w", type=float, default=0.5)
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val", choices=["holdout", "test", "none"], default="holdout")
    p.add_argument("--holdout-per-class", type=int, default=500)
    p.add_argument("--ckpt-dir", type=str, default="")
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--final-copy", type=str, default="")
    p.add_argument("--metrics-out", type=str, default="")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    args = build_args().parse_args(argv)
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=dev).manual_seed(args.seed + 1234)

    # ---- data -----------------------------------------------------------
    xtr_u8, ytr = load_split("train", dev)
    if args.val == "holdout":
        vi, ti = stratified_holdout(ytr, args.holdout_per_class, seed=0)
        x_val = standardize(xtr_u8[vi], dev)
        y_val = ytr[vi]
        xtr_u8, ytr = xtr_u8[ti], ytr[ti]
    elif args.val == "test":
        xte_u8, y_val = load_split("test", dev)
        x_val = standardize(xte_u8, dev)
    else:
        x_val = y_val = None
    x_std = standardize(xtr_u8, dev)
    n = len(x_std)
    aug = GPUAugmenter(x_std, pad=args.pad)

    # ---- model ----------------------------------------------------------
    widths = tuple(int(v) for v in args.widths.split(","))
    net = Net(widths, args.bn_mom, args.scale,
              tta=not args.no_tta, tta_w=args.tta_w).to(dev).to(memory_format=torch.channels_last)
    init_whitening(net, x_std)
    net.whiten.bias.requires_grad = True

    # ---- optimizer ------------------------------------------------------
    steps_per_epoch = n // args.bs
    total_steps = steps_per_epoch * args.epochs
    kilo = 1024 * (1 + 1 / (1 - args.momentum))
    lr = args.lr / kilo
    wd = args.wd * args.bs / kilo
    lr_bias = lr * args.bias_scaler
    norm_biases = [p for k, p in net.named_parameters()
                   if ("norm" in k or "whiten.bias" in k) and p.requires_grad]
    others = [p for k, p in net.named_parameters()
              if not ("norm" in k or "whiten.bias" in k) and p.requires_grad]
    opt = torch.optim.SGD(
        [dict(params=norm_biases, lr=lr_bias, weight_decay=wd / lr_bias),
         dict(params=others, lr=lr, weight_decay=wd / lr)],
        momentum=args.momentum, nesterov=True)
    base_lrs = [lr_bias, lr]
    lr_fn = make_lr_fn(total_steps, args.warm_frac, args.lr_start, args.lr_end)

    ckpt_dir = args.ckpt_dir
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    # ---- warmup: pay cudnn autotuning + kernel JIT BEFORE the timed run --
    # Uses synthetic data so no real epoch is ever oversized; every timed epoch
    # is then a genuine, uniform single pass over the training split.
    dummy = torch.randn(args.bs, 3, 32, 32, device=dev).to(memory_format=torch.channels_last)
    dummy_y = torch.zeros(args.bs, dtype=torch.long, device=dev)
    for _ in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(net.features(dummy), dummy_y,
                                   label_smoothing=args.label_smoothing, reduction="sum")
        loss.backward()
    opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        net.eval()
        net(torch.randn(1000, 3, 32, 32, device=dev).to(memory_format=torch.channels_last))
        net.train()
    # undo warmup damage: re-init everything the dummy pass could have moved
    torch.manual_seed(args.seed)
    for m in net.modules():
        if isinstance(m, (nn.BatchNorm2d,)):
            m.reset_running_stats()
        if isinstance(m, Conv):
            m.reset_parameters()
    net.head.weight.data.zero_()
    net.whiten.bias.data.zero_()
    init_whitening(net, x_std)
    for g in opt.param_groups:
        for p in g["params"]:
            opt.state.pop(p, None)
    torch.cuda.synchronize()

    # ---- timed training loop -------------------------------------------
    scripted = torch.jit.script(net)
    hist = []
    t0 = time.perf_counter()
    step = 0
    whiten_bias_steps = int(args.whiten_bias_epochs * steps_per_epoch)
    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(n, device=dev, generator=gen)
        for bi in range(steps_per_epoch):
            if step == whiten_bias_steps:
                net.whiten.bias.requires_grad = False
            m = lr_fn(step)
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * m
            idx = perm[bi * args.bs:(bi + 1) * args.bs]
            xb = aug.batch(idx, gen).to(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net.features(xb)
                loss = F.cross_entropy(out, ytr[idx],
                                       label_smoothing=args.label_smoothing,
                                       reduction="sum")
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
        if ckpt_dir:
            net.eval()
            scripted.eval()
            torch.jit.save(scripted, f"{ckpt_dir}/ckpt_epoch_{ep:04d}.pt")
            net.train()
        hist.append(dict(epoch=ep, t=time.perf_counter() - t0))
    torch.cuda.synchronize()
    train_wall = time.perf_counter() - t0

    # ---- evaluation (never inside the timed loop) -----------------------
    accs = {}
    if x_val is not None:
        accs["final"] = evaluate(net, x_val, y_val)
    res = dict(args=vars(args), train_wall=train_wall, hist=hist, acc=accs,
               params=sum(p.numel() for p in net.parameters()))
    # Handles for in-process sweeps (underscore keys are never serialized).
    res["_net"] = net
    res["_val"] = (x_val, y_val)
    if not args.quiet:
        print(json.dumps({k: v for k, v in res.items()
                          if k != "hist" and not k.startswith("_")}), flush=True)
    if args.final_copy and ckpt_dir:
        shutil.copy(f"{ckpt_dir}/ckpt_epoch_{args.epochs:04d}.pt", args.final_copy)
    if args.metrics_out:
        with open(args.metrics_out, "w") as f:
            json.dump({k: v for k, v in res.items() if not k.startswith("_")},
                      f, indent=2)
    return res


if __name__ == "__main__":
    main()
