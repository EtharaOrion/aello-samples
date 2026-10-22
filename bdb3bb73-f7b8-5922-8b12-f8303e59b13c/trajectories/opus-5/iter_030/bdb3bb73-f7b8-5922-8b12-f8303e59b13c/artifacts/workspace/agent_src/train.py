#!/usr/bin/env python
"""CIFAR-10 -> 94% top-1 from scratch on one H100.

Design notes
------------
* Architecture is a compact 3-group convnet in the airbench lineage: a frozen
  2x2 patch-ZCA whitening conv, three (conv -> pool -> BN -> GELU -> conv -> BN
  -> GELU) groups, global max-pool, bias-free head with a fixed logit scale.
* The whole train split lives on the GPU as a pre-normalized, pre-padded fp16
  tensor, so an "epoch" is a permutation + slicing; there is no host dataloader.
* Augmentation (random translate, horizontal flip, optional cutout) runs on
  device with index gathers, matching the airbench formulation.
* The artifact written every epoch is a `torch.jit.script`ed inference module
  that folds 6-way TTA (mirror x {centre, -1px, +1px}) into `forward`.
  It is scripted ONCE before training and shares parameter storage with the
  live model, so the per-epoch checkpoint cost is just the serialize+write.

Input convention: the saved module expects float32 NCHW already standardized
with the CIFAR-10 mean/std (the verifier's convention). It additionally detects
and adapts [0,1] and [0,255] inputs so that a mis-fed batch degrades gracefully
instead of silently collapsing; a standardized batch takes the no-op path.
"""
import argparse
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROC_T0 = time.time()  # process start, before torch/data/warmup: the honest zero

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CACHE = "/workspace/data/cache"


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class Conv(nn.Conv2d):
    """3x3 bias-free conv with a partial identity (dirac) init."""

    def __init__(self, cin, cout, k=3):
        super().__init__(cin, cout, k, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[: w.size(1)])


class BN(nn.BatchNorm2d):
    """BatchNorm with a frozen unit scale; only the shift is learned.

    Freezing `weight` matters because the biases sit in a param group with a
    64x larger LR -- a trainable scale there would be driven far too hard.
    """

    def __init__(self, c, momentum):
        super().__init__(c, eps=1e-12, momentum=1 - momentum)
        self.weight.requires_grad = False


class ConvGroup(nn.Module):
    def __init__(self, cin, cout, bn_mom):
        super().__init__()
        self.conv1 = Conv(cin, cout)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BN(cout, bn_mom)
        self.conv2 = Conv(cout, cout)
        self.norm2 = BN(cout, bn_mom)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.norm1(self.pool(self.conv1(x))))
        return self.act(self.norm2(self.conv2(x)))


class Net(nn.Module):
    def __init__(self, widths=(128, 384, 512), w0=24, bn_mom=0.6, scale=1 / 9):
        super().__init__()
        self.whiten = nn.Conv2d(3, w0, 2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.act = nn.GELU()
        self.g1 = ConvGroup(w0, widths[0], bn_mom)
        self.g2 = ConvGroup(widths[0], widths[1], bn_mom)
        self.g3 = ConvGroup(widths[1], widths[2], bn_mom)
        self.pool = nn.MaxPool2d(3)
        self.head = nn.Linear(widths[2], 10, bias=False)
        self.head.weight.data.zero_()
        self.scale = scale

    def forward(self, x):
        x = self.act(self.whiten(x))
        x = self.g3(self.g2(self.g1(x)))
        x = self.pool(x).flatten(1)
        return self.head(x) * self.scale


class TTANet(nn.Module):
    """Inference wrapper that is what actually gets scripted and saved.

    level 0 = no TTA, 1 = mirror (2 views), 2 = mirror x 1px translate (6 views).
    """

    def __init__(self, net, level=2):
        super().__init__()
        self.net = net
        self.level = level
        self.register_buffer("mean", torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR_STD).view(1, 3, 1, 1))

    def standardize(self, x):
        # Range-detect so we never double-normalize the verifier's already
        # standardized batch (min ~ -2.0) but still handle [0,1] / [0,255].
        if float(x.min()) < -0.5:
            return x
        if float(x.max()) > 1.5:
            x = x / 255.0
        return (x - self.mean) / self.std

    def mirror(self, x):
        return 0.5 * self.net(x) + 0.5 * self.net(x.flip(-1))

    def forward(self, x):
        x = self.standardize(x.float())
        if self.level == 0:
            return self.net(x)
        logits = self.mirror(x)
        if self.level == 1:
            return logits
        h, w = x.size(2), x.size(3)
        p = F.pad(x, (1, 1, 1, 1), "reflect")
        shifted = 0.5 * (
            self.mirror(p[:, :, 0:h, 0:w]) + self.mirror(p[:, :, 2 : 2 + h, 2 : 2 + w])
        )
        return 0.5 * logits + 0.5 * shifted


def init_whitening(layer, images, eps=5e-4):
    """Set the frozen first conv to a 2x2 patch-ZCA basis (+/- eigenvector pairs)."""
    patches = (
        images.unfold(2, 2, 1).unfold(3, 2, 1).transpose(1, 3).reshape(-1, 3, 2, 2).float()
    )
    flat = patches.reshape(len(patches), -1)
    cov = (flat.T @ flat) / len(flat)
    eigval, eigvec = torch.linalg.eigh(cov, UPLO="U")
    scaled = eigvec.T.reshape(-1, 3, 2, 2) / torch.sqrt(eigval.view(-1, 1, 1, 1) + eps)
    layer.weight.data[:] = torch.cat((scaled, -scaled))
    layer.bias.data.zero_()


# --------------------------------------------------------------------------- #
# data (GPU-resident)
# --------------------------------------------------------------------------- #
def load_split(split, dev):
    x = np.load(f"{CACHE}/{split}_x.npy")
    y = np.load(f"{CACHE}/{split}_y.npy")
    x = torch.from_numpy(x).to(dev).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(CIFAR_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=dev).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x, torch.from_numpy(y).to(dev)


def batch_flip(x, gen):
    m = (torch.rand(len(x), device=x.device, generator=gen) < 0.5).view(-1, 1, 1, 1)
    return torch.where(m, x.flip(-1), x)


def batch_crop(padded, size, gen):
    """Per-sample random translate as two axis-wise `gather`s.

    The obvious formulation -- loop over the (2r+1) shift values and use boolean
    mask assignment -- costs a `nonzero` (and a host sync) per shift value and
    measured 49 ms/1k images here, 3x the cost of the fwd+bwd it feeds. Two
    gathers with broadcast index tensors do the same thing branch-free.
    """
    r = (padded.size(-1) - size) // 2
    if r == 0:
        return padded
    n, c, _, w = padded.shape
    dev = padded.device
    sh = torch.randint(0, 2 * r + 1, (n, 2), device=dev, generator=gen)
    rows = (torch.arange(size, device=dev).view(1, 1, size, 1) + sh[:, 0].view(n, 1, 1, 1))
    tmp = padded.gather(2, rows.expand(n, c, size, w))
    cols = (torch.arange(size, device=dev).view(1, 1, 1, size) + sh[:, 1].view(n, 1, 1, 1))
    return tmp.gather(3, cols.expand(n, c, size, size))


def batch_cutout(x, size, gen):
    if size == 0:
        return x
    n, _, h, w = x.shape
    cy = torch.randint(0, h, (n, 1, 1), device=x.device, generator=gen)
    cx = torch.randint(0, w, (n, 1, 1), device=x.device, generator=gen)
    ys = torch.arange(h, device=x.device).view(1, h, 1)
    xs = torch.arange(w, device=x.device).view(1, 1, w)
    m = ((ys - cy).abs() < size // 2) & ((xs - cx).abs() < size // 2)
    return x.masked_fill(m.unsqueeze(1), 0.0)


# --------------------------------------------------------------------------- #
# schedule
# --------------------------------------------------------------------------- #
def lr_at(step, total, warm_frac, lo_start, lo_end, hold_frac=0.0):
    """Triangular ramp: lo_start -> 1.0 (at warm_frac) -> hold -> lo_end."""
    wu = max(1, int(total * warm_frac))
    hold = int(total * hold_frac)
    if step < wu:
        f = step / wu
        return lo_start * (1 - f) + f
    if step < wu + hold:
        return 1.0
    f = (step - wu - hold) / max(1, total - wu - hold)
    return 1.0 * (1 - f) + lo_end * f


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def main(a):
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(a.seed)
    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed)

    xtr, ytr = load_split("train", dev)
    # Optional held-out slice, carved deterministically from the train split so
    # that no test image is ever seen by the model or by model selection.
    if a.holdout > 0:
        g = torch.Generator().manual_seed(1234)
        perm = torch.randperm(len(xtr), generator=g).to(dev)
        hi, ti = perm[: a.holdout], perm[a.holdout :]
        xva, yva = xtr[hi], ytr[hi]
        xtr, ytr = xtr[ti], ytr[ti]
    else:
        xva = yva = None

    net = Net(tuple(a.widths), bn_mom=a.bn_mom, scale=a.scale).to(dev).to(
        memory_format=torch.channels_last
    )
    init_whitening(net.whiten, xtr[: a.whiten_n])
    if a.freeze_whiten_bias:
        net.whiten.bias.requires_grad = False

    n_par = sum(p.numel() for p in net.parameters())
    pad = a.translate
    # fp16 + channels_last: the augmentation pipeline is pure memory traffic, so
    # halving the element size halves its cost. Autocast consumes fp16 anyway.
    xtr_pad = (
        F.pad(xtr, (pad,) * 4, "reflect").half().to(memory_format=torch.channels_last)
    )
    del xtr

    # ---- optimizer -------------------------------------------------------- #
    biases, weights = [], []
    for m in net.modules():
        if isinstance(m, (BN, nn.Conv2d)) and getattr(m, "bias", None) is not None:
            if m.bias.requires_grad:
                biases.append(m.bias)
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            weights.append(m.weight) if m.weight.requires_grad else None
    kilostep = 1024 * (1 + 1 / (1 - a.momentum))
    lr = a.lr / kilostep
    wd = a.wd * a.batch / kilostep
    lr_b = lr * a.bias_scaler
    opt = torch.optim.SGD(
        [
            dict(params=biases, lr=lr_b, weight_decay=wd / lr_b),
            dict(params=weights, lr=lr, weight_decay=wd / lr),
        ],
        momentum=a.momentum,
        nesterov=True,
    )
    steps_per_epoch = len(xtr_pad) // a.batch
    total_steps = steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_at(s, total_steps, a.warm_frac, a.lo_start, a.lo_end, a.hold_frac)
    )

    # ---- the saved artifact: scripted once, shares storage with `net` ------ #
    os.makedirs(a.ckpt_dir, exist_ok=True)
    for f in os.listdir(a.ckpt_dir):
        if f.startswith("ckpt_epoch_") and f.endswith(".pt"):
            os.remove(os.path.join(a.ckpt_dir, f))
    saver = torch.jit.script(TTANet(net, a.tta).eval())
    saver.eval()

    # ---- warm up kernels BEFORE the timed run (random data, no train pass) - #
    dummy = torch.randn(a.batch, 3, 32, 32, device=dev).to(memory_format=torch.channels_last)
    for _ in range(a.warmup):
        with torch.autocast("cuda", dtype=torch.float16):
            out = net(dummy)
            F.cross_entropy(out, torch.zeros(a.batch, dtype=torch.long, device=dev)).backward()
        net.zero_grad(set_to_none=True)
    with torch.no_grad():
        saver(dummy[:64])
    torch.cuda.synchronize()

    # ---- timed training run ----------------------------------------------- #
    def holdout_acc():
        net.eval()
        with torch.no_grad():
            pr = torch.cat(
                [saver(xva[i : i + 2500]).argmax(1) for i in range(0, len(xva), 2500)]
            )
        return (pr == yva).float().mean().item()

    hist = []
    eval_time = 0.0  # dev-only holdout evals, subtracted so timings stay honest
    run_t0 = time.time()
    for ep in range(1, a.epochs + 1):
        net.train()
        # Augment the whole permuted epoch in one shot: one shuffle-gather plus
        # a handful of large kernels, instead of ~5 small launches per step.
        idx = torch.randperm(len(xtr_pad), device=dev, generator=gen)
        xa = batch_flip(batch_crop(xtr_pad[idx], 32, gen), gen)
        if a.cutout:
            xa = batch_cutout(xa, a.cutout, gen)
        ya = ytr[idx]
        for i in range(steps_per_epoch):
            sl = slice(i * a.batch, (i + 1) * a.batch)
            xb, yb = xa[sl], ya[sl]
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(
                    net(xb), yb, label_smoothing=a.smooth, reduction="sum"
                )
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
        saver.save(os.path.join(a.ckpt_dir, f"ckpt_epoch_{ep:04d}.pt"))
        last_save = time.time()
        row = dict(epoch=ep, t=last_save - run_t0 - eval_time)
        if a.eval_each and xva is not None:
            e0 = time.time()
            row["acc"] = holdout_acc()
            eval_time += time.time() - e0
        hist.append(row)
    torch.cuda.synchronize()
    wall = time.time() - run_t0 - eval_time

    # ---- held-out evaluation (never the test split) ------------------------ #
    acc = holdout_acc() if xva is not None else None

    per_ep = (hist[-1]["t"] - hist[0]["t"]) / max(1, len(hist) - 1)
    t_star_proxy = per_ep * a.epochs
    res = dict(
        tag=a.tag, seed=a.seed, epochs=a.epochs, params=n_par, holdout=a.holdout,
        wall_clock_sec=wall, per_epoch_sec=per_ep, t_star_proxy=t_star_proxy,
        # End-to-end: interpreter start -> last checkpoint on disk. This is what
        # gets reported externally; it is strictly larger than both the training
        # loop and the checkpoint span, so the report can never under-state.
        proc_wall_sec=last_save - PROC_T0,
        holdout_acc=acc, hist=hist, cfg={k: v for k, v in vars(a).items()},
    )
    print(json.dumps({k: v for k, v in res.items() if k not in ("hist", "cfg")}))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")
    if a.final_copy:
        last = os.path.join(a.ckpt_dir, f"ckpt_epoch_{a.epochs:04d}.pt")
        shutil.copy(last, a.final_copy)
    return res


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--batch", type=int, default=1000)
    p.add_argument("--lr", type=float, default=11.5)
    p.add_argument("--wd", type=float, default=0.0153)
    p.add_argument("--momentum", type=float, default=0.85)
    p.add_argument("--bias-scaler", type=float, default=64.0)
    p.add_argument("--smooth", type=float, default=0.2)
    p.add_argument("--scale", type=float, default=1 / 9)
    p.add_argument("--bn-mom", type=float, default=0.6)
    p.add_argument("--widths", type=int, nargs=3, default=[128, 384, 512])
    p.add_argument("--translate", type=int, default=2)
    p.add_argument("--cutout", type=int, default=0)
    p.add_argument("--tta", type=int, default=2)
    p.add_argument("--warm-frac", type=float, default=0.23)
    p.add_argument("--hold-frac", type=float, default=0.0)
    p.add_argument("--lo-start", type=float, default=0.2)
    p.add_argument("--lo-end", type=float, default=0.07)
    p.add_argument("--whiten-n", type=int, default=5000)
    p.add_argument("--freeze-whiten-bias", type=int, default=0)
    p.add_argument("--warmup", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout", type=int, default=0)
    p.add_argument("--eval-each", type=int, default=0)
    p.add_argument("--ckpt-dir", default="/workspace/scratch_ckpt")
    p.add_argument("--final-copy", default="")
    p.add_argument("--out", default="")
    p.add_argument("--tag", default="run")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
