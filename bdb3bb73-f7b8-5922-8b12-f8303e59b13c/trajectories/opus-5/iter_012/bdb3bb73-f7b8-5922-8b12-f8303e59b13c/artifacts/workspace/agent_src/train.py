"""CIFAR-10 from scratch to 94% top-1 on one H100, as fast as possible.

Design notes
------------
* Architecture: a compact "airbench"-style convnet -- a frozen 2x2 ZCA-whitening conv
  (initialised from the *training* split's patch covariance only), three
  conv/pool/BN/GELU groups, global max-pool and a bias-free linear head.
* Data lives on the GPU as normalised fp16 tensors; augmentation (random translate,
  horizontal flip, cutout) is done on-device with two gathers + a couple of masks, so
  there is no dataloader in the loop at all.
* Inputs follow the verifier convention EXACTLY: float32 NCHW already standardised with
  mean=(0.4914,0.4822,0.4465), std=(0.2470,0.2435,0.2616). `forward` does NOT normalise.
* Test-time augmentation (mirror + 1px translate, 6 views) is baked into `forward` so a
  plain `torch.jit.load(path)(x).argmax(1)` gets it for free.
* One scripted checkpoint per epoch. The ScriptModule shares parameter storage with the
  eager model, so it is built once and re-saved each epoch (checked by check_artifacts.py).
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

T_PROC_START = time.time()  # process entry: everything after this counts as the run

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CACHE = "/workspace/data/cache"
LAST = {}  # trained model + holdout tensors of the most recent train() call (used by ablate.py)


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------
class ConvGroup(nn.Module):
    def __init__(self, cin: int, cout: int, bn_mom: float):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = nn.BatchNorm2d(cout, momentum=bn_mom)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(cout, momentum=bn_mom)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.norm1(self.pool(self.conv1(x))))
        x = self.act(self.norm2(self.conv2(x)))
        return x


class Net(nn.Module):
    def __init__(self, widths=(64, 256, 256), scale: float = 0.111, bn_mom: float = 0.4,
                 tta: int = 2, bn_affine: bool = False):
        super().__init__()
        w1, w2, w3 = widths
        self.whiten = nn.Conv2d(3, 24, 2, padding=0, bias=True)
        self.act0 = nn.GELU()
        self.g1 = ConvGroup(24, w1, bn_mom)
        self.g2 = ConvGroup(w1, w2, bn_mom)
        self.g3 = ConvGroup(w2, w3, bn_mom)
        self.pool = nn.MaxPool2d(3)
        self.head = nn.Linear(w3, 10, bias=False)
        self.scale = float(scale)
        self.tta = int(tta)
        for m in self.modules():  # BN scale frozen at 1, only the shift is learned
            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                m.weight.requires_grad_(bn_affine)
        nn.init.zeros_(self.head.weight)

    def core(self, x):
        x = self.act0(self.whiten(x))
        x = self.g3(self.g2(self.g1(x)))
        x = self.pool(x).flatten(1)
        return self.head(x) * self.scale

    def forward(self, x):
        # Verifier convention: x is float32 NCHW, already per-channel standardised.
        if self.training or self.tta < 1:
            return self.core(x)
        base = self.core(x) + self.core(x.flip(-1))
        if self.tta < 2:
            return base
        xp = F.pad(x, (1, 1, 1, 1), mode='reflect')
        a = xp[:, :, 0:32, 0:32]
        b = xp[:, :, 2:34, 2:34]
        shifted = (self.core(a) + self.core(a.flip(-1))
                   + self.core(b) + self.core(b.flip(-1))) * 0.5
        return 0.6 * base + 0.4 * shifted


@torch.no_grad()
def init_whitening(conv: nn.Conv2d, images: torch.Tensor, eps: float = 5e-4):
    """Set the frozen first layer to a ZCA whitening filter bank over 2x2 patches.

    `images`: (N,3,32,32) normalised float tensor from the TRAIN split.
    Output channels are +/- pairs of the scaled eigenvectors so that the following GELU
    keeps both signs of every whitened component.
    """
    k = conv.weight.shape[-1]
    patches = images.unfold(2, k, 1).unfold(3, k, 1)          # N,3,H',W',k,k
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3 * k * k).float()
    patches = patches[torch.randperm(patches.shape[0], device=patches.device)[:200_000]]
    patches = patches - patches.mean(dim=0, keepdim=True)     # (D,) broadcast over rows
    cov = (patches.T @ patches) / (patches.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(cov.double())
    filt = (evecs.T.reshape(-1, 3, k, k) / (evals + eps).sqrt().view(-1, 1, 1, 1)).float()
    conv.weight.data[:] = torch.cat([filt, -filt])
    conv.bias.data.zero_()
    conv.weight.requires_grad_(False)
    return patches, filt


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
def load_split(split, device):
    d = torch.load(os.path.join(CACHE, f"{split}.pt"))
    return d["x"].to(device), d["y"].to(device)


def normalize(x_u8):
    mean = torch.tensor(CIFAR_MEAN, device=x_u8.device).view(1, 3, 1, 1) * 255.0
    std = torch.tensor(CIFAR_STD, device=x_u8.device).view(1, 3, 1, 1) * 255.0
    return (x_u8.float() - mean) / std


def rand_translate(xp, out, gen):
    """Random crop of `out` from a reflect-padded batch, via two gathers."""
    B, C, _, W = xp.shape
    r = (xp.shape[-1] - out)
    ar = torch.arange(out, device=xp.device)
    sy = torch.randint(0, r + 1, (B, 1), device=xp.device, generator=gen)
    sx = torch.randint(0, r + 1, (B, 1), device=xp.device, generator=gen)
    iy = (ar.view(1, -1) + sy).view(B, 1, out, 1).expand(B, C, out, W)
    x = xp.gather(2, iy)
    ix = (ar.view(1, -1) + sx).view(B, 1, 1, out).expand(B, C, out, out)
    return x.gather(3, ix)


def rand_flip(x, gen):
    m = torch.rand(x.shape[0], 1, 1, 1, device=x.device, generator=gen) < 0.5
    return torch.where(m, x.flip(-1), x)


def rand_cutout(x, size, gen):
    if size <= 0:
        return x
    B, _, H, W = x.shape
    cy = torch.randint(0, H, (B, 1, 1), device=x.device, generator=gen)
    cx = torch.randint(0, W, (B, 1, 1), device=x.device, generator=gen)
    yy = torch.arange(H, device=x.device).view(1, H, 1)
    xx = torch.arange(W, device=x.device).view(1, 1, W)
    m = ((yy - cy).abs() < size / 2) & ((xx - cx).abs() < size / 2)
    return x.masked_fill(m.unsqueeze(1), 0.0)


# --------------------------------------------------------------------------------------
# eval helpers (used for held-out validation only; the graded ladder is scored by the
# verifier itself)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, x_norm, y, bs=1000):
    model.eval()
    correct = 0
    for i in range(0, x_norm.shape[0], bs):
        out = model(x_norm[i:i + bs].float())
        correct += (out.argmax(1) == y[i:i + bs]).sum().item()
    model.train()
    return correct / x_norm.shape[0]


# --------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------
def train(args):
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed + 1)

    xtr_u8, ytr = load_split("train", dev)
    if args.holdout > 0:
        g = torch.Generator().manual_seed(1234)          # fixed split, independent of --seed
        perm = torch.randperm(xtr_u8.shape[0], generator=g).to(dev)
        val_idx, tr_idx = perm[:args.holdout], perm[args.holdout:]
        xva, yva = normalize(xtr_u8[val_idx]), ytr[val_idx]
        xtr_u8, ytr = xtr_u8[tr_idx], ytr[tr_idx]
    else:
        xva = yva = None

    xtr = normalize(xtr_u8)
    pad = args.translate
    xpad = F.pad(xtr, (pad,) * 4, mode='reflect').half().contiguous(memory_format=torch.channels_last)
    N = xtr.shape[0]

    model = Net(widths=tuple(args.widths), scale=args.scale, bn_mom=args.bn_mom, tta=args.tta,
                bn_affine=bool(args.bn_affine))
    model = model.to(dev).to(memory_format=torch.channels_last)
    if args.whiten == "zca":
        init_whitening(model.whiten, xtr[:10000])   # frozen ZCA filters from TRAIN patches only
    # else: leave the default random init and let the layer train (ablation)
    del xtr

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.SGD([
        {"params": decay, "weight_decay": args.wd, "lr": args.lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": args.lr * args.bias_scaler},
    ], lr=args.lr, momentum=args.momentum, nesterov=True)
    base_lrs = [g["lr"] for g in opt.param_groups]

    steps_per_epoch = N // args.bs
    total = steps_per_epoch * args.epochs
    warm = max(1, int(round(args.warmup * total)))

    def lr_at(step):
        if step < warm:
            return step / warm
        return max(0.0, (total - step) / (total - warm))

    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.ls)
    os.makedirs(args.out, exist_ok=True)
    scripted = torch.jit.script(model) if args.save else None

    step = 0
    t_start = time.time()
    stamps, val_hist = [], []
    model.train()
    for epoch in range(1, args.epochs + 1):
        idx = torch.randperm(N, device=dev, generator=gen)
        for i in range(steps_per_epoch):
            b = idx[i * args.bs:(i + 1) * args.bs]
            xb = rand_translate(xpad[b], 32, gen)
            xb = rand_cutout(rand_flip(xb, gen), args.cutout, gen)
            f = lr_at(step)
            for gp, bl in zip(opt.param_groups, base_lrs):
                gp["lr"] = bl * f
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(xb), ytr[b])
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
        if args.save:
            scripted.save(os.path.join(args.out, f"ckpt_epoch_{epoch:04d}.pt"))
        stamps.append(time.time() - t_start)
        if args.val_every and xva is not None and (epoch % args.val_every == 0 or epoch == args.epochs):
            val_hist.append((epoch, round(stamps[-1], 2), round(evaluate(model, xva, yva), 4)))
            print(f"  ep{epoch:3d} t={stamps[-1]:6.2f}s val={val_hist[-1][2]:.4f}", flush=True)

    torch.cuda.synchronize()
    wall = time.time() - t_start
    # honest total: process entry (imports/data load/whitening included) -> last checkpoint
    res = {"args": vars(args), "wall_clock_sec": wall,
           "wall_clock_sec_total": stamps[-1] + (t_start - T_PROC_START) if stamps else None,
           "epoch_stamps": [round(s, 3) for s in stamps],
           "val_hist": val_hist, "params": sum(p.numel() for p in model.parameters())}
    if xva is not None:
        res["val_acc"] = evaluate(model, xva, yva)
    if len(stamps) > 1:
        n = len(stamps)
        res["t_star_if_last"] = (stamps[-1] - stamps[0]) * n / (n - 1)
    print(json.dumps({k: v for k, v in res.items() if k != "args"}), flush=True)
    LAST["model"], LAST["val"] = model, (xva, yva)
    if args.log:
        with open(args.log, "a") as f:
            f.write(json.dumps(res) + "\n")
    return res


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=32)
    p.add_argument("--bs", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.25)
    p.add_argument("--wd", type=float, default=5e-3)
    p.add_argument("--momentum", type=float, default=0.85)
    p.add_argument("--bias-scaler", type=float, default=8.0, dest="bias_scaler")
    p.add_argument("--ls", type=float, default=0.2)
    p.add_argument("--warmup", type=float, default=0.15)
    p.add_argument("--widths", type=int, nargs=3, default=[64, 256, 256])
    p.add_argument("--scale", type=float, default=1 / 9)
    p.add_argument("--bn-mom", type=float, default=0.4, dest="bn_mom")
    p.add_argument("--tta", type=int, default=2)
    p.add_argument("--whiten", type=str, default="zca", choices=["zca", "learned"])
    p.add_argument("--bn-affine", type=int, default=0, dest="bn_affine")
    p.add_argument("--translate", type=int, default=2)
    p.add_argument("--cutout", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout", type=int, default=0)
    p.add_argument("--val-every", type=int, default=0, dest="val_every")
    p.add_argument("--out", type=str, default="/workspace/scratch_ckpt")
    p.add_argument("--save", type=int, default=1)
    p.add_argument("--log", type=str, default="/workspace/runs.jsonl")
    return p.parse_args(argv)


if __name__ == "__main__":
    train(get_args())
