"""Train the CIFAR-10 convnet and write one scripted checkpoint per epoch.

Everything (images, augmentation, evaluation) lives on the GPU; there is no
DataLoader in the hot loop.  Each epoch is exactly one pass over the training
split, after which ``torch.jit.script(model).save(...)`` writes
``ckpt_epoch_{NNNN}.pt`` so the checkpoint ladder is contiguous from epoch 1.

Usage
-----
    python train.py --out DIR [--epochs N] [--seed S] [--holdout K] ...

``--holdout K`` removes the last K images of a fixed shuffled ordering of the
train split and reports accuracy on them instead of on the test split.  All
hyperparameter decisions in this project were made with ``--holdout 5000``;
the test split is never read by this script.
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from net import CIFAR_MEAN, CIFAR_STD, Net, init_whitening

CACHE = "/workspace/cache"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_split(name: str, device: str = "cuda"):
    d = torch.load(os.path.join(CACHE, f"{name}.pt"))
    return d["x"].to(device, non_blocking=True), d["y"].to(device, non_blocking=True)


def standardize(x_uint8: torch.Tensor, dtype=torch.float16) -> torch.Tensor:
    """uint8 NCHW -> per-channel standardized float, matching the verifier's
    ``ToTensor() + Normalize(CIFAR_MEAN, CIFAR_STD)`` transform exactly."""
    mean = torch.tensor(CIFAR_MEAN, device=x_uint8.device).view(1, 3, 1, 1) * 255
    std = torch.tensor(CIFAR_STD, device=x_uint8.device).view(1, 3, 1, 1) * 255
    return ((x_uint8.float() - mean) / std).to(dtype)


# --------------------------------------------------------------------------- #
# GPU augmentation
# --------------------------------------------------------------------------- #
def random_crop(padded: torch.Tensor, r: int, size: int = 32) -> torch.Tensor:
    """Random +/-r pixel translation, implemented as two gathers over a
    pre-reflection-padded batch (cheaper than building per-sample index grids)."""
    if r == 0:
        return padded
    n = len(padded)
    shifts = torch.randint(-r, r + 1, (n, 2), device=padded.device)
    tmp = torch.empty((n, 3, size, size + 2 * r), device=padded.device, dtype=padded.dtype)
    for s in range(-r, r + 1):
        m = shifts[:, 0] == s
        tmp[m] = padded[m, :, r + s:r + s + size, :]
    out = torch.empty((n, 3, size, size), device=padded.device, dtype=padded.dtype)
    for s in range(-r, r + 1):
        m = shifts[:, 1] == s
        out[m] = tmp[m, :, :, r + s:r + s + size]
    return out


def random_flip(x: torch.Tensor) -> torch.Tensor:
    m = (torch.rand(len(x), 1, 1, 1, device=x.device) < 0.5)
    return torch.where(m, x.flip(-1), x)


def random_cutout(x: torch.Tensor, size: int) -> torch.Tensor:
    """Zero a size x size square at a random center; 0 == the dataset mean colour
    because the images are standardized."""
    if size == 0:
        return x
    n, _, h, w = x.shape
    cy = torch.randint(0, h, (n, 1, 1), device=x.device)
    cx = torch.randint(0, w, (n, 1, 1), device=x.device)
    ys = torch.arange(h, device=x.device).view(1, h, 1)
    xs = torch.arange(w, device=x.device).view(1, 1, w)
    mask = ((ys - cy).abs() < size / 2) & ((xs - cx).abs() < size / 2)
    return x.masked_fill(mask.unsqueeze(1), 0.0)


# --------------------------------------------------------------------------- #
# schedule / optimizer
# --------------------------------------------------------------------------- #
def lr_at(step: int, total: int, peak_frac: float, start: float, end: float) -> float:
    """Triangular schedule: `start` -> 1.0 over the first `peak_frac` of steps,
    then 1.0 -> `end`."""
    warm = max(1, int(total * peak_frac))
    if step < warm:
        f = step / warm
        return start * (1 - f) + f
    f = (step - warm) / max(1, total - warm)
    return 1.0 * (1 - f) + end * f


def build_optimizer(model: Net, lr: float, wd: float, momentum: float, bias_scaler: float):
    """Two groups: decayed conv/linear weights, and undecayed BN/whitening biases
    trained at `bias_scaler` times the base learning rate."""
    biases, weights = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (biases if p.ndim <= 1 else weights).append(p)
    groups = [
        dict(params=weights, lr=lr, weight_decay=wd),
        dict(params=biases, lr=lr * bias_scaler, weight_decay=0.0),
    ]
    return torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=True)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model: Net, x: torch.Tensor, y: torch.Tensor, bs: int = 1000) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(x), bs):
        correct += (model(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum()
    model.train()
    return correct.item() / len(x)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train(cfg):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"

    os.makedirs(cfg.out, exist_ok=True)
    for f in os.listdir(cfg.out):                      # one ladder == one run
        if f.startswith("ckpt_epoch_"):
            os.remove(os.path.join(cfg.out, f))

    xtr_u8, ytr = load_split("train", dev)
    # A fixed permutation (seed 1234, independent of --seed) defines the holdout
    # so that every experiment validates on the same images.
    perm = torch.randperm(len(xtr_u8), generator=torch.Generator().manual_seed(1234)).to(dev)
    xtr_u8, ytr = xtr_u8[perm], ytr[perm]
    if cfg.holdout:
        xev_u8, yev = xtr_u8[-cfg.holdout:], ytr[-cfg.holdout:]
        xtr_u8, ytr = xtr_u8[:-cfg.holdout], ytr[:-cfg.holdout]
    else:
        xev_u8, yev = load_split("test", dev)

    xtr = standardize(xtr_u8)
    xev = standardize(xev_u8, dtype=torch.float32).contiguous(memory_format=torch.channels_last)
    del xtr_u8, xev_u8
    n = len(xtr)

    torch.manual_seed(cfg.seed)
    model = Net(cfg.w1, cfg.w2, cfg.w3, cfg.bn_momentum, cfg.scaling_factor, tta=not cfg.no_tta)
    model = model.to(dev).to(memory_format=torch.channels_last)
    init_whitening(model, xtr[:10000].float())
    model.train()
    scripted = torch.jit.script(model)   # shares parameter storage with `model`

    # Reflection-pad once; every epoch crops random 32x32 windows out of this.
    padded = F.pad(xtr, (cfg.translate,) * 4, "reflect").contiguous(memory_format=torch.channels_last)
    del xtr

    steps_per_epoch = (n + cfg.bs - 1) // cfg.bs
    total_steps = steps_per_epoch * cfg.epochs
    opt = build_optimizer(model, cfg.lr, cfg.wd, cfg.momentum, cfg.bias_scaler)
    base_lrs = [g["lr"] for g in opt.param_groups]

    t0 = time.time()
    step = 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        order = torch.randperm(n, device=dev)
        for i in range(0, n, cfg.bs):
            idx = order[i:i + cfg.bs]
            xb = random_crop(padded[idx], cfg.translate)
            xb = random_cutout(random_flip(xb), cfg.cutout)
            scale = lr_at(step, total_steps, cfg.peak_frac, cfg.lr_start, cfg.lr_end)
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * scale
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xb)
            loss = F.cross_entropy(logits.float(), ytr[idx], label_smoothing=cfg.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
        scripted.save(os.path.join(cfg.out, f"ckpt_epoch_{epoch:04d}.pt"))
        history.append({"epoch": epoch, "t": time.time() - t0})

    wall = time.time() - t0
    acc = evaluate(model, xev, yev)
    summary = dict(vars(cfg), acc=acc, wall_clock_sec=wall,
                   sec_per_epoch=wall / cfg.epochs, history=history)
    with open(os.path.join(cfg.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}), flush=True)
    return summary


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout", type=int, default=0)
    p.add_argument("--bs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1.5)
    p.add_argument("--wd", type=float, default=1.3e-3)
    p.add_argument("--momentum", type=float, default=0.85)
    p.add_argument("--bias_scaler", type=float, default=32.0)
    p.add_argument("--label_smoothing", type=float, default=0.2)
    p.add_argument("--peak_frac", type=float, default=0.23)
    p.add_argument("--lr_start", type=float, default=0.2)
    p.add_argument("--lr_end", type=float, default=0.07)
    p.add_argument("--translate", type=int, default=2)
    p.add_argument("--cutout", type=int, default=0)
    p.add_argument("--w1", type=int, default=128)
    p.add_argument("--w2", type=int, default=384)
    p.add_argument("--w3", type=int, default=512)
    p.add_argument("--bn_momentum", type=float, default=0.6)
    p.add_argument("--scaling_factor", type=float, default=1 / 9)
    p.add_argument("--no_tta", action="store_true")
    return p


if __name__ == "__main__":
    train(get_parser().parse_args())
