"""CIFAR-10 -> 94% top-1 speedrun on 1xH100.

Design notes
------------
* Input convention: the model is trained and evaluated on inputs that are ALREADY
  per-channel standardized with CIFAR-10 train statistics, exactly matching the
  verifier's `ToTensor() -> Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))`.
  `forward` therefore performs NO normalization of its own.
* All 50k train images live on the GPU as uint8; augmentation is done on-device,
  so there is no DataLoader in the training loop.
* The first conv is a frozen 2x2 ZCA-style whitening filter initialized from the
  patch covariance of the TRAIN split only (no test data is ever touched).
* Test-time augmentation is folded into the scripted `forward`, so the verifier
  gets it for free with a plain `model(x)` call.
* torch.compile is deliberately NOT used: its warmup would land inside epoch 1,
  which the verifier's `(save[N]-save[1])*N/(N-1)` estimator excludes, shrinking
  reported t* without any genuine speedup. See REPORT.md.
"""
import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    epochs: int = 16
    batch_size: int = 1024
    lr: float = 2.0                 # peak LR for (unit-scaled) weight groups
    momentum: float = 0.85
    nesterov: bool = True
    wd: float = 0.0153              # decoupled-ish; applied to conv/linear weights
    bias_lr_scale: float = 32.0     # BN affine + head get a larger LR
    label_smoothing: float = 0.20
    logit_scale: float = 0.111      # tempers logits so LS-CE is well conditioned
    warmup_frac: float = 0.16
    final_lr_frac: float = 0.02
    widths: tuple = (128, 384, 512)
    whiten_ch: int = 24
    whiten_eps: float = 5e-4
    bn_momentum: float = 0.2
    pad: int = 2                    # random-translate padding
    flip: bool = True
    cutout: int = 0                 # 0 = off, else square side
    tta: int = 2                    # 0 none, 1 mirror, 2 mirror+1px translate
    seed: int = 0
    holdout: int = 0                # >0 => train on 50k-holdout, eval on holdout
    dtype: str = "bf16"


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class ConvGroup(nn.Module):
    def __init__(self, cin: int, cout: int, bn_mom: float):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout, momentum=bn_mom)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout, momentum=bn_mom)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.gelu(self.bn1(x))
        x = self.conv2(x)
        x = F.gelu(self.bn2(x))
        return x


class Net(nn.Module):
    """Expects per-channel STANDARDIZED NCHW float input (verifier convention)."""

    def __init__(self, cfg: Config):
        super().__init__()
        w1, w2, w3 = cfg.widths
        self.whiten = nn.Conv2d(3, cfg.whiten_ch, 2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.whiten.bias.requires_grad = False
        self.g1 = ConvGroup(cfg.whiten_ch, w1, cfg.bn_momentum)
        self.g2 = ConvGroup(w1, w2, cfg.bn_momentum)
        self.g3 = ConvGroup(w2, w3, cfg.bn_momentum)
        self.head = nn.Linear(w3, 10, bias=False)
        self.tta = int(cfg.tta)
        self.logit_scale = float(cfg.logit_scale)

    def features(self, x: Tensor) -> Tensor:
        x = F.gelu(self.whiten(x))
        x = self.g1(x)
        x = self.g2(x)
        x = self.g3(x)
        x = F.max_pool2d(x, 3)
        x = torch.flatten(x, 1)
        return self.head(x) * self.logit_scale

    def forward(self, x: Tensor) -> Tensor:
        # Inputs are ALREADY per-channel standardized (verifier convention); do
        # NOT normalize here. The verifier feeds float32 while the weights are
        # bf16, so cast to the parameter dtype; logits are accumulated in fp32.
        x = x.to(self.whiten.weight.dtype)
        if self.tta == 0:
            return self.features(x).float()
        if self.tta == 1:
            return self.features(x).float() + self.features(torch.flip(x, [3])).float()
        # tta == 2: mirror pair, plus a 1px-translated mirror pair (half weight)
        out = self.features(x).float() + self.features(torch.flip(x, [3])).float()
        xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
        xs = xp[:, :, 1:33, 0:32]
        out = out + 0.5 * (self.features(xs).float()
                           + self.features(torch.flip(xs, [3])).float())
        return out


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
_CACHE = {}


def load_cached(split: str, device):
    """Loads once per process; sweeps reuse the GPU-resident copy."""
    if split not in _CACHE:
        d = torch.load(f"/workspace/cache/{split}.pt", weights_only=False)
        _CACHE[split] = (d["x"].to(device), d["y"].to(device))
    return _CACHE[split]


def standardize(x_uint8: Tensor, dtype) -> Tensor:
    mean = torch.tensor(CIFAR_MEAN, device=x_uint8.device).view(1, 3, 1, 1) * 255.0
    std = torch.tensor(CIFAR_STD, device=x_uint8.device).view(1, 3, 1, 1) * 255.0
    return ((x_uint8.float() - mean) / std).to(dtype)


def init_whitening(model: Net, x_std: Tensor, eps: float):
    """Frozen 2x2 whitening conv from TRAIN-split patch covariance."""
    with torch.no_grad():
        patches = (x_std[:20000].float()
                   .unfold(2, 2, 1).unfold(3, 2, 1)          # N,C,H',W',2,2
                   .permute(0, 2, 3, 1, 4, 5)
                   .reshape(-1, 3 * 2 * 2))
        patches = patches - patches.mean(0, keepdim=True)
        cov = (patches.T @ patches) / (patches.shape[0] - 1)
        evals, evecs = torch.linalg.eigh(cov)
        scale = (evals + eps).rsqrt()
        filt = (evecs * scale).T.reshape(12, 3, 2, 2)        # 12 whitening filters
        n = model.whiten.weight.shape[0] // 2
        w = torch.cat([filt[:n], -filt[:n]], dim=0)
        model.whiten.weight.copy_(w.to(model.whiten.weight.dtype))
        model.whiten.bias.zero_()


# --------------------------------------------------------------------------- #
# GPU augmentation
# --------------------------------------------------------------------------- #
def augment(x: Tensor, cfg: Config, gen: torch.Generator) -> Tensor:
    b, c, h, w = x.shape
    dev = x.device
    if cfg.flip:
        m = torch.rand(b, 1, 1, 1, device=dev, generator=gen) < 0.5
        x = torch.where(m, torch.flip(x, [3]), x)
    p = cfg.pad
    if p > 0:
        xp = F.pad(x, (p, p, p, p))          # zeros == dataset mean in std space
        wp = w + 2 * p
        ar = torch.arange(h, device=dev)
        r = torch.randint(0, 2 * p + 1, (b,), device=dev, generator=gen)
        cc = torch.randint(0, 2 * p + 1, (b,), device=dev, generator=gen)
        ir = (r[:, None] + ar)[:, None, :, None].expand(b, c, h, wp)
        xp = xp.gather(2, ir)
        ic = (cc[:, None] + ar)[:, None, None, :].expand(b, c, h, w)
        x = xp.gather(3, ic)
    if cfg.cutout > 0:
        s = cfg.cutout
        cy = torch.randint(0, h, (b, 1, 1, 1), device=dev, generator=gen)
        cx = torch.randint(0, w, (b, 1, 1, 1), device=dev, generator=gen)
        yy = torch.arange(h, device=dev).view(1, 1, h, 1)
        xx = torch.arange(w, device=dev).view(1, 1, 1, w)
        keep = ~(((yy - cy).abs() < s // 2) & ((xx - cx).abs() < s // 2))
        x = x * keep
    return x


# --------------------------------------------------------------------------- #
# Eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model: nn.Module, x_std: Tensor, y: Tensor, bs: int = 1000) -> float:
    model.eval()
    correct = 0
    for i in range(0, x_std.shape[0], bs):
        logits = model(x_std[i:i + bs])
        correct += (logits.argmax(1) == y[i:i + bs]).sum().item()
    model.train()
    return correct / x_std.shape[0]


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def train(cfg: Config, ckpt_dir=None, eval_each_epoch=False, log=print):
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    dt = torch.bfloat16 if cfg.dtype == "bf16" else torch.float16

    xtr_u8, ytr = load_cached("train", dev)
    x_all = standardize(xtr_u8, dt)

    if cfg.holdout > 0:
        g = torch.Generator().manual_seed(1234)          # fixed split, seed-independent
        perm = torch.randperm(x_all.shape[0], generator=g).to(dev)
        v_idx, t_idx = perm[:cfg.holdout], perm[cfg.holdout:]
        x_val, y_val = x_all[v_idx], ytr[v_idx]
        x_std, y = x_all[t_idx].contiguous(), ytr[t_idx].contiguous()
    else:
        x_std, y = x_all, ytr
        x_val = y_val = None

    n = x_std.shape[0]
    model = Net(cfg).to(dev).to(memory_format=torch.channels_last)
    init_whitening(model, x_std, cfg.whiten_eps)
    model = model.to(dt)
    for m in model.modules():                            # keep BN in fp32
        if isinstance(m, nn.BatchNorm2d):
            m.float()

    decay, no_decay = [], []
    for name, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        (no_decay if prm.ndim <= 1 else decay).append(prm)
    opt = torch.optim.SGD(
        [{"params": decay, "lr": cfg.lr, "weight_decay": cfg.wd},
         {"params": no_decay, "lr": cfg.lr * cfg.bias_lr_scale, "weight_decay": 0.0}],
        momentum=cfg.momentum, nesterov=cfg.nesterov)

    steps_per_epoch = n // cfg.batch_size                # drop-last: uniform epochs
    total = steps_per_epoch * cfg.epochs
    warm = max(1, int(total * cfg.warmup_frac))

    def lr_at(s):
        if s < warm:
            return s / warm
        f = (s - warm) / max(1, total - warm)
        return 1.0 + f * (cfg.final_lr_frac - 1.0)

    lossf = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    gen = torch.Generator(device=dev).manual_seed(cfg.seed + 7)
    base_lrs = [g_["lr"] for g_ in opt.param_groups]

    # Script ONCE; the ScriptModule shares parameter/buffer storage with `model`,
    # so each epoch's .save() serializes that epoch's live weights. Verified by
    # tests/test_share.py and by the strictly-increasing accuracy ladder.
    smodel = torch.jit.script(model) if ckpt_dir else None
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    hist, step = [], 0
    torch.cuda.synchronize()
    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        idx = torch.randperm(n, device=dev, generator=gen)
        for i in range(steps_per_epoch):
            b = idx[i * cfg.batch_size:(i + 1) * cfg.batch_size]
            xb = augment(x_std[b], cfg, gen).to(memory_format=torch.channels_last)
            for gi, gp in enumerate(opt.param_groups):
                gp["lr"] = base_lrs[gi] * lr_at(step)
            out = model.features(xb)
            loss = lossf(out.float(), y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1

        if ckpt_dir:
            smodel.eval()
            smodel.save(os.path.join(ckpt_dir, f"ckpt_epoch_{ep:04d}.pt"))
            smodel.train()
        if eval_each_epoch and x_val is not None:
            hist.append((ep, evaluate(model, x_val, y_val), time.time() - t0))
            log(f"  ep{ep:3d} acc={hist[-1][1]*100:.2f} t={hist[-1][2]:.1f}s")

    torch.cuda.synchronize()
    wall = time.time() - t0
    acc = evaluate(model, x_val, y_val) if x_val is not None else None
    return {"model": model, "acc": acc, "wall": wall, "hist": hist,
            "steps_per_epoch": steps_per_epoch}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=0)
    ap.add_argument("--ckpt-dir", type=str, default=None)
    ap.add_argument("--eval-each", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()

    cfg = Config(epochs=a.epochs, seed=a.seed, holdout=a.holdout)
    for kv in a.set:
        k, v = kv.split("=", 1)
        cur = getattr(cfg, k)
        setattr(cfg, k, type(cur)(eval(v)) if not isinstance(cur, tuple) else eval(v))
    print(json.dumps({k: str(v) for k, v in asdict(cfg).items()}))

    r = train(cfg, ckpt_dir=a.ckpt_dir, eval_each_epoch=a.eval_each)
    print(f"wall={r['wall']:.2f}s steps/ep={r['steps_per_epoch']} acc={r['acc']}")
