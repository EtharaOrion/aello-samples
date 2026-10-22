"""CIFAR-10 -> 94% top-1 on a single H100, from scratch.

Design notes
------------
* Architecture: a frozen 2x2 ZCA-whitening conv followed by three
  conv/pool/BN/GELU groups and a bias-free linear head (airbench-style).
* All 50k training images live on the GPU as a pre-padded bf16 tensor; the
  whole epoch is augmented in two `gather`s plus one `where` (random 32x32
  crop from a 36x36 reflect-padded image + random horizontal mirror).
  There is no DataLoader and no host<->device traffic in the training loop.
* INPUT CONVENTION: the model consumes inputs that are ALREADY per-channel
  standardized with CIFAR-10 mean/std, exactly matching the evaluation
  pipeline (`ToTensor()` then `Normalize(MEAN, STD)`).  `forward` therefore
  performs NO normalization of its own.
* Test-time augmentation (mirror + 1-px horizontal translations, 6 views)
  is folded into `forward` so that a plain `torch.jit.load(p)(x)` call
  reproduces the reported accuracy.
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
CACHE = "/workspace/cache"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_split_standardized(split, device="cuda"):
    """uint8 HWC cache -> standardized float32 NCHW tensor on `device`."""
    x = np.load(os.path.join(CACHE, f"{split}_x.npy"))
    y = np.load(os.path.join(CACHE, f"{split}_y.npy"))
    xt = torch.from_numpy(x).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    xt = (xt - mean) / std
    return xt, torch.from_numpy(y).to(device)


def stratified_holdout(x, y, n_holdout=5000, seed=1234):
    """Class-balanced split of the TRAIN set into (train, holdout).

    The cache is stored class-by-class, so a naive tail slice would hold out
    an entire class; this keeps both sides balanced and deterministic.
    """
    g = torch.Generator().manual_seed(seed)
    per_class = n_holdout // 10
    hold_idx = []
    for c in range(10):
        idx = (y == c).nonzero(as_tuple=True)[0].cpu()
        perm = idx[torch.randperm(idx.numel(), generator=g)]
        hold_idx.append(perm[:per_class])
    hold_idx = torch.cat(hold_idx).to(x.device)
    mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    mask[hold_idx] = False
    keep = mask.nonzero(as_tuple=True)[0]
    keep = keep[torch.randperm(keep.numel(), generator=g).to(x.device)]
    return x[keep], y[keep], x[hold_idx], y[hold_idx]


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class BatchNorm(nn.BatchNorm2d):
    """BN with a frozen unit scale; only the shift is learned (airbench trick)."""

    def __init__(self, ch, bn_momentum=0.6, eps=1e-12, learn_weight=False):
        super().__init__(ch, eps=eps, momentum=1.0 - bn_momentum)
        self.weight.requires_grad = learn_weight
        self.bias.requires_grad = True


class ConvGroup(nn.Module):
    def __init__(self, c_in, c_out, bn_momentum):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(c_out, bn_momentum)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.norm2 = BatchNorm(c_out, bn_momentum)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.activ(self.norm1(self.pool(self.conv1(x))))
        x = self.activ(self.norm2(self.conv2(x)))
        return x


class Net(nn.Module):
    """Whitening conv -> 3 conv groups -> global max-pool -> linear head.

    `tta` and `logit_scale` are plain attributes so the scripted graph keeps
    them as constants of the saved artifact.
    """

    def __init__(self, widths=(128, 384, 512), bn_momentum=0.6, logit_scale=1 / 9,
                 tta=True):
        super().__init__()
        self.whiten = nn.Conv2d(3, 24, kernel_size=2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.activ = nn.GELU()
        self.group1 = ConvGroup(24, widths[0], bn_momentum)
        self.group2 = ConvGroup(widths[0], widths[1], bn_momentum)
        self.group3 = ConvGroup(widths[1], widths[2], bn_momentum)
        self.pool = nn.MaxPool2d(3)
        self.head = nn.Linear(widths[2], 10, bias=False)
        self.logit_scale = float(logit_scale)
        self.tta = bool(tta)

    def _trunk(self, x):
        x = self.activ(self.whiten(x))
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.head(x) * self.logit_scale

    def forward(self, x):
        # x: float32 NCHW, ALREADY standardized with CIFAR-10 mean/std.
        if not self.tta or self.training:
            return self._trunk(x)
        # 6-view TTA: {identity, shift +1px, shift -1px} x {identity, mirror}.
        xf = x.flip(3)
        left = torch.cat([x[:, :, :, :1], x[:, :, :, :-1]], dim=3)
        right = torch.cat([x[:, :, :, 1:], x[:, :, :, -1:]], dim=3)
        lf = torch.cat([xf[:, :, :, :1], xf[:, :, :, :-1]], dim=3)
        rf = torch.cat([xf[:, :, :, 1:], xf[:, :, :, -1:]], dim=3)
        big = torch.cat([x, xf, left, lf, right, rf], dim=0)
        out = self._trunk(big)
        n = x.shape[0]
        acc = out[:n] + out[n:2 * n] + out[2 * n:3 * n]
        acc = acc + out[3 * n:4 * n] + out[4 * n:5 * n] + out[5 * n:]
        return acc / 6.0


# --------------------------------------------------------------------------
# whitening initialisation (computed from TRAIN data only)
# --------------------------------------------------------------------------
@torch.no_grad()
def init_whitening(conv, train_x, eps=5e-4, n_patch_images=10000):
    k = conv.weight.shape[2]
    x = train_x[:n_patch_images]
    patches = x.unfold(2, k, 1).unfold(3, k, 1)            # N,C,H',W',k,k
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3 * k * k).double()
    patches = patches - patches.mean(0, keepdim=True)
    cov = (patches.T @ patches) / patches.shape[0]
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.flip(0)
    eigvecs = eigvecs.T.reshape(3 * k * k, 3, k, k).flip(0)
    filt = (eigvecs / torch.sqrt(eigvals.view(-1, 1, 1, 1) + eps)).float()
    conv.weight.data[:] = torch.cat([filt, -filt])
    conv.bias.data.zero_()


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
DEFAULTS = dict(
    epochs=24,
    batch_size=1024,
    lr=0.60,                 # peak LR for the (scale-invariant) conv weights
    bias_scaler=32.0,        # BN-bias / whiten-bias LR multiplier
    momentum=0.85,
    weight_decay=0.024,      # L2 on conv+head weights only
    label_smoothing=0.20,
    warmup_frac=0.23,
    final_lr_frac=0.0,
    widths=(128, 384, 512),
    bn_momentum=0.6,
    logit_scale=1 / 9,
    translate=2,
    whiten_bias_epochs=3,
    seed=0,
)


def make_optimizer(model, cfg):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if ("bias" in name) else decay).append(p)
    groups = [
        dict(params=decay, lr=cfg["lr"], weight_decay=cfg["weight_decay"]),
        dict(params=no_decay, lr=cfg["lr"] * cfg["bias_scaler"], weight_decay=0.0),
    ]
    return torch.optim.SGD(groups, momentum=cfg["momentum"], nesterov=True)


def lr_multiplier(step, total, cfg):
    warm = max(1, int(cfg["warmup_frac"] * total))
    if step < warm:
        return 0.2 + 0.8 * step / warm
    frac = (step - warm) / max(1, total - warm)
    return 1.0 + (cfg["final_lr_frac"] - 1.0) * frac


@torch.no_grad()
def augment_epoch(padded, perm, pad, gen):
    """Random crop + random mirror for a whole (permuted) epoch at once."""
    x = padded[perm]                                    # N,3,32+2p,32+2p
    n = x.shape[0]
    dev = x.device
    dy = torch.randint(0, 2 * pad + 1, (n,), device=dev, generator=gen)
    dx = torch.randint(0, 2 * pad + 1, (n,), device=dev, generator=gen)
    ar = torch.arange(32, device=dev)
    ih = (dy[:, None] + ar[None, :]).view(n, 1, 32, 1).expand(n, 3, 32, x.shape[3])
    x = x.gather(2, ih)
    iw = (dx[:, None] + ar[None, :]).view(n, 1, 1, 32).expand(n, 3, 32, 32)
    x = x.gather(3, iw)
    flip = torch.rand(n, 1, 1, 1, device=dev, generator=gen) < 0.5
    x = torch.where(flip, x.flip(3), x)
    return x.contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def evaluate(model, x, y, bs=1024):
    model.eval()
    correct = 0
    for i in range(0, x.shape[0], bs):
        out = model(x[i:i + bs])
        correct += (out.argmax(1) == y[i:i + bs]).sum().item()
    model.train()
    return correct / x.shape[0]


def warmup(cfg, iters=6, device="cuda"):
    """Trigger cudnn autotuning / allocator growth on RANDOM data.

    This runs before training starts so that per-epoch wall times reflect
    steady state.  It touches no dataset image and its weights are thrown
    away -- the trained model is constructed afresh inside `train`.
    """
    m = Net(cfg["widths"], cfg["bn_momentum"], cfg["logit_scale"]).to(device)
    m = m.to(memory_format=torch.channels_last)
    m.train()
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    xb = torch.randn(cfg["batch_size"], 3, 32, 32, device=device,
                     dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
    yb = torch.zeros(cfg["batch_size"], dtype=torch.long, device=device)
    for _ in range(iters):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(m(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    del m, opt, xb, yb
    torch.cuda.synchronize()


def train(cfg, train_x, train_y, ckpt_dir=None, eval_x=None, eval_y=None,
          log=None, save_every=1):
    """Train one model.  Returns (per-epoch history, final model, timings).

    If `ckpt_dir` is given, one scripted checkpoint is written per epoch.
    `eval_x`/`eval_y` are optional and are ONLY ever a held-out slice of the
    training split (never the test split).
    """
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    dev = train_x.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(cfg["seed"] + 1234)

    # bf16 storage for the augmentation buffer: the two gathers are pure
    # memory traffic, and the conv stack consumes bf16 under autocast anyway.
    pad = cfg["translate"]
    padded = F.pad(train_x, (pad,) * 4, mode="reflect").to(torch.bfloat16).contiguous()

    model = Net(cfg["widths"], cfg["bn_momentum"], cfg["logit_scale"]).to(dev)
    model = model.to(memory_format=torch.channels_last)
    init_whitening(model.whiten, train_x)
    model.train()

    opt = make_optimizer(model, cfg)
    lossfn = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    n = train_x.shape[0]
    bs = cfg["batch_size"]
    steps_per_epoch = n // bs
    total_steps = steps_per_epoch * cfg["epochs"]
    base_lrs = [g["lr"] for g in opt.param_groups]

    # Checkpointing: one eval-mode replica is scripted once, then refreshed
    # in place each epoch and saved.  Scripting once keeps the per-epoch save
    # cost at a few tens of ms instead of re-compiling the graph every time.
    scripted, proto_sd = None, None
    if ckpt_dir is not None:
        os.makedirs(ckpt_dir, exist_ok=True)
        proto = Net(cfg["widths"], cfg["bn_momentum"], cfg["logit_scale"]).to(dev)
        proto.eval()
        scripted = torch.jit.script(proto)
        proto_sd = proto.state_dict()

    history, step = [], 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        if epoch == cfg["whiten_bias_epochs"] + 1:
            model.whiten.bias.requires_grad = False
        perm = torch.randperm(n, device=dev, generator=gen)
        xb_all = augment_epoch(padded, perm, pad, gen)
        yb_all = train_y[perm]
        for i in range(steps_per_epoch):
            mult = lr_multiplier(step, total_steps, cfg)
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * mult
            xb = xb_all[i * bs:(i + 1) * bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lossfn(model(xb), yb_all[i * bs:(i + 1) * bs])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
        del xb_all, yb_all

        if scripted is not None and (epoch % save_every == 0):
            with torch.no_grad():
                live = model.state_dict()
                for k, v in proto_sd.items():
                    v.copy_(live[k])
            scripted.save(os.path.join(ckpt_dir, f"ckpt_epoch_{epoch:04d}.pt"))

        rec = dict(epoch=epoch)
        if eval_x is not None:
            rec["acc"] = evaluate(model, eval_x, eval_y)
        torch.cuda.synchronize()
        rec["t"] = time.perf_counter() - t0
        history.append(rec)
        if log:
            log(rec)

    torch.cuda.synchronize()
    return history, model, time.perf_counter() - t0
