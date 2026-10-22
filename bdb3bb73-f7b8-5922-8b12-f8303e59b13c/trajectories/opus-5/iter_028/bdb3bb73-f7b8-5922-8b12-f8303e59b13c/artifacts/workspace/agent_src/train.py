"""CIFAR-10 from scratch to 94%+ top-1 on a single H100.

Architecture lineage: Keller Jordan's "airbench" CIFAR speedrun net -- a frozen
2x2 ZCA-whitening conv followed by three conv/pool/BN/GELU groups and a
bias-free linear head.

Conventions that matter for grading:
  * The exported model consumes ALREADY-STANDARDIZED float32 NCHW input, i.e.
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)) applied
    to a [0,1] tensor. `forward` therefore does NOT normalize again. The ZCA
    statistics are estimated on standardized training images so that the train
    and verification pipelines agree exactly.
  * One periodic checkpoint per epoch, each a torch.jit.script'ed module saved
    with `.save()`. The scripted wrapper is built ONCE (it shares parameter
    storage with the live training net) so per-epoch export cost is just
    serialization, not recompilation.
  * Every epoch is exactly one pass over the training split.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CACHE = "/workspace/data_cache"


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class BatchNorm(nn.BatchNorm2d):
    """BN with a frozen (all-ones) scale; only the shift is learned."""

    def __init__(self, num_features, momentum, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, c_in, c_out, kernel_size=3):
        super().__init__(c_in, c_out, kernel_size=kernel_size,
                         padding=kernel_size // 2, bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        # Partial identity init: the first c_in output filters start as a
        # pass-through, which speeds up early optimisation.
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, c_in, c_out, bn_momentum):
        super().__init__()
        self.conv1 = Conv(c_in, c_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(c_out, bn_momentum)
        self.conv2 = Conv(c_out, c_out)
        self.norm2 = BatchNorm(c_out, bn_momentum)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.activ(self.norm1(self.pool(self.conv1(x))))
        x = self.activ(self.norm2(self.conv2(x)))
        return x


class Net(nn.Module):
    def __init__(self, widths=(128, 384, 512), bn_momentum=0.6, scale=1 / 9):
        super().__init__()
        w1, w2, w3 = widths
        self.whiten = nn.Conv2d(3, 24, kernel_size=2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.whiten_activ = nn.GELU()
        self.group1 = ConvGroup(24, w1, bn_momentum)
        self.group2 = ConvGroup(w1, w2, bn_momentum)
        self.group3 = ConvGroup(w2, w3, bn_momentum)
        self.pool = nn.MaxPool2d(3)
        self.head = nn.Linear(w3, 10, bias=False)
        self.head.weight.data.zero_()
        self.scale = scale

    def forward(self, x):
        x = self.whiten_activ(self.whiten(x))
        x = self.group3(self.group2(self.group1(x)))
        x = self.pool(x).flatten(1)
        return self.head(x) * self.scale


class Exported(nn.Module):
    """Inference wrapper: casts to the trunk dtype and applies 6-way TTA.

    TTA = {identity, mirror} x {centre, translate(-1,-1), translate(+1,+1)},
    all six variants evaluated in a single batched trunk pass.
    """

    def __init__(self, net, tta_weight=0.6):
        super().__init__()
        self.net = net
        self.tta_weight = tta_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is already per-channel standardized float32 NCHW; do not
        # re-normalize.
        x = x.to(memory_format=torch.channels_last).half()
        pad = F.pad(x, (1, 1, 1, 1), "reflect")
        v0 = x
        v1 = pad[:, :, 0:32, 0:32]
        v2 = pad[:, :, 2:34, 2:34]
        variants = torch.cat([v0, v0.flip(-1), v1, v1.flip(-1),
                              v2, v2.flip(-1)], 0)
        out = self.net(variants).float()
        n = x.size(0)
        centre = out[0:n] + out[n:2 * n]
        trans = (out[2 * n:3 * n] + out[3 * n:4 * n]
                 + out[4 * n:5 * n] + out[5 * n:6 * n]) / 2.0
        return self.tta_weight * centre + (1.0 - self.tta_weight) * trans


# --------------------------------------------------------------------------- #
# Data (kept resident on the GPU as uint8; standardized once into fp16)
# --------------------------------------------------------------------------- #
def load_split(split, device):
    x = torch.from_numpy(np.load(f"{CACHE}/{split}_x.npy")).to(device)
    y = torch.from_numpy(np.load(f"{CACHE}/{split}_y.npy")).to(device)
    x = x.permute(0, 3, 1, 2).contiguous()  # NHWC uint8 -> NCHW uint8
    return x, y


def standardize(x_uint8):
    mean = torch.tensor(CIFAR_MEAN, device=x_uint8.device).view(1, 3, 1, 1) * 255
    std = torch.tensor(CIFAR_STD, device=x_uint8.device).view(1, 3, 1, 1) * 255
    x = (x_uint8.float() - mean) / std
    return x.to(memory_format=torch.channels_last).half()


def init_whitening(layer, images, eps=5e-4):
    """Set the frozen 2x2 conv to the ZCA filters of the training patches."""
    p = images.float().unfold(2, 2, 1).unfold(3, 2, 1)
    p = p.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, 2, 2)
    flat = p.reshape(len(p), -1)
    cov = (flat.T @ flat) / len(flat)
    evals, evecs = torch.linalg.eigh(cov)
    filt = evecs.T.reshape(12, 3, 2, 2) / torch.sqrt(evals.view(-1, 1, 1, 1) + eps)
    layer.weight.data[:] = torch.cat([filt, -filt]).to(layer.weight.dtype)
    layer.bias.data.zero_()


# --------------------------------------------------------------------------- #
# GPU-side augmentation: per-sample random translate (two gathers) + mirror
# --------------------------------------------------------------------------- #
class Augment:
    def __init__(self, x, pad, device, cutout=0):
        self.pad = pad
        self.cutout = cutout
        self.side = 32 + 2 * pad
        self.padded = F.pad(x, (pad,) * 4, "reflect") if pad > 0 else x
        self.ar = torch.arange(32, device=device)

    def __call__(self, idx, gen):
        n = idx.numel()
        img = self.padded[idx]
        if self.pad > 0:
            off = torch.randint(0, 2 * self.pad + 1, (2, n), device=idx.device,
                                generator=gen)
            iy = (off[0].view(n, 1) + self.ar).view(n, 1, 32, 1)
            ix = (off[1].view(n, 1) + self.ar).view(n, 1, 1, 32)
            c = img.size(1)
            img = img.gather(2, iy.expand(n, c, 32, self.side))
            img = img.gather(3, ix.expand(n, c, 32, 32))
        flip = torch.rand(n, 1, 1, 1, device=idx.device, generator=gen) < 0.5
        img = torch.where(flip, img.flip(-1), img)
        if self.cutout > 0:
            r = self.cutout // 2
            cy = torch.randint(0, 32, (n, 1, 1), device=idx.device, generator=gen)
            cx = torch.randint(0, 32, (n, 1, 1), device=idx.device, generator=gen)
            m = (((self.ar.view(1, 32, 1) - cy).abs() < r)
                 & ((self.ar.view(1, 1, 32) - cx).abs() < r))
            # Zero in standardized space == replacing the patch with the
            # per-channel dataset mean.
            img = img.masked_fill(m.unsqueeze(1), 0.0)
        return img.contiguous(memory_format=torch.channels_last)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def build_optimizer(net, cfg, batch_size):
    # Nesterov momentum travels 1/(1-m) further than plain SGD for a constant
    # gradient; airbench normalises lr/wd by that factor so they are
    # batch-size- and momentum-independent.
    kilostep = 1024 * (1 + 1 / (1 - cfg.momentum))
    lr = cfg.lr / kilostep
    wd = cfg.wd * batch_size / kilostep
    biases = [p for n, p in net.named_parameters()
              if p.requires_grad and p.ndim <= 1]
    weights = [p for n, p in net.named_parameters()
               if p.requires_grad and p.ndim > 1]
    groups = [dict(params=weights, lr=lr, weight_decay=wd),
              dict(params=biases, lr=lr * cfg.bias_scaler, weight_decay=0.0)]
    return torch.optim.SGD(groups, momentum=cfg.momentum, nesterov=True)


def lr_curve(step, total, cfg):
    peak_at = cfg.warmup_frac * total
    if step <= peak_at:
        return cfg.lr_start + (1.0 - cfg.lr_start) * step / max(peak_at, 1.0)
    tail = cfg.hold_frac * total          # final fraction held at lr_end
    decay_end = total - tail
    if step >= decay_end:
        return cfg.lr_end
    f = (step - peak_at) / max(decay_end - peak_at, 1.0)
    return 1.0 + (cfg.lr_end - 1.0) * f


def train(cfg, train_x, train_y, ckpt_dir=None, eval_x=None, eval_y=None,
          log=print):
    device = "cuda"
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    xs = standardize(train_x)
    net = Net(cfg.widths, cfg.bn_momentum, cfg.scale).to(device)
    net = net.to(memory_format=torch.channels_last).half()
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.float()
    init_whitening(net.whiten, xs[:5000])

    aug = Augment(xs, cfg.translate, device, cfg.cutout)
    n = xs.size(0)
    steps_per_epoch = n // cfg.batch_size          # one genuine pass per epoch
    total_steps = steps_per_epoch * cfg.epochs
    opt = build_optimizer(net, cfg, cfg.batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_curve(s, total_steps, cfg))

    exported = None
    if ckpt_dir is not None:
        os.makedirs(ckpt_dir, exist_ok=True)
        # Scripted ONCE; shares parameter storage with `net`, so later epochs
        # only pay serialization cost.
        exported = torch.jit.script(Exported(net, cfg.tta_weight).eval())

    history = []
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        net.train()
        net.whiten.bias.requires_grad = (epoch <= cfg.whiten_bias_epochs)
        perm = torch.randperm(n, device=device, generator=gen)
        for i in range(steps_per_epoch):
            idx = perm[i * cfg.batch_size:(i + 1) * cfg.batch_size]
            out = net(aug(idx, gen))
            loss = F.cross_entropy(out, train_y[idx],
                                   label_smoothing=cfg.label_smoothing,
                                   reduction="sum")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

        if exported is not None:
            torch.jit.save(exported, f"{ckpt_dir}/ckpt_epoch_{epoch:04d}.pt")

        rec = dict(epoch=epoch, wall=time.time() - t0)
        if eval_x is not None and (epoch % cfg.eval_every == 0
                                   or epoch >= cfg.epochs - cfg.eval_last):
            rec["acc"] = evaluate(net, eval_x, eval_y, cfg.tta_weight)
        history.append(rec)
        log(f"  epoch {epoch:3d}  t={rec['wall']:7.2f}s"
            + (f"  acc={rec['acc'] * 100:.2f}" if "acc" in rec else ""))
    return net, history


@torch.no_grad()
def evaluate(net, x_uint8, y, tta_weight=0.6, batch=1000):
    """Accuracy with the same 6-way TTA the exported model uses."""
    net.eval()
    wrapper = Exported(net, tta_weight).eval()
    xs = (x_uint8.float() - torch.tensor(CIFAR_MEAN, device=x_uint8.device)
          .view(1, 3, 1, 1) * 255) / (
        torch.tensor(CIFAR_STD, device=x_uint8.device).view(1, 3, 1, 1) * 255)
    correct = 0
    for i in range(0, xs.size(0), batch):
        out = wrapper(xs[i:i + batch])
        correct += (out.argmax(1) == y[i:i + batch]).sum().item()
    net.train()
    return correct / xs.size(0)


def get_cfg(**over):
    d = dict(epochs=18, batch_size=1024, lr=11.5, momentum=0.85, wd=0.0153,
             bias_scaler=64.0, label_smoothing=0.2, whiten_bias_epochs=3,
             translate=2, cutout=0, widths=(128, 384, 512), bn_momentum=0.6,
             scale=1 / 9, tta_weight=0.6, seed=0,
             lr_start=0.2, lr_end=0.07, warmup_frac=0.23, hold_frac=0.0,
             eval_every=1000, eval_last=3)
    d.update(over)
    return argparse.Namespace(**d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=str, default="/workspace/periodic_checkpoints")
    ap.add_argument("--holdout", type=int, default=0,
                    help="if >0, train on 50000-N and validate on the last N")
    ap.add_argument("--set", action="append", default=[],
                    metavar="KEY=VAL", help="override any get_cfg() field")
    args = ap.parse_args()
    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        overrides[k] = eval(v)

    torch.backends.cudnn.benchmark = True
    run_t0 = time.time()
    tx, ty = load_split("train", "cuda")
    # The test split is NEVER loaded here. Held-out validation always comes
    # from a fixed slice of the train split; test accuracy is measured only
    # afterwards, by verify.py, on already-saved checkpoints.
    ex = ey = None
    if args.holdout:
        g = torch.Generator().manual_seed(1234)
        perm = torch.randperm(50000, generator=g).cuda()
        tr, va = perm[:-args.holdout], perm[-args.holdout:]
        ex, ey = tx[va], ty[va]
        tx, ty = tx[tr], ty[tr]

    cfg = get_cfg(epochs=args.epochs, seed=args.seed, **overrides)
    net, hist = train(cfg, tx, ty, ckpt_dir=args.ckpt, eval_x=ex, eval_y=ey)
    print(f"total wall clock {time.time() - run_t0:.2f}s")
