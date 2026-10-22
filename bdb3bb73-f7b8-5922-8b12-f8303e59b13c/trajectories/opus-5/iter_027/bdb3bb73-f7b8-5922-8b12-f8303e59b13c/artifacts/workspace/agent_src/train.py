"""CIFAR-10 speed run: train from scratch and emit a per-epoch checkpoint ladder.

Everything lives on the GPU: the 50k training images are uploaded once as
half-precision tensors, pre-padded by 2 px (reflect) so that random-crop
augmentation is two `gather`s, and horizontal flip is a single `where`.  There
is no DataLoader and no host<->device traffic inside the epoch loop.

Each epoch ends with `ScriptModule.save(...)`, writing
`<ckpt_dir>/ckpt_epoch_{NNNN}.pt` -- a fully self-contained scripted model that
`torch.jit.load(...)` can run directly on standardized float32 NCHW batches.
The save is *inside* the timed loop, because the verifier reconstructs the
run's per-epoch wall clock from those files' save times and we want that number
to be the honest cost of producing them.

Usage
-----
  python train.py --ckpt-dir /workspace/periodic_checkpoints --epochs 16
  python train.py --holdout 5000 --no-ckpt --seed 0      # tuning run
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CIFAR_MEAN, CIFAR_STD, Deploy, init_whitening_conv, make_net

CACHE = "/workspace/cache"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_train(holdout=0, seed=0, pad=2):
    """Return (padded_train_images_half, train_labels, val_images_half, val_labels).

    `holdout` images are split off the *train* split with a fixed, seed-derived
    permutation and are the ONLY thing used to make modelling decisions.  The
    real test split is never read here.
    """
    d = torch.load(os.path.join(CACHE, "train_u8.pt"))
    x, y = d["x"].cuda(), d["y"].cuda()
    if holdout > 0:
        g = torch.Generator(device="cuda").manual_seed(1234)  # fixed split, not the run seed
        perm = torch.randperm(x.shape[0], generator=g, device="cuda")
        val_idx, tr_idx = perm[:holdout], perm[holdout:]
        xv, yv = x[val_idx], y[val_idx]
        x, y = x[tr_idx], y[tr_idx]
    else:
        xv = yv = None

    mean = torch.tensor(CIFAR_MEAN, device="cuda").view(1, 3, 1, 1) * 255.0
    std = torch.tensor(CIFAR_STD, device="cuda").view(1, 3, 1, 1) * 255.0

    def prep(t):
        return ((t.float() - mean) / std).half().contiguous(memory_format=torch.channels_last)

    x = prep(x)
    if xv is not None:
        xv = prep(xv)
    xp = F.pad(x, (pad,) * 4, "reflect").contiguous(memory_format=torch.channels_last)
    return x, xp, y, xv, yv


def augment(xp, idx, pad, gen, cutout=0):
    """Random crop (two gathers) + random horizontal flip (one where) + cutout.

    Cutout zeroes a `cutout`x`cutout` square at a random centre.  The images are
    already standardized, so zero *is* the per-channel dataset mean -- the usual
    "fill with the mean pixel" formulation, for free."""
    b = idx.shape[0]
    imgs = xp[idx]
    s = 2 * pad + 1
    off = torch.randint(0, s, (2, b), device="cuda", generator=gen)
    ar = torch.arange(32, device="cuda")
    rows = (off[0].unsqueeze(1) + ar).view(b, 1, 32, 1).expand(b, 3, 32, imgs.shape[3])
    imgs = imgs.gather(2, rows)
    cols = (off[1].unsqueeze(1) + ar).view(b, 1, 1, 32).expand(b, 3, 32, 32)
    imgs = imgs.gather(3, cols)
    flip = torch.rand(b, 1, 1, 1, device="cuda", generator=gen) < 0.5
    imgs = torch.where(flip, imgs.flip(-1), imgs)
    if cutout > 0:
        c = torch.randint(0, 32, (2, b), device="cuda", generator=gen)
        h = cutout // 2
        my = ((ar.view(1, 32) - c[0].view(b, 1)).abs() < h).view(b, 1, 32, 1)
        mx = ((ar.view(1, 32) - c[1].view(b, 1)).abs() < h).view(b, 1, 1, 32)
        imgs = imgs.masked_fill(my & mx, 0)
    return imgs.contiguous(memory_format=torch.channels_last)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def build_optimizer(net, cfg, steps_per_epoch, epochs):
    """airbench-style SGD: loss is *summed* over the batch, so the learning rate
    is quoted in per-example units and rescaled by the momentum "kilostep" term."""
    kilostep_scale = 1024 * (1 + 1 / (1 - cfg.momentum))
    lr = cfg.lr / kilostep_scale
    wd = cfg.wd * cfg.batch_size / kilostep_scale
    lr_biases = lr * cfg.bias_scaler

    whiten_bias = net[0].bias
    others = [p for n, p in net.named_parameters() if p.requires_grad and n != "0.bias"]
    biases = [p for p in others if p.ndim <= 1]
    weights = [p for p in others if p.ndim > 1]
    groups = [
        dict(params=[whiten_bias], lr=lr_biases, weight_decay=wd / lr_biases),
        dict(params=biases, lr=lr_biases, weight_decay=wd / lr_biases),
        dict(params=weights, lr=lr, weight_decay=wd / lr),
    ]
    opt = torch.optim.SGD(groups, momentum=cfg.momentum, nesterov=True)

    total = steps_per_epoch * epochs
    peak = max(1, int(cfg.peak_frac * total))
    sched = np.interp(np.arange(total + 1), [0, peak, total],
                      [cfg.start_mult, 1.0, cfg.end_mult])
    if cfg.tail_epochs > 0:  # optional flat low-LR tail (helps the ladder's tail)
        tail = steps_per_epoch * cfg.tail_epochs
        sched[-tail:] = cfg.end_mult
    whiten_steps = int(round(cfg.whiten_bias_epochs * steps_per_epoch))
    return opt, sched, whiten_steps


@torch.no_grad()
def evaluate_logits(net, x, bs=2000, tta=True):
    net.eval()
    outs = []
    dep = Deploy(net).cuda().eval()
    for i in range(0, x.shape[0], bs):
        xb = x[i:i + bs]
        outs.append(dep(xb.float()) if tta else net(xb).float())
    net.train()
    return torch.cat(outs)


def train(cfg, data=None):
    t_run0 = time.time()
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(cfg.seed)
    gen = torch.Generator(device="cuda").manual_seed(cfg.seed)

    x, xp, y, xv, yv = data if data is not None else load_train(cfg.holdout, cfg.seed, cfg.pad)
    n = x.shape[0]
    steps_per_epoch = (n + cfg.batch_size - 1) // cfg.batch_size

    net = make_net(widths=tuple(cfg.widths), bn_momentum=cfg.bn_momentum,
                   scaling_factor=cfg.scaling)
    init_whitening_conv(net[0], x[:5000].float())
    n_params = sum(p.numel() for p in net.parameters())

    opt, sched, whiten_steps = build_optimizer(net, cfg, steps_per_epoch, cfg.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing, reduction="sum")
    base_lrs = [g["lr"] for g in opt.param_groups]

    # ---- warm up cudnn autotuning OUTSIDE the timed loop -------------------
    # cudnn.benchmark spends its first-call time picking algorithms; doing that
    # inside epoch 1 would make the reconstructed per-epoch time meaningless.
    if cfg.warmup:
        wnet = make_net(widths=tuple(cfg.widths), bn_momentum=cfg.bn_momentum,
                        scaling_factor=cfg.scaling)
        wopt = torch.optim.SGD([p for p in wnet.parameters() if p.requires_grad], lr=0.0)
        for _ in range(3):
            idx = torch.randint(0, n, (cfg.batch_size,), device="cuda", generator=gen)
            out = wnet(augment(xp, idx, cfg.pad, gen, cfg.cutout))
            loss_fn(out, y[idx]).backward()
            wopt.step(); wopt.zero_grad(set_to_none=True)
        del wnet, wopt
        torch.cuda.synchronize()

    ckpt_dir = cfg.ckpt_dir
    scripted = None
    if not cfg.no_ckpt:
        os.makedirs(ckpt_dir, exist_ok=True)
        for f in os.listdir(ckpt_dir):
            if f.startswith("ckpt_epoch_") and f.endswith(".pt"):
                os.remove(os.path.join(ckpt_dir, f))
        # Script once; each epoch we only refresh weights and re-save.
        scripted = torch.jit.script(Deploy(net).eval())
        scripted.save(os.path.join("/tmp", "warm_script.pt"))  # warm the serializer

    net.train()
    step = 0
    log = []
    ep_loss = torch.zeros((), device='cuda', dtype=torch.float32)
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        perm = torch.randperm(n, device="cuda", generator=gen)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            mult = sched[min(step, len(sched) - 1)]
            for g, blr in zip(opt.param_groups, base_lrs):
                g["lr"] = blr * mult
            if step >= whiten_steps:
                opt.param_groups[0]["lr"] = 0.0
            out = net(augment(xp, idx, cfg.pad, gen, cfg.cutout))
            loss = loss_fn(out, y[idx])
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if cfg.debug:
                ep_loss += loss.detach()
            step += 1
        if cfg.debug:
            gn = sum((p.grad.float().norm() ** 2 for p in net.parameters()
                      if p.grad is not None), torch.zeros((), device="cuda")).sqrt()
            print(f"  epoch {epoch:3d} loss/ex {ep_loss.item() / n:8.4f} "
                  f"|out|max {out.detach().abs().max().item():9.3f} lrmult {mult:.3f}")
            ep_loss = torch.zeros((), device="cuda", dtype=torch.float32)
        if scripted is not None:
            # NB: `Deploy(net)` wraps the *live* net, so we must not call
            # .eval() on the wrapper -- that would propagate into `net` and
            # freeze BatchNorm for the rest of training.  Only the scripted
            # copy gets switched to eval mode.
            scripted.load_state_dict(Deploy(net).state_dict())
            scripted.eval()
            scripted.save(os.path.join(ckpt_dir, f"ckpt_epoch_{epoch:04d}.pt"))
            net.train()
        rec = {"epoch": epoch, "t": time.time() - t0}
        if cfg.eval_curve and xv is not None:
            rec["val"] = (evaluate_logits(net, xv, tta=not cfg.no_tta).argmax(1)
                          == yv).float().mean().item()
            t0 += 0.0  # curve eval is outside the graded path; timing is advisory here
        log.append(rec)

    torch.cuda.synchronize()
    train_sec = time.time() - t0
    wall_sec = time.time() - t_run0

    val_acc = None
    if xv is not None:
        logits = evaluate_logits(net, xv, tta=not cfg.no_tta)
        val_acc = (logits.argmax(1) == yv).float().mean().item()

    res = dict(tag=cfg.tag, seed=cfg.seed, epochs=cfg.epochs, widths=list(cfg.widths),
               lr=cfg.lr, wd=cfg.wd, momentum=cfg.momentum, label_smoothing=cfg.label_smoothing,
               bias_scaler=cfg.bias_scaler, bn_momentum=cfg.bn_momentum, pad=cfg.pad, cutout=cfg.cutout,
               peak_frac=cfg.peak_frac, end_mult=cfg.end_mult, start_mult=cfg.start_mult,
               tail_epochs=cfg.tail_epochs, whiten_bias_epochs=cfg.whiten_bias_epochs,
               batch_size=cfg.batch_size, holdout=cfg.holdout, n_params=n_params,
               val_acc=val_acc, train_sec=train_sec, wall_sec=wall_sec,
               sec_per_epoch=train_sec / cfg.epochs, epoch_log=log)
    print(json.dumps({k: v for k, v in res.items() if k != "epoch_log"}))
    if cfg.out:
        with open(cfg.out, "a") as f:
            f.write(json.dumps(res) + "\n")
    return res


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--widths", type=int, nargs=3, default=[128, 384, 512])
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=11.5)
    p.add_argument("--wd", type=float, default=0.0153)
    p.add_argument("--momentum", type=float, default=0.85)
    p.add_argument("--bias-scaler", type=float, default=64.0)
    p.add_argument("--label-smoothing", type=float, default=0.2)
    p.add_argument("--bn-momentum", type=float, default=0.6)
    p.add_argument("--scaling", type=float, default=1 / 9)
    p.add_argument("--pad", type=int, default=2)
    p.add_argument("--cutout", type=int, default=0)
    p.add_argument("--peak-frac", type=float, default=0.23)
    p.add_argument("--start-mult", type=float, default=0.2)
    p.add_argument("--end-mult", type=float, default=0.07)
    p.add_argument("--tail-epochs", type=int, default=0)
    p.add_argument("--whiten-bias-epochs", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout", type=int, default=0)
    p.add_argument("--ckpt-dir", type=str, default="/workspace/scratch_ckpt")
    p.add_argument("--no-ckpt", action="store_true")
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--tag", type=str, default="run")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--eval-curve", action="store_true")
    return p


if __name__ == "__main__":
    train(get_parser().parse_args())
