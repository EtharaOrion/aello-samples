"""Train the CIFAR-10 speedrun net and write one scripted checkpoint per epoch.

Two modes:
  --holdout 5000   carve a class-balanced holdout out of train, train on the
                   rest, report holdout accuracy.  Used for every selection
                   decision.  Checkpoints go to a scratch dir.
  --holdout 0      train on all 50 000 images; this is what produces the graded
                   ladder.

Batch size 1000 divides both 50 000 and 45 000, so every epoch is exactly one
full pass with no dropped tail and no ragged final batch.
"""
import argparse
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from net import CIFAR_MEAN, CIFAR_STD, Net, init_whitening

CACHE = "/workspace/cache"
HOLDOUT_SEED = 1234          # fixed forever: every sweep sees the same holdout


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--bs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=11.5)        # airbench units
    p.add_argument("--wd", type=float, default=0.0153)      # airbench units
    p.add_argument("--momentum", type=float, default=0.85)
    p.add_argument("--bias-scaler", type=float, default=64.0)
    p.add_argument("--label-smoothing", type=float, default=0.2)
    p.add_argument("--widths", type=int, nargs=3, default=[128, 384, 512])
    p.add_argument("--scale", type=float, default=1.0 / 9.0)
    p.add_argument("--bn-momentum", type=float, default=0.1)
    p.add_argument("--peak-frac", type=float, default=0.23)
    p.add_argument("--lr-start", type=float, default=0.2)
    p.add_argument("--lr-end", type=float, default=0.07)
    p.add_argument("--translate", type=int, default=2)
    p.add_argument("--tta-level", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout", type=int, default=0)
    p.add_argument("--ckpt-dir", type=str, default="/workspace/scratch_ckpt")
    p.add_argument("--save-ckpts", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=0)     # 0 = only at end
    p.add_argument("--tag", type=str, default="run")
    p.add_argument("--out-json", type=str, default="")
    return p.parse_args()


def standardize(x_u8, device):
    """uint8 NHWC numpy -> standardized bf16 NCHW cuda tensor."""
    x = torch.from_numpy(x_u8).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)
    return x.sub_(mean).div_(std)


def balanced_holdout(y, n, seed):
    """Indices of a class-balanced holdout of size `n`."""
    rng = np.random.default_rng(seed)
    per = n // 10
    idx = []
    for c in range(10):
        cls = np.where(y == c)[0]
        idx.append(rng.choice(cls, per, replace=False))
    idx = np.concatenate(idx)
    mask = np.ones(len(y), dtype=bool)
    mask[idx] = False
    return np.where(mask)[0], idx


@torch.no_grad()
def evaluate(model, x, y, bs=1000):
    model.eval()
    correct = 0
    for i in range(0, len(x), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x[i:i + bs])
        correct += (out.argmax(1) == y[i:i + bs]).sum().item()
    model.train()
    return correct / len(x)


def run(a):
    t0 = time.time()
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    xtr = np.load(f"{CACHE}/train_x.npy")
    ytr = np.load(f"{CACHE}/train_y.npy")
    if a.holdout > 0:
        tr_idx, ho_idx = balanced_holdout(ytr, a.holdout, HOLDOUT_SEED)
        xho, yho = xtr[ho_idx], ytr[ho_idx]
        xtr, ytr = xtr[tr_idx], ytr[tr_idx]
    else:
        xho = yho = None

    X = standardize(xtr, dev)
    Y = torch.from_numpy(ytr).to(dev)
    P = a.translate
    Xpad = F.pad(X, (P, P, P, P), mode="reflect").to(torch.bfloat16).contiguous()
    del X

    model = Net(tuple(a.widths), a.scale, a.bn_momentum, a.tta_level).to(dev)
    init_whitening(model, standardize(xtr[:10000], dev))
    model = model.to(memory_format=torch.channels_last).train()
    n_par = sum(p.numel() for p in model.parameters())

    norm_biases = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    filters = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    ks = a.bs * (1 + 1 / (1 - a.momentum))
    lr = a.lr / ks
    wd = a.wd * a.bs / ks
    lr_b = lr * a.bias_scaler
    opt = torch.optim.SGD(
        [dict(params=norm_biases, lr=lr_b, weight_decay=wd / lr_b),
         dict(params=filters, lr=lr, weight_decay=wd / lr)],
        momentum=a.momentum, nesterov=True)

    N = len(Y)
    steps_per_epoch = N // a.bs
    total = steps_per_epoch * a.epochs
    knots = [0, int(a.peak_frac * total), total]
    vals = [a.lr_start, 1.0, a.lr_end]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: float(np.interp(s, knots, vals)))

    os.makedirs(a.ckpt_dir, exist_ok=True)
    saver = None
    if a.save_ckpts:
        saver = torch.jit.script(
            Net(tuple(a.widths), a.scale, a.bn_momentum, a.tta_level).to(dev).eval())

    ar32 = torch.arange(32, device=dev)
    S = 32 + 2 * P
    hist = []
    for ep in range(1, a.epochs + 1):
        perm = torch.randperm(N, device=dev)
        for s in range(steps_per_epoch):
            idx = perm[s * a.bs:(s + 1) * a.bs]
            xb = Xpad[idx]
            B = xb.shape[0]
            if P > 0:
                oy = torch.randint(0, 2 * P + 1, (B,), device=dev)
                ox = torch.randint(0, 2 * P + 1, (B,), device=dev)
                r = (oy[:, None] + ar32)[:, None, :, None].expand(B, 3, 32, S)
                xb = xb.gather(2, r)
                c = (ox[:, None] + ar32)[:, None, None, :].expand(B, 3, 32, 32)
                xb = xb.gather(3, c)
            flip = torch.rand(B, 1, 1, 1, device=dev) < 0.5
            xb = torch.where(flip, xb.flip(3), xb).contiguous(
                memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.logits(xb)
                loss = F.cross_entropy(out.float(), Y[idx],
                                       label_smoothing=a.label_smoothing,
                                       reduction="sum")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

        if saver is not None:
            saver.load_state_dict(model.state_dict())
            saver.save(os.path.join(a.ckpt_dir, f"ckpt_epoch_{ep:04d}.pt"))
        if a.eval_every and (ep % a.eval_every == 0) and xho is not None:
            hist.append((ep, time.time() - t0,
                         evaluate(model, standardize(xho, dev),
                                  torch.from_numpy(yho).to(dev))))

    torch.cuda.synchronize()
    wall = time.time() - t0

    acc = None
    if xho is not None:
        acc = evaluate(model, standardize(xho, dev), torch.from_numpy(yho).to(dev))

    res = dict(tag=a.tag, seed=a.seed, epochs=a.epochs, params=n_par,
               holdout=a.holdout, holdout_acc=acc, wall_clock_sec=wall,
               hist=hist, cfg={k: v for k, v in vars(a).items()})
    print(json.dumps({k: v for k, v in res.items() if k != "cfg"}), flush=True)
    if a.out_json:
        with open(a.out_json, "a") as f:
            f.write(json.dumps(res) + "\n")
    return res


if __name__ == "__main__":
    run(get_args())
