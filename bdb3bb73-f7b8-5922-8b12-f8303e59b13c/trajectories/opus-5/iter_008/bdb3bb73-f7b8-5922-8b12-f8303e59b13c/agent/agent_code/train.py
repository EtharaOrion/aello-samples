"""Train the CIFAR-10 speedrun net and write one scripted checkpoint per epoch.

Everything lives on the GPU: the 50k train images are cached once as a reflect-padded
fp16 tensor, and augmentation (random 32x32 crop from the 36x36 pad + random mirror) is
two gathers and a `where`.  There is no DataLoader in the timed loop.

Usage examples
--------------
  # held-out validation run (45k train / 5k val, no graded checkpoints)
  python train.py --holdout 5000 --epochs 18 --ckpt-dir /workspace/scratch_ckpt --no-save

  # graded run over the full 50k train split
  python train.py --epochs 18 --ckpt-dir /workspace/periodic_checkpoints
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import CIFAR_MEAN, CIFAR_STD, Net, init_whitening

CACHE = "/workspace/cache"
PAD = 2


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--bs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1.5)          # mean-loss LR at bs=1000
    p.add_argument("--wd", type=float, default=2e-4)
    p.add_argument("--bias-scaler", type=float, default=32.0)
    p.add_argument("--momentum", type=float, default=0.85)
    p.add_argument("--label-smoothing", type=float, default=0.2)
    p.add_argument("--widths", type=int, nargs=3, default=[128, 384, 512])
    p.add_argument("--logit-scale", type=float, default=1 / 9)
    p.add_argument("--bn-momentum", type=float, default=0.6)
    p.add_argument("--warmup-frac", type=float, default=0.23)
    p.add_argument("--lr-start", type=float, default=0.2)
    p.add_argument("--lr-end", type=float, default=0.07)
    p.add_argument("--whiten-bias-epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout", type=int, default=0)
    p.add_argument("--ckpt-dir", type=str, default="/workspace/scratch_ckpt")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--eval-test", action="store_true")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--json-out", type=str, default="")
    return p.parse_args()


_CACHE_GPU = {}


def load_gpu(split):
    if split in _CACHE_GPU:                      # sweeps reuse one process
        return _CACHE_GPU[split]
    x = np.load(f"{CACHE}/{split}_x.npy")
    y = np.load(f"{CACHE}/{split}_y.npy")
    x = torch.from_numpy(x).cuda().permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(CIFAR_MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device="cuda").view(1, 3, 1, 1)
    x = (x - mean) / std
    _CACHE_GPU[split] = (x, torch.from_numpy(y).cuda())
    return _CACHE_GPU[split]


def make_batch(padded, idx, ar, gen):
    """Random 32x32 crop + random horizontal mirror, straight out of the padded cache."""
    b = idx.numel()
    imgs = padded[idx]
    sy = torch.randint(0, 2 * PAD + 1, (b,), device="cuda", generator=gen)
    sx = torch.randint(0, 2 * PAD + 1, (b,), device="cuda", generator=gen)
    h = imgs.shape[-1]
    iy = (sy[:, None] + ar[None, :])[:, None, :, None].expand(b, 3, 32, h)
    imgs = torch.gather(imgs, 2, iy)
    ix = (sx[:, None] + ar[None, :])[:, None, None, :].expand(b, 3, 32, 32)
    imgs = torch.gather(imgs, 3, ix)
    flip = (torch.rand(b, device="cuda", generator=gen) < 0.5).view(b, 1, 1, 1)
    return torch.where(flip, imgs.flip(-1), imgs)


@torch.no_grad()
def evaluate(model, x, y, bs=1000):
    model.eval()
    correct = 0
    for i in range(0, len(x), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x[i:i + bs])
        correct += (out.float().argmax(1) == y[i:i + bs]).sum().item()
    model.train()
    return correct / len(x)


class Saver:
    """Script the model once, then just `.save()` each epoch.

    `torch.jit.script` returns a module that *shares* parameter and buffer storage with
    the source module (verified: matching `data_ptr()`s, and BN running stats updated by
    training show up in the saved file).  So one scripted view, permanently in eval mode,
    always reflects the live weights -- and re-scripting every epoch is pure waste.
    Measured: 57 ms -> 25 ms per checkpoint, i.e. ~7% off the mean epoch time that the
    verifier's t* is built from.
    """

    def __init__(self, model):
        self.sm = torch.jit.script(model)
        self.sm.eval()          # independent of the source module's train/eval flag

    def save(self, path):
        self.sm.save(path)


def main(args=None, verbose=True):
    args = args or get_args()
    t_start = time.time()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(args.seed + 1)

    xtr, ytr = load_gpu("train")
    if args.holdout:
        # The cache was shuffled once with a fixed seed, so the tail slice is a
        # class-balanced held-out sample that the model never sees.
        xva, yva = xtr[-args.holdout:], ytr[-args.holdout:]
        xtr, ytr = xtr[:-args.holdout], ytr[:-args.holdout]
    else:
        xva = yva = None
    n = len(xtr)
    padded = F.pad(xtr, (PAD,) * 4, mode="reflect").half().contiguous()
    xtr_eval = xtr.half()
    del xtr

    model = Net(*args.widths, logit_scale=args.logit_scale,
                bn_momentum=args.bn_momentum).cuda().to(memory_format=torch.channels_last)
    init_whitening(model.whiten, xtr_eval[:10000].float())
    model.train()

    decay, no_decay = [], []
    for k, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if ("norm" in k or k.endswith("bias")) else decay).append(p)
    lr = args.lr * args.bs / 1000.0
    opt = torch.optim.SGD(
        [dict(params=decay, lr=lr, weight_decay=args.wd),
         dict(params=no_decay, lr=lr * args.bias_scaler, weight_decay=0.0)],
        momentum=args.momentum, nesterov=True)

    steps_per_epoch = n // args.bs
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_frac))

    def lr_mult(step):
        if step < warmup:
            f = step / warmup
            return args.lr_start * (1 - f) + f
        f = (step - warmup) / max(1, total_steps - warmup)
        return (1 - f) + args.lr_end * f

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_mult)

    saver = None
    if not args.no_save:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        saver = Saver(model)
    ar = torch.arange(32, device="cuda")
    hist = []
    step = 0
    for epoch in range(1, args.epochs + 1):
        if epoch == args.whiten_bias_epochs + 1:
            model.whiten.bias.requires_grad_(False)
            model.whiten.bias.grad = None
        perm = torch.randperm(n, device="cuda", generator=gen)
        for i in range(steps_per_epoch):
            idx = perm[i * args.bs:(i + 1) * args.bs]
            xb = make_batch(padded, idx, ar, gen).to(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(xb), ytr[idx],
                                       label_smoothing=args.label_smoothing)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
        if saver is not None:
            saver.save(os.path.join(args.ckpt_dir, f"ckpt_epoch_{epoch:04d}.pt"))
        rec = {"epoch": epoch, "t": round(time.time() - t_start, 3)}
        if xva is not None and (epoch % args.eval_every == 0 or epoch == args.epochs):
            rec["val"] = evaluate(model, xva, yva)
        hist.append(rec)
        if verbose:
            print(json.dumps(rec), flush=True)

    total = time.time() - t_start
    out = {"tag": args.tag, "args": vars(args), "hist": hist, "wall_clock_sec": total}
    if xva is not None:
        out["final_val"] = hist[-1].get("val", evaluate(model, xva, yva))
    if args.eval_test:
        xte, yte = load_gpu("test")
        out["final_test"] = evaluate(model, xte.half(), yte)
    if verbose:
        print(json.dumps({k: v for k, v in out.items() if k != "args"}), flush=True)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f)
    return out


if __name__ == "__main__":
    main()
