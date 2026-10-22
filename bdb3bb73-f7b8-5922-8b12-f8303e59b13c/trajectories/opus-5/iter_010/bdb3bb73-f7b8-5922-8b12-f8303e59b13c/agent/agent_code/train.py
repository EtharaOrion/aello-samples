"""Train CIFAR-10 from scratch on one H100 and emit a per-epoch checkpoint ladder.

Every epoch is exactly one pass over the training split. After each epoch the
model is exported with `torch.jit.script(...).save(...)` to
<ckpt-dir>/ckpt_epoch_{NNNN}.pt, so each rung is a standalone loadable module.

Evaluation modes
  --eval holdout : score against a held-out slice of the TRAIN split (used for
                   every tuning decision; the test split is never read).
  --eval test    : score against the real test split (only used to report the
                   accuracy of an already-promoted ladder).
  --eval none    : no scoring at all (fastest; ladder timing only).
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from net import CifarNet, init_whitening

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
CACHE = "/workspace/data_cache"


def load_split(split, dev):
    x = torch.from_numpy(np.load(f"{CACHE}/{split}_x.npy")).to(dev)
    y = torch.from_numpy(np.load(f"{CACHE}/{split}_y.npy")).to(dev)
    return x, y


def standardize(x_uint8, dev, dtype):
    """uint8 NCHW -> the exact float tensor the verifier feeds the model."""
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1) * 255.0
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1) * 255.0
    return ((x_uint8.float() - mean) / std).to(dtype)


def make_optimizer(model, lr, wd, momentum):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    return torch.optim.SGD(
        [{"params": decay, "weight_decay": wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, momentum=momentum, nesterov=True)


@torch.no_grad()
def evaluate(model, x, y, bs=1000):
    """Score in eval mode + fp32, i.e. the same numerics as the verifier."""
    was_training = model.training
    model.eval()
    correct = 0
    for i in range(0, x.shape[0], bs):
        logits = model(x[i:i + bs])
        correct += (logits.argmax(1) == y[i:i + bs]).sum().item()
    if was_training:
        model.train()
    return correct / x.shape[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.4)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.85)
    p.add_argument("--ls", type=float, default=0.2)
    p.add_argument("--warmup", type=float, default=0.15)
    p.add_argument("--final-lr-frac", type=float, default=0.02)
    p.add_argument("--bn-mom", type=float, default=0.15)
    p.add_argument("--logit-scale", type=float, default=1.0)
    p.add_argument("--widths", type=int, nargs=3, default=[128, 384, 512])
    p.add_argument("--pad", type=int, default=2)
    p.add_argument("--cutout", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-dir", type=str, default="/workspace/scratch_ckpt")
    p.add_argument("--eval", choices=["none", "holdout", "test"], default="holdout")
    p.add_argument("--holdout", type=int, default=5000)
    p.add_argument("--save", type=int, default=1, help="write the ckpt ladder")
    p.add_argument("--tag", type=str, default="run")
    p.add_argument("--log", type=str, default="")
    args = p.parse_args()

    t_start = time.time()
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ---------------- data (all resident on the GPU) ----------------
    xtr_u8, ytr = load_split("train", dev)
    if args.eval == "holdout" and args.holdout > 0:
        # deterministic, class-balanced-ish split of the TRAIN set only
        g = torch.Generator(device="cpu").manual_seed(1234)
        perm = torch.randperm(xtr_u8.shape[0], generator=g).to(dev)
        val_idx, tr_idx = perm[:args.holdout], perm[args.holdout:]
        xva, yva = standardize(xtr_u8[val_idx], dev, torch.float32), ytr[val_idx]
        xtr_u8, ytr = xtr_u8[tr_idx], ytr[tr_idx]
    elif args.eval == "test":
        xte_u8, yte = load_split("test", dev)
        xva, yva = standardize(xte_u8, dev, torch.float32), yte
    else:
        xva = yva = None

    n = xtr_u8.shape[0]
    xtr = standardize(xtr_u8, dev, torch.float16)
    pad = args.pad
    xpad = F.pad(xtr, (pad,) * 4, mode="reflect").contiguous()
    side = 32 + 2 * pad
    base = (torch.arange(32, device=dev).view(32, 1) * side
            + torch.arange(32, device=dev).view(1, 32)).view(1, 1024)
    xpad_flat = xpad.view(n, 3, side * side)
    del xtr

    # ---------------- model ----------------
    model = CifarNet(*args.widths, bn_mom=args.bn_mom,
                     logit_scale=args.logit_scale).to(dev).to(memory_format=torch.channels_last)
    init_whitening(model, standardize(xtr_u8[:5000], dev, torch.float32))
    opt = make_optimizer(model, args.lr, args.wd, args.momentum)

    steps_per_epoch = (n + args.bs - 1) // args.bs
    total_steps = steps_per_epoch * args.epochs
    warm = max(1, int(args.warmup * total_steps))

    def lr_at(step):
        if step < warm:
            return args.lr * (step + 1) / warm
        frac = (step - warm) / max(1, total_steps - warm)
        return args.lr * ((1 - frac) + frac * args.final_lr_frac)

    # scripted view of the same parameter tensors: scripting once and calling
    # .save() each epoch keeps per-epoch checkpoint overhead to serialization only
    os.makedirs(args.ckpt_dir, exist_ok=True)
    scripted = torch.jit.script(model) if args.save else None

    # warm up cudnn autotuning + allocator before the ladder starts
    model.train()
    for _ in range(3):
        with torch.autocast("cuda", torch.bfloat16):
            loss = F.cross_entropy(model(xpad_flat[:args.bs].gather(
                2, base.expand(args.bs, 3, 1024)).view(args.bs, 3, 32, 32).float()
                .to(memory_format=torch.channels_last)), ytr[:args.bs])
        loss.backward()
    opt.zero_grad(set_to_none=True)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
    torch.cuda.synchronize()

    # ---------------- training ----------------
    history = []
    step = 0
    t_train0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, args.bs):
            idx = perm[i:i + args.bs]
            b = idx.shape[0]
            # random translate (one gather) + random mirror (one where)
            off = torch.randint(0, 2 * pad + 1, (b, 2), device=dev)
            gidx = (base + (off[:, 0:1] * side + off[:, 1:2])).view(b, 1, 1024).expand(b, 3, 1024)
            xb = xpad_flat[idx].gather(2, gidx).view(b, 3, 32, 32)
            flip = torch.rand(b, 1, 1, 1, device=dev) < 0.5
            xb = torch.where(flip, xb.flip(3), xb)
            if args.cutout > 0:
                c = args.cutout
                cy = torch.randint(0, 32, (b, 1, 1, 1), device=dev)
                cx = torch.randint(0, 32, (b, 1, 1, 1), device=dev)
                ar = torch.arange(32, device=dev)
                my = (ar.view(1, 1, 32, 1) - cy).abs() < c / 2
                mx = (ar.view(1, 1, 1, 32) - cx).abs() < c / 2
                xb = xb.masked_fill(my & mx, 0.0)
            xb = xb.float().to(memory_format=torch.channels_last)

            for gp in opt.param_groups:
                gp["lr"] = lr_at(step)
            with torch.autocast("cuda", torch.bfloat16):
                loss = F.cross_entropy(model(xb), ytr[idx], label_smoothing=args.ls)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

        if args.save:
            model.eval()
            scripted.save(os.path.join(args.ckpt_dir, f"ckpt_epoch_{epoch:04d}.pt"))
        acc = evaluate(model, xva, yva) if xva is not None else float("nan")
        history.append({"epoch": epoch, "acc": acc, "t": time.time() - t_train0})
        print(f"[{args.tag}] epoch {epoch:3d}  acc {acc:.4f}  "
              f"t {time.time() - t_train0:6.2f}s", flush=True)

    wall = time.time() - t_start
    result = {"tag": args.tag, "args": vars(args), "history": history,
              "final_acc": history[-1]["acc"], "wall_clock_sec": wall,
              "train_loop_sec": time.time() - t_train0}
    print(json.dumps({k: result[k] for k in
                      ("tag", "final_acc", "wall_clock_sec", "train_loop_sec")}), flush=True)
    if args.log:
        with open(args.log, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
