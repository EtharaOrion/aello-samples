"""Train the CIFAR-10 speedrun net and write a per-epoch checkpoint ladder.

Everything -- data, augmentation, evaluation -- lives on the GPU; there is no
DataLoader in the hot loop.  Each epoch is exactly one pass over the training
split, and each epoch ends by writing one scripted checkpoint.

Protocol note: the test split is never read by this script.  ``--holdout N``
carves a stratified N-image validation slice out of the *train* split, which is
the only signal used for hyperparameter decisions.  Test accuracy is measured
afterwards, from the saved checkpoints, by ``eval_ladder.py``.
"""

import argparse
import copy
import json
import math
import os
import time

PROC_START = time.perf_counter()  # process start, for honest total wall-clock

import numpy as np
import torch
import torch.nn.functional as F

from model import CIFAR_MEAN, CIFAR_STD, Net, init_whitening

CACHE = "/workspace/cache"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=float, default=30)
    p.add_argument("--bs", type=int, default=1024)
    p.add_argument("--lr", type=float, default=11.5)          # per-1024-example LR
    p.add_argument("--wd", type=float, default=0.0153)        # per-1024-example WD
    p.add_argument("--momentum", type=float, default=0.85)
    p.add_argument("--bias-scaler", type=float, default=64.0)
    p.add_argument("--bias-wd-scale", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.2)
    p.add_argument("--widths", type=int, nargs=3, default=[128, 384, 512])
    p.add_argument("--bn-momentum", type=float, default=0.4)
    p.add_argument("--scale", type=float, default=1.0 / 9)
    p.add_argument("--translate", type=int, default=2)
    p.add_argument("--cutout", type=int, default=0)
    p.add_argument("--warmup-frac", type=float, default=0.23)
    p.add_argument("--final-lr-frac", type=float, default=0.07)
    p.add_argument("--start-lr-frac", type=float, default=0.2)
    p.add_argument("--whiten-bias-epochs", type=float, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout", type=int, default=0)
    p.add_argument("--outdir", type=str, default="/workspace/scratch_ckpt")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--eval-every", type=int, default=0, help="holdout eval cadence (0=last only)")
    p.add_argument("--ablate", type=str, default="", choices=["", "whiten", "dirac", "gamma"])
    p.add_argument("--no-flip", action="store_true")
    p.add_argument("--tag", type=str, default="")
    return p.parse_args()


def stratified_holdout(y, n, seed):
    """Return (train_idx, val_idx) with ``n`` validation images, class-balanced."""
    rng = np.random.RandomState(seed)
    per = n // 10
    val = []
    for c in range(10):
        idx = np.where(y == c)[0]
        val.append(rng.permutation(idx)[:per])
    val = np.concatenate(val)
    mask = np.ones(len(y), dtype=bool)
    mask[val] = False
    return np.where(mask)[0], val


def load_split(holdout, seed):
    x = np.load(f"{CACHE}/train_x.npy")
    y = np.load(f"{CACHE}/train_y.npy")
    if holdout > 0:
        tr, va = stratified_holdout(y, holdout, seed=1234)  # fixed split across seeds
        return (x[tr], y[tr]), (x[va], y[va])
    return (x, y), None


def to_gpu_standardized(x_np, y_np):
    x = torch.from_numpy(x_np).cuda().permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(CIFAR_MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device="cuda").view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.contiguous(memory_format=torch.channels_last), torch.from_numpy(y_np).cuda()


def make_augmenter(x_pad, translate, flip=True):
    """Per-image random crop (two gathers) + random mirror (one where)."""
    ar = torch.arange(32, device="cuda")
    span = 2 * translate + 1

    def aug(idx):
        b = idx.numel()
        img = x_pad.index_select(0, idx)
        if translate > 0:
            oy = torch.randint(0, span, (b,), device="cuda")
            ox = torch.randint(0, span, (b,), device="cuda")
            rows = (oy[:, None] + ar)[:, None, :, None].expand(b, 3, 32, img.shape[3])
            img = img.gather(2, rows)
            cols = (ox[:, None] + ar)[:, None, None, :].expand(b, 3, 32, 32)
            img = img.gather(3, cols)
        if not flip:
            return img.contiguous(memory_format=torch.channels_last)
        m = torch.rand(b, 1, 1, 1, device="cuda") < 0.5
        return torch.where(m, img.flip(3), img).contiguous(memory_format=torch.channels_last)

    return aug


def apply_cutout(img, size):
    b = img.shape[0]
    cy = torch.randint(0, 32, (b, 1, 1), device="cuda")
    cx = torch.randint(0, 32, (b, 1, 1), device="cuda")
    ys = torch.arange(32, device="cuda").view(1, 32, 1)
    xs = torch.arange(32, device="cuda").view(1, 1, 32)
    half = size // 2
    keep = ~(((ys - cy).abs() < half) & ((xs - cx).abs() < half))
    return img * keep.unsqueeze(1)


@torch.no_grad()
def evaluate(model, x, y, bs=1000, tta=False):
    was_tta, was_train = model.tta, model.training
    model.tta, correct = tta, 0
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(x), bs):
            out = model(x[i:i + bs])
            correct += (out.float().argmax(1) == y[i:i + bs]).sum().item()
    model.tta = was_tta
    model.train(was_train)
    return correct / len(x)


def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    (xtr_np, ytr_np), val = load_split(args.holdout, args.seed)
    xtr, ytr = to_gpu_standardized(xtr_np, ytr_np)
    if val is not None:
        xva, yva = to_gpu_standardized(*val)
    n = len(xtr)

    pad = args.translate
    x_pad = F.pad(xtr, (pad,) * 4, mode="reflect").contiguous(memory_format=torch.channels_last) \
        if pad > 0 else xtr
    aug = make_augmenter(x_pad, pad, flip=not args.no_flip)

    model = Net(tuple(args.widths), args.bn_momentum, args.scale, ablate=args.ablate).cuda()
    model = model.to(memory_format=torch.channels_last)
    if args.ablate != "whiten":
        init_whitening(model, xtr[:5000])
    n_params = sum(p.numel() for p in model.parameters())

    # Two param groups split by rank: 1-D params (BN betas, whitening bias) run
    # at bias_scaler x the base LR, conv/linear weights at the base LR. Both
    # groups get the same absolute weight decay unless --bias-wd-scale says otherwise.
    bias_params, weight_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bias_params if p.ndim <= 1 else weight_params).append(p)
    kilostep = args.bs * (1 + 1 / (1 - args.momentum))
    lr = args.lr / kilostep
    # Decoupled weight decay: PyTorch SGD multiplies the decay term by the LR,
    # so we divide it back out and keep the *absolute* per-step shrinkage fixed
    # while the LR is scheduled. Without this the decay is ~700x too weak.
    wd_abs = args.wd * args.bs / kilostep
    opt = torch.optim.SGD(
        [dict(params=weight_params, lr=lr, weight_decay=0.0),
         dict(params=bias_params, lr=lr * args.bias_scaler, weight_decay=0.0)],
        momentum=args.momentum, nesterov=True)
    base_lrs = [g["lr"] for g in opt.param_groups]
    wd_groups = [1.0, args.bias_wd_scale]  # relative decay for weights / biases

    steps_per_epoch = n // args.bs
    total_steps = int(steps_per_epoch * args.epochs)
    warmup = max(1, int(total_steps * args.warmup_frac))

    def lr_mult(step):
        if step < warmup:
            f = step / warmup
            return args.start_lr_frac * (1 - f) + f
        f = (step - warmup) / max(1, total_steps - warmup)
        return (1 - f) + args.final_lr_frac * f

    os.makedirs(args.outdir, exist_ok=True)
    saver = None
    if not args.no_save:
        for f in os.listdir(args.outdir):
            if f.startswith("ckpt_epoch_"):
                os.remove(os.path.join(args.outdir, f))
        # Script the serialization target ONCE so that per-epoch saves are pure
        # I/O -- torch.jit.script costs ~1 s and would otherwise land inside the
        # measured epoch time.
        proto = copy.deepcopy(model).float().eval()
        proto.tta = True
        saver = torch.jit.script(proto)
        saver.eval()

    lossf = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction="sum")
    log = []
    step = 0
    epochs_int = math.ceil(args.epochs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for epoch in range(1, epochs_int + 1):
        model.train()
        perm = torch.randperm(n, device="cuda")
        for i in range(steps_per_epoch):
            if step >= total_steps:
                break
            mult = lr_mult(step)
            for g, b, wg in zip(opt.param_groups, base_lrs, wd_groups):
                g["lr"] = b * mult
                g["weight_decay"] = wd_abs * wg / g["lr"] if wg else 0.0
            if args.whiten_bias_epochs and epoch > args.whiten_bias_epochs:
                model.whiten.bias.requires_grad_(False)
            idx = perm[i * args.bs:(i + 1) * args.bs]
            img = aug(idx)
            if args.cutout:
                img = apply_cutout(img, args.cutout)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lossf(model(img), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1

        if saver is not None:
            saver.load_state_dict(model.state_dict())
            saver.save(os.path.join(args.outdir, f"ckpt_epoch_{epoch:04d}.pt"))
        rec = {"epoch": epoch, "t": time.perf_counter() - t0}
        if val is not None and (args.eval_every and epoch % args.eval_every == 0
                                or epoch == epochs_int):
            rec["val"] = evaluate(model, xva, yva, tta=True)
            rec["val_notta"] = evaluate(model, xva, yva, tta=False)
        log.append(rec)
        print(json.dumps(rec), flush=True)

    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    # Reported wall clock spans process start -> last checkpoint, so it can
    # never understate the checkpoint ladder's own span.
    summary = {"tag": args.tag, "seed": args.seed, "params": n_params, "epochs": epochs_int,
               "wall_clock_sec": time.perf_counter() - PROC_START,
               "train_loop_sec": wall, "sec_per_epoch": wall / epochs_int,
               "final_val": log[-1].get("val"), "final_val_notta": log[-1].get("val_notta"),
               "args": vars(args)}
    print("SUMMARY " + json.dumps(summary), flush=True)
    if not args.no_save:
        with open(os.path.join(args.outdir, "train_log.json"), "w") as f:
            json.dump({"summary": summary, "epochs": log}, f, indent=1)


if __name__ == "__main__":
    main()
