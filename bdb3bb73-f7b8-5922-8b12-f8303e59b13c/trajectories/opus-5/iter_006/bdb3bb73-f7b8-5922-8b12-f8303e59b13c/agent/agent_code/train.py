"""CIFAR-10 speedrun training.

Reads ONLY `/workspace/cache/train_{x,y}.npy` (the 50k train split).  The test split
is never opened by this file, so there is no path for test data to influence the
weights or any decision made here.

Writes one TorchScript checkpoint per epoch to `--out-dir` as
`ckpt_epoch_{NNNN}.pt`, numbered contiguously from 0001, plus (unless
`--no-final-copy`) `/workspace/checkpoint.pt` as a copy of the last one.

Timing note: the scored clock is reconstructed by the verifier from checkpoint
mtimes, so every millisecond between the epoch-1 and epoch-N saves is real training
work.  We deliberately do NOT use `torch.compile`: its ~15 s warmup would sit
*before* the epoch-1 save and would shrink the measured t* without making training
any faster.  cuDNN autotuning is left to happen inside epoch 1 for the same reason
(honest, and epoch 1 is excluded by the verifier's own (N-1)-interval formula).
"""
import argparse
import json
import os
import shutil
import time
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F

from model import CIFAR_MEAN, CIFAR_STD, init_whitening, make_net

CACHE = "/workspace/cache"


# --------------------------------------------------------------------------------
# Data (all resident on the GPU; no DataLoader, no host round-trips per step)
# --------------------------------------------------------------------------------
def load_train_split(pad: int):
    """Return (padded standardized fp16 images, labels) for the 50k train split."""
    x = np.load(f"{CACHE}/train_x.npy")  # (N,32,32,3) uint8
    y = np.load(f"{CACHE}/train_y.npy")
    x = torch.from_numpy(x).cuda().permute(0, 3, 1, 2).float()  # (N,3,32,32) in [0,255]
    mean = torch.tensor(CIFAR_MEAN, device="cuda").view(1, 3, 1, 1) * 255.0
    std = torch.tensor(CIFAR_STD, device="cuda").view(1, 3, 1, 1) * 255.0
    x = (x - mean) / std  # identical to ToTensor() + Normalize(), what the verifier feeds
    x_pad = F.pad(x, (pad,) * 4, mode="reflect")
    return x_pad.half().contiguous(memory_format=torch.channels_last), torch.from_numpy(y).cuda()


def augment(x_pad, idx, pad, ar32, gen):
    """Random `pad`-pixel translation + horizontal mirror, done entirely on-GPU.

    Two gathers (rows, then columns) give every image its own independent offset in
    a single fused pass -- no python loop over the (2*pad+1)^2 shift combinations.
    """
    imgs = x_pad[idx]
    b, c, h, w = imgs.shape
    span = 2 * pad + 1
    oy = torch.randint(0, span, (b,), device=imgs.device, generator=gen)
    ox = torch.randint(0, span, (b,), device=imgs.device, generator=gen)
    iy = (oy.view(b, 1, 1, 1) + ar32.view(1, 1, -1, 1)).expand(b, c, 32, w)
    imgs = imgs.gather(2, iy)
    ix = (ox.view(b, 1, 1, 1) + ar32.view(1, 1, 1, -1)).expand(b, c, 32, 32)
    imgs = imgs.gather(3, ix)
    flip = torch.rand(b, device=imgs.device, generator=gen) < 0.5
    imgs = torch.where(flip.view(b, 1, 1, 1), imgs.flip(-1), imgs)
    return imgs.contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def evaluate(net, x, y, bs=1000):
    """Accuracy of the deployed (TTA) forward pass. Used only on held-out TRAIN data."""
    was_training = net.training
    net.eval()
    correct = 0
    for i in range(0, len(x), bs):
        correct += (net(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
    if was_training:
        net.train()
    return correct / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--out-dir", default="/workspace/periodic_checkpoints")
    ap.add_argument("--holdout", type=int, default=0,
                    help=">0 holds out that many TRAIN images for validation instead "
                         "of training on them; 0 trains on all 50k (final-run mode).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--widths", type=int, nargs=3, default=[64, 256, 256])
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=11.5, help="peak LR per 1024 examples")
    ap.add_argument("--momentum", type=float, default=0.85)
    ap.add_argument("--wd", type=float, default=0.0153)
    ap.add_argument("--bias-scaler", type=float, default=64.0)
    ap.add_argument("--label-smoothing", type=float, default=0.2)
    ap.add_argument("--logit-scale", type=float, default=1 / 9)
    ap.add_argument("--bn-momentum", type=float, default=0.6)
    ap.add_argument("--whiten-bias-epochs", type=int, default=3)
    ap.add_argument("--pad", type=int, default=2, help="max random translation in px")
    ap.add_argument("--lr-peak-frac", type=float, default=0.23)
    ap.add_argument("--lr-start-frac", type=float, default=0.2)
    ap.add_argument("--lr-end-frac", type=float, default=0.07)
    ap.add_argument("--no-tta", action="store_true", help="ablation: disable test-time aug")
    ap.add_argument("--no-whiten", action="store_true",
                    help="ablation: leave the 2x2 conv randomly initialized and trainable")
    ap.add_argument("--no-final-copy", action="store_true")
    ap.add_argument("--log", default=None, help="write per-epoch JSON log here")
    args = ap.parse_args()

    t_run_start = time.time()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out_dir, exist_ok=True)

    x_pad, y_all = load_train_split(args.pad)

    # Held-out validation slice (train images only, fixed split independent of --seed
    # so every config is scored on exactly the same images).
    if args.holdout > 0:
        rng = np.random.default_rng(1234)
        perm = torch.from_numpy(rng.permutation(len(y_all))).cuda()
        val_idx, tr_idx = perm[: args.holdout], perm[args.holdout:]
        p = args.pad
        x_val = x_pad[val_idx][:, :, p:p + 32, p:p + 32].contiguous()
        y_val = y_all[val_idx]
        x_pad, y_all = x_pad[tr_idx].contiguous(), y_all[tr_idx]
    else:
        x_val = y_val = None

    n = len(y_all)
    bs = args.batch_size
    steps_per_epoch = ceil(n / bs)
    total_steps = steps_per_epoch * args.epochs

    net = make_net(tuple(args.widths), args.bn_momentum, args.logit_scale,
                   tta=not args.no_tta)
    # Data-dependent init of the frozen whitening filters, from TRAIN images only.
    p = args.pad
    if args.no_whiten:
        net.whiten.weight.requires_grad = True
    else:
        init_whitening(net.whiten, x_pad[:5000, :, p:p + 32, p:p + 32].float())

    # --- optimizer -------------------------------------------------------------
    # LR is specified per 1024 examples with a sum-reduced loss; `kilostep_scale`
    # converts it to a PyTorch SGD lr, dividing out the factor by which Nesterov
    # momentum inflates the effective step relative to the raw gradient.
    kilostep_scale = 1024 * (1 + 1 / (1 - args.momentum))
    lr = args.lr / kilostep_scale
    wd = args.wd * bs / kilostep_scale
    lr_biases = lr * args.bias_scaler
    biases = [p_ for k, p_ in net.named_parameters() if k.endswith("bias")]
    weights = [p_ for k, p_ in net.named_parameters() if not k.endswith("bias")]
    # weight_decay is divided by lr because PyTorch SGD applies lr*wd*param; this
    # keeps the decay decoupled from the (heavily scheduled) learning rate.
    opt = torch.optim.SGD(
        [dict(params=weights, lr=lr, weight_decay=wd / lr),
         dict(params=biases, lr=lr_biases, weight_decay=wd / lr_biases)],
        momentum=args.momentum, nesterov=True)
    # Triangular schedule: warm up from `lr_start_frac` to the peak at
    # `lr_peak_frac` of the run, then decay linearly to `lr_end_frac`.
    sched_pts = np.interp(np.arange(total_steps + 1),
                          [0, int(args.lr_peak_frac * total_steps), total_steps],
                          [args.lr_start_frac, 1.0, args.lr_end_frac])
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, sched_pts.__getitem__)

    gen = torch.Generator(device="cuda")
    gen.manual_seed(args.seed)
    ar32 = torch.arange(32, device="cuda")

    # Script once: the ScriptModule shares parameter/buffer storage with `net`, so
    # each per-epoch `.save()` serializes that epoch's actual weights.  (Verified in
    # verify_ladder.py, which checks the ladder's accuracies are strictly evolving.)
    scripted = torch.jit.script(net)

    log = {"args": vars(args), "epochs": []}
    net.train()
    for epoch in range(1, args.epochs + 1):
        # airbench's trick: let the whitening layer's bias adapt for a few epochs,
        # then freeze it so the whitening transform is fully static thereafter.
        net.whiten.bias.requires_grad = epoch <= args.whiten_bias_epochs
        order = torch.randperm(n, device="cuda", generator=gen)
        for i in range(steps_per_epoch):
            idx = order[i * bs:(i + 1) * bs]
            imgs = augment(x_pad, idx, args.pad, ar32, gen)
            out = net.core(imgs)
            loss = F.cross_entropy(out.float(), y_all[idx],
                                   label_smoothing=args.label_smoothing, reduction="sum")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            scheduler.step()

        path = os.path.join(args.out_dir, f"ckpt_epoch_{epoch:04d}.pt")
        scripted.save(path)
        rec = {"epoch": epoch, "t_since_start": time.time() - t_run_start}
        if x_val is not None:
            rec["val_acc"] = evaluate(net, x_val, y_val)
        log["epochs"].append(rec)
        print(json.dumps(rec), flush=True)

    total = time.time() - t_run_start
    log["wall_clock_sec"] = total
    last = os.path.join(args.out_dir, f"ckpt_epoch_{args.epochs:04d}.pt")
    if not args.no_final_copy:
        shutil.copy(last, "/workspace/checkpoint.pt")
    if args.log:
        with open(args.log, "w") as f:
            json.dump(log, f, indent=1)
    print(f"TOTAL_WALL_CLOCK_SEC {total:.3f}", flush=True)


if __name__ == "__main__":
    main()
