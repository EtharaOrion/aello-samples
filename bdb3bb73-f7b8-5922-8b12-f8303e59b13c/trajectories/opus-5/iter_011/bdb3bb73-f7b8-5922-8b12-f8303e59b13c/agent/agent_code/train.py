"""CIFAR-10 speedrun trainer.

Usage
-----
  python train.py --ckpt-dir /workspace/scratch_ckpt --holdout 5000   # tuning
  python train.py --ckpt-dir /workspace/periodic_checkpoints          # graded

Every epoch is one genuine pass over the training split and is followed by a
``torch.jit.script`` checkpoint save to ``ckpt_epoch_{NNNN}.pt``.  The
ScriptModule is built once before the loop and re-saved each epoch: the
optimiser updates parameters in place, so the ScriptModule (which aliases the
same storages) always serialises the current weights.  This keeps the
per-epoch save at ~30 ms instead of re-tracing the graph every epoch.
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from common import (GpuTrainData, Net, evaluate, load_split, lr_at,
                    make_optimizer, standardize)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=26)
    p.add_argument("--bs", type=int, default=500)  # divides 50000 and 45000 exactly
    p.add_argument("--lr", type=float, default=0.8)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.85)
    p.add_argument("--label-smoothing", type=float, default=0.2)
    p.add_argument("--translate", type=int, default=3)
    p.add_argument("--cutout", type=int, default=0)
    p.add_argument("--warmup-frac", type=float, default=0.2)
    p.add_argument("--final-frac", type=float, default=0.02)
    p.add_argument("--logit-scale", type=float, default=1 / 9.0)
    p.add_argument("--widths", type=int, nargs=3, default=[128, 384, 512])
    p.add_argument("--dirac", type=int, default=1, help="partial identity conv init")
    p.add_argument("--bn-momentum", type=float, default=0.4)
    p.add_argument("--bn-learn-scale", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout", type=int, default=0,
                   help="images held out of train for validation (0 = use all 50k)")
    p.add_argument("--ckpt-dir", type=str, default="/workspace/scratch_ckpt")
    p.add_argument("--no-ckpt", action="store_true")
    p.add_argument("--eval-test", action="store_true",
                   help="report final test accuracy (never used for tuning)")
    p.add_argument("--tag", type=str, default="")
    return p.parse_args()


def main():
    a = parse()
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"

    xtr, ytr = load_split("train", dev)
    if a.holdout > 0:
        g = torch.Generator(device="cpu").manual_seed(1234)  # fixed split across runs
        perm = torch.randperm(xtr.shape[0], generator=g).to(dev)
        val_idx, tr_idx = perm[: a.holdout], perm[a.holdout:]
        xva, yva = xtr[val_idx], ytr[val_idx]
        xtr, ytr = xtr[tr_idx], ytr[tr_idx]
    else:
        xva = yva = None

    pad = max(4, a.translate)
    data = GpuTrainData(xtr, ytr, pad=pad, device=dev)
    n = data.n

    model = Net(widths=tuple(a.widths), logit_scale=a.logit_scale, dirac=bool(a.dirac),
                bn_momentum=a.bn_momentum, bn_freeze_scale=not a.bn_learn_scale).to(dev)
    model.init_whiten(standardize(xtr[:10000]))
    model = model.to(memory_format=torch.channels_last)
    nparams = sum(p.numel() for p in model.parameters())

    opt = make_optimizer(model, a.lr, a.wd, a.momentum)
    steps_per_epoch = (n + a.bs - 1) // a.bs
    total_steps = steps_per_epoch * a.epochs
    gen = torch.Generator(device=dev).manual_seed(a.seed + 777)

    os.makedirs(a.ckpt_dir, exist_ok=True)

    # ---- warm up cudnn autotuning on a THROWAWAY model ---------------------
    # cudnn.benchmark caches plans per (shape, dtype, algo) globally, so a
    # discarded model is enough; the real run then starts with a hot cache and
    # the timed loop contains no autotuning.
    warm = Net(widths=tuple(a.widths), logit_scale=a.logit_scale, dirac=bool(a.dirac),
               bn_momentum=a.bn_momentum, bn_freeze_scale=not a.bn_learn_scale).to(dev)
    warm = warm.to(memory_format=torch.channels_last)
    wopt = make_optimizer(warm, a.lr, a.wd, a.momentum)
    wu_idx = torch.arange(a.bs, device=dev)
    for _ in range(3):
        xb, yb = data.batch(wu_idx, a.translate, True, a.cutout, gen)
        with torch.autocast("cuda", torch.bfloat16):
            loss = F.cross_entropy(warm(xb), yb, label_smoothing=a.label_smoothing)
        wopt.zero_grad(set_to_none=True)
        loss.backward()
        wopt.step()
    del warm, wopt

    scripted = None
    if not a.no_ckpt:
        model.eval()
        scripted = torch.jit.script(model)
        model.train()
    gen = torch.Generator(device=dev).manual_seed(a.seed + 777)  # reset after warm-up
    torch.cuda.synchronize()

    # ---- timed training loop ----------------------------------------------
    t0 = time.time()
    step = 0
    save_times = []
    for epoch in range(1, a.epochs + 1):
        perm = torch.randperm(n, device=dev, generator=gen)
        for i in range(0, n, a.bs):
            idx = perm[i:i + a.bs]
            xb, yb = data.batch(idx, a.translate, True, a.cutout, gen)
            for gp in opt.param_groups:
                gp["lr"] = lr_at(step, total_steps, a.lr, a.warmup_frac, a.final_frac)
            with torch.autocast("cuda", torch.bfloat16):
                loss = F.cross_entropy(model(xb), yb, label_smoothing=a.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
        if not a.no_ckpt:
            torch.cuda.synchronize()
            scripted.save(os.path.join(a.ckpt_dir, f"ckpt_epoch_{epoch:04d}.pt"))
            save_times.append(time.time() - t0)
    torch.cuda.synchronize()
    train_wall = time.time() - t0

    # ---- reporting (evaluation happens AFTER the timed loop) ---------------
    res = {"tag": a.tag, "params": nparams, "epochs": a.epochs, "seed": a.seed,
           "train_wall_sec": train_wall, "sec_per_epoch": train_wall / a.epochs,
           "cfg": vars(a)}
    if xva is not None:
        res["val_acc"] = evaluate(model, xva, yva)
    if a.eval_test:
        xte, yte = load_split("test", dev)
        res["test_acc"] = evaluate(model, xte, yte)
    if save_times:
        res["save_times"] = save_times
    print(json.dumps(res))


if __name__ == "__main__":
    main()
