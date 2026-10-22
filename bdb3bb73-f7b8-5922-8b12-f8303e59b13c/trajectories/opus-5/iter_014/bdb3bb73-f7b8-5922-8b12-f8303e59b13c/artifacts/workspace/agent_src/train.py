"""CIFAR-10 speedrun trainer.

Everything lives on the GPU: the training split is cached as uint8, standardized
once, reflection-padded once, and augmented with two gathers + two `where`s per
epoch. No DataLoader, no host round-trips.

Two modes:
  --split holdout : train on 45k, score the held-out 5k. Used for every
                    hyperparameter decision. The test split is never touched.
  --split full    : train on all 50k and write the scored checkpoint ladder.

The checkpoint ladder is what the grader clocks, so the timed loop does nothing
but train and save. All evaluation happens afterwards, from the saved files.
"""
import argparse
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Net, init_whitening

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
CACHE = "/workspace/cache"


# ---------------------------------------------------------------- data

def load_train(device):
    x = np.load(f"{CACHE}/train_x.npy")  # (50000, 32, 32, 3) uint8
    y = np.load(f"{CACHE}/train_y.npy")
    x = torch.from_numpy(x).to(device).permute(0, 3, 1, 2).contiguous()
    return x, torch.from_numpy(y).to(device)


def standardize(x_uint8):
    mean = torch.tensor(MEAN, device=x_uint8.device).view(1, 3, 1, 1) * 255.0
    std = torch.tensor(STD, device=x_uint8.device).view(1, 3, 1, 1) * 255.0
    return (x_uint8.float() - mean) / std


def holdout_split(x, y, n_val=5000, seed=0):
    """Class-balanced held-out slice, fixed across all experiments."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    per = n_val // 10
    val_idx = []
    for c in range(10):
        idx = (y == c).nonzero(as_tuple=True)[0].cpu()
        perm = torch.randperm(idx.numel(), generator=g)
        val_idx.append(idx[perm[:per]])
    val_idx = torch.cat(val_idx).to(x.device)
    mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    mask[val_idx] = False
    tr_idx = mask.nonzero(as_tuple=True)[0]
    return (x[tr_idx], y[tr_idx]), (x[val_idx], y[val_idx])


# ---------------------------------------------------------------- augmentation

class Augmenter:
    """Random translate (+/- pad px) + horizontal flip + cutout, whole-epoch."""

    def __init__(self, x_std, pad=2, cutout=0, device="cuda"):
        self.n = x_std.shape[0]
        self.pad = pad
        self.cutout = cutout
        self.device = device
        self.padded = F.pad(x_std, [pad] * 4, mode="reflect").half().contiguous()
        self.ar = torch.arange(32, device=device)

    def epoch(self, gen):
        n, p = self.n, self.pad
        perm = torch.randperm(n, device=self.device, generator=gen)
        src = self.padded[perm]  # (n,3,32+2p,32+2p)
        side = 32 + 2 * p
        oy = torch.randint(0, 2 * p + 1, (n,), device=self.device, generator=gen)
        ox = torch.randint(0, 2 * p + 1, (n,), device=self.device, generator=gen)
        iy = (oy.view(n, 1, 1, 1) + self.ar.view(1, 1, 32, 1)).expand(n, 3, 32, side)
        out = src.gather(2, iy)
        ix = (ox.view(n, 1, 1, 1) + self.ar.view(1, 1, 1, 32)).expand(n, 3, 32, 32)
        out = out.gather(3, ix)
        flip = torch.rand(n, 1, 1, 1, device=self.device, generator=gen) < 0.5
        out = torch.where(flip, out.flip(3), out)
        if self.cutout > 0:
            c = self.cutout
            cy = torch.randint(0, 32, (n, 1, 1, 1), device=self.device, generator=gen)
            cx = torch.randint(0, 32, (n, 1, 1, 1), device=self.device, generator=gen)
            ry = self.ar.view(1, 1, 32, 1)
            rx = self.ar.view(1, 1, 1, 32)
            hole = (((ry - cy).abs() < c // 2) & ((rx - cx).abs() < c // 2))
            out = torch.where(hole, torch.zeros((), dtype=out.dtype, device=out.device), out)
        return out.to(memory_format=torch.channels_last), perm


# ---------------------------------------------------------------- training

def build_model(cfg, x_std_train, device):
    model = Net(widths=cfg["widths"], bn_momentum=cfg["bn_momentum"],
                logit_scale=cfg["logit_scale"], tta=cfg["tta"]).to(device)
    # The cache is ordered by class, so stride the ZCA sample instead of taking a
    # prefix -- a prefix would estimate the patch statistics from 2 classes only.
    sub = x_std_train[:: max(1, x_std_train.shape[0] // 10000)][:10000]
    init_whitening(model.whiten, sub, eps=cfg["zca_eps"])
    return model.to(memory_format=torch.channels_last)


def make_optimizer(model, cfg, steps_per_epoch, epochs):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    groups = [
        dict(params=decay, lr=cfg["lr"], weight_decay=cfg["wd"]),
        dict(params=no_decay, lr=cfg["lr"] * cfg["bias_scaler"], weight_decay=0.0),
    ]
    opt = torch.optim.SGD(groups, lr=cfg["lr"], momentum=cfg["momentum"],
                          nesterov=True, weight_decay=cfg["wd"])
    total = steps_per_epoch * epochs
    warm = max(1, int(round(cfg["warmup"] * total)))

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        t = (step - warm) / max(1, total - warm)
        return max(cfg["lr_floor"], 1.0 - t)

    return opt, lr_at


def train(cfg, mode, ckpt_dir, log_path=None, eval_val=False, seed=0, save_ladder=True):
    device = "cuda"
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    x_u8, y = load_train(device)
    if mode == "holdout":
        (xtr_u8, ytr), (xva_u8, yva) = holdout_split(x_u8, y, seed=0)
    else:
        xtr_u8, ytr = x_u8, y
        xva_u8 = yva = None

    xtr = standardize(xtr_u8)
    model = build_model(cfg, xtr, device)
    aug = Augmenter(xtr, pad=cfg["pad"], cutout=cfg["cutout"], device=device)
    del xtr

    bs = cfg["bs"]
    n = xtr_u8.shape[0]
    steps_per_epoch = n // bs
    epochs = cfg["epochs"]
    opt, lr_at = make_optimizer(model, cfg, steps_per_epoch, epochs)
    base_lrs = [g["lr"] for g in opt.param_groups]
    crit = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    feats = torch.compile(model.features, dynamic=False) if cfg["compile"] else model.features

    def step(xb, yb):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = crit(feats(xb), yb)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    # ---- warmup: trigger compilation/autotuning on synthetic data, then restore
    # ---- the exact initial weights so the timed run starts from scratch.
    if cfg["compile"]:
        init_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.train()
        xw = torch.randn(bs, 3, 32, 32, device=device).to(memory_format=torch.channels_last)
        yw = torch.randint(0, 10, (bs,), device=device)
        for _ in range(3):
            step(xw, yw)
        opt.zero_grad(set_to_none=True)
        opt.state.clear()
        model.load_state_dict(init_state)
        del init_state, xw, yw
        torch.cuda.synchronize()

    scripted = torch.jit.script(Net(widths=cfg["widths"], bn_momentum=cfg["bn_momentum"],
                                    logit_scale=cfg["logit_scale"], tta=cfg["tta"]).cuda())
    scripted.eval()

    os.makedirs(ckpt_dir, exist_ok=True)
    gen = torch.Generator(device=device).manual_seed(seed + 1234)
    history = []
    torch.cuda.synchronize()
    t0 = time.time()
    gstep = 0
    for ep in range(1, epochs + 1):
        model.train()
        xa, perm = aug.epoch(gen)
        ya = ytr[perm]
        for i in range(steps_per_epoch):
            f = lr_at(gstep)
            for g, bl in zip(opt.param_groups, base_lrs):
                g["lr"] = bl * f
            sl = slice(i * bs, (i + 1) * bs)
            step(xa[sl].float(), ya[sl])
            gstep += 1
        del xa
        if save_ladder:
            scripted.load_state_dict(model.state_dict())
            scripted.save(os.path.join(ckpt_dir, f"ckpt_epoch_{ep:04d}.pt"))
        history.append(time.time() - t0)
    torch.cuda.synchronize()
    wall = time.time() - t0

    result = dict(cfg=cfg, seed=seed, mode=mode, wall=wall, epoch_times=history)
    if eval_val and xva_u8 is not None:
        model.eval()
        xva = standardize(xva_u8).to(memory_format=torch.channels_last)
        accs = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            preds = []
            for i in range(0, xva.shape[0], 1000):
                preds.append(model(xva[i:i + 1000]).float().argmax(1))
            accs.append((torch.cat(preds) == yva).float().mean().item())
        result["val_acc"] = accs[0]
    if log_path:
        with open(log_path, "a") as fh:
            fh.write(json.dumps(result) + "\n")
    return result, model


DEFAULT = dict(
    widths=(128, 384, 512), bn_momentum=0.6, logit_scale=1 / 9, tta=6,
    zca_eps=5e-4, pad=2, cutout=0, bs=1000, epochs=20,
    lr=0.4, wd=0.02, momentum=0.85, bias_scaler=16.0, warmup=0.15,
    lr_floor=0.0, label_smoothing=0.2, compile=True,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="holdout", choices=["holdout", "full"])
    ap.add_argument("--ckpt-dir", default="/workspace/scratch_ckpt")
    ap.add_argument("--log", default="/workspace/logs/runs.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--cutout", type=int, default=None)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    cfg = dict(DEFAULT)
    for k in ("epochs", "lr", "wd", "cutout"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    if os.path.isdir(args.ckpt_dir):
        shutil.rmtree(args.ckpt_dir)
    res, _ = train(cfg, args.split, args.ckpt_dir, args.log,
                   eval_val=(args.split == "holdout"), seed=args.seed,
                   save_ladder=not args.no_save)
    print(json.dumps({k: v for k, v in res.items() if k != "epoch_times"}))
    print("mean epoch s:", res["wall"] / cfg["epochs"])


if __name__ == "__main__":
    main()
