#!/usr/bin/env python
"""CIFAR-10 -> 94% top-1 on 1x H100, from scratch.

Architecture (airbench-lineage, Keller Jordan's CIFAR-10 speedrun family):
  frozen 2x2 ZCA-whitening conv (3 -> 24)  ->  3 x [conv-pool-BN-GELU-conv-BN-GELU]
  with widths 128 / 384 / 512  ->  global max-pool  ->  bias-free linear head.
  ~6.08M parameters. Convs run in fp16, BatchNorm in fp32, channels-last.

Everything lives on the GPU: the whole train split is cached as a uint8 tensor and
augmentation (random translate via two gathers, mirror via a where, optional cutout)
is done on-device, so there is no DataLoader in the loop.

I/O CONTRACT WITH THE VERIFIER
  The verifier does:  torch.jit.load(path).eval()  then  model(x)  where x is
  float32 NCHW, already standardized by
  transforms.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616)).
  So the model is TRAINED on identically standardized inputs and forward() does NOT
  re-normalize.  forward() also folds in 6-way test-time augmentation
  (mirror x {no shift, +1px, -1px}), which is why it is defined inline rather than
  applied by the caller.

All hyperparameter decisions in this file were made against a held-out slice of the
TRAIN split (--holdout).  The test split is never read during training.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CACHE = "/workspace/data_cache"


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
def load_split(split, device):
    x = np.load(f"{CACHE}/{split}_x.npy")          # N,32,32,3 uint8
    y = np.load(f"{CACHE}/{split}_y.npy")
    x = torch.from_numpy(x).to(device).permute(0, 3, 1, 2).contiguous()   # N,3,32,32 uint8
    y = torch.from_numpy(y).to(device)
    return x, y


def standardize(x_uint8, mean, std):
    """uint8 NCHW -> the exact float tensor the verifier will hand us (fp16 for speed)."""
    return (x_uint8.half() / 255.0 - mean) / std


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------
def make_conv(cin, cout, dirac):
    c = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
    if dirac:
        # airbench initializes the first `cin` output filters as an identity (dirac)
        # map, so each conv starts out partially preserving its input.
        nn.init.dirac_(c.weight.data[:cin])
    return c


def make_bn(cout, bn_momentum, bn_weight_grad):
    # NOTE: airbench's "bn_momentum=0.6" is an EMA retention factor; nn.BatchNorm2d's
    # `momentum` is the update weight, hence 1 - bn_momentum.
    b = nn.BatchNorm2d(cout, eps=1e-12, momentum=1.0 - bn_momentum)
    # airbench freezes the BN scale at 1 and trains only the shift.
    b.weight.requires_grad = bool(bn_weight_grad)
    return b


class ConvGroup(nn.Module):
    def __init__(self, cin, cout, bn_momentum, dirac=True, bn_weight_grad=False):
        super().__init__()
        self.conv1 = make_conv(cin, cout, dirac)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = make_bn(cout, bn_momentum, bn_weight_grad)
        self.conv2 = make_conv(cout, cout, dirac)
        self.norm2 = make_bn(cout, bn_momentum, bn_weight_grad)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x


class CifarNet(nn.Module):
    def __init__(self, w1=128, w2=384, w3=512, scale=1.0 / 9.0, bn_momentum=0.6,
                 tta_level=2, head_init_scale=-1.0, dirac=True, bn_weight_grad=False):
        super().__init__()
        self.whiten = nn.Conv2d(3, 24, 2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.activ = nn.GELU()
        self.group1 = ConvGroup(24, w1, bn_momentum, dirac, bn_weight_grad)
        self.group2 = ConvGroup(w1, w2, bn_momentum, dirac, bn_weight_grad)
        self.group3 = ConvGroup(w2, w3, bn_momentum, dirac, bn_weight_grad)
        self.head = nn.Linear(w3, 10, bias=False)
        self.scale = scale
        self.tta_level = tta_level
        # head_init_scale < 0 keeps the legacy zero-init; >= 0 rescales the default init
        # so that the trunk receives an undamped gradient (airbench does the latter).
        if head_init_scale < 0:
            nn.init.zeros_(self.head.weight)
        else:
            w = self.head.weight.data
            w *= head_init_scale / w.std()

    def _trunk(self, x):
        x = self.whiten(x)
        x = self.activ(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = x.amax(dim=(2, 3))
        x = self.head(x)
        return x.float() * self.scale

    def forward(self, x):
        # Inputs arrive already standardized (see module docstring): do NOT re-normalize.
        x = x.half()
        if self.training:
            return self._trunk(x)
        # ---- test-time augmentation, folded into forward so the verifier gets it ----
        logits = self._trunk(x) + self._trunk(x.flip(3))
        if self.tta_level >= 2:
            # +-1 px horizontal translate (edge-replicate), each also mirrored.
            xp = F.pad(x, (1, 0, 0, 0), mode="replicate")[:, :, :, :-1]
            xm = F.pad(x, (0, 1, 0, 0), mode="replicate")[:, :, :, 1:]
            logits = logits + self._trunk(xp) + self._trunk(xp.flip(3))
            logits = logits + self._trunk(xm) + self._trunk(xm.flip(3))
        return logits


def get_patches(x, k):
    return x.unfold(2, k, 1).unfold(3, k, 1).transpose(1, 3).reshape(-1, x.size(1), k, k).float()


@torch.no_grad()
def init_whitening(layer, train_images, eps=5e-4):
    """Freeze the first conv to a ZCA-style whitening transform of 2x2 image patches."""
    k = layer.weight.shape[2]
    patches = get_patches(train_images[:5000], k)
    flat = patches.reshape(patches.size(0), -1)
    cov = (flat.T @ flat) / flat.size(0)
    evals, evecs = torch.linalg.eigh(cov.double())
    evals = evals.flip(0).view(-1, 1, 1, 1).float()
    evecs = evecs.T.reshape(-1, 3, k, k).flip(0).float()
    scaled = evecs / torch.sqrt(evals + eps)
    layer.weight.data[:] = torch.cat((scaled, -scaled)).to(layer.weight.dtype)
    layer.bias.data.zero_()


# --------------------------------------------------------------------------------------
# GPU augmentation
# --------------------------------------------------------------------------------------
class Augmenter:
    """Random translate (two gathers) + mirror (one where) + optional cutout."""

    def __init__(self, images, translate, cutout, device, gen):
        self.p = translate
        self.cutout = cutout
        self.gen = gen
        self.device = device
        if translate > 0:
            self.padded = F.pad(images, (translate,) * 4, mode="reflect")
        else:
            self.padded = images
        self.n, self.c, _, _ = images.shape
        self.out = images.shape[-1]
        self.ar = torch.arange(self.out, device=device)

    def __call__(self, idx):
        b = idx.numel()
        x = self.padded[idx]
        if self.p > 0:
            dy = torch.randint(0, 2 * self.p + 1, (b,), device=self.device, generator=self.gen)
            dx = torch.randint(0, 2 * self.p + 1, (b,), device=self.device, generator=self.gen)
            iy = (dy[:, None] + self.ar[None, :])
            ix = (dx[:, None] + self.ar[None, :])
            W = x.size(3)
            x = x.gather(2, iy[:, None, :, None].expand(b, self.c, self.out, W))
            x = x.gather(3, ix[:, None, None, :].expand(b, self.c, self.out, self.out))
        flip = torch.rand(b, 1, 1, 1, device=self.device, generator=self.gen) < 0.5
        x = torch.where(flip, x.flip(3), x)
        if self.cutout > 0:
            s = self.cutout
            cy = torch.randint(0, self.out, (b, 1), device=self.device, generator=self.gen)
            cx = torch.randint(0, self.out, (b, 1), device=self.device, generator=self.gen)
            ry = (self.ar[None, :] - cy).abs() < s // 2
            rx = (self.ar[None, :] - cx).abs() < s // 2
            mask = (ry[:, None, :, None] & rx[:, None, None, :])
            x = x.masked_fill(mask, 0.0)
        return x


# --------------------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, x, y, bs=1000):
    model.eval()
    correct = 0
    for i in range(0, x.size(0), bs):
        out = model(x[i:i + bs])
        correct += (out.argmax(1) == y[i:i + bs]).sum()
    model.train()
    return correct.item() / x.size(0)


# --------------------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=11.5)
    ap.add_argument("--momentum", type=float, default=0.85)
    ap.add_argument("--wd", type=float, default=0.0153)
    ap.add_argument("--bias_scaler", type=float, default=64.0)
    ap.add_argument("--label_smoothing", type=float, default=0.2)
    ap.add_argument("--lr_peak_frac", type=float, default=0.23)
    ap.add_argument("--lr_init", type=float, default=0.2)
    ap.add_argument("--lr_final", type=float, default=0.07)
    ap.add_argument("--lr_hold_epochs", type=int, default=0,
                    help="run the last N epochs flat at lr_final instead of still decaying into it")
    ap.add_argument("--widths", type=str, default="128,384,512")
    ap.add_argument("--scale", type=float, default=1.0 / 9.0)
    ap.add_argument("--head_init_scale", type=float, default=-1.0)
    ap.add_argument("--dirac", type=int, default=1)
    ap.add_argument("--bn_weight_grad", type=int, default=0)
    ap.add_argument("--bn_momentum", type=float, default=0.6)
    ap.add_argument("--translate", type=int, default=2)
    ap.add_argument("--cutout", type=int, default=0)
    ap.add_argument("--tta_level", type=int, default=2)
    ap.add_argument("--whiten_bias_epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=0,
                    help="held-out TRAIN images used for all tuning decisions (0 = train on all 50k)")
    ap.add_argument("--ckpt_dir", type=str, default="")
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--tag", type=str, default="run")
    ap.add_argument("--log", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed + 1234)

    # ---------------- data ----------------
    xtr_u8, ytr = load_split("train", dev)
    if args.holdout > 0:
        g = torch.Generator().manual_seed(1234)          # fixed split, independent of --seed
        perm = torch.randperm(xtr_u8.size(0), generator=g).to(dev)
        val_idx, tr_idx = perm[:args.holdout], perm[args.holdout:]
        xva_u8, yva = xtr_u8[val_idx], ytr[val_idx]
        xtr_u8, ytr = xtr_u8[tr_idx], ytr[tr_idx]
    else:
        xva_u8, yva = None, None

    mean = torch.tensor(CIFAR_MEAN, device=dev).view(1, 3, 1, 1).half()
    std = torch.tensor(CIFAR_STD, device=dev).view(1, 3, 1, 1).half()
    xtr = standardize(xtr_u8, mean, std).to(memory_format=torch.channels_last)
    xva = standardize(xva_u8, mean, std).to(memory_format=torch.channels_last) if xva_u8 is not None else None
    del xtr_u8, xva_u8
    n = xtr.size(0)

    # ---------------- model ----------------
    w1, w2, w3 = [int(v) for v in args.widths.split(",")]
    model = CifarNet(w1, w2, w3, args.scale, args.bn_momentum, args.tta_level, args.head_init_scale,
                     args.dirac, args.bn_weight_grad).to(dev)
    init_whitening(model.whiten, xtr.float())
    model = model.to(memory_format=torch.channels_last).half()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.float()

    steps_per_epoch = n // args.bs
    total_steps = steps_per_epoch * args.epochs

    # airbench-style coupling: the loss is a SUM over the batch, so lr/wd are rescaled.
    kilostep = 1024 * (1 + 1 / (1 - args.momentum))
    lr = args.lr / kilostep
    wd = args.wd * args.bs / kilostep
    lr_bias = lr * args.bias_scaler

    norm_biases = [p for k, p in model.named_parameters() if "norm" in k and p.requires_grad]
    other = [p for k, p in model.named_parameters() if "norm" not in k and p.requires_grad]
    opt = torch.optim.SGD(
        [dict(params=norm_biases, lr=lr_bias, weight_decay=wd / lr_bias),
         dict(params=other, lr=lr, weight_decay=wd / lr)],
        momentum=args.momentum, nesterov=True)
    # 3-phase schedule: warm up to peak, decay linearly to lr_final, then HOLD flat.
    # np.interp clamps past its last x-point, so the hold is expressed by ending the
    # decay early.  hold=0 reproduces the plain 2-phase triangular schedule.
    decay_end = total_steps - args.lr_hold_epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: float(np.interp(s, [0, int(args.lr_peak_frac * total_steps), decay_end],
                                       [args.lr_init, 1.0, args.lr_final])))
    lossfn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction="none")

    aug = Augmenter(xtr, args.translate, args.cutout, dev, gen)

    ckpt_dir = args.ckpt_dir
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    # ---- warm up cudnn autotune + TorchScript compile BEFORE the timed loop, so that
    # ---- epoch 1 is exactly the same amount of work as every other epoch.
    warm = aug(torch.arange(args.bs, device=dev))
    model.train()
    lossfn(model(warm), ytr[:args.bs]).sum().backward()
    opt.zero_grad(set_to_none=True)
    scripted = torch.jit.script(model)
    if xva is not None:
        evaluate(scripted, xva[:args.bs], yva[:args.bs])
    torch.cuda.synchronize()

    # roll the model and optimizer back to a pristine state after the warmup
    torch.manual_seed(args.seed)
    gen.manual_seed(args.seed + 1234)
    model2 = CifarNet(w1, w2, w3, args.scale, args.bn_momentum, args.tta_level, args.head_init_scale,
                     args.dirac, args.bn_weight_grad).to(dev)
    init_whitening(model2.whiten, xtr.float())
    model2 = model2.to(memory_format=torch.channels_last).half()
    for m in model2.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.float()
    model.load_state_dict(model2.state_dict())
    del model2
    for g_ in opt.param_groups:
        for p in g_["params"]:
            opt.state.pop(p, None)
    torch.cuda.synchronize()

    hist = []
    t0 = time.time()
    step = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        # the whitening bias is trained for the first few epochs, then frozen
        model.whiten.bias.requires_grad = (ep <= args.whiten_bias_epochs)
        idx = torch.randperm(n, device=dev, generator=gen)
        for i in range(steps_per_epoch):
            b = idx[i * args.bs:(i + 1) * args.bs]
            x = aug(b)
            out = model(x)
            loss = lossfn(out, ytr[b]).sum()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
        if ckpt_dir:
            scripted.eval()
            scripted.save(os.path.join(ckpt_dir, f"ckpt_epoch_{ep:04d}.pt"))
        el = time.time() - t0
        acc = None
        if xva is not None and (ep % args.eval_every == 0 or ep == args.epochs):
            acc = evaluate(scripted, xva, yva)
        hist.append(dict(epoch=ep, t=el, val=acc))
        print(f"[{args.tag}] ep {ep:3d}  t {el:7.2f}s  val {acc if acc is not None else float('nan'):.4f}",
              flush=True)
    total = time.time() - t0
    torch.cuda.synchronize()

    out = dict(args=vars(args), hist=hist, wall_clock_sec=total,
               n_params=sum(p.numel() for p in model.parameters()))
    if args.log:
        with open(args.log, "a") as f:
            f.write(json.dumps(out) + "\n")
    print(f"[{args.tag}] TOTAL {total:.2f}s", flush=True)


if __name__ == "__main__":
    main()
