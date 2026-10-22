"""CIFAR-10 -> 94% top-1 speedrun on 1xH100.

Everything lives on the GPU: the whole training split is resident as a
pre-normalized, pre-padded fp16 tensor and augmentation (random translate +
horizontal flip + optional cutout) is done with tensor ops, so there is no
dataloader in the training loop and no host<->device syncs inside an epoch.

One TorchScript checkpoint is written per epoch to --ckpt-dir.
"""
import argparse
import copy
import json
import os
import time

import torch
import torch.nn.functional as F

from model import Net, CIFAR_MEAN, CIFAR_STD, init_whitening_conv

CACHE = "/workspace/data_cache"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--bs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=4.0)          # peak lr, mean-reduction loss
    p.add_argument("--wd", type=float, default=1.2e-3)       # on >=2d weights only
    p.add_argument("--bias-lr-scale", type=float, default=16.0)
    p.add_argument("--momentum", type=float, default=0.85)
    p.add_argument("--label-smoothing", type=float, default=0.2)
    p.add_argument("--warmup-frac", type=float, default=0.20)
    p.add_argument("--final-lr-frac", type=float, default=0.02)
    p.add_argument("--translate", type=int, default=2)
    p.add_argument("--cutout", type=int, default=0)
    p.add_argument("--widths", type=int, nargs=3, default=[64, 256, 256])
    p.add_argument("--scaling-factor", type=float, default=1 / 9)
    p.add_argument("--bn-momentum", type=float, default=0.4)
    p.add_argument("--val", type=int, default=0, help="hold out this many train images for validation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-dir", type=str, default="/workspace/scratch_ckpt")
    p.add_argument("--compile", type=int, default=1)
    p.add_argument("--tag", type=str, default="run")
    return p.parse_args()


def load_split(split, device):
    d = torch.load(os.path.join(CACHE, f"{split}.pt"))
    x = d["x"].to(device).permute(0, 3, 1, 2).contiguous()   # N,3,32,32 uint8
    return x, d["y"].to(device)


def standardize(x_uint8):
    mean = torch.tensor(CIFAR_MEAN, device=x_uint8.device).view(1, 3, 1, 1) * 255.0
    std = torch.tensor(CIFAR_STD, device=x_uint8.device).view(1, 3, 1, 1) * 255.0
    return (x_uint8.float() - mean) / std


class Augment:
    """Random translate (via two gathers) + horizontal flip + optional cutout, all on GPU."""

    def __init__(self, pad, crop, cutout, device):
        self.pad, self.crop, self.cutout = pad, crop, cutout
        self.ar = torch.arange(crop, device=device)

    def __call__(self, imgs, g):
        n, _, h, w = imgs.shape
        out = imgs
        if self.pad > 0:
            span = 2 * self.pad + 1
            sy = torch.randint(0, span, (n, 1), device=imgs.device, generator=g)
            sx = torch.randint(0, span, (n, 1), device=imgs.device, generator=g)
            ridx = (sy + self.ar.view(1, -1)).view(n, 1, self.crop, 1).expand(n, 3, self.crop, w)
            out = out.gather(2, ridx)
            cidx = (sx + self.ar.view(1, -1)).view(n, 1, 1, self.crop).expand(n, 3, self.crop, self.crop)
            out = out.gather(3, cidx)
        flip = torch.rand(n, 1, 1, 1, device=imgs.device, generator=g) < 0.5
        out = torch.where(flip, out.flip(3), out)
        if self.cutout > 0:
            r = self.cutout // 2
            cy = torch.randint(0, self.crop, (n, 1, 1), device=imgs.device, generator=g)
            cx = torch.randint(0, self.crop, (n, 1, 1), device=imgs.device, generator=g)
            my = (self.ar.view(1, -1, 1) - cy).abs() < r
            mx = (self.ar.view(1, 1, -1) - cx).abs() < r
            out = out.masked_fill((my & mx).unsqueeze(1), 0.0)
        return out.contiguous(memory_format=torch.channels_last)


def build_opt(model, args):
    w = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    b = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return torch.optim.SGD(
        [dict(params=w, lr=args.lr, weight_decay=args.wd),
         dict(params=b, lr=args.lr * args.bias_lr_scale, weight_decay=0.0)],
        momentum=args.momentum, nesterov=True)


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    xtr_u8, ytr = load_split("train", dev)
    xva = yva = None
    if args.val > 0:
        g0 = torch.Generator(device=dev).manual_seed(1234)
        perm = torch.randperm(len(ytr), device=dev, generator=g0)
        vi, ti = perm[:args.val], perm[args.val:]
        xva, yva = standardize(xtr_u8[vi]), ytr[vi]
        xtr_u8, ytr = xtr_u8[ti], ytr[ti]

    xtr = standardize(xtr_u8)
    del xtr_u8
    pad = args.translate
    src = F.pad(xtr, (pad,) * 4, mode="reflect") if pad > 0 else xtr
    xtr_pad = src.half().contiguous(memory_format=torch.channels_last)
    del src
    ntr = xtr_pad.shape[0]

    model = Net(tuple(args.widths), args.scaling_factor, args.bn_momentum).to(dev)
    model = model.to(memory_format=torch.channels_last)
    init_whitening_conv(model.body[0], xtr[:10000])
    del xtr
    init_sd = copy.deepcopy(model.state_dict())

    steps_per_epoch = ntr // args.bs           # one genuine pass; order reshuffled every epoch
    total_steps = steps_per_epoch * args.epochs
    warm = max(1, int(args.warmup_frac * total_steps))

    def lr_frac(step):
        if step < warm:
            return 0.10 + 0.90 * step / warm
        t = (step - warm) / max(1, total_steps - warm)
        return 1.0 + (args.final_lr_frac - 1.0) * t

    aug = Augment(pad, 32, args.cutout, dev)
    g = torch.Generator(device=dev).manual_seed(args.seed + 7)

    fwd = torch.compile(model, dynamic=False) if args.compile else model

    def train_step(xb, yb):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(fwd(xb), yb, label_smoothing=args.label_smoothing)
        loss.backward()

    # Warm up compile + cudnn autotune, then rewind to the pristine init so the
    # timed run is unaffected.
    opt = build_opt(model, args)
    model.train()
    for _ in range(3):
        train_step(aug(xtr_pad[:args.bs], g), ytr[:args.bs])
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    model.load_state_dict(init_sd)
    opt = build_opt(model, args)
    g = torch.Generator(device=dev).manual_seed(args.seed + 7)

    # Script once; weights are refreshed from the live model before every save.
    smodel = torch.jit.script(Net(tuple(args.widths), args.scaling_factor, args.bn_momentum).eval())
    torch.cuda.synchronize()

    step = 0
    t0 = time.time()
    ckpt_times = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(ntr, device=dev, generator=g)[:steps_per_epoch * args.bs]
        order = order.view(steps_per_epoch, args.bs)
        for i in range(steps_per_epoch):
            idx = order[i]
            f = lr_frac(step)
            opt.param_groups[0]["lr"] = args.lr * f
            opt.param_groups[1]["lr"] = args.lr * args.bias_lr_scale * f
            train_step(aug(xtr_pad[idx], g), ytr[idx])
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
        smodel.load_state_dict(model.state_dict())
        smodel.save(os.path.join(args.ckpt_dir, f"ckpt_epoch_{epoch:04d}.pt"))
        ckpt_times.append(time.time() - t0)

    torch.cuda.synchronize()
    total = time.time() - t0
    span = ckpt_times[-1] - ckpt_times[0]
    mean_epoch = span / max(1, len(ckpt_times) - 1)
    out = {"tag": args.tag, "total_sec": round(total, 3), "ckpt_span_sec": round(span, 3),
           "mean_epoch_sec": round(mean_epoch, 4), "epochs": args.epochs,
           "steps_per_epoch": steps_per_epoch}

    if xva is not None:
        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, len(yva), 1000):
                correct += (model(xva[i:i + 1000]).argmax(1) == yva[i:i + 1000]).sum().item()
        out["val_acc"] = correct / len(yva)
    print(json.dumps(out), flush=True)

    with open(os.path.join(args.ckpt_dir, "run_meta.json"), "w") as f:
        json.dump({**out, "args": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()
