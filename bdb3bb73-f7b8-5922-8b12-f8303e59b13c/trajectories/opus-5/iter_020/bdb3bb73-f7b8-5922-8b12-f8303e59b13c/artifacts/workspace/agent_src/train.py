"""CIFAR-10 from scratch -> 94%+ top-1 on a single H100, fast.

Design notes
------------
* Architecture: a compact 3-group convnet in the style of the CIFAR-10 speedrun
  literature (frozen 2x2 ZCA whitening stem -> 128/384/512 conv groups -> bias-free
  linear head).  ~6.1M params, no pretrained weights, no external data.
* Data: the whole train split lives on the GPU as one uint8 tensor; augmentation
  (random 2px translate + horizontal flip) is done on-device with two gathers and a
  `where`, once per epoch for the entire split.  There is no DataLoader in the hot loop.
* Input convention: the model consumes ALREADY per-channel-standardized NCHW float
  tensors, i.e. exactly `ToTensor() -> Normalize(CIFAR_MEAN, CIFAR_STD)`.  Training uses
  the identical normalization, so the exported model matches the evaluation convention.
* Test-time augmentation (6-way: {identity, mirror} x {center, -1px, +1px}) is folded
  into `forward`, so a plain `model(x)` call from the evaluator already gets TTA.
* Checkpoints: one TorchScript module per epoch, written to `--out`.

Usage
    python train.py --out DIR                 # full 50k train split, writes a ladder
    python train.py --holdout 5000 --seed 1   # 45k/5k split, prints val acc, no ladder
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
CACHE = "/workspace/cache"

# ---------------------------------------------------------------- model


class BatchNorm(nn.BatchNorm2d):
    """BN with frozen unit scale (gamma) and a trainable shift -- the extra scale
    degree of freedom is redundant with the following conv and only adds gradient noise."""

    def __init__(self, num_features, momentum=0.2, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.requires_grad = False
        self.bias.requires_grad = True


class ConvGroup(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.norm2 = BatchNorm(c_out)

    def forward(self, x):
        x = F.gelu(self.norm1(self.pool(self.conv1(x))))
        x = F.gelu(self.norm2(self.conv2(x)))
        return x


class Net(nn.Module):
    def __init__(self, widths=(128, 384, 512), whiten_width=24, scale=1 / 9.0, scale_mode="forward"):
        super().__init__()
        self.whiten = nn.Conv2d(3, whiten_width, 2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.group1 = ConvGroup(whiten_width, widths[0])
        self.group2 = ConvGroup(widths[0], widths[1])
        self.group3 = ConvGroup(widths[1], widths[2])
        self.pool = nn.MaxPool2d(3)
        self.head = nn.Linear(widths[2], 10, bias=False)
        self.head.weight.data *= scale
        # Logit temperature applied in the forward pass (not just folded into the init):
        # it permanently softens the softmax and scales head gradients, which is what makes
        # the large-LR / label-smoothing regime stable.
        # "init" reproduces the earlier variant that only shrank the head at init time;
        # kept as an ablation switch, see RESULTS.md.
        self.scale = float(scale) if scale_mode == "forward" else 1.0

    def forward(self, x):
        x = F.gelu(self.whiten(x))
        x = self.group3(self.group2(self.group1(x)))
        x = self.pool(x).flatten(1)
        return self.head(x) * self.scale


class TTAWrapper(nn.Module):
    """Exported module: 6-way TTA ({identity, mirror} x {center, -1px, +1px}) in forward.

    Input is a float NCHW batch already standardized with (CIFAR_MEAN, CIFAR_STD)
    -- the same convention the evaluator uses.  No re-normalization happens here.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def _mirror(self, x):
        return self.net(x) + self.net(x.flip(3))

    def forward(self, x):
        center = self._mirror(x)
        p = F.pad(x, (1, 1, 1, 1), mode="reflect")
        shifted = self._mirror(p[:, :, 0:32, 0:32]) + self._mirror(p[:, :, 2:34, 2:34])
        return 0.5 * center + 0.25 * shifted


@torch.no_grad()
def init_whitening(conv, images, eps=5e-4):
    """Frozen ZCA-whitening stem: eigenvectors of the 2x2 patch covariance, scaled by
    1/sqrt(lambda), duplicated with opposite sign so GELU sees both polarities."""
    patches = images.unfold(2, 2, 1).unfold(3, 2, 1)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, 2, 2).float()
    flat = patches.reshape(len(patches), -1)
    cov = (flat.T @ flat) / len(flat)
    evals, evecs = torch.linalg.eigh(cov, UPLO="U")
    filt = evecs.T.reshape(-1, 3, 2, 2).flip(0) / torch.sqrt(evals.flip(0).view(-1, 1, 1, 1) + eps)
    conv.weight.data[:] = torch.cat((filt, -filt))
    conv.bias.data[:] = 0.0


# ---------------------------------------------------------------- data


def load_train(holdout, split_seed=1234):
    x = torch.from_numpy(np.load(f"{CACHE}/train_x.npy"))
    y = torch.from_numpy(np.load(f"{CACHE}/train_y.npy"))
    if holdout <= 0:
        return x, y, None, None
    # class-balanced held-out slice carved out of the TRAIN split only.
    g = np.random.default_rng(split_seed)
    per_class = holdout // 10
    val_idx = np.concatenate([g.permutation(np.where(y.numpy() == c)[0])[:per_class] for c in range(10)])
    mask = np.ones(len(y), dtype=bool)
    mask[val_idx] = False
    tr_idx = np.where(mask)[0]
    return x[tr_idx], y[tr_idx], x[val_idx], y[val_idx]


def to_gpu_norm(x_u8):
    """uint8 NHWC -> standardized bf16 NCHW on GPU (identical to ToTensor+Normalize)."""
    x = x_u8.cuda().permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(CIFAR_MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device="cuda").view(1, 3, 1, 1)
    return x.sub_(mean).div_(std)


# ---------------------------------------------------------------- train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=11.5)
    ap.add_argument("--momentum", type=float, default=0.85)
    ap.add_argument("--wd", type=float, default=0.0153)
    ap.add_argument("--bias-scaler", type=float, default=64.0)
    ap.add_argument("--label-smoothing", type=float, default=0.2)
    ap.add_argument("--peak", type=float, default=0.23)
    ap.add_argument("--lr-end", type=float, default=0.07)
    ap.add_argument("--lr-start", type=float, default=0.2)
    ap.add_argument("--whiten-bias-epochs", type=int, default=3)
    ap.add_argument("--widths", type=int, nargs=3, default=[128, 384, 512])
    ap.add_argument("--scale", type=float, default=1 / 9.0, help="logit temperature")
    ap.add_argument("--scale-mode", choices=["forward", "init"], default="forward")
    ap.add_argument("--tx", type=int, default=2, help="max random translation (px)")
    ap.add_argument("--cutout", type=int, default=0, help="cutout square size (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=0)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--log", type=str, default="")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)

    xtr_u8, ytr, xva_u8, yva = load_train(args.holdout)
    xtr = to_gpu_norm(xtr_u8)
    ytr = ytr.cuda()
    N = len(xtr)
    # reflect-pad once; per-epoch random crops are gathers into this buffer
    pad = args.tx
    xpad = F.pad(xtr, (pad,) * 4, mode="reflect").to(memory_format=torch.channels_last).bfloat16()
    del xtr
    xva = yva_g = None
    if xva_u8 is not None:
        xva = to_gpu_norm(xva_u8).to(memory_format=torch.channels_last)
        yva_g = yva.cuda()

    net = Net(tuple(args.widths), scale=args.scale, scale_mode=args.scale_mode).cuda().to(memory_format=torch.channels_last)
    init_whitening(net.whiten, xpad[:5000, :, pad : pad + 32, pad : pad + 32].float())
    net.whiten.bias.requires_grad = False
    model = TTAWrapper(net).cuda().to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in net.parameters())

    # optimizer: airbench-style decoupled scaling.  Loss is summed over the batch, so the
    # LR carries an explicit 1/(batch * momentum-amplification) factor; BN shifts and the
    # whitening bias get a much larger LR and no effective decay penalty change.
    kilostep = args.batch * (1 + 1 / (1 - args.momentum))
    lr = args.lr / kilostep
    wd = args.wd * args.batch / kilostep
    lr_bias = lr * args.bias_scaler
    biases = [p for p in net.parameters() if p.requires_grad and p.ndim < 2]
    weights = [p for p in net.parameters() if p.requires_grad and p.ndim >= 2]
    biases.append(net.whiten.bias)  # frozen for the first few epochs, then unfrozen
    opt = torch.optim.SGD(
        [
            dict(params=biases, lr=lr_bias, weight_decay=wd / lr_bias),
            dict(params=weights, lr=lr, weight_decay=wd / lr),
        ],
        momentum=args.momentum,
        nesterov=True,
    )
    steps_per_epoch = (N + args.batch - 1) // args.batch  # every image is seen once
    total_steps = args.epochs * steps_per_epoch
    sched = np.interp(
        np.arange(total_steps + 1),
        [0, int(args.peak * total_steps), total_steps],
        [args.lr_start, 1.0, args.lr_end],
    )
    base_lrs = [g["lr"] for g in opt.param_groups]

    out_dir = args.out
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    scripted = None
    if out_dir:
        model.eval()
        scripted = torch.jit.script(model)
        model.train()

    ar = torch.arange(32, device="cuda")
    history = []
    t0 = time.time()
    step = 0
    for epoch in range(1, args.epochs + 1):
        net.train()
        if epoch == args.whiten_bias_epochs + 1:
            net.whiten.bias.requires_grad = True
        # ---- augment the whole split on-device: random crop (2 gathers) + mirror
        with torch.no_grad():
            dy = torch.randint(0, 2 * pad + 1, (N,), device="cuda")
            dx = torch.randint(0, 2 * pad + 1, (N,), device="cuda")
            ih = (dy[:, None] + ar).view(N, 1, 32, 1).expand(N, 3, 32, 32 + 2 * pad)
            xa = xpad.gather(2, ih)
            iw = (dx[:, None] + ar).view(N, 1, 1, 32).expand(N, 3, 32, 32)
            xa = xa.gather(3, iw)
            flip = torch.rand(N, device="cuda") < 0.5
            xa = torch.where(flip.view(N, 1, 1, 1), xa.flip(3), xa)
            if args.cutout > 0:
                # square erase filled with 0 == the per-channel mean in standardized space
                r = args.cutout / 2.0
                cy = torch.randint(0, 32, (N, 1), device="cuda")
                cx = torch.randint(0, 32, (N, 1), device="cuda")
                my = ((ar[None, :] - cy).abs() < r).view(N, 1, 32, 1)
                mx = ((ar[None, :] - cx).abs() < r).view(N, 1, 1, 32)
                xa = xa.masked_fill(my & mx, 0.0)
            xa = xa.contiguous(memory_format=torch.channels_last)
            perm = torch.randperm(N, device="cuda")
        for i in range(steps_per_epoch):
            idx = perm[i * args.batch : (i + 1) * args.batch]
            m = sched[step]
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * m
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(xa[idx])
                loss = F.cross_entropy(
                    out.float(), ytr[idx], label_smoothing=args.label_smoothing, reduction="sum"
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
        rec = {"epoch": epoch, "loss": round(loss.item() / len(idx), 4)}
        if out_dir:
            scripted.load_state_dict(model.state_dict())
            scripted.save(os.path.join(out_dir, f"ckpt_epoch_{epoch:04d}.pt"))
        rec["t"] = round(time.time() - t0, 3)
        if xva is not None:
            net.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                preds = torch.cat([model(xva[i : i + 1000]).float().argmax(1) for i in range(0, len(xva), 1000)])
            rec["val"] = round((preds == yva_g).float().mean().item(), 4)
        history.append(rec)
        print(
            f"ep {epoch:3d}  t={rec['t']:7.3f}s  loss={rec['loss']:.4f}"
            + (f"  val={rec['val']:.4f}" if "val" in rec else ""),
            flush=True,
        )

    total = time.time() - t0
    summary = dict(
        tag=args.tag, seed=args.seed, epochs=args.epochs, holdout=args.holdout,
        params=n_params, wall_clock_sec=round(total, 3),
        final_val=history[-1].get("val"), history=history,
        args={k: v for k, v in vars(args).items()},
    )
    print("SUMMARY " + json.dumps({k: v for k, v in summary.items() if k != "history"}), flush=True)
    if args.log:
        with open(args.log, "a") as f:
            f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
