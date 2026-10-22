"""CIFAR-100 under a fixed step budget (8000 updates x 128 batch = 1,024,000 presentations).

ResNet-34-D (basic blocks, avgpool-shortcut downsample), 32 -> RES bilinear upsample,
GPU-resident data + GPU-batched TrivialAugment/cutout, SGD-Nesterov + cosine, EMA.

Run:  python3 train.py --instance i20k --seed 0 \
          --graded ../data/graded_images.npz --out /submission/preds_i20k.npy
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BUDGET_STEPS = 8000
BUDGET_PRESENTATIONS = 1_024_000
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
N_CLASSES = 100


# ---------------------------------------------------------------- model
class Basic(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        nn.init.zeros_(self.b2.weight)  # zero-init residual
        if stride != 1 or cin != cout:
            # ResNet-D: anti-aliased shortcut (avgpool then 1x1) instead of strided 1x1
            pool = [nn.AvgPool2d(2, 2)] if stride != 1 else []
            self.sc = nn.Sequential(*pool, nn.Conv2d(cin, cout, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(cout))
        else:
            self.sc = None

    def forward(self, x):
        r = x if self.sc is None else self.sc(x)
        x = F.relu(self.b1(self.c1(x)), inplace=True)
        x = self.b2(self.c2(x))
        return F.relu(x + r, inplace=True)


class Net(nn.Module):
    def __init__(self, width=128, blocks=(3, 4, 6, 3), n_classes=N_CLASSES,
                 strides=(1, 2, 2, 2)):
        super().__init__()
        ws = [width, width * 2, width * 4, width * 8]
        self.stem = nn.Sequential(nn.Conv2d(3, ws[0], 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(ws[0]), nn.ReLU(inplace=True))
        layers, cin = [], ws[0]
        for i, (n, w) in enumerate(zip(blocks, ws)):
            for j in range(n):
                layers.append(Basic(cin, w, strides[i] if j == 0 else 1))
                cin = w
        self.body = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, n_classes)

    def forward(self, x):
        x = self.body(self.stem(x))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


# ---------------------------------------------------------------- GPU augmentation
def rand_crop_flip(x, pad=4):
    """x: (B,3,H,W) float in [0,1]. Reflect-pad + per-sample crop + per-sample hflip."""
    b, c, h, w = x.shape
    xp = F.pad(x, (pad,) * 4, mode="reflect")
    oy = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    ox = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    ar = torch.arange(b, device=x.device)
    rows = (oy[:, None] + torch.arange(h, device=x.device)[None, :])[:, None, :, None]
    cols = (ox[:, None] + torch.arange(w, device=x.device)[None, :])[:, None, None, :]
    x = xp[ar[:, None, None, None], torch.arange(c, device=x.device)[None, :, None, None],
           rows, cols]
    fl = torch.rand(b, device=x.device) < 0.5
    return torch.where(fl[:, None, None, None], x.flip(3), x)


_BLUR = None


def _sharpness(x, f):
    global _BLUR
    if _BLUR is None or _BLUR.device != x.device:
        k = torch.tensor([[1., 1., 1.], [1., 5., 1.], [1., 1., 1.]], device=x.device) / 13.
        _BLUR = k.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    bl = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), _BLUR.to(x.dtype), groups=3)
    return (bl + (x - bl) * f.view(-1, 1, 1, 1)).clamp(0, 1)


# op ids: 0 identity, 1 shearX, 2 shearY, 3 transX, 4 transY, 5 rotate,
#         6 brightness, 7 color, 8 contrast, 9 sharpness, 10 posterize,
#         11 solarize, 12 autocontrast
N_OPS = 13


def trivial_augment(x, gen=None):
    """Batched TrivialAugment: one uniformly-random op at a uniformly-random magnitude
    per image. x: (B,3,H,W) float in [0,1]."""
    b = x.shape[0]
    dev = x.device
    op = torch.randint(0, N_OPS, (b,), device=dev)
    m = torch.rand(b, device=dev)                      # magnitude in [0,1]
    sgn = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)

    # ---- geometric (single batched grid_sample) ----
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    sm = sgn * m
    is1, is2, is3, is4, is5 = (op == 1), (op == 2), (op == 3), (op == 4), (op == 5)
    theta[:, 0, 1] = torch.where(is1, sm * 0.30, theta[:, 0, 1])           # shear X
    theta[:, 1, 0] = torch.where(is2, sm * 0.30, theta[:, 1, 0])           # shear Y
    theta[:, 0, 2] = torch.where(is3, sm * 0.55, theta[:, 0, 2])           # translate X
    theta[:, 1, 2] = torch.where(is4, sm * 0.55, theta[:, 1, 2])           # translate Y
    ang = torch.where(is5, sm * (30.0 * math.pi / 180.0), torch.zeros_like(sm))
    ca, sa = torch.cos(ang), torch.sin(ang)   # ca==1, sa==0 for non-rotate rows
    theta[:, 0, 0] = ca
    theta[:, 1, 1] = ca
    theta[:, 0, 1] = torch.where(is5, -sa, theta[:, 0, 1])
    theta[:, 1, 0] = torch.where(is5, sa, theta[:, 1, 0])
    geo = is1 | is2 | is3 | is4 | is5
    if geo.any():
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        xg = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros",
                           align_corners=False)
        x = torch.where(geo[:, None, None, None], xg, x)

    # ---- photometric (compute per-op on the whole batch, select with mask) ----
    def blend(a, bimg, f):  # a + (b-a)*? -> torchvision style: bimg is the degenerate img
        return (bimg + (a - bimg) * f.view(-1, 1, 1, 1)).clamp(0, 1)

    f = 1.0 + sgn * m * 0.9
    if (op == 6).any():                                                     # brightness
        x = torch.where((op == 6)[:, None, None, None], blend(x, torch.zeros_like(x), f), x)
    if (op == 7).any():                                                     # color / sat
        g = (x * torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)).sum(
            1, keepdim=True).expand_as(x)
        x = torch.where((op == 7)[:, None, None, None], blend(x, g, f), x)
    if (op == 8).any():                                                     # contrast
        mu = x.mean(dim=(1, 2, 3), keepdim=True).expand_as(x)
        x = torch.where((op == 8)[:, None, None, None], blend(x, mu, f), x)
    if (op == 9).any():                                                     # sharpness
        x = torch.where((op == 9)[:, None, None, None], _sharpness(x, f), x)
    if (op == 10).any():                                                    # posterize
        bits = (8 - (m * 4).floor()).clamp(4, 8)                            # 8..4 bits
        q = (2.0 ** (8 - bits)).view(-1, 1, 1, 1)
        xp = ((x * 255.0 / q).floor() * q / 255.0).clamp(0, 1)
        x = torch.where((op == 10)[:, None, None, None], xp, x)
    if (op == 11).any():                                                    # solarize
        thr = (1.0 - m).view(-1, 1, 1, 1)
        x = torch.where((op == 11)[:, None, None, None], torch.where(x >= thr, 1.0 - x, x), x)
    if (op == 12).any():                                                    # autocontrast
        lo = x.amin(dim=(2, 3), keepdim=True)
        hi = x.amax(dim=(2, 3), keepdim=True)
        xa = ((x - lo) / (hi - lo).clamp_min(1e-5)).clamp(0, 1)
        x = torch.where((op == 12)[:, None, None, None], xa, x)
    return x


def cutout(x, size):
    b, _, h, w = x.shape
    cy = torch.randint(0, h, (b,), device=x.device)
    cx = torch.randint(0, w, (b,), device=x.device)
    ys = torch.arange(h, device=x.device).view(1, h, 1)
    xs = torch.arange(w, device=x.device).view(1, 1, w)
    half = size // 2
    m = ((ys - cy.view(-1, 1, 1)).abs() < half) & ((xs - cx.view(-1, 1, 1)).abs() < half)
    return x * (~m).unsqueeze(1).to(x.dtype)


# ---------------------------------------------------------------- eval helpers
def balanced_acc(pred, y, n=N_CLASSES):
    rec = []
    for c in range(n):
        m = y == c
        if m.sum():
            rec.append((pred[m] == c).mean())
    return float(np.mean(rec))


@torch.no_grad()
def probs_of(model, x_u8, mean, std, res, device, tta="flip", bs=1000):
    """x_u8: uint8 GPU tensor (N,3,32,32). Returns softmax probs averaged over TTA views."""
    model.eval()
    shifts = [(0, 0)] if tta == "flip" else [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)]
    out = torch.zeros(x_u8.shape[0], N_CLASSES, device=device)
    for s in range(0, x_u8.shape[0], bs):
        xb0 = x_u8[s:s + bs].float().div_(255.)
        acc = torch.zeros(xb0.shape[0], N_CLASSES, device=device)
        for dy, dx in shifts:
            xb = xb0 if (dy == 0 and dx == 0) else torch.roll(xb0, (dy, dx), dims=(2, 3))
            xb = F.interpolate(xb, size=(res, res), mode="bilinear", align_corners=False)
            xb = ((xb - mean) / std).contiguous(memory_format=torch.channels_last)
            for fl in (False, True):
                xi = xb.flip(3) if fl else xb
                with torch.autocast(device, dtype=torch.bfloat16):
                    acc += F.softmax(model(xi).float(), 1)
        out[s:s + bs] = acc / (len(shifts) * 2)
    model.train()
    return out


# ---------------------------------------------------------------- training
def train(args, device="cuda"):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    d = np.load(os.path.join(DATA, "train_i20k.npz"))
    xtr = torch.from_numpy(d["x"]).permute(0, 3, 1, 2).contiguous().to(device)  # uint8
    ytr = torch.from_numpy(d["y"]).to(device)
    dv = np.load(os.path.join(DATA, "val_public.npz"))
    xva = torch.from_numpy(dv["x"]).permute(0, 3, 1, 2).contiguous().to(device)
    yva = dv["y"]

    mean = (xtr.float() / 255.).mean(dim=(0, 2, 3)).view(1, 3, 1, 1)
    std = (xtr.float() / 255.).std(dim=(0, 2, 3)).view(1, 3, 1, 1)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    blocks = tuple(int(t) for t in args.blocks.split(","))
    strides = tuple(int(t) for t in args.strides.split(","))
    model = Net(args.width, blocks, strides=strides).to(device).to(memory_format=torch.channels_last)
    nparam = sum(p.numel() for p in model.parameters())

    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.SGD([{"params": decay, "weight_decay": args.wd},
                           {"params": no_decay, "weight_decay": 0.0}],
                          lr=args.lr, momentum=0.9, nesterov=True)

    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    # capture stable tensor references once so the per-step EMA is a fused foreach op
    _msd = model.state_dict()
    _fk = [k for k, v in _msd.items() if v.dtype.is_floating_point]
    _ik = [k for k in _msd if k not in _fk]
    ema_f = [ema[k] for k in _fk]
    src_f = [_msd[k] for k in _fk]
    plist = [p for p in model.parameters() if p.requires_grad]
    warm = args.warmup
    N = xtr.shape[0]
    g = torch.Generator(device=device); g.manual_seed(args.seed + 1234)
    perm = torch.randperm(N, device=device, generator=g)
    ptr = 0
    steps_done = 0
    presentations = 0
    t0 = time.time()

    for step in range(args.steps):
        per_step = args.batch * (2 if args.sam > 0 else 1)
        if presentations + per_step > BUDGET_PRESENTATIONS or steps_done >= BUDGET_STEPS:
            break
        if ptr + args.batch > N:
            perm = torch.randperm(N, device=device, generator=g)
            ptr = 0
        idx = perm[ptr:ptr + args.batch]; ptr += args.batch

        lr = (args.lr * (step + 1) / warm) if step < warm else \
            args.lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, args.steps - warm)))
        for gp in opt.param_groups:
            gp["lr"] = lr

        xb = xtr[idx].float().div_(255.)
        xb = rand_crop_flip(xb, 4)
        if args.ta:
            xb = trivial_augment(xb)
        xb = F.interpolate(xb, size=(args.res, args.res), mode="bilinear", align_corners=False)
        xb = ((xb - mean) / std)
        if args.cutout:
            xb = cutout(xb, args.cutout)
        xb = xb.contiguous(memory_format=torch.channels_last)
        yb = ytr[idx]

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device, dtype=torch.bfloat16):
            loss = F.cross_entropy(model(xb), yb, label_smoothing=args.ls)
        loss.backward()
        if args.sam > 0:
            # SAM: ascend to the local worst case on THIS batch, take the gradient there,
            # then one optimizer update. Both passes see the same augmented views; we
            # still bill each pass as a presentation (see cost accounting below).
            with torch.no_grad():
                grads = [p.grad for p in plist]
                gn = torch.norm(torch.stack(torch._foreach_norm(grads)))
                eps = torch._foreach_mul(grads, args.sam / (gn + 1e-12))
                torch._foreach_add_(plist, eps)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device, dtype=torch.bfloat16):
                F.cross_entropy(model(xb), yb, label_smoothing=args.ls).backward()
            with torch.no_grad():
                torch._foreach_sub_(plist, eps)
        opt.step()
        steps_done += 1
        presentations += args.batch * (2 if args.sam > 0 else 1)

        dcy = min(args.ema, (1 + steps_done) / (10 + steps_done))
        with torch.no_grad():
            torch._foreach_mul_(ema_f, dcy)
            torch._foreach_add_(ema_f, src_f, alpha=1 - dcy)

    for k in _ik:                      # non-float buffers (num_batches_tracked)
        ema[k].copy_(_msd[k])
    torch.cuda.synchronize()
    t_train = time.time() - t0
    print(f"[{args.tag}] params={nparam/1e6:.1f}M steps={steps_done}/{BUDGET_STEPS} "
          f"presentations={presentations}/{BUDGET_PRESENTATIONS} train_s={t_train:.1f}",
          flush=True)

    ema_model = Net(args.width, blocks, strides=strides).to(device).to(memory_format=torch.channels_last)
    ema_model.load_state_dict({k: v for k, v in ema.items()})

    xg = np.load(args.graded)["x"]
    xg = torch.from_numpy(xg).permute(0, 3, 1, 2).contiguous().to(device)

    res = {}
    for tta in ("flip", "shift5"):
        pv_r = probs_of(model, xva, mean, std, args.res, device, tta)
        pv_e = probs_of(ema_model, xva, mean, std, args.res, device, tta)
        pg_r = probs_of(model, xg, mean, std, args.res, device, tta)
        pg_e = probs_of(ema_model, xg, mean, std, args.res, device, tta)
        res[tta] = dict(vr=pv_r.cpu().numpy(), ve=pv_e.cpu().numpy(),
                        gr=pg_r.cpu().numpy(), ge=pg_e.cpu().numpy())
        for nm, p in (("raw", pv_r), ("ema", pv_e), ("ens", pv_r + pv_e)):
            print(f"[{args.tag}] {tta:6s} {nm:4s} val_bacc="
                  f"{balanced_acc(p.argmax(1).cpu().numpy(), yva):.4f}", flush=True)

    os.makedirs(args.probs_dir, exist_ok=True)
    np.savez(os.path.join(args.probs_dir, f"{args.tag}.npz"),
             **{f"{t}_{k}": v for t, dd in res.items() for k, v in dd.items()},
             yva=yva, ytr=d["y"], steps=steps_done, presentations=presentations,
             train_s=t_train)

    pens = res["flip"]["gr"] + res["flip"]["ge"]
    pred = pens.argmax(1).astype(np.int64)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        np.save(args.out, pred)
        print(f"[{args.tag}] wrote {args.out}", flush=True)
    print(f"[{args.tag}] TOTAL_S={time.time()-t0:.1f}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="i20k")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graded", default=os.path.join(DATA, "graded_images.npz"))
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--probs_dir", default="/workspace/probs")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--res", type=int, default=48)
    ap.add_argument("--steps", type=int, default=BUDGET_STEPS)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--wd", type=float, default=2e-3)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--cutout", type=int, default=12)
    ap.add_argument("--ta", type=int, default=1)
    ap.add_argument("--sam", type=float, default=0.0, help="SAM rho; 0 disables")
    ap.add_argument("--blocks", default="3,4,6,3")
    ap.add_argument("--strides", default="1,2,2,2")
    return train(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
