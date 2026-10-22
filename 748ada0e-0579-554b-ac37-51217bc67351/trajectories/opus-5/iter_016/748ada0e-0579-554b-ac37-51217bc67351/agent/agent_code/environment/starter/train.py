"""CIFAR-100 @ 20k images, fixed budget: 8000 steps x batch 128 = 1,024,000 presentations.

Single training run; all data GPU-resident; batched augmentation (no dataloader).
Saves val + graded probability tensors for raw and EMA weights so post-processing
(TTA combination, prior/logit adjustment) can be chosen without retraining.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BUDGET_STEPS = 8000
BATCH = 128
N_CLASSES = 100
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        nn.init.zeros_(self.b2.weight)
        if stride != 1 or cin != cout:
            pool = [nn.AvgPool2d(2, 2)] if stride != 1 else []          # ResNet-D shortcut
            self.sc = nn.Sequential(*pool, nn.Conv2d(cin, cout, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(cout))
        else:
            self.sc = None

    def forward(self, x):
        o = F.relu(self.b1(self.c1(x)), inplace=True)
        o = self.b2(self.c2(o))
        return F.relu(o + (x if self.sc is None else self.sc(x)), inplace=True)


class Net(nn.Module):
    def __init__(self, w=128, blocks=(3, 4, 6, 3), n_classes=N_CLASSES):
        super().__init__()
        ws = [w, 2 * w, 4 * w, 8 * w]
        self.stem = nn.Sequential(nn.Conv2d(3, w, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(w), nn.ReLU(inplace=True))
        layers, cin = [], w
        for i, (n, c) in enumerate(zip(blocks, ws)):
            for j in range(n):
                layers.append(Block(cin, c, 2 if (j == 0 and i > 0) else 1))
                cin = c
        self.body = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, n_classes)

    def forward(self, x):
        x = self.body(self.stem(x))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


# ---------------------------------------------------------------- augmentation (GPU, batched)
def trivial_aug(x, gen):
    """Batched TrivialAugment-Wide: one random op per sample, magnitude ~ U[0,1]. x in [0,1]."""
    b, c, h, w = x.shape
    dev = x.device
    op = torch.randint(0, 12, (b,), device=dev, generator=gen)
    m = torch.rand(b, device=dev, generator=gen)
    sgn = torch.where(torch.rand(b, device=dev, generator=gen) < 0.5, -1.0, 1.0)
    ms = m * sgn

    th = torch.zeros(b, 2, 3, device=dev)
    th[:, 0, 0] = 1.0
    th[:, 1, 1] = 1.0
    sh = 0.99 * ms
    s1 = op == 1
    th[:, 0, 1] = torch.where(s1, sh, th[:, 0, 1])
    s2 = op == 2
    th[:, 1, 0] = torch.where(s2, sh, th[:, 1, 0])
    tr = 0.64 * ms                                   # +-0.32 of image size (grid units are 2x)
    th[:, 0, 2] = torch.where(op == 3, tr, th[:, 0, 2])
    th[:, 1, 2] = torch.where(op == 4, tr, th[:, 1, 2])
    ang = (135.0 * np.pi / 180.0) * ms
    ca, sa = torch.cos(ang), torch.sin(ang)
    s5 = op == 5
    th[:, 0, 0] = torch.where(s5, ca, th[:, 0, 0])
    th[:, 0, 1] = torch.where(s5, -sa, th[:, 0, 1])
    th[:, 1, 0] = torch.where(s5, sa, th[:, 1, 0])
    th[:, 1, 1] = torch.where(s5, ca, th[:, 1, 1])
    grid = F.affine_grid(th, (b, c, h, w), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    f = (1.0 + 0.99 * ms).view(b, 1, 1, 1)
    gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    o = op.view(b, 1, 1, 1)
    x = torch.where(o == 6, x * f, x)                                    # brightness
    x = torch.where(o == 7, gray + (x - gray) * f, x)                    # colour / saturation
    x = torch.where(o == 8, gray.mean((2, 3), keepdim=True) + (x - gray.mean((2, 3), keepdim=True)) * f, x)
    bits = (8 - (m * 6).round()).view(b, 1, 1, 1)                        # posterize 8..2 bits
    lv = torch.pow(2.0, bits)
    x = torch.where(o == 9, (x.clamp(0, 1) * lv).floor() / lv, x)
    thr = (1.0 - m).view(b, 1, 1, 1)                                     # solarize
    x = torch.where(o == 10, torch.where(x >= thr, 1.0 - x, x), x)
    lo = x.amin((2, 3), keepdim=True)
    hi = x.amax((2, 3), keepdim=True)
    x = torch.where(o == 11, (x - lo) / (hi - lo).clamp_min(1e-3), x)    # autocontrast
    return x.clamp_(0, 1)


def augment(x, pad, cut, res, gen, mean=None, std=None, ta=0):
    """x: (B,3,32,32) float in [0,1]. Random crop + hflip (+TA) + normalise + cutout + resize."""
    b, c, h, w = x.shape
    xp = F.pad(x, (pad,) * 4, mode="reflect")
    oy = torch.randint(0, 2 * pad + 1, (b,), device=x.device, generator=gen)
    ox = torch.randint(0, 2 * pad + 1, (b,), device=x.device, generator=gen)
    ar = torch.arange(h, device=x.device)
    rows = (oy[:, None] + ar)[:, None, :, None].expand(b, c, h, xp.shape[3])
    x = xp.gather(2, rows)
    cols = (ox[:, None] + ar)[:, None, None, :].expand(b, c, h, w)
    x = x.gather(3, cols)

    flip = torch.rand(b, 1, 1, 1, device=x.device, generator=gen) < 0.5
    x = torch.where(flip, x.flip(3), x)

    if ta:
        x = trivial_aug(x, gen)
    if mean is not None:
        x = (x - mean) / std

    if cut > 0:
        cy = torch.randint(0, h, (b, 1, 1), device=x.device, generator=gen)
        cx = torch.randint(0, w, (b, 1, 1), device=x.device, generator=gen)
        yy = ar.view(1, h, 1)
        xx = ar.view(1, 1, w)
        m = ((yy - cy).abs() < cut // 2) & ((xx - cx).abs() < cut // 2)
        x = x * (~m).unsqueeze(1).to(x.dtype)

    if res != h:
        x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
    return x.contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def probs(model, x, res, mean, std, flip_tta=True, bs=500, shift=0):
    """x: uint8 GPU tensor (N,32,32,3) -> softmax probs (N,100). Optional flip/shift TTA."""
    model.eval()
    offs = [(0, 0)] if shift == 0 else [(0, 0), (-shift, 0), (shift, 0), (0, -shift), (0, shift)]
    out = []
    for s in range(0, len(x), bs):
        x0 = x[s:s + bs].permute(0, 3, 1, 2).float().div_(255.).sub_(mean).div_(std)
        xp = F.pad(x0, (shift,) * 4, mode="reflect") if shift else x0
        h = x0.shape[-1]
        acc, n = 0.0, 0
        for dy, dx in offs:
            xb = xp[:, :, shift + dy:shift + dy + h, shift + dx:shift + dx + h] if shift else x0
            if res != h:
                xb = F.interpolate(xb, size=(res, res), mode="bilinear", align_corners=False)
            xb = xb.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                acc = acc + model(xb).float().softmax(1)
                n += 1
                if flip_tta:
                    acc = acc + model(xb.flip(3)).float().softmax(1)
                    n += 1
        out.append((acc / n).cpu())
    model.train()
    return torch.cat(out).numpy()


def bal_acc(p, y):
    pred = p.argmax(1)
    return float(np.mean([(pred[y == c] == c).mean() for c in range(N_CLASSES) if (y == c).any()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="i20k")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graded", default=os.path.join(DATA, "graded_images.npz"))
    ap.add_argument("--out", default="/submission/preds_i20k.npy")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--blocks", default="3,4,6,3")
    ap.add_argument("--res", type=int, default=48)
    ap.add_argument("--steps", type=int, default=BUDGET_STEPS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--cut", type=int, default=12)
    ap.add_argument("--ta", type=int, default=0)
    ap.add_argument("--shift", type=int, default=0)
    ap.add_argument("--pad", type=int, default=4)
    ap.add_argument("--warm", type=int, default=300)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--ema2", type=float, default=0.9997)
    ap.add_argument("--probdir", default="/workspace/probs")
    ap.add_argument("--write", type=int, default=1)
    a = ap.parse_args()

    t0 = time.time()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    d = np.load(os.path.join(DATA, "train_i20k.npz"))
    xtr = torch.from_numpy(d["x"]).to(dev)                       # (N,32,32,3) uint8
    ytr = torch.from_numpy(d["y"]).to(dev)
    dv = np.load(os.path.join(DATA, "val_public.npz"))
    xva = torch.from_numpy(dv["x"]).to(dev)
    yva = dv["y"]
    xgr = torch.from_numpy(np.load(a.graded)["x"]).to(dev)

    xf = xtr.permute(0, 3, 1, 2).float().div_(255.)
    mean = xf.mean((0, 2, 3)).view(1, 3, 1, 1)
    std = xf.std((0, 2, 3)).view(1, 3, 1, 1)
    xf = xf.contiguous()                                         # [0,1] train, GPU-resident
    del d

    blocks = tuple(int(v) for v in a.blocks.split(","))
    model = Net(a.width, blocks).to(dev).to(memory_format=torch.channels_last)
    nparam = sum(p.numel() for p in model.parameters())

    decay, nodecay = [], []
    for n_, p_ in model.named_parameters():
        (nodecay if p_.ndim <= 1 else decay).append(p_)
    opt = torch.optim.SGD([{"params": decay, "weight_decay": a.wd},
                           {"params": nodecay, "weight_decay": 0.0}],
                          lr=a.lr, momentum=0.9, nesterov=True)

    emas = []
    for dcy in (a.ema, a.ema2):
        if dcy > 0:
            m = [p.detach().clone().float() for p in model.parameters()]
            emas.append([dcy, m])
    mparams = list(model.parameters())

    gen = torch.Generator(device=dev); gen.manual_seed(a.seed + 1234)
    N = xf.shape[0]
    perm = torch.randperm(N, device=dev, generator=gen)
    ptr = 0
    presentations = 0
    model.train()
    for step in range(a.steps):
        if ptr + a.batch > N:
            perm = torch.randperm(N, device=dev, generator=gen)
            ptr = 0
        idx = perm[ptr:ptr + a.batch]; ptr += a.batch
        xb = augment(xf[idx], a.pad, a.cut, a.res, gen, mean, std, a.ta)
        yb = ytr[idx]
        presentations += xb.shape[0]

        lr = a.lr * (step + 1) / a.warm if step < a.warm else \
            0.5 * a.lr * (1 + np.cos(np.pi * (step - a.warm) / max(1, a.steps - a.warm)))
        for g in opt.param_groups:
            g["lr"] = lr

        with torch.autocast("cuda", dtype=torch.bfloat16):
            if a.mixup > 0:
                lam = float(np.random.beta(a.mixup, a.mixup))
                pi = torch.randperm(xb.shape[0], device=dev, generator=gen)
                xb = lam * xb + (1 - lam) * xb[pi]
                o = model(xb)
                loss = lam * F.cross_entropy(o, yb, label_smoothing=a.ls) + \
                    (1 - lam) * F.cross_entropy(o, yb[pi], label_smoothing=a.ls)
            else:
                loss = F.cross_entropy(model(xb), yb, label_smoothing=a.ls)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            for dcy, mv in emas:
                d_ = dcy if step >= 200 else min(dcy, (step + 1) / (step + 10))
                torch._foreach_mul_(mv, d_)
                torch._foreach_add_(mv, [p.detach() for p in mparams], alpha=1 - d_)

    ttrain = time.time() - t0

    os.makedirs(a.probdir, exist_ok=True)
    res = {"tag": a.tag, "params": nparam, "steps": a.steps,
           "presentations": presentations, "train_s": ttrain, "cfg": vars(a)}
    store = {}
    store["val_raw"] = probs(model, xva, a.res, mean, std)
    store["gr_raw"] = probs(model, xgr, a.res, mean, std)
    if a.shift:
        store["val_rawS"] = probs(model, xva, a.res, mean, std, shift=a.shift)
        store["gr_rawS"] = probs(model, xgr, a.res, mean, std, shift=a.shift)
    res["val_raw"] = bal_acc(store["val_raw"], yva)

    bk = [p.detach().clone() for p in mparams]
    for k, (dcy, mv) in enumerate(emas):
        with torch.no_grad():
            for p, m in zip(mparams, mv):
                p.copy_(m)
        # BN buffers are shared with the raw model (running stats already smoothed)
        store[f"val_ema{k}"] = probs(model, xva, a.res, mean, std)
        store[f"gr_ema{k}"] = probs(model, xgr, a.res, mean, std)
        if a.shift:
            store[f"val_ema{k}S"] = probs(model, xva, a.res, mean, std, shift=a.shift)
            store[f"gr_ema{k}S"] = probs(model, xgr, a.res, mean, std, shift=a.shift)
        res[f"val_ema{k}"] = bal_acc(store[f"val_ema{k}"], yva)
    with torch.no_grad():
        for p, b in zip(mparams, bk):
            p.copy_(b)

    ens = store["val_raw"].copy()
    for k in range(len(emas)):
        ens += store[f"val_ema{k}"]
    for k_ in [k_ for k_ in store if k_.startswith("val_") and k_.endswith("S")]:
        res[k_] = bal_acc(store[k_], yva)
    res["val_ens"] = bal_acc(ens, yva)
    if len(emas) > 0:
        res["val_raw_ema0"] = bal_acc(store["val_raw"] + store["val_ema0"], yva)

    prior = np.bincount(ytr.cpu().numpy(), minlength=N_CLASSES).astype(np.float64)
    prior /= prior.sum()
    for tau in (0.5, 1.0):
        res[f"val_ens_la{tau}"] = bal_acc(ens / prior[None, :] ** tau, yva)

    np.savez(os.path.join(a.probdir, f"{a.tag}.npz"), yva=yva, prior=prior, **store)
    res["total_s"] = time.time() - t0
    print(json.dumps(res, default=str))
    with open(os.path.join(a.probdir, f"{a.tag}.json"), "w") as f:
        json.dump(res, f, default=str)

    if a.write:
        g = store["gr_raw"].copy()
        for k in range(len(emas)):
            g += store[f"gr_ema{k}"]
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        np.save(a.out, (g / prior[None, :] ** 0.5).argmax(1).astype(np.int64))
        print("wrote", a.out, "in", round(time.time() - t0, 1), "s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
