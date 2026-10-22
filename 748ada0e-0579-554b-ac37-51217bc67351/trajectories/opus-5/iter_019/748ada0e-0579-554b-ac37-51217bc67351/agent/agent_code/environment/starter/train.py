"""CIFAR-100 @ 20k images, fixed budget: 8000 steps x batch 128 = 1,024,000 presentations.

Single training run; all data GPU-resident; batched augmentation (no dataloader).
Saves val + graded probability tensors for every (weight-set x TTA) variant so the
inference combination is *chosen on val_public* and then applied to the graded images.
"""
import argparse, itertools, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BUDGET_STEPS = 8000
BUDGET_PRESENTATIONS = 1024000
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
    th[:, 0, 1] = torch.where(op == 1, sh, th[:, 0, 1])
    th[:, 1, 0] = torch.where(op == 2, sh, th[:, 1, 0])
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
    gm = gray.mean((2, 3), keepdim=True)
    x = torch.where(o == 8, gm + (x - gm) * f, x)                        # contrast
    bits = (8 - (m * 6).round()).view(b, 1, 1, 1)                        # posterize 8..2 bits
    lv = torch.pow(2.0, bits)
    x = torch.where(o == 9, (x.clamp(0, 1) * lv).floor() / lv, x)
    thr = (1.0 - m).view(b, 1, 1, 1)                                     # solarize
    x = torch.where(o == 10, torch.where(x >= thr, 1.0 - x, x), x)
    lo = x.amin((2, 3), keepdim=True)
    hi = x.amax((2, 3), keepdim=True)
    x = torch.where(o == 11, (x - lo) / (hi - lo).clamp_min(1e-3), x)    # autocontrast
    return x.clamp_(0, 1)


def rrc(x, pad, res, gen, smin, smax, rmax=4.0 / 3.0):
    """Batched random-resized-crop (+hflip) straight to res x res, one grid_sample.

    `s` is the box area as a fraction of the original 32x32 frame, so s>1 zooms *out*
    into the reflect-padded border -- a superset of the plain pad-`pad` random crop.
    """
    b, c, h, w = x.shape
    dev = x.device
    xp = F.pad(x, (pad,) * 4, mode="reflect")
    H = xp.shape[-1]
    s = smin + (smax - smin) * torch.rand(b, device=dev, generator=gen)
    ar = torch.exp((2 * torch.rand(b, device=dev, generator=gen) - 1) * np.log(rmax))
    bw = (h * torch.sqrt(s * ar)).clamp(max=float(H))
    bh = (h * torch.sqrt(s / ar)).clamp(max=float(H))
    cx = bw / 2 + (H - bw) * torch.rand(b, device=dev, generator=gen)
    cy = bh / 2 + (H - bh) * torch.rand(b, device=dev, generator=gen)
    sx = bw / H
    sy = bh / H
    flip = torch.rand(b, device=dev, generator=gen) < 0.5
    th = torch.zeros(b, 2, 3, device=dev)
    th[:, 0, 0] = torch.where(flip, -sx, sx)
    th[:, 1, 1] = sy
    th[:, 0, 2] = 2 * cx / H - 1
    th[:, 1, 2] = 2 * cy / H - 1
    grid = F.affine_grid(th, (b, c, res, res), align_corners=False)
    return F.grid_sample(xp, grid, mode="bilinear", padding_mode="reflection",
                         align_corners=False)


def augment_rrc(x, pad, cut, res, gen, mean, std, ta, smin, smax):
    """RRC variant: crop/scale/flip to `res` first, then TA / normalise / cutout."""
    x = rrc(x, pad, res, gen, smin, smax).clamp_(0, 1)
    if ta:
        xa = trivial_aug(x, gen)
        if ta < 1.0:
            keep = (torch.rand(x.shape[0], 1, 1, 1, device=x.device, generator=gen) < ta).to(x.dtype)
            xa = keep * xa + (1 - keep) * x
        x = xa
    x = (x - mean) / std
    k = int(round(cut * res / 32.0))
    if k > 0:
        ar = torch.arange(res, device=x.device)
        cy = torch.randint(0, res, (x.shape[0], 1, 1), device=x.device, generator=gen)
        cx = torch.randint(0, res, (x.shape[0], 1, 1), device=x.device, generator=gen)
        m = ((ar.view(1, res, 1) - cy).abs() < k // 2) & ((ar.view(1, 1, res) - cx).abs() < k // 2)
        x = x * (~m).unsqueeze(1).to(x.dtype)
    return x.contiguous(memory_format=torch.channels_last)


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
        if ta < 1.0:                       # apply TA to a random subset of the batch
            xa = trivial_aug(x, gen)
            keep = (torch.rand(b, 1, 1, 1, device=x.device, generator=gen) < ta).to(x.dtype)
            x = keep * xa + (1 - keep) * x
        else:
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


def cutmix_batch(x, alpha, gen):
    """Paste a random box from a shuffled copy. Returns (x, perm, lam_by_area)."""
    b, _, h, w = x.shape
    lam = float(np.random.beta(alpha, alpha))
    rh, rw = int(h * np.sqrt(1 - lam)), int(w * np.sqrt(1 - lam))
    if rh < 1 or rw < 1:
        return x, None, 1.0
    cy = int(torch.randint(0, h, (1,), device=x.device, generator=gen).item())
    cx = int(torch.randint(0, w, (1,), device=x.device, generator=gen).item())
    y0, y1 = max(cy - rh // 2, 0), min(cy + rh // 2, h)
    x0, x1 = max(cx - rw // 2, 0), min(cx + rw // 2, w)
    if y1 <= y0 or x1 <= x0:
        return x, None, 1.0
    perm = torch.randperm(b, device=x.device, generator=gen)
    x = x.clone()
    x[:, :, y0:y1, x0:x1] = x[perm][:, :, y0:y1, x0:x1]
    return x, perm, 1.0 - (y1 - y0) * (x1 - x0) / (h * w)


# ---------------------------------------------------------------- inference
@torch.no_grad()
def probs(model, x, res, mean, std, bs=1000, shift=0, scale=1.0):
    """x: uint8 GPU tensor (N,32,32,3) -> softmax probs (N,100). Flip TTA always;
    `shift` adds 4 translated views; `scale` rescales the evaluation resolution."""
    model.eval()
    if shift == 0:
        offs = [(0, 0)]
    elif shift > 0:                     # centre + 4 axis-aligned translations
        offs = [(0, 0), (-shift, 0), (shift, 0), (0, -shift), (0, shift)]
    else:                               # full 3x3 grid of translations
        s = -shift
        offs = [(dy, dx) for dy in (-s, 0, s) for dx in (-s, 0, s)]
    shift = abs(shift)
    r = int(round(res * scale / 8)) * 8
    out = []
    for s in range(0, len(x), bs):
        x0 = x[s:s + bs].permute(0, 3, 1, 2).float().div_(255.).sub_(mean).div_(std)
        xp = F.pad(x0, (shift,) * 4, mode="reflect") if shift else x0
        h = x0.shape[-1]
        acc, n = 0.0, 0
        for dy, dx in offs:
            xb = xp[:, :, shift + dy:shift + dy + h, shift + dx:shift + dx + h] if shift else x0
            if r != h:
                xb = F.interpolate(xb, size=(r, r), mode="bilinear", align_corners=False)
            xb = xb.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                acc = acc + model(xb).float().softmax(1)
                acc = acc + model(xb.flip(3)).float().softmax(1)
                n += 2
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
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--blocks", default="3,4,6,3")
    ap.add_argument("--res", type=int, default=48)
    ap.add_argument("--res0", type=int, default=0)      # progressive resize start (0 = off)
    ap.add_argument("--resramp", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=BUDGET_STEPS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--cut", type=int, default=12)
    ap.add_argument("--ta", type=int, default=1)
    ap.add_argument("--shift", type=int, default=2)
    ap.add_argument("--pad", type=int, default=4)
    ap.add_argument("--warm", type=int, default=300)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--cutmix", type=float, default=0.0)
    ap.add_argument("--mixp", type=float, default=0.5)
    ap.add_argument("--ema", type=float, default=0.0)
    ap.add_argument("--cycles", default="")      # e.g. "3000,5000": SGDR restarts + snapshots
    ap.add_argument("--tap", type=float, default=1.0)   # prob. of applying TrivialAugment
    ap.add_argument("--traindiag", type=int, default=1)
    ap.add_argument("--rrc", type=float, default=0.0)     # min area frac (0 = plain pad-crop)
    ap.add_argument("--rrcmax", type=float, default=1.2)
    ap.add_argument("--save", default="")     # dump final weights (offline TTA exploration)
    ap.add_argument("--ttaspec", default=":0:1.0,S:2:1.0,T:3:1.0,Z:2:1.17")
    ap.add_argument("--probdir", default="/workspace/probs")
    ap.add_argument("--write", type=int, default=1)
    ap.add_argument("--compile", type=int, default=0)
    a = ap.parse_args()

    assert a.steps <= BUDGET_STEPS, "step budget"
    assert a.steps * a.batch <= BUDGET_PRESENTATIONS, "presentation budget"

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
    net = torch.compile(model) if a.compile else model

    decay, nodecay = [], []
    for n_, p_ in model.named_parameters():
        (nodecay if p_.ndim <= 1 else decay).append(p_)
    opt = torch.optim.SGD([{"params": decay, "weight_decay": a.wd},
                           {"params": nodecay, "weight_decay": 0.0}],
                          lr=a.lr, momentum=0.9, nesterov=True)

    mparams = list(model.parameters())
    ema = [p.detach().clone().float() for p in mparams] if a.ema > 0 else None

    cl = [int(v) for v in a.cycles.split(",")] if a.cycles else [a.steps]
    assert sum(cl) == a.steps, (cl, a.steps)
    cyc_of, snap_at, c0 = [], set(), 0
    for L in cl:
        cyc_of += [(c0, L)] * L
        c0 += L
        if c0 < a.steps:
            snap_at.add(c0 - 1)                # snapshot at the end of every non-final cycle
    snaps = []

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

        if a.res0:                                   # progressive resize: res0 -> res
            fr = min(1.0, step / max(1, a.resramp * a.steps))
            r = int(round((a.res0 + fr * (a.res - a.res0)) / 8)) * 8
        else:
            r = a.res
        xb = augment_rrc(xf[idx], a.pad, a.cut, r, gen, mean, std, a.ta * a.tap,
                         a.rrc, a.rrcmax) if a.rrc > 0 else \
            augment(xf[idx], a.pad, a.cut, r, gen, mean, std, a.ta * a.tap)
        yb = ytr[idx]
        presentations += xb.shape[0]           # one augmented view per sample per step

        c0, clen = cyc_of[step]                # cycle start / length (single cycle by default)
        w_ = a.warm if c0 == 0 else 100        # short re-warm-up after an SGDR restart
        s_ = step - c0
        lr = a.lr * (s_ + 1) / w_ if s_ < w_ else \
            0.5 * a.lr * (1 + np.cos(np.pi * (s_ - w_) / max(1, clen - w_)))
        for g in opt.param_groups:
            g["lr"] = lr

        mixp, mixc = None, 1.0
        if (a.cutmix > 0 or a.mixup > 0) and np.random.rand() < a.mixp:
            if a.cutmix > 0 and (a.mixup <= 0 or np.random.rand() < 0.5):
                xb, mixp, mixc = cutmix_batch(xb, a.cutmix, gen)
            else:
                mixc = float(np.random.beta(a.mixup, a.mixup))
                mixp = torch.randperm(xb.shape[0], device=dev, generator=gen)
                xb = mixc * xb + (1 - mixc) * xb[mixp]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = net(xb)
            loss = F.cross_entropy(o, yb, label_smoothing=a.ls)
            if mixp is not None:
                loss = mixc * loss + (1 - mixc) * F.cross_entropy(o, yb[mixp],
                                                                  label_smoothing=a.ls)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if ema is not None:
            with torch.no_grad():
                d_ = a.ema if step >= 200 else min(a.ema, (step + 1) / (step + 10))
                torch._foreach_mul_(ema, d_)
                torch._foreach_add_(ema, [p.detach() for p in mparams], alpha=1 - d_)

        if step in snap_at:                    # SGDR snapshot (weights + BN buffers)
            snaps.append({k: v.detach().clone() for k, v in model.state_dict().items()})

    torch.cuda.synchronize()
    ttrain = time.time() - t0
    if a.save:
        torch.save({"sd": model.state_dict(), "mean": mean, "std": std, "cfg": vars(a)}, a.save)
    assert presentations <= BUDGET_PRESENTATIONS, presentations

    # ---------------- inference: store every (weight-set x TTA) variant
    os.makedirs(a.probdir, exist_ok=True)
    res_j = {"tag": a.tag, "params": nparam, "steps": a.steps,
             "presentations": presentations, "train_s": ttrain, "cfg": vars(a)}
    store = {}

    def evaluate(suffix):
        for spec in a.ttaspec.split(","):
            nm, sh, sc = spec.split(":")
            k = suffix + nm
            store["val_" + k] = probs(model, xva, a.res, mean, std, shift=int(sh), scale=float(sc))
            store["gr_" + k] = probs(model, xgr, a.res, mean, std, shift=int(sh), scale=float(sc))

    evaluate("raw")
    if snaps:
        final_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for i, sd in enumerate(snaps):
            model.load_state_dict(sd)
            evaluate(f"c{i}")
        model.load_state_dict(final_sd)
    if a.traindiag:                            # clean train accuracy -> under/over-fit regime
        ntd = min(2000, N)
        ptd = probs(model, xtr[:ntd], a.res, mean, std, shift=0)
        res_j["train_acc"] = float((ptd.argmax(1) == ytr[:ntd].cpu().numpy()).mean())
    if ema is not None:
        bk = [p.detach().clone() for p in mparams]
        with torch.no_grad():
            for p, m in zip(mparams, ema):
                p.copy_(m)
        evaluate("ema")            # BN buffers shared with raw model (running stats smoothed)
        with torch.no_grad():
            for p, b in zip(mparams, bk):
                p.copy_(b)

    prior = np.bincount(ytr.cpu().numpy(), minlength=N_CLASSES).astype(np.float64)
    prior /= prior.sum()
    np.savez(os.path.join(a.probdir, f"{a.tag}.npz"), yva=yva, prior=prior, **store)

    # ---------------- choose the inference combination ON VAL, apply to graded
    keys = sorted(k[4:] for k in store if k.startswith("val_"))
    for k in keys:
        res_j["val_" + k] = bal_acc(store["val_" + k], yva)
    cands = []
    for r in range(1, len(keys) + 1):
        for c in itertools.combinations(keys, r):
            v = sum(store["val_" + k] for k in c)
            v /= v.sum(1, keepdims=True)
            for tau in (0.0, 0.5, 1.0):
                cands.append((bal_acc(v / prior[None, :] ** tau, yva), c, tau))
    cands.sort(key=lambda t: -t[0])
    best = cands[0]
    res_j["best_combo"] = "+".join(best[1])
    res_j["best_tau"] = best[2]
    res_j["val_best"] = best[0]
    res_j["top5"] = [("+".join(c), t, round(s, 4)) for s, c, t in cands[:5]]
    # all-views average: robust fallback, reported for comparison
    allv = sum(store["val_" + k] for k in keys)
    allv /= allv.sum(1, keepdims=True)
    for tau in (0.0, 0.5, 1.0):
        res_j[f"val_allviews_t{tau}"] = bal_acc(allv / prior[None, :] ** tau, yva)
    res_j["val_allviews"] = res_j["val_allviews_t0.0"]
    rawk = [k for k in keys if k.startswith("raw")]          # final weight-set, all TTA views
    rv = sum(store["val_" + k] for k in rawk)
    rv /= rv.sum(1, keepdims=True)
    res_j["val_rawviews_t0.5"] = bal_acc(rv / prior[None, :] ** 0.5, yva)

    res_j["total_s"] = time.time() - t0
    print(json.dumps(res_j, default=str))
    with open(os.path.join(a.probdir, f"{a.tag}.json"), "w") as f:
        json.dump(res_j, f, default=str)

    if a.write:
        g = sum(store["gr_" + k] for k in best[1])
        g /= g.sum(1, keepdims=True)
        pred = (g / prior[None, :] ** best[2]).argmax(1).astype(np.int64)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        np.save(a.out, pred)
        print("wrote", a.out, "combo", res_j["best_combo"], "tau", best[2],
              "val", round(best[0], 4), "in", round(time.time() - t0, 1), "s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
