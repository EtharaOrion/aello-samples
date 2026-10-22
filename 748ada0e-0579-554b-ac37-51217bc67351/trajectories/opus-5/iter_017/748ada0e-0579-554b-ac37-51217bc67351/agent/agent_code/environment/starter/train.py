"""CIFAR-100 @ 20k images, fixed budget: 8000 steps x batch 128 = 1,024,000 presentations.

Single training run; all data GPU-resident; batched augmentation (no dataloader).
After training, val probabilities are computed for several inference variants
(weights x test-resolution x TTA), the best combination is chosen on val_public,
and graded probabilities are computed only for that combination.
"""
import argparse, json, os, sys, time
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
    def __init__(self, cin, cout, stride, se=0):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        nn.init.zeros_(self.b2.weight)
        if se:
            h = max(8, cout // se)
            self.se = nn.Sequential(nn.Linear(cout, h), nn.ReLU(inplace=True),
                                    nn.Linear(h, cout), nn.Sigmoid())
        else:
            self.se = None
        if stride != 1 or cin != cout:
            pool = [nn.AvgPool2d(2, 2)] if stride != 1 else []          # ResNet-D shortcut
            self.sc = nn.Sequential(*pool, nn.Conv2d(cin, cout, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(cout))
        else:
            self.sc = None

    def forward(self, x):
        o = F.relu(self.b1(self.c1(x)), inplace=True)
        o = self.b2(self.c2(o))
        if self.se is not None:
            w = self.se(F.adaptive_avg_pool2d(o, 1).flatten(1))
            o = o * w[:, :, None, None]
        return F.relu(o + (x if self.sc is None else self.sc(x)), inplace=True)


class Net(nn.Module):
    def __init__(self, w=128, blocks=(3, 4, 6, 3), n_classes=N_CLASSES, se=0, mult=(1, 2, 4, 8)):
        super().__init__()
        ws = [int(round(w * m)) for m in mult]
        self.stem = nn.Sequential(nn.Conv2d(3, w, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(w), nn.ReLU(inplace=True))
        layers, cin = [], w
        for i, (n, c) in enumerate(zip(blocks, ws)):
            for j in range(n):
                layers.append(Block(cin, c, 2 if (j == 0 and i > 0) else 1, se))
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
def probs(model, x, res, mean, std, flip_tta=True, bs=1000, shift=0, mode="plus"):
    """x: uint8 GPU tensor (N,32,32,3) -> softmax probs (N,100). Optional flip/shift TTA."""
    model.eval()
    if shift == 0:
        offs = [(0, 0)]
    elif mode == "grid":
        offs = [(dy, dx) for dy in (-shift, 0, shift) for dx in (-shift, 0, shift)]
    else:
        offs = [(0, 0), (-shift, 0), (shift, 0), (0, -shift), (0, shift)]
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


@torch.no_grad()
def recal_bn(model, xf, mean, std, res, gen, nbatch, batch):
    """Re-estimate BN running stats on CLEAN training views.

    Training BN stats are biased: they see cutout holes and TrivialAugment colour
    distortions, while test images are clean. Uses `nbatch*batch` extra training-sample
    presentations, which are counted against the budget by the caller.
    """
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    saved = [(m.running_mean.clone(), m.running_var.clone(),
              m.num_batches_tracked.clone(), m.momentum) for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None                      # cumulative moving average
    model.train()
    N = xf.shape[0]
    for _ in range(nbatch):
        idx = torch.randint(0, N, (batch,), device=xf.device, generator=gen)
        xb = (xf[idx] - mean) / std
        if res != xb.shape[-1]:
            xb = F.interpolate(xb, size=(res, res), mode="bilinear", align_corners=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(xb.contiguous(memory_format=torch.channels_last))
    return bns, saved


def restore_bn(bns, saved):
    for m, (rm, rv, nb, mom) in zip(bns, saved):
        m.running_mean.copy_(rm); m.running_var.copy_(rv)
        m.num_batches_tracked.copy_(nb); m.momentum = mom


def bal_acc(p, y):
    pred = p.argmax(1)
    return float(np.mean([(pred[y == c] == c).mean() for c in range(N_CLASSES) if (y == c).any()]))


def res_at(step, total, sched):
    """sched: 'r1:frac1,r2:frac2,...' cumulative fractions of training. '' -> constant."""
    f = (step + 1) / total
    for part in sched.split(","):
        r, fr = part.split(":")
        if f <= float(fr) + 1e-9:
            return int(r)
    return int(sched.split(",")[-1].split(":")[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="i20k")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graded", default=os.path.join(DATA, "graded_images.npz"))
    ap.add_argument("--out", default="/submission/preds_i20k.npy")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--blocks", default="3,4,6,3")
    ap.add_argument("--se", type=int, default=0)          # SE reduction ratio; 0 = off
    ap.add_argument("--mult", default="1,2,4,8")          # per-stage width multipliers
    ap.add_argument("--res", type=int, default=48)
    ap.add_argument("--resched", default="")        # e.g. "32:0.35,48:0.8,56:1.0"
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
    ap.add_argument("--cutmix", type=float, default=0.0)      # Beta alpha; 0 = off
    ap.add_argument("--cutmix_p", type=float, default=0.5)    # per-step probability
    ap.add_argument("--ema", type=float, default=0.99)
    ap.add_argument("--testres", default="")
    ap.add_argument("--recal", type=int, default=0)    # BN-recalibration batches (0 = off)
    ap.add_argument("--maxkeys", type=int, default=2)   # cap graded inference passes
    ap.add_argument("--ttas", default="s2")   # comma list: s<N>=plus shifts, g<N>=3x3 grid        # extra test resolutions, e.g. "48,56"
    ap.add_argument("--probdir", default="/workspace/probs")
    ap.add_argument("--write", type=int, default=1)
    ap.add_argument("--tmax", type=float, default=395.0)   # wall-clock guard for eval stage
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
    model = Net(a.width, blocks, se=a.se,
                mult=tuple(float(v) for v in a.mult.split(","))).to(dev).to(memory_format=torch.channels_last)
    nparam = sum(p.numel() for p in model.parameters())

    decay, nodecay = [], []
    for n_, p_ in model.named_parameters():
        (nodecay if p_.ndim <= 1 else decay).append(p_)
    opt = torch.optim.SGD([{"params": decay, "weight_decay": a.wd},
                           {"params": nodecay, "weight_decay": 0.0}],
                          lr=a.lr, momentum=0.9, nesterov=True)

    ema_v = [p.detach().clone().float() for p in model.parameters()] if a.ema > 0 else None
    mparams = list(model.parameters())

    gen = torch.Generator(device=dev); gen.manual_seed(a.seed + 1234)
    N = xf.shape[0]
    perm = torch.randperm(N, device=dev, generator=gen)
    ptr = 0
    presentations = 0
    updates = 0
    model.train()
    for step in range(a.steps):
        if ptr + a.batch > N:
            perm = torch.randperm(N, device=dev, generator=gen)
            ptr = 0
        idx = perm[ptr:ptr + a.batch]; ptr += a.batch
        r = res_at(step, a.steps, a.resched) if a.resched else a.res
        xb = augment(xf[idx], a.pad, a.cut, r, gen, mean, std, a.ta)
        yb = ytr[idx]
        presentations += xb.shape[0]

        lr = a.lr * (step + 1) / a.warm if step < a.warm else \
            0.5 * a.lr * (1 + np.cos(np.pi * (step - a.warm) / max(1, a.steps - a.warm)))
        for g in opt.param_groups:
            g["lr"] = lr

        with torch.autocast("cuda", dtype=torch.bfloat16):
            if a.cutmix > 0 and np.random.rand() < a.cutmix_p:
                lam = float(np.random.beta(a.cutmix, a.cutmix))
                pi = torch.randperm(xb.shape[0], device=dev, generator=gen)
                H = xb.shape[2]
                rh = int(H * np.sqrt(1.0 - lam))
                if rh > 0:
                    cy = np.random.randint(H); cx = np.random.randint(H)
                    y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, H)
                    x1, x2 = max(cx - rh // 2, 0), min(cx + rh // 2, H)
                    xb[:, :, y1:y2, x1:x2] = xb[pi][:, :, y1:y2, x1:x2]
                    lam = 1.0 - ((y2 - y1) * (x2 - x1) / (H * H))
                o = model(xb)
                loss = lam * F.cross_entropy(o, yb, label_smoothing=a.ls) + \
                    (1 - lam) * F.cross_entropy(o, yb[pi], label_smoothing=a.ls)
            elif a.mixup > 0:
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
        updates += 1

        if ema_v is not None:
            with torch.no_grad():
                d_ = a.ema if step >= 200 else min(a.ema, (step + 1) / (step + 10))
                torch._foreach_mul_(ema_v, d_)
                torch._foreach_add_(ema_v, [p.detach() for p in mparams], alpha=1 - d_)

    ttrain = time.time() - t0
    final_res = res_at(a.steps - 1, a.steps, a.resched) if a.resched else a.res

    # ------------------------------------------------------------ inference-variant selection
    os.makedirs(a.probdir, exist_ok=True)
    res_info = {"tag": a.tag, "params": nparam, "steps": a.steps, "updates": updates,
                "presentations": presentations, "train_s": ttrain, "final_res": final_res,
                "cfg": vars(a)}
    prior = np.bincount(ytr.cpu().numpy(), minlength=N_CLASSES).astype(np.float64)
    prior /= prior.sum()

    test_res = sorted({final_res} | {int(v) for v in a.testres.split(",") if v.strip()})
    ttas = [t.strip() for t in a.ttas.split(",") if t.strip()]
    weight_sets = [("raw", None)]
    if ema_v is not None:
        weight_sets.append(("ema", ema_v))
    backup = [p.detach().clone() for p in mparams]

    valp, scores, recal_state = {}, {}, {}
    for wname, wv in weight_sets:
        with torch.no_grad():
            for p, b in zip(mparams, backup if wv is None else wv):
                p.copy_(b)
        for tail, do_recal in ([("", 0)] + ([("R", 1)] if a.recal else [])):
            if do_recal:
                bns, saved = recal_bn(model, xf, mean, std, final_res, gen, a.recal, a.batch)
                presentations += a.recal * a.batch
                recal_state[wname] = [(m.running_mean.clone(), m.running_var.clone(),
                                       m.num_batches_tracked.clone(), m.momentum) for m in bns]
            for r in test_res:
                for t in ttas:
                    k = f"{wname}{tail}|{r}|{t}"
                    valp[k] = probs(model, xva, r, mean, std, shift=int(t[1:]),
                                    mode="grid" if t[0] == "g" else "plus")
                    scores[k] = bal_acc(valp[k], yva)
            if do_recal:
                restore_bn(bns, saved)
    with torch.no_grad():
        for p, b in zip(mparams, backup):
            p.copy_(b)

    # candidate combinations: single variants plus averages over weights / resolutions
    cands = {k: (k,) for k in valp}
    for r in test_res:                                   # average the two weight sets
        for t in ttas:
            for tail in ("", "R"):
                ks = tuple(f"{w}{tail}|{r}|{t}" for w, _ in weight_sets)
                if len(ks) > 1 and all(k in valp for k in ks):
                    cands[f"both{tail}|{r}|{t}"] = ks
    for w, _ in weight_sets:                             # average TTA variants
        if len(ttas) > 1:
            cands[f"{w}|{final_res}|allt"] = tuple(f"{w}|{final_res}|{t}" for t in ttas)
    cands["all"] = tuple(valp)
    ranked = []
    cands = {c: ks for c, ks in cands.items() if len(ks) <= a.maxkeys}
    for cname, ks in cands.items():
        v = sum(valp[k] for k in ks)
        v = v / v.sum(1, keepdims=True)
        for tau in (0.0, 0.5, 1.0):
            ranked.append((bal_acc(v / prior[None, :] ** tau, yva), cname, tau, ks))
    ranked.sort(key=lambda t: -t[0])
    res_info["val_variants"] = {k: round(v, 4) for k, v in scores.items()}
    res_info["val_top"] = [(round(s, 4), c, t) for s, c, t, _ in ranked[:8]]

    # val is only 2000 images (bootstrap sd ~0.01), so picking the single argmax of ~90
    # candidates overfits it. Among candidates within `band` of the top, take the one that
    # averages the most variants -- averaging is the more robust estimator.
    band = 0.005
    top = ranked[0][0]
    near = [c for c in ranked if c[0] >= top - band]
    best_s, best_c, best_tau, best_ks = max(near, key=lambda t: (len(t[3]), t[0]))
    res_info["argmax"] = [round(ranked[0][0], 4), ranked[0][1], ranked[0][2]]
    res_info["chosen"] = {"combo": best_c, "tau": best_tau, "val": round(best_s, 4),
                          "keys": list(best_ks)}

    # ------------------------------------------------------------ graded predictions
    if a.write:
        gacc = None
        for wname, wv in weight_sets:
            for tail in ("", "R"):
                ks = [k for k in best_ks if k.split("|")[0] == wname + tail]
                if not ks:
                    continue
                with torch.no_grad():
                    for p, b in zip(mparams, backup if wv is None else wv):
                        p.copy_(b)
                if tail == "R":
                    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
                    saved = [(m.running_mean.clone(), m.running_var.clone(),
                              m.num_batches_tracked.clone(), m.momentum) for m in bns]
                    restore_bn(bns, recal_state[wname])
                for k in ks:
                    _, r, t = k.split("|")
                    g = probs(model, xgr, int(r), mean, std, shift=int(t[1:]),
                              mode="grid" if t[0] == "g" else "plus")
                    gacc = g if gacc is None else gacc + g
                if tail == "R":
                    restore_bn(bns, saved)
        with torch.no_grad():
            for p, b in zip(mparams, backup):
                p.copy_(b)
        gacc = gacc / gacc.sum(1, keepdims=True)
        pred = (gacc / prior[None, :] ** best_tau).argmax(1).astype(np.int64)
        assert pred.shape == (len(xgr),) and pred.min() >= 0 and pred.max() < N_CLASSES
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        np.save(a.out, pred)
        res_info["wrote"] = a.out

    np.savez(os.path.join(a.probdir, f"{a.tag}.npz"), yva=yva, prior=prior, **valp)
    res_info["presentations"] = presentations
    assert presentations <= BUDGET_PRESENTATIONS, ("presentation budget", presentations)
    assert updates <= BUDGET_STEPS
    res_info["total_s"] = time.time() - t0
    print(json.dumps(res_info, default=str))
    with open(os.path.join(a.probdir, f"{a.tag}.json"), "w") as f:
        json.dump(res_info, f, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
