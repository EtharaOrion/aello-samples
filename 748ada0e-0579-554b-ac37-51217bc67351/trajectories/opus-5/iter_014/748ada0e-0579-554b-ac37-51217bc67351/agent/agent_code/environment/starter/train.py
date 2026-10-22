"""CIFAR-100 under a fixed step budget: 8000 updates, 1,024,000 presentations.

Everything is GPU-resident (no dataloader). Model is a ResNet-34 layout with
ResNet-D style anti-aliased downsampling, trained at 48px (32px inputs upsampled).
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BUDGET_STEPS = 8000
BUDGET_PRESENTATIONS = 1024000
N_CLASSES = 100


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, 1, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        nn.init.zeros_(self.b2.weight)
        self.stride = stride
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(cout))
        else:
            self.sc = None

    def forward(self, x):
        # anti-aliased transition: blur/avg-pool both paths, convs stay stride 1
        if self.stride != 1:
            x = F.avg_pool2d(x, 2, 2)
        s = x if self.sc is None else self.sc(x)
        o = F.relu(self.b1(self.c1(x)), inplace=True)
        o = self.b2(self.c2(o))
        return F.relu(o + s, inplace=True)


class Net(nn.Module):
    def __init__(self, width=128, depths=(3, 4, 6, 3), n_classes=N_CLASSES):
        super().__init__()
        w = [width, width * 2, width * 4, width * 8]
        self.stem = nn.Sequential(nn.Conv2d(3, w[0], 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(w[0]), nn.ReLU(inplace=True))
        layers, cin = [], w[0]
        for si, (d, c) in enumerate(zip(depths, w)):
            for bi in range(d):
                layers.append(Block(cin, c, stride=2 if (bi == 0 and si > 0) else 1))
                cin = c
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(cin, n_classes)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.body(self.stem(x))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


# ------------------------------------------------------- GPU augmentation
def rand_crop(x, pad=4):
    b, c, h, w = x.shape
    xp = F.pad(x, (pad,) * 4, mode="reflect")
    oy = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    ox = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    ar = torch.arange(h, device=x.device)
    rows = (oy[:, None] + ar).view(b, h, 1).expand(b, h, w)
    cols = (ox[:, None] + ar).view(b, 1, w).expand(b, h, w)
    bi = torch.arange(b, device=x.device).view(b, 1, 1).expand(b, h, w)
    return xp[bi, :, rows, cols].permute(0, 3, 1, 2).contiguous()


def rand_flip(x):
    m = torch.rand(x.shape[0], device=x.device) < 0.5
    return torch.where(m.view(-1, 1, 1, 1), x.flip(-1), x)


def cutout(x, size):
    if size <= 0:
        return x
    b, c, h, w = x.shape
    cy = torch.randint(0, h, (b, 1, 1), device=x.device)
    cx = torch.randint(0, w, (b, 1, 1), device=x.device)
    ar_y = torch.arange(h, device=x.device).view(1, h, 1)
    ar_x = torch.arange(w, device=x.device).view(1, 1, w)
    m = ((ar_y - cy).abs() < size // 2) & ((ar_x - cx).abs() < size // 2)
    return x * (~m).unsqueeze(1).to(x.dtype)


def _blend(a, b, f):
    return (a + (b - a) * f).clamp(0, 1)


def trivial_augment(x, n_ops=1, strength=1.0):
    """Batched TrivialAugment: each sample gets one op with a uniformly random magnitude."""
    for _ in range(n_ops):
        x = _ta_once(x, strength)
    return x


_TA_N = 13  # 0 id, 1 shearX, 2 shearY, 3 transX, 4 transY, 5 rot,
#             6 bright, 7 contrast, 8 color, 9 sharp, 10 poster, 11 solar, 12 autoc


def _ta_once(x, s=1.0):
    b = x.shape[0]
    dev = x.device
    op = torch.randint(0, _TA_N, (b,), device=dev)
    m = torch.rand(b, device=dev)                       # magnitude in [0,1]
    sgn = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)

    # --- geometric ops folded into a single affine grid_sample
    z = torch.zeros(b, device=dev)
    shx = torch.where(op == 1, sgn * m * 0.30 * s, z)
    shy = torch.where(op == 2, sgn * m * 0.30 * s, z)
    tx = torch.where(op == 3, sgn * m * 0.30 * s, z)
    ty = torch.where(op == 4, sgn * m * 0.30 * s, z)
    ang = torch.where(op == 5, sgn * m * (30.0 * s) * np.pi / 180.0, z)
    ca, sa = torch.cos(ang), torch.sin(ang)
    # rotation composed with shear
    theta = torch.stack([torch.stack([ca, -sa + shx, tx * 2], 1),
                         torch.stack([sa + shy, ca, ty * 2], 1)], 1)
    if bool((op <= 5).any() & (op >= 1).any()):
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        xg = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection",
                           align_corners=False)
        x = torch.where((op >= 1).logical_and(op <= 5).view(-1, 1, 1, 1), xg, x)

    # --- photometric ops (cheap, compute all then select)
    f = (1.0 + sgn * m * 0.9 * s).view(-1, 1, 1, 1)
    out = x
    mean_gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    out = torch.where((op == 6).view(-1, 1, 1, 1), _blend(torch.zeros_like(x), x, f), out)
    cmean = mean_gray.mean(dim=(1, 2, 3), keepdim=True)
    out = torch.where((op == 7).view(-1, 1, 1, 1),
                      _blend(cmean.expand_as(x), x, f), out)
    out = torch.where((op == 8).view(-1, 1, 1, 1),
                      _blend(mean_gray.expand_as(x), x, f), out)
    k = x.new_tensor([[1., 1., 1.], [1., 5., 1.], [1., 1., 1.]]) / 13.0
    blur = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"),
                    k.view(1, 1, 3, 3).expand(3, 1, 3, 3), groups=3)
    out = torch.where((op == 9).view(-1, 1, 1, 1), _blend(blur, x, f), out)
    bits = (8 - (m * 4.0 * s).long()).clamp(2, 8).view(-1, 1, 1, 1)
    lv = (2.0 ** bits)
    out = torch.where((op == 10).view(-1, 1, 1, 1), (x * lv).floor() / lv, out)
    thr = (1.0 - m * 0.8 * s).view(-1, 1, 1, 1)
    out = torch.where((op == 11).view(-1, 1, 1, 1), torch.where(x >= thr, 1.0 - x, x), out)
    lo = x.amin(dim=(2, 3), keepdim=True)
    hi = x.amax(dim=(2, 3), keepdim=True)
    out = torch.where((op == 12).view(-1, 1, 1, 1),
                      ((x - lo) / (hi - lo).clamp_min(1e-3)).clamp(0, 1), out)
    return out


# ---------------------------------------------------------------- helpers
def bal_acc(pred, y, n=N_CLASSES):
    rec = []
    for c in range(n):
        m = y == c
        if m.sum():
            rec.append((pred[m] == c).mean())
    return float(np.mean(rec))


@torch.no_grad()
def logits_of(model, x_u8, mean, std, res, bs=1000, flip_tta=True):
    """x_u8: uint8 CUDA tensor N,32,32,3 -> float32 softmax-probs on CPU."""
    model.eval()
    outs = []
    for s in range(0, x_u8.shape[0], bs):
        xb = x_u8[s:s + bs].permute(0, 3, 1, 2).float().div_(255.)
        if res != 32:
            xb = F.interpolate(xb, size=(res, res), mode="bilinear", align_corners=False)
        xb = ((xb - mean) / std).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(xb).float().softmax(1)
            if flip_tta:
                p = p + model(xb.flip(-1)).float().softmax(1)
        outs.append(p.float())
    model.train()
    return torch.cat(outs)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                s.copy_(v)

    def copy_to(self, model):
        model.load_state_dict({k: v.to(dtype=p.dtype) for (k, v), p
                               in zip(self.shadow.items(), model.state_dict().values())})


# ---------------------------------------------------------------- training
def run(a):
    t0 = time.time()
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    tr = np.load(os.path.join(DATA, "train_i20k.npz"))
    xtr = torch.from_numpy(tr["x"]).to(dev)                      # N,32,32,3 uint8
    ytr = torch.from_numpy(tr["y"]).to(dev)
    va = np.load(os.path.join(DATA, "val_public.npz"))
    xva = torch.from_numpy(va["x"]).to(dev)
    yva = va["y"]

    xf = xtr.permute(0, 3, 1, 2).float().div(255.)
    mean = xf.mean(dim=(0, 2, 3)).view(1, 3, 1, 1)
    std = xf.std(dim=(0, 2, 3)).view(1, 3, 1, 1)
    del xf

    model = Net(a.width, tuple(int(d) for d in a.depths.split(","))).to(dev)
    model = model.to(memory_format=torch.channels_last)
    nparam = sum(p.numel() for p in model.parameters())

    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.SGD([{"params": decay, "weight_decay": a.wd},
                           {"params": no_decay, "weight_decay": 0.0}],
                          lr=a.lr, momentum=0.9, nesterov=True)

    steps, bs = a.steps, a.batch
    assert steps <= BUDGET_STEPS and steps * bs <= BUDGET_PRESENTATIONS, "over budget"
    warm = int(0.04 * steps)

    def lr_at(i):
        if i < warm:
            return a.lr * (i + 1) / warm
        t = (i - warm) / max(1, steps - warm)
        return a.lr * 0.5 * (1 + np.cos(np.pi * t))

    ema1 = EMA(model, a.ema1)
    ema2 = EMA(model, a.ema2)

    N = xtr.shape[0]
    cls_idx = None
    if a.balanced:
        order = torch.argsort(ytr)
        ys = ytr[order]
        counts = torch.bincount(ytr, minlength=N_CLASSES)
        starts = torch.cumsum(counts, 0) - counts
        cls_idx = (order, starts, counts)

    g = torch.Generator(device=dev); g.manual_seed(a.seed + 1234)
    presentations = 0
    model.train()
    for it in range(steps):
        for gparam in opt.param_groups:
            gparam["lr"] = lr_at(it)
        if a.balanced:
            order, starts, counts = cls_idx
            c = torch.randint(0, N_CLASSES, (bs,), device=dev, generator=g)
            off = (torch.rand(bs, device=dev, generator=g) * counts[c].float()).long()
            idx = order[starts[c] + off]
        else:
            idx = torch.randint(0, N, (bs,), device=dev, generator=g)
        xb = xtr[idx].permute(0, 3, 1, 2).float().div_(255.)
        yb = ytr[idx]
        xb = rand_crop(xb, a.pad)
        xb = rand_flip(xb)
        if a.ta > 0:
            xb = trivial_augment(xb, a.ta, a.ta_strength)
        if a.cutout > 0:
            xb = cutout(xb, a.cutout)
        if a.res != 32:
            xb = F.interpolate(xb, size=(a.res, a.res), mode="bilinear", align_corners=False)
        xb = ((xb - mean) / std).contiguous(memory_format=torch.channels_last)

        lam = None
        if a.mixup > 0 and np.random.rand() < a.mixup_p:
            lam = float(np.random.beta(a.mixup, a.mixup))
            perm = torch.randperm(bs, device=dev, generator=g)
            xb = lam * xb + (1 - lam) * xb[perm]
            yb2 = yb[perm]

        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(xb)
            if lam is None:
                loss = F.cross_entropy(out, yb, label_smoothing=a.ls)
            else:
                loss = (lam * F.cross_entropy(out, yb, label_smoothing=a.ls)
                        + (1 - lam) * F.cross_entropy(out, yb2, label_smoothing=a.ls))
        loss.backward()
        opt.step()
        presentations += bs
        ema1.update(model)
        ema2.update(model)
        if a.log and (it + 1) % 1000 == 0:
            print(f"  step {it+1}/{steps} loss {loss.item():.3f} "
                  f"lr {lr_at(it):.4f} {time.time()-t0:.0f}s", flush=True)

    t_train = time.time() - t0
    print(f"[budget] steps {steps}/{BUDGET_STEPS} presentations "
          f"{presentations}/{BUDGET_PRESENTATIONS} params {nparam/1e6:.1f}M "
          f"train {t_train:.1f}s", flush=True)

    # ---- evaluate raw + two EMAs, save probability tensors
    import copy
    raw_sd = copy.deepcopy(model.state_dict())
    res, probs = a.res, {}
    probs["raw"] = logits_of(model, xva, mean, std, res)
    gp = {}
    xg = torch.from_numpy(np.load(a.graded)["x"]).to(dev) if a.graded else None
    if xg is not None:
        gp["raw"] = logits_of(model, xg, mean, std, res)
    for name, e in (("ema1", ema1), ("ema2", ema2)):
        e.copy_to(model)
        probs[name] = logits_of(model, xva, mean, std, res)
        if xg is not None:
            gp[name] = logits_of(model, xg, mean, std, res)
    model.load_state_dict(raw_sd)

    prior = torch.bincount(ytr, minlength=N_CLASSES).float()
    prior = (prior / prior.sum())

    def report(p, tag):
        pr = p.argmax(1).cpu().numpy()
        b = bal_acc(pr, yva)
        padj = (p / prior.to(p.device).pow(a.prior_pow).unsqueeze(0)).argmax(1).cpu().numpy()
        b2 = bal_acc(padj, yva)
        print(f"  {tag:16s} bal_acc {b:.4f}   prior-adj {b2:.4f}", flush=True)
        return b, b2

    results = {}
    for k in ["raw", "ema1", "ema2"]:
        results[k] = report(probs[k], k)
    ens = probs["raw"] + probs["ema1"] + probs["ema2"]
    results["ens3"] = report(ens, "raw+ema1+ema2")
    ens2 = probs["raw"] + probs["ema1"]
    results["ens2"] = report(ens2, "raw+ema1")

    if a.save_ckpt:
        torch.save({"raw": raw_sd, "ema1": ema1.shadow, "ema2": ema2.shadow,
                    "mean": mean.cpu(), "std": std.cpu(),
                    "cfg": vars(a)}, a.save_ckpt)
    if a.save_probs:
        torch.save({"val": {k: v.cpu() for k, v in probs.items()},
                    "graded": {k: v.cpu() for k, v in gp.items()},
                    "yva": yva, "prior": prior.cpu()}, a.save_probs)
        print(f"  saved probs -> {a.save_probs}", flush=True)

    if a.out and xg is not None:
        # fixed a-priori rule: average raw + both EMAs, flip-TTA, prior-corrected
        pg = gp["raw"] + gp["ema1"] + gp["ema2"]
        pg = pg / prior.to(pg.device).pow(a.prior_pow).unsqueeze(0)
        pred = pg.argmax(1).cpu().numpy().astype(np.int64)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        np.save(a.out, pred)
        print(f"wrote {a.out} shape {pred.shape} uniq {len(np.unique(pred))}", flush=True)

    print(f"[total] {time.time()-t0:.1f}s", flush=True)
    print("RESULT " + json.dumps({k: [round(x, 4) for x in v] for k, v in results.items()}))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="i20k")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graded", default=os.path.join(DATA, "graded_images.npz"))
    ap.add_argument("--out", default="")
    ap.add_argument("--steps", type=int, default=BUDGET_STEPS)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--wd", type=float, default=2e-3)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depths", default="3,4,6,3")
    ap.add_argument("--res", type=int, default=48)
    ap.add_argument("--pad", type=int, default=4)
    ap.add_argument("--cutout", type=int, default=12)
    ap.add_argument("--ta", type=int, default=1)
    ap.add_argument("--ta_strength", type=float, default=1.0)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--mixup_p", type=float, default=0.5)
    ap.add_argument("--ema1", type=float, default=0.999)
    ap.add_argument("--ema2", type=float, default=0.9995)
    ap.add_argument("--balanced", type=int, default=0)
    ap.add_argument("--prior_pow", type=float, default=0.0)
    ap.add_argument("--save_probs", default="")
    ap.add_argument("--save_ckpt", default="")
    ap.add_argument("--log", type=int, default=1)
    a = ap.parse_args()
    run(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
