"""CIFAR-100 @ 20k images, fixed budget: <=8000 optimizer steps, <=1,024,000 presentations.

Everything (data + augmentation) lives on the GPU; no dataloader.
"""
import argparse, os, sys, time, math, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = "/workspace/environment/data"
N_CLASSES = 100
BUDGET_STEPS = 8000
BUDGET_PRESENT = 1024000


# --------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.pool = nn.AvgPool2d(2) if stride == 2 else nn.Identity()
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        nn.init.zeros_(self.bn2.weight)
        if stride != 1 or cin != cout:
            lay = []
            if stride == 2:
                lay.append(nn.AvgPool2d(2))
            lay += [nn.Conv2d(cin, cout, 1, 1, 0, bias=False), nn.BatchNorm2d(cout)]
            self.short = nn.Sequential(*lay)
        else:
            self.short = nn.Identity()

    def forward(self, x):
        o = F.relu(self.bn1(self.conv1(x)), inplace=True)
        o = self.pool(o)
        o = self.bn2(self.conv2(o))
        return F.relu(o + self.short(x), inplace=True)


class Net(nn.Module):
    def __init__(self, width=128, depths=(3, 4, 6, 3), n_classes=N_CLASSES, drop=0.0):
        super().__init__()
        w = [width, width * 2, width * 4, width * 8]
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        blocks, cin = [], width
        for si, (d, c) in enumerate(zip(depths, w)):
            for bi in range(d):
                blocks.append(Block(cin, c, 2 if (bi == 0 and si > 0) else 1))
                cin = c
        self.blocks = nn.Sequential(*blocks)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.fc = nn.Linear(cin, n_classes)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(self.drop(x))


# ---------------------------------------------------------------------- augmentation
BLUR = None


def make_blur(dev, dt):
    k = torch.tensor([[1., 1., 1.], [1., 5., 1.], [1., 1., 1.]], device=dev, dtype=dt) / 13.
    return k.view(1, 1, 3, 3).repeat(3, 1, 1, 1)


def geo_mat(idx, mag, sgn, cfg, B, dev):
    z = torch.zeros(B, device=dev)
    ang = torch.where(idx == 2, sgn * mag * cfg["rot"], z)
    shx = torch.where(idx == 3, sgn * mag * cfg["shear"], z)
    shy = torch.where(idx == 4, sgn * mag * cfg["shear"], z)
    ttx = torch.where(idx == 5, sgn * mag * cfg["trans"] * 2.0, z)
    tty = torch.where(idx == 6, sgn * mag * cfg["trans"] * 2.0, z)
    M = torch.zeros(B, 3, 3, device=dev)
    ca, sa = torch.cos(ang), torch.sin(ang)
    M[:, 0, 0] = ca
    M[:, 0, 1] = -sa + shx
    M[:, 1, 0] = sa + shy
    M[:, 1, 1] = ca
    M[:, 0, 2] = ttx
    M[:, 1, 2] = tty
    M[:, 2, 2] = 1.0
    return M


def color_ops(x, idx, mag, sgn, cfg):
    global BLUR
    B = x.shape[0]
    dev = x.device
    one = torch.ones(B, 1, 1, 1, device=dev)
    m4, s4, i4 = mag.view(B, 1, 1, 1), sgn.view(B, 1, 1, 1), idx.view(B, 1, 1, 1)
    mn = x.amin(dim=(2, 3), keepdim=True)
    mx = x.amax(dim=(2, 3), keepdim=True)
    x = torch.where(i4 == 1, (x - mn) / (mx - mn).clamp_min(1e-3), x)
    x = x * torch.where(i4 == 7, one + s4 * m4 * cfg["cj"], one)
    gray = (x * torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)).sum(1, keepdim=True)
    m = gray.mean(dim=(2, 3), keepdim=True)
    x = m + (x - m) * torch.where(i4 == 8, one + s4 * m4 * cfg["cj"], one)
    x = gray + (x - gray) * torch.where(i4 == 9, one + s4 * m4 * cfg["cj"], one)
    if BLUR is None or BLUR.device != dev:
        BLUR = make_blur(dev, x.dtype)
    bl = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), BLUR.to(x.dtype), groups=3)
    x = bl + (x - bl) * torch.where(i4 == 10, one + s4 * m4 * cfg["cj"], one)
    x = x.clamp(0, 1)
    lv = 2.0 ** (8.0 - torch.round(m4 * 4.0))
    x = torch.where(i4 == 11, torch.floor(x * lv) / lv, x)
    thr = torch.where(i4 == 12, 1.0 - m4 * 0.6, one * 2.0)
    return torch.where(x >= thr, 1.0 - x, x)


def augment(x, cfg, gen):
    """x: float (B,3,32,32) in [0,1] -> augmented float in [0,1]."""
    B, dev, H = x.shape[0], x.device, x.shape[2]
    flip = torch.where(torch.rand(B, device=dev, generator=gen) < 0.5, -1.0, 1.0)
    pad = cfg["pad"]
    tbx = torch.randint(-pad, pad + 1, (B,), device=dev, generator=gen).float() * (2.0 / H)
    tby = torch.randint(-pad, pad + 1, (B,), device=dev, generator=gen).float() * (2.0 / H)

    draws = []
    for _ in range(cfg["ta_n"] if cfg["nops"] > 0 else 0):
        idx = torch.randint(0, cfg["nops"], (B,), device=dev, generator=gen)
        on = torch.rand(B, device=dev, generator=gen) < cfg["ta_p"]
        idx = torch.where(on, idx, torch.zeros_like(idx))          # op 0 == identity
        mag = torch.rand(B, device=dev, generator=gen)
        sgn = torch.where(torch.rand(B, device=dev, generator=gen) < 0.5, -1.0, 1.0)
        draws.append((idx, mag, sgn))

    M = torch.eye(3, device=dev).expand(B, 3, 3).contiguous()
    for idx, mag, sgn in draws:
        M = torch.bmm(M, geo_mat(idx, mag, sgn, cfg, B, dev))
    M[:, :, 0] = M[:, :, 0] * flip[:, None]
    M[:, 0, 2] = M[:, 0, 2] + tbx
    M[:, 1, 2] = M[:, 1, 2] + tby
    grid = F.affine_grid(M[:, :2, :], (B, 3, H, H), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    for idx, mag, sgn in draws:
        x = color_ops(x, idx, mag, sgn, cfg)
    return x.clamp(0, 1)


def cutout_(x, size, gen):
    if size <= 0:
        return x
    B, _, H, W = x.shape
    dev = x.device
    cy = torch.randint(0, H, (B, 1, 1), device=dev, generator=gen)
    cx = torch.randint(0, W, (B, 1, 1), device=dev, generator=gen)
    ys = torch.arange(H, device=dev).view(1, H, 1)
    xs = torch.arange(W, device=dev).view(1, 1, W)
    h = size // 2
    mask = ((ys - cy).abs() < h) & ((xs - cx).abs() < h)
    return x.masked_fill(mask.unsqueeze(1), 0.0)


# --------------------------------------------------------------------------- helpers
MEAN = torch.tensor([0.5071, 0.4866, 0.4409]).view(1, 3, 1, 1)
STD = torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1)


def prep(x, res, mean, std):
    x = (x - mean) / std
    if res != x.shape[-1]:
        x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
    return x.contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def infer_probs(model, xu8, res, mean, std, bs=500, tta=True):
    model.eval()
    out = []
    for s in range(0, xu8.shape[0], bs):
        xb = xu8[s:s + bs].float().div_(255.)
        views = [xb, torch.flip(xb, dims=[3])] if tta else [xb]
        p = 0
        for v in views:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = model(prep(v, res, mean, std))
            p = p + F.softmax(lo.float(), dim=1)
        out.append(p / len(views))
    model.train()
    return torch.cat(out)


def bal_acc(pred, y, n=N_CLASSES):
    pred = np.asarray(pred)
    y = np.asarray(y)
    rec = []
    for c in range(n):
        m = y == c
        if m.sum():
            rec.append((pred[m] == c).mean())
    return float(np.mean(rec))


# --------------------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="i20k")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graded", default=f"{DATA}/graded_images.npz")
    ap.add_argument("--out", default="")
    ap.add_argument("--steps", type=int, default=BUDGET_STEPS)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depths", default="3,4,6,3")
    ap.add_argument("--res", type=int, default=48)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--wd", type=float, default=2e-3)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--warm", type=int, default=250)
    ap.add_argument("--pad", type=int, default=4)
    ap.add_argument("--cut", type=int, default=12)
    ap.add_argument("--ta", type=int, default=0, help="0=off, else number of TA ops (13)")
    ap.add_argument("--ta_p", type=float, default=1.0)
    ap.add_argument("--ta_n", type=int, default=1)
    ap.add_argument("--rot", type=float, default=0.35)
    ap.add_argument("--shear", type=float, default=0.25)
    ap.add_argument("--trans", type=float, default=0.15)
    ap.add_argument("--cj", type=float, default=0.7)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--ema", default="0.999,0.9995")
    ap.add_argument("--balanced", type=int, default=1)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--cutmix", type=float, default=0.0)
    ap.add_argument("--mix_p", type=float, default=0.5)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--save_probs", default="")
    ap.add_argument("--save_ckpt", default="")
    a = ap.parse_args()

    t0 = time.time()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed * 7919 + 13)

    d = np.load(f"{DATA}/train_i20k.npz")
    xtr = torch.from_numpy(d["x"]).to(dev).permute(0, 3, 1, 2).contiguous()
    ytr = torch.from_numpy(d["y"]).to(dev)
    dv = np.load(f"{DATA}/val_public.npz")
    xva = torch.from_numpy(dv["x"]).to(dev).permute(0, 3, 1, 2).contiguous()
    yva = dv["y"]
    xg = torch.from_numpy(np.load(a.graded)["x"]).to(dev).permute(0, 3, 1, 2).contiguous()

    mean = MEAN.to(dev)
    std = STD.to(dev)

    # class-balanced sampling table
    ynp = d["y"]
    per = [np.where(ynp == c)[0] for c in range(N_CLASSES)]
    mx = max(len(p) for p in per)
    tab = np.zeros((N_CLASSES, mx), dtype=np.int64)
    cnt = np.zeros(N_CLASSES, dtype=np.int64)
    for c, p in enumerate(per):
        tab[c, :len(p)] = p
        tab[c, len(p):] = np.random.choice(p, mx - len(p))
        cnt[c] = len(p)
    tab_t = torch.from_numpy(tab).to(dev)
    cnt_t = torch.from_numpy(cnt).to(dev)

    depths = tuple(int(v) for v in a.depths.split(","))
    model = Net(a.width, depths, drop=a.drop).to(dev).to(memory_format=torch.channels_last)
    nparam = sum(p.numel() for p in model.parameters())

    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.SGD([{"params": decay, "weight_decay": a.wd},
                           {"params": no_decay, "weight_decay": 0.0}],
                          lr=a.lr, momentum=0.9, nesterov=True)

    fkeys = [k for k, v in model.state_dict().items() if v.is_floating_point()]
    msd = [model.state_dict()[k] for k in fkeys]
    emas = []
    for dcy in [float(v) for v in a.ema.split(",") if v]:
        m = Net(a.width, depths, drop=a.drop).to(dev).to(memory_format=torch.channels_last)
        m.load_state_dict(model.state_dict())
        for p in m.parameters():
            p.requires_grad_(False)
        esd = m.state_dict()
        emas.append([dcy, m, [esd[k] for k in fkeys]])

    cfg = dict(pad=a.pad, nops=(13 if a.ta else 0), ta_p=a.ta_p, ta_n=a.ta_n, rot=a.rot,
               shear=a.shear, trans=a.trans, cj=a.cj)

    steps = a.steps
    present = 0
    assert steps <= BUDGET_STEPS
    assert steps * a.batch <= BUDGET_PRESENT, "presentation budget"

    model.train()
    for it in range(steps):
        # lr schedule: linear warmup then cosine
        if it < a.warm:
            lr = a.lr * (it + 1) / a.warm
        else:
            t = (it - a.warm) / max(1, steps - a.warm)
            lr = a.lr * 0.5 * (1 + math.cos(math.pi * t))
        for g in opt.param_groups:
            g["lr"] = lr

        if a.balanced:
            cl = torch.randint(0, N_CLASSES, (a.batch,), device=dev, generator=gen)
            off = (torch.rand(a.batch, device=dev, generator=gen) * cnt_t[cl].float()).long()
            i = tab_t[cl, off]
        else:
            i = torch.randint(0, xtr.shape[0], (a.batch,), device=dev, generator=gen)
        xb = xtr[i].float().div_(255.)
        yb = ytr[i]
        present += a.batch

        xb = augment(xb, cfg, gen)
        xb = (xb - mean) / std
        xb = cutout_(xb, a.cut, gen)
        if a.res != xb.shape[-1]:
            xb = F.interpolate(xb, size=(a.res, a.res), mode="bilinear", align_corners=False)
        xb = xb.contiguous(memory_format=torch.channels_last)

        lam, perm, mode = 1.0, None, None
        if (a.mixup > 0 or a.cutmix > 0) and torch.rand(1, generator=gen, device=dev).item() < a.mix_p:
            mode = "cutmix" if (a.cutmix > 0 and (a.mixup == 0 or np.random.rand() < 0.5)) else "mixup"
            alpha = a.cutmix if mode == "cutmix" else a.mixup
            lam = float(np.random.beta(alpha, alpha))
            perm = torch.randperm(a.batch, device=dev, generator=gen)
            if mode == "mixup":
                xb = lam * xb + (1 - lam) * xb[perm]
            else:
                R = a.res
                rr = int(R * math.sqrt(1 - lam))
                if rr > 0:
                    cy0 = np.random.randint(R)
                    cx0 = np.random.randint(R)
                    y1, y2 = max(0, cy0 - rr // 2), min(R, cy0 + rr // 2)
                    x1, x2 = max(0, cx0 - rr // 2), min(R, cx0 + rr // 2)
                    xb[:, :, y1:y2, x1:x2] = xb[perm][:, :, y1:y2, x1:x2]
                    lam = 1 - ((y2 - y1) * (x2 - x1) / (R * R))
                else:
                    perm = None

        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(xb)
            loss = F.cross_entropy(logit, yb, label_smoothing=a.ls)
            if perm is not None:
                loss = lam * loss + (1 - lam) * F.cross_entropy(logit, yb[perm], label_smoothing=a.ls)
        loss.backward()
        opt.step()

        with torch.no_grad():
            for dcy, m, esd in emas:
                dd = min(dcy, (it + 1) / (it + 10.0))
                torch._foreach_mul_(esd, dd)
                torch._foreach_add_(esd, msd, alpha=1 - dd)

    ttrain = time.time() - t0
    # ------------------------------------------------------------------ evaluation
    res = {}
    pv = {}
    pg = {}
    pv["raw"] = infer_probs(model, xva, a.res, mean, std)
    pg["raw"] = infer_probs(model, xg, a.res, mean, std)
    for k, (dcy, m, _) in enumerate(emas):
        pv[f"ema{k}"] = infer_probs(m, xva, a.res, mean, std)
        pg[f"ema{k}"] = infer_probs(m, xg, a.res, mean, std)
    for k in pv:
        res[k] = bal_acc(pv[k].argmax(1).cpu().numpy(), yva)
    ens_v = sum(pv.values()) / len(pv)
    ens_g = sum(pg.values()) / len(pg)
    res["ens"] = bal_acc(ens_v.argmax(1).cpu().numpy(), yva)

    best_key = max(res, key=res.get)
    print(json.dumps(dict(tag=a.tag, seed=a.seed, params=nparam / 1e6, steps=steps,
                          present=present, t_train=round(ttrain, 1),
                          t_total=round(time.time() - t0, 1),
                          acc={k: round(v, 4) for k, v in res.items()}, best=best_key)))

    if a.save_ckpt:
        torch.save(dict(args=vars(a), raw=model.state_dict(),
                        emas=[m.state_dict() for _, m, _ in emas]), a.save_ckpt)
    if a.save_probs:
        np.savez(a.save_probs, **{f"v_{k}": v.cpu().numpy().astype(np.float16) for k, v in pv.items()},
                 **{f"g_{k}": v.cpu().numpy().astype(np.float16) for k, v in pg.items()}, yva=yva)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        np.save(a.out, ens_g.argmax(1).cpu().numpy().astype(np.int64))
        print(f"wrote {a.out}  total {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
