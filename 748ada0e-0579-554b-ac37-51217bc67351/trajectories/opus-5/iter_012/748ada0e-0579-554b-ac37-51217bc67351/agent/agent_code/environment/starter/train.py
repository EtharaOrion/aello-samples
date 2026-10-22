"""CIFAR-100 @ 8000 optimizer steps / 1,024,000 sample presentations.

ResNet-34-style net (basic blocks, [3,4,6,3]) at base width W, anti-aliased stage
transitions (avg-pool both paths, then stride-1 convs), inputs upsampled 32->RES.
All data lives on the GPU as uint8; augmentation is done on-device.

Run: python3 train.py --instance i20k --seed 1 \
        --graded ../data/graded_images.npz --out /submission/preds_i20k.npy
"""
import argparse, os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aello_data as data

BUDGET_STEPS = 8000
BUDGET_PRESENTATIONS = 1_024_000


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.pool = nn.AvgPool2d(2) if stride == 2 else None
        self.c1 = nn.Conv2d(cin, cout, 3, 1, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        nn.init.zeros_(self.b2.weight)
        if cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(cout))
        else:
            self.sc = None

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        h = F.relu(self.b1(self.c1(x)), inplace=True)
        h = self.b2(self.c2(h))
        s = x if self.sc is None else self.sc(x)
        return F.relu(h + s, inplace=True)


class Net(nn.Module):
    def __init__(self, n_classes, width=160, depths=(3, 4, 6, 3), stem=3):
        super().__init__()
        w = [width, width * 2, width * 4, width * 8]
        layers = [nn.Conv2d(3, width, stem, 1, stem // 2, bias=False),
                  nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
        cin = width
        for si, (d, cw) in enumerate(zip(depths, w)):
            for bi in range(d):
                layers.append(Block(cin, cw, stride=(2 if (bi == 0 and si > 0) else 1)))
                cin = cw
        self.body = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, n_classes)

    def forward(self, x):
        h = self.body(x)
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.fc(h)


def make_model(seed, n_classes, width, depths, stem, device):
    torch.manual_seed(seed)
    m = Net(n_classes, width, depths, stem).to(device).to(memory_format=torch.channels_last)
    return m


# ---------------------------------------------------------------- gpu augmentation
class Aug:
    """All-on-GPU: random crop (reflect pad), hflip, cutout, then upsample to RES."""

    def __init__(self, res, pad, cut, device, rrc=0.0, ar=0.0):
        self.res, self.pad, self.cut, self.device = res, pad, cut, device
        self.rrc, self.ar = rrc, ar

    def __call__(self, xb, gen, cut_scale=1.0):
        if self.rrc > 0:
            return self.warp(xb, gen, cut_scale)
        b, c, h, w = xb.shape
        p = self.pad
        if p > 0:
            xp = F.pad(xb, (p,) * 4, mode="reflect")
            oy = torch.randint(0, 2 * p + 1, (b,), device=self.device, generator=gen)
            ox = torch.randint(0, 2 * p + 1, (b,), device=self.device, generator=gen)
            ar = torch.arange(b, device=self.device)
            ch = torch.arange(c, device=self.device)
            rows = (oy[:, None] + torch.arange(h, device=self.device)[None, :])[:, None, :, None]
            cols = (ox[:, None] + torch.arange(w, device=self.device)[None, :])[:, None, None, :]
            xb = xp[ar[:, None, None, None], ch[None, :, None, None], rows, cols]
        flip = torch.rand(b, 1, 1, 1, device=self.device, generator=gen) < 0.5
        xb = torch.where(flip, xb.flip(3), xb)
        cs = int(round(self.cut * cut_scale))
        if cs > 0:
            cy = torch.randint(0, h, (b, 1, 1), device=self.device, generator=gen)
            cx = torch.randint(0, w, (b, 1, 1), device=self.device, generator=gen)
            yy = torch.arange(h, device=self.device).view(1, h, 1)
            xx = torch.arange(w, device=self.device).view(1, 1, w)
            m = ((yy - cy).abs() < cs / 2) & ((xx - cx).abs() < cs / 2)
            xb = xb * (~m).unsqueeze(1).to(xb.dtype)
        if self.res != h:
            xb = F.interpolate(xb, size=(self.res, self.res), mode="bilinear",
                              align_corners=False)
        return xb.contiguous(memory_format=torch.channels_last)


    def warp(self, xb, gen, cut_scale=1.0):
        """Random-resized-crop + aspect jitter + hflip in one grid_sample, straight to RES.

        Scale jitter is free here because the pipeline already resamples 32 -> RES.
        """
        b = xb.shape[0]
        dev, res = self.device, self.res
        u = torch.rand(b, 4, device=dev, generator=gen)
        s = self.rrc + (1.0 - self.rrc) * u[:, 0]           # side as a fraction of the image
        if self.ar > 0:
            lr = (u[:, 3] * 2 - 1) * np.log(1 + self.ar)
            sx, sy = s * torch.exp(lr * 0.5), s * torch.exp(-lr * 0.5)
        else:
            sx, sy = s, s
        sx, sy = sx.clamp(max=1.0), sy.clamp(max=1.0)
        tx = (u[:, 1] * 2 - 1) * (1 - sx)
        ty = (u[:, 2] * 2 - 1) * (1 - sy)
        flip = torch.where(torch.rand(b, device=dev, generator=gen) < 0.5, -1.0, 1.0)
        th = torch.zeros(b, 2, 3, device=dev)
        th[:, 0, 0] = sx * flip
        th[:, 0, 2] = tx
        th[:, 1, 1] = sy
        th[:, 1, 2] = ty
        grid = F.affine_grid(th, (b, xb.shape[1], res, res), align_corners=False)
        out = F.grid_sample(xb, grid, mode="bilinear", padding_mode="reflection",
                            align_corners=False)
        cs = int(round(self.cut * cut_scale * res / 32.0))
        if cs > 0:
            cy = torch.randint(0, res, (b, 1, 1), device=dev, generator=gen)
            cx = torch.randint(0, res, (b, 1, 1), device=dev, generator=gen)
            yy = torch.arange(res, device=dev).view(1, res, 1)
            xx = torch.arange(res, device=dev).view(1, 1, res)
            m = ((yy - cy).abs() < cs / 2) & ((xx - cx).abs() < cs / 2)
            out = out * (~m).unsqueeze(1).to(out.dtype)
        return out.contiguous(memory_format=torch.channels_last)


def normalize(x, mean, std):
    return (x - mean) / std


# ---------------------------------------------------------------- inference
@torch.no_grad()
def logits_tta(model, x_u8, res, mean, std, device, views="flip", batch=500):
    """x_u8: uint8 NCHW cuda tensor at 32x32. Returns averaged log-softmax [N,100]."""
    model.eval()
    offs = [(0, 0)]
    if views == "flip5":
        offs = [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)]
    elif views == "flip9":
        offs = [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2),
                (2, 2), (2, -2), (-2, 2), (-2, -2)]
    acc = None
    n = x_u8.shape[0]
    for s in range(0, n, batch):
        xb0 = x_u8[s:s + batch].float().div_(255.)
        tot = None
        for (dy, dx) in offs:
            if dy or dx:
                xs = torch.roll(F.pad(xb0, (4, 4, 4, 4), mode="reflect"),
                                shifts=(dy, dx), dims=(2, 3))[:, :, 4:-4, 4:-4]
            else:
                xs = xb0
            for f in (False, True):
                xf = xs.flip(3) if f else xs
                if res != xf.shape[-1]:
                    xf = F.interpolate(xf, size=(res, res), mode="bilinear",
                                       align_corners=False)
                xf = normalize(xf, mean, std).contiguous(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = model(xf)
                o = F.log_softmax(o.float(), dim=1)
                tot = o if tot is None else tot + o
        tot = tot / (len(offs) * 2)
        acc = tot if acc is None else torch.cat([acc, tot], 0)
    model.train()
    return acc


def bal_acc(pred, y, k=100):
    rec = []
    for c in range(k):
        m = y == c
        if m.sum():
            rec.append((pred[m] == c).mean())
    return float(np.mean(rec))


def sinkhorn_balance(logp, iters=12, temp=1.0, k=100):
    """Nudge column marginals of the prediction matrix toward uniform."""
    p = torch.softmax(torch.as_tensor(logp) / temp, dim=1).double()
    n = p.shape[0]
    for _ in range(iters):
        col = p.sum(0, keepdim=True)
        p = p / col.clamp_min(1e-12) * (n / k)
        p = p / p.sum(1, keepdim=True).clamp_min(1e-12)
    return p.numpy()


# ---------------------------------------------------------------- training
def train(a, device="cuda"):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    x, y = data.load_train(a.instance)
    xv, yv = data.load_val()
    xg = np.load(a.graded)["x"] if a.graded else None

    X = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).contiguous().to(device)
    Y = torch.from_numpy(y).to(device)
    XV = torch.from_numpy(np.ascontiguousarray(xv)).permute(0, 3, 1, 2).contiguous().to(device)
    XG = (torch.from_numpy(np.ascontiguousarray(xg)).permute(0, 3, 1, 2).contiguous().to(device)
          if xg is not None else None)

    mean = (X.float() / 255.).mean((0, 2, 3)).view(1, 3, 1, 1)
    std = (X.float() / 255.).std((0, 2, 3)).view(1, 3, 1, 1)

    depths = tuple(int(v) for v in a.depths.split(","))
    model = make_model(a.seed, data.N_CLASSES, a.width, depths, a.stem, device)
    nparam = sum(p.numel() for p in model.parameters())

    decay, no_decay = [], []
    for n_, p_ in model.named_parameters():
        (no_decay if p_.ndim <= 1 else decay).append(p_)
    opt = torch.optim.SGD([{"params": decay, "weight_decay": a.wd},
                           {"params": no_decay, "weight_decay": 0.0}],
                          lr=a.lr, momentum=0.9, nesterov=True)

    steps = min(a.steps, BUDGET_STEPS, BUDGET_PRESENTATIONS // a.batch)
    warm = max(1, int(a.warm * steps))

    def lr_at(t):
        if t < warm:
            return a.lr * (t + 1) / warm
        q = (t - warm) / max(1, steps - warm)
        return a.lr * (a.lr_min + (1 - a.lr_min) * 0.5 * (1 + np.cos(np.pi * q)))

    aug = Aug(a.res, a.pad, a.cut, device, a.rrc, a.ar)
    gen = torch.Generator(device=device); gen.manual_seed(a.seed + 1234)
    rng = np.random.default_rng(a.seed)

    # EMA over float params/buffers via _foreach for speed; int buffers copied at the end.
    sd0 = model.state_dict()
    fkeys = [k for k, v in sd0.items() if v.dtype.is_floating_point]
    ikeys = [k for k in sd0 if k not in fkeys]
    src = [sd0[k] for k in fkeys]
    emas = []
    for d in [float(v) for v in a.ema.split(",") if v]:
        emas.append([d, [v.detach().clone().float() for v in src]])

    presented = 0
    t0 = time.time()
    perm = torch.from_numpy(rng.permutation(len(y))).to(device)
    ptr = 0
    model.train()
    for t in range(steps):
        if ptr + a.batch > len(y):
            perm = torch.from_numpy(rng.permutation(len(y))).to(device)
            ptr = 0
        idx = perm[ptr:ptr + a.batch]; ptr += a.batch

        frac = t / steps
        cut_scale = 1.0
        if a.aug_off > 0 and frac > 1 - a.aug_off:
            cut_scale = 0.0
        xb = X[idx].float().div_(255.)
        xb = aug(xb, gen, cut_scale=cut_scale)
        xb = normalize(xb, mean, std).contiguous(memory_format=torch.channels_last)
        yb = Y[idx]

        lam = None
        if a.mixup > 0 and cut_scale > 0 and rng.random() < a.mix_prob:
            lam = float(np.random.default_rng(rng.integers(1 << 30)).beta(a.mixup, a.mixup))
            lam = max(lam, 1 - lam)
            ridx = torch.randperm(xb.shape[0], device=device, generator=gen)
            # cutmix box
            rh = int(round(a.res * np.sqrt(1 - lam)))
            if rh > 0:
                cy = int(torch.randint(0, a.res, (1,), generator=gen, device=device).item())
                cx = int(torch.randint(0, a.res, (1,), generator=gen, device=device).item())
                y0, y1 = max(0, cy - rh // 2), min(a.res, cy + rh // 2)
                x0, x1 = max(0, cx - rh // 2), min(a.res, cx + rh // 2)
                xb[:, :, y0:y1, x0:x1] = xb[ridx][:, :, y0:y1, x0:x1]
                lam = 1 - ((y1 - y0) * (x1 - x0)) / (a.res * a.res)
                yb2 = yb[ridx]
            else:
                lam = None

        for g in opt.param_groups:
            g["lr"] = lr_at(t)
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
        presented += a.batch

        if emas:
            with torch.no_grad():
                for d, e in emas:
                    dd = d if t > 200 else 0.9
                    torch._foreach_mul_(e, dd)
                    torch._foreach_add_(e, src, alpha=1 - dd)

        if t % 1000 == 0:
            print("step %5d  lr %.4f  loss %.3f  %.1fs" % (t, lr_at(t), loss.item(),
                                                           time.time() - t0), flush=True)

    train_time = time.time() - t0
    print("TRAIN done: %d/%d steps, %d/%d presentations, %.1fs, params %.1fM"
          % (steps, BUDGET_STEPS, presented, BUDGET_PRESENTATIONS, train_time, nparam / 1e6),
          flush=True)

    # -------- collect logits for raw + each EMA
    named = [("raw", model)]
    for d, e in emas:
        m2 = make_model(a.seed, data.N_CLASSES, a.width, depths, a.stem, device)
        sd = dict(zip(fkeys, e))
        sd.update({k: sd0[k] for k in ikeys})
        m2.load_state_dict(sd)
        named.append(("ema%g" % d, m2))

    if a.ckpt:
        torch.save({"cfg": vars(a), "mean": mean.cpu(), "std": std.cpu(),
                    "models": {nm: {k: v.cpu() for k, v in m.state_dict().items()}
                               for nm, m in named}}, a.ckpt)
        print("wrote ckpt %s" % a.ckpt, flush=True)

    out = {"steps": steps, "presented": presented, "train_time": train_time,
           "params": nparam, "res": a.res, "tag": a.tag}

    # Stage 1: cheap single-view val pass to pick the snapshot combo.
    cheap = {}
    for nm, m in named:
        lv = logits_tta(m, XV, a.res, mean, std, device, views="flip1").cpu().numpy()
        cheap[nm] = lv
        print("  %-10s val(flip1) bal-acc %.4f" % (nm, bal_acc(lv.argmax(1), yv)), flush=True)
    keys = [nm for nm, _ in named]
    combos = {k: [k] for k in keys}
    for k in keys[1:]:
        combos["raw+" + k] = ["raw", k]
    if len(keys) > 2:
        combos["all"] = keys
    best, best_s = None, -1
    for cn, ks in sorted(combos.items()):
        s = bal_acc((sum(cheap[k] for k in ks) / len(ks)).argmax(1), yv)
        print("  combo %-22s val(flip1) %.4f" % (cn, s), flush=True)
        if s > best_s:
            best_s, best = s, (cn, ks)
    print("picked combo %s (val flip1 %.4f)" % (best[0], best_s), flush=True)

    # Stage 2: full multi-crop+flip TTA for the picked combo only.
    store = {}
    for nm, m in named:
        if nm not in best[1]:
            continue
        store["val_" + nm] = logits_tta(m, XV, a.res, mean, std, device,
                                        views=a.tta).cpu().numpy()
        if XG is not None:
            store["graded_" + nm] = logits_tta(m, XG, a.res, mean, std, device,
                                               views=a.tta).cpu().numpy()
    lv = sum(store["val_" + k] for k in best[1]) / len(best[1])
    val_tta = bal_acc(lv.argmax(1), yv)
    sk = a.sk_temp if a.sk_temp > 0 else 0.7
    val_tta_sk = bal_acc(sinkhorn_balance(lv, temp=sk).argmax(1), yv)
    print("FINAL %s + %s TTA: val bal-acc %.4f  (sinkhorn@%.2f %.4f, unused unless --sk-temp>0)"
          " [%.1fs]" % (best[0], a.tta, val_tta, sk, val_tta_sk, time.time() - t0), flush=True)
    out["best_combo"] = best[0]
    out["best_val"] = val_tta

    if a.logits:
        np.savez_compressed(a.logits, yv=yv, **store,
                            meta=json.dumps({k: v for k, v in out.items()}))
        print("wrote logits %s" % a.logits, flush=True)

    if a.out and XG is not None:
        lg = sum(store["graded_" + k] for k in best[1]) / len(best[1])
        pg = (sinkhorn_balance(lg, temp=a.sk_temp).argmax(1) if a.sk_temp > 0
              else lg.argmax(1))
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        np.save(a.out, pg.astype(np.int64))
        print("wrote %s  (%d distinct classes)" % (a.out, len(np.unique(pg))), flush=True)
    print("WALL train+predict %.1fs" % (time.time() - t0), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="i20k", choices=list(data.INSTANCES))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--graded", default="../data/graded_images.npz")
    ap.add_argument("--out", default="")
    ap.add_argument("--logits", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--lr-min", type=float, default=0.0)
    ap.add_argument("--wd", type=float, default=2e-3)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--warm", type=float, default=0.03)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--depths", default="3,4,6,3")
    ap.add_argument("--stem", type=int, default=3)
    ap.add_argument("--res", type=int, default=48)
    ap.add_argument("--pad", type=int, default=4)
    ap.add_argument("--cut", type=int, default=12)
    ap.add_argument("--rrc", type=float, default=0.0)
    ap.add_argument("--ar", type=float, default=0.0)
    ap.add_argument("--aug-off", type=float, default=0.0)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--mix-prob", type=float, default=0.5)
    ap.add_argument("--ema", default="0.999,0.998")
    ap.add_argument("--tta", default="flip5")
    ap.add_argument("--sk-temp", type=float, default=0.0)
    a = ap.parse_args()
    train(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
