"""AELLO-C8-S1 pipeline (CIFAR-100 at a fixed step budget).

Budget accounting (honest, hard-capped in code):
  * At most BUDGET_STEPS optimiser updates.
  * At most BUDGET_PRESENT training-sample presentations.  Every step draws exactly
    `batch` images from the training set (sampled without replacement inside an epoch),
    and each drawn image is forwarded exactly once, so a step costs `batch`
    presentations.  The loop stops as soon as either cap would be exceeded.
    Mixing augmentations (mixup / cutmix), when enabled, mix a batch with a *permutation
    of itself* -- no additional images are read, the batch is still `batch` images wide.

Run:  python3 train.py --instance i20k --seed 0 \
          --graded ../data/graded_images.npz --out /submission/preds_i20k.npy
"""
import argparse, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aello_data as data

BUDGET_STEPS = 8000            # graded budget: optimiser updates
BUDGET_PRESENT = 1024000       # graded budget: training-sample presentations


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        nn.init.zeros_(self.b2.weight)          # identity-init residual branch
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                       nn.BatchNorm2d(cout))

    def forward(self, x):
        r = x if self.short is None else self.short(x)
        x = F.relu(self.b1(self.c1(x)), inplace=True)
        x = self.b2(self.c2(x))
        return F.relu(x + r, inplace=True)


class ResNet(nn.Module):
    """CIFAR ResNet: 3x3 stem (no downsample), 3 or 4 stages of BasicBlocks."""

    def __init__(self, n_classes, width=64, blocks=(2, 2, 2, 2), stem=None, dropout=0.0):
        super().__init__()
        stem = stem or width
        self.stem = nn.Sequential(nn.Conv2d(3, stem, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(stem), nn.ReLU(inplace=True))
        layers, cin = [], stem
        for i, n in enumerate(blocks):
            cout = width * (2 ** i)
            for j in range(n):
                layers.append(BasicBlock(cin, cout, stride=(2 if (j == 0 and i > 0) else 1)))
                cin = cout
        self.body = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(cin, n_classes)

    def forward(self, x):
        x = self.body(self.stem(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(self.drop(x))


def make_model(seed, n_classes, arch="r18", width=64, dropout=0.0):
    torch.manual_seed(seed)
    cfg = {
        "r18":  dict(width=width, blocks=(2, 2, 2, 2)),
        "r18t": dict(width=width, blocks=(2, 2, 2), stem=width),      # 3 stages
        "r26":  dict(width=width, blocks=(3, 3, 3, 3)),
        "r34":  dict(width=width, blocks=(3, 4, 6, 3)),
        "r10":  dict(width=width, blocks=(1, 1, 1, 1)),
    }[arch]
    return ResNet(n_classes, dropout=dropout, **cfg)


# --------------------------------------------------------------------------------------
# GPU-resident data + augmentation
# --------------------------------------------------------------------------------------
class GPUData:
    def __init__(self, x, y, device, pad=4):
        self.n = len(y)
        self.pad = pad
        t = torch.from_numpy(np.ascontiguousarray(x)).to(device)         # N,H,W,C uint8
        t = t.permute(0, 3, 1, 2).float().div_(255.)
        self.mean = t.mean(dim=(0, 2, 3), keepdim=True)
        self.std = t.std(dim=(0, 2, 3), keepdim=True)
        t = (t - self.mean) / self.std
        self.xp = F.pad(t, (pad,) * 4, mode="reflect").contiguous()      # N,C,H+2p,W+2p
        self.y = torch.from_numpy(y).to(device)
        self.h = x.shape[1]
        self.device = device

    def norm(self, x_uint8_nhwc):
        t = torch.from_numpy(np.ascontiguousarray(x_uint8_nhwc)).to(self.device)
        t = t.permute(0, 3, 1, 2).float().div_(255.)
        return ((t - self.mean) / self.std).contiguous(memory_format=torch.channels_last)

    def batch(self, idx, gen, cutout=0):
        """Random crop + hflip (+ cutout), all per-sample, on GPU."""
        b = idx.numel()
        h, p, dev = self.h, self.pad, self.device
        oy = torch.randint(0, 2 * p + 1, (b,), device=dev, generator=gen)
        ox = torch.randint(0, 2 * p + 1, (b,), device=dev, generator=gen)
        ar = torch.arange(h, device=dev)
        rows = (oy[:, None] + ar[None, :])[:, None, :, None]
        cols = (ox[:, None] + ar[None, :])[:, None, None, :]
        sel = self.xp[idx]
        ch = torch.arange(3, device=dev)[None, :, None, None]
        bi = torch.arange(b, device=dev)[:, None, None, None]
        xb = sel[bi, ch, rows, cols]
        flip = torch.rand(b, device=dev, generator=gen) < 0.5
        xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
        if cutout > 0:
            cy = torch.randint(0, h, (b,), device=dev, generator=gen)
            cx = torch.randint(0, h, (b,), device=dev, generator=gen)
            gy = ar[None, :]
            my = (gy >= (cy[:, None] - cutout // 2)) & (gy < (cy[:, None] + cutout - cutout // 2))
            mx = (gy >= (cx[:, None] - cutout // 2)) & (gy < (cx[:, None] + cutout - cutout // 2))
            mask = ~(my[:, :, None] & mx[:, None, :])
            xb = xb * mask[:, None, :, :].to(xb.dtype)
        return xb.contiguous(memory_format=torch.channels_last), self.y[idx]


def epoch_index_stream(n, batch, gen, device):
    """Yield index tensors of size `batch`, reshuffling each epoch (sampling w/o
    replacement inside an epoch -> even coverage of the training set)."""
    perm = torch.randperm(n, device=device, generator=gen)
    pos = 0
    while True:
        if pos + batch > n:
            perm = torch.randperm(n, device=device, generator=gen)
            pos = 0
        yield perm[pos:pos + batch]
        pos += batch


# --------------------------------------------------------------------------------------
# train / predict
# --------------------------------------------------------------------------------------
CROPS5 = [(4, 4), (2, 2), (2, 6), (6, 2), (6, 6)]


@torch.no_grad()
def logits_for(model, xb_norm, tta="flip", bs=1000, pad=4):
    """tta: 'none' | 'flip' | 'flip5' (5 crops x {id, hflip} = 10 views)."""
    model.eval()
    out = []
    h = xb_norm.shape[-1]
    for s in range(0, xb_norm.shape[0], bs):
        xb = xb_norm[s:s + bs]
        views = [xb]
        if tta == "flip5":
            xp = F.pad(xb, (pad,) * 4, mode="reflect")
            views = [xp[:, :, oy:oy + h, ox:ox + h].contiguous(
                memory_format=torch.channels_last) for (oy, ox) in CROPS5]
        acc = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for v in views:
                lg = F.log_softmax(model(v).float(), 1)
                if tta != "none":
                    lg = lg + F.log_softmax(model(v.flip(3)).float(), 1)
                acc = lg if acc is None else acc + lg
        out.append(acc.float())
    model.train()
    return torch.cat(out)


def balanced_acc(pred, y, n_classes=100):
    rec = []
    for c in range(n_classes):
        m = y == c
        if m.sum() > 0:
            rec.append((pred[m] == c).mean())
    return float(np.mean(rec))


def train(args, device="cuda"):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    x, y = data.load_train(args.instance)
    ds = GPUData(x, y, device, pad=args.pad)

    model = make_model(args.seed, data.N_CLASSES, args.arch, args.width,
                       args.dropout).to(device).to(memory_format=torch.channels_last)
    nparam = sum(p.numel() for p in model.parameters())

    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.SGD([{"params": decay, "weight_decay": args.wd},
                           {"params": no_decay, "weight_decay": 0.0}],
                          lr=args.lr, momentum=0.9, nesterov=True)

    # honest budget cap
    steps = min(args.steps, BUDGET_STEPS, BUDGET_PRESENT // args.batch)
    warm = max(1, int(args.warmup * steps))

    def lr_at(t):
        if t < warm:
            return args.lr * (t + 1) / warm
        q = (t - warm) / max(1, steps - warm)
        return args.final_lr_frac * args.lr + (1 - args.final_lr_frac) * args.lr * \
            0.5 * (1 + math.cos(math.pi * q))

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 1234)
    torch.manual_seed(args.seed)
    stream = epoch_index_stream(ds.n, args.batch, gen, device)

    emas = []
    for d in args.emas:
        emas.append((d, torch.optim.swa_utils.AveragedModel(
            model, avg_fn=(lambda dd: (lambda a, c, n: dd * a + (1 - dd) * c))(d))))

    presented = 0
    t0 = time.time()
    for t in range(steps):
        if presented + args.batch > BUDGET_PRESENT:
            steps = t
            break
        idx = next(stream)
        xb, yb = ds.batch(idx, gen, cutout=args.cutout)
        presented += args.batch

        lam, yb2 = 1.0, yb
        if args.mix_p > 0 and torch.rand(1, generator=gen, device=device).item() < args.mix_p:
            perm = torch.randperm(xb.shape[0], device=device, generator=gen)
            yb2 = yb[perm]
            if args.mix_mode == "cutmix":
                lam = float(np.random.default_rng(args.seed * 100003 + t).beta(args.mix_a, args.mix_a))
                h = ds.h
                rh = int(h * math.sqrt(1 - lam))
                if rh > 0:
                    cy = int(torch.randint(0, h, (1,), device=device, generator=gen).item())
                    cx = int(torch.randint(0, h, (1,), device=device, generator=gen).item())
                    y1, y2 = max(0, cy - rh // 2), min(h, cy + rh // 2)
                    x1, x2 = max(0, cx - rh // 2), min(h, cx + rh // 2)
                    xb = xb.clone()
                    xb[:, :, y1:y2, x1:x2] = xb[perm][:, :, y1:y2, x1:x2]
                    lam = 1 - ((y2 - y1) * (x2 - x1) / (h * h))
            else:
                lam = float(np.random.default_rng(args.seed * 100003 + t).beta(args.mix_a, args.mix_a))
                xb = lam * xb + (1 - lam) * xb[perm]

        for g in opt.param_groups:
            g["lr"] = lr_at(t)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device, dtype=torch.bfloat16):
            out = model(xb)
            loss = F.cross_entropy(out, yb, label_smoothing=args.ls)
            if lam < 1.0:
                loss = lam * loss + (1 - lam) * F.cross_entropy(out, yb2,
                                                                label_smoothing=args.ls)
        loss.backward()
        opt.step()
        if emas and t >= int(steps * args.ema_start):
            for _, e in emas:
                e.update_parameters(model)

    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"[train] arch={args.arch} w={args.width} params={nparam/1e6:.2f}M steps={steps} "
          f"present={presented} time={dt:.1f}s", flush=True)

    # BN stats for the EMA weights: copy running stats from the live model
    outs = []
    for d, e in emas:
        em = e.module
        for (mb, mm) in zip(em.modules(), model.modules()):
            if isinstance(mb, nn.BatchNorm2d):
                mb.running_mean.copy_(mm.running_mean)
                mb.running_var.copy_(mm.running_var)
                mb.num_batches_tracked.copy_(mm.num_batches_tracked)
        outs.append(("ema%g" % d, em))
    return model, outs, ds, presented, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="i20k", choices=list(data.INSTANCES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graded", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--logits-out", default=None)
    ap.add_argument("--arch", default="r18")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=BUDGET_STEPS)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--warmup", type=float, default=0.04)
    ap.add_argument("--final-lr-frac", type=float, default=0.0)
    ap.add_argument("--pad", type=int, default=4)
    ap.add_argument("--cutout", type=int, default=8)
    ap.add_argument("--mix-p", type=float, default=0.0)
    ap.add_argument("--mix-a", type=float, default=1.0)
    ap.add_argument("--mix-mode", default="cutmix", choices=["cutmix", "mixup"])
    ap.add_argument("--emas", type=float, nargs="*", default=[])
    ap.add_argument("--ema-start", type=float, default=0.5)
    ap.add_argument("--tta", default="flip", choices=["none", "flip", "flip5"])
    ap.add_argument("--pick", default=None, help="force a variant name instead of val argmax")
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()

    t_all = time.time()
    model, emas, ds, presented, steps = train(a)

    xv, yv = data.load_val()
    xvn = ds.norm(xv)
    variants = [("raw", model)] + emas
    res, cache = {}, {}
    for name, m in variants:
        for tta in (["flip", "flip5"] if a.tta == "flip5" else [a.tta]):
            key = f"{name}/{tta}"
            lg = logits_for(m, xvn, tta=tta, pad=a.pad)
            cache[key] = (m, tta)
            res[key] = balanced_acc(lg.argmax(1).cpu().numpy(), yv)
    pick = a.pick if (a.pick in res) else max(res, key=res.get)
    best_model, best_tta = cache[pick]
    print(f"[val] tag={a.tag} " + " ".join(f"{k}={v:.4f}" for k, v in sorted(res.items())) +
          f" pick={pick} total={time.time()-t_all:.1f}s", flush=True)

    if a.graded and a.out:
        xg = np.load(a.graded)["x"]
        lg = logits_for(best_model, ds.norm(xg), tta=best_tta, pad=a.pad)
        pred = lg.argmax(1).cpu().numpy().astype(np.int64)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        np.save(a.out, pred)
        print("wrote %s  shape=%s range=(%d,%d)" % (a.out, pred.shape, pred.min(), pred.max()),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
