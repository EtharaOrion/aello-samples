"""CIFAR-100 @ 20k images, fixed budget: 8000 optimizer steps / 1,024,000 presentations.

Recipe (one run, batch 128 x 8000 steps = exactly the presentation budget):
  CIFAR-style ResNet-34 (BasicBlock, stages [3,4,6,3]), base width 160 (133M params),
  inputs upsampled 32 -> 48 px, ResNet-D shortcuts (avgpool -> 1x1 conv) at the stage
  transitions.  SGD + Nesterov, lr 0.2, cosine with 3% warmup, wd 2e-3, label smoothing
  0.1.  Augmentation: reflect-pad-4 random crop + horizontal flip + cutout(12)
  (+ optional GPU-batched TrivialAugment).
  Prediction: horizontal-flip TTA, averaging the log-probs of the final weights and of an
  EMA copy of the same trajectory, then per-class Sinkhorn balancing.

MEASURED IN EARLIER SESSIONS (val_public balanced accuracy; sigma between runs ~0.5-0.6%)
  A_base   no cutout, seed 0        raw 0.7058  ema 0.7086  raw+ema 0.7137
  B_cut    + cutout(12), seed 1     raw 0.7251  ema 0.7173  raw+ema 0.7343   <-- submitted
  C_cutmix + mixup(0.2, p 0.5)      raw 0.7289  ema 0.7082  raw+ema 0.7329
  E_cut16  cutout(16)               raw 0.7258  ema 0.7015  raw+ema 0.7200
  Sinkhorn balancing on top of B: +0.5% at tau 1.0, +0.8% at tau 1.5 (paired resamples).

  * Cutout is the single biggest lever: +1.9% raw, +2.1% ensembled.  At 51 epochs over
    20k images this 133M-param net memorises, so masking augmentation binds first.
  * EMA is worth ~+0.9%, but only as an ENSEMBLE MEMBER -- the EMA weights alone are
    worse than the final weights.  Both come from the same trajectory, so it is free.
  * Mixup on top of cutout is a wash.  A third EMA at a shorter horizon adds nothing.
  * flip5 TTA (flip + 2px shifts) is WORSE than plain flip TTA: 0.7282 vs 0.7343.

TRIED IN EARLIER ATTEMPTS AND NOT BETTER: width 224 @ res32, scale jitter, stochastic
depth 0.10, lr 0.4 + long warmup, wd 1e-3, two 4000-step models, depth 56 @ width 144,
two-cycle snapshot ensembling, CutMix annealed off, anti-aliased main path (also ~20%
slower, which pushes the run past the 432s wall-clock allowance).
"""
import argparse, itertools, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aello_data as data

BUDGET_STEPS = 8000
BUDGET_PRESENTATIONS = 1_024_000


# --------------------------------------------------------------------------- model
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        nn.init.zeros_(self.bn2.weight)          # zero-init residual
        if stride != 1 or cin != cout:
            # ResNet-D shortcut: pool first, then 1x1 (no information thrown away)
            short = []
            if stride != 1:
                short.append(nn.AvgPool2d(2, 2))
            short += [nn.Conv2d(cin, cout, 1, 1, 0, bias=False), nn.BatchNorm2d(cout)]
            self.short = nn.Sequential(*short)
        else:
            self.short = nn.Identity()

    def forward(self, x):
        o = F.relu(self.bn1(self.conv1(x)), inplace=True)
        o = self.bn2(self.conv2(o))
        return F.relu(o + self.short(x), inplace=True)


class ResNet(nn.Module):
    def __init__(self, blocks=(3, 4, 6, 3), width=160, n_classes=100):
        super().__init__()
        w = [width, width * 2, width * 4, width * 8]
        self.stem = nn.Sequential(nn.Conv2d(3, w[0], 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(w[0]), nn.ReLU(inplace=True))
        layers, cin = [], w[0]
        for i, n in enumerate(blocks):
            for j in range(n):
                layers.append(BasicBlock(cin, w[i], stride=2 if (j == 0 and i > 0) else 1))
                cin = w[i]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(cin, n_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.body(self.stem(x))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


def make_model(seed, width, blocks, n_classes=100):
    torch.manual_seed(seed)
    return ResNet(blocks=blocks, width=width, n_classes=n_classes)


# ------------------------------------------------------------------------ data aug
MEAN = torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1) * 255.
STD = torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1) * 255.


def random_crop(x, pad):
    b, c, h, w = x.shape
    xp = F.pad(x, (pad,) * 4, mode="reflect")
    oy = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    ox = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    ar = torch.arange(b, device=x.device)
    rows = (oy[:, None] + torch.arange(h, device=x.device)[None, :])[:, None, :, None]
    cols = (ox[:, None] + torch.arange(w, device=x.device)[None, :])[:, None, None, :]
    return xp[ar[:, None, None, None], torch.arange(c, device=x.device)[None, :, None, None],
              rows, cols]


def cutout(x, size):
    b, c, h, w = x.shape
    cy = torch.randint(0, h, (b,), device=x.device)
    cx = torch.randint(0, w, (b,), device=x.device)
    ys = torch.arange(h, device=x.device)[None, :]
    xs = torch.arange(w, device=x.device)[None, :]
    my = ((ys - cy[:, None]).abs() < size // 2)
    mx = ((xs - cx[:, None]).abs() < size // 2)
    mask = ~(my[:, None, :, None] & mx[:, None, None, :])
    return x * mask


_SHARP_K = None


def trivial_augment(x, strength=1.0, p=1.0):
    """GPU-batched TrivialAugment(-ish): one random op per sample, random magnitude.

    x is float in [0, 255], NCHW.  Geometry goes through a single batched affine
    grid_sample (identity theta for the colour ops, which is exact), colour ops are
    computed for the whole batch and selected with torch.where -- both are pennies
    next to the 133M-param forward pass.
    """
    global _SHARP_K
    b, c, h, w = x.shape
    dev = x.device
    n_ops = 12
    op = torch.randint(0, n_ops, (b,), device=dev)
    if p < 1.0:                                   # samples that get no op at all
        op = torch.where(torch.rand(b, device=dev) < p, op, torch.full_like(op, -1))
    m = torch.rand(b, device=dev) * strength      # magnitude in [0, strength)
    sgn = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)

    # ---- geometry (ops 0..4) via one affine warp
    one = torch.ones(b, device=dev)
    zero = torch.zeros(b, device=dev)
    ang = sgn * m * (20.0 * np.pi / 180.0)
    cos, sin = torch.cos(ang), torch.sin(ang)
    sh = sgn * m * 0.30
    tr = sgn * m * 0.25
    is_rot, is_shx, is_shy = op == 0, op == 1, op == 2
    is_tx, is_ty = op == 3, op == 4
    a00 = torch.where(is_rot, cos, one)
    a11 = torch.where(is_rot, cos, one)
    a01 = torch.where(is_rot, -sin, torch.where(is_shx, sh, zero))
    a10 = torch.where(is_rot, sin, torch.where(is_shy, sh, zero))
    a02 = torch.where(is_tx, tr, zero)
    a12 = torch.where(is_ty, tr, zero)
    theta = torch.stack([torch.stack([a00, a01, a02], 1),
                         torch.stack([a10, a11, a12], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection",
                      align_corners=False)

    # ---- colour (ops 5..11)
    f = (1.0 + sgn * m * 0.6).view(b, 1, 1, 1)
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    out = x
    def pick(cond, val):
        return torch.where(cond.view(b, 1, 1, 1), val, out)
    out = pick(op == 5, x * f)                                        # brightness
    out = pick(op == 6, (x - gray.mean((1, 2, 3), keepdim=True)) * f  # contrast
               + gray.mean((1, 2, 3), keepdim=True))
    out = pick(op == 7, (x - gray) * f + gray)                        # colour/saturation
    q = torch.pow(2.0, torch.round(m * 4.0)).view(b, 1, 1, 1)         # posterize 8..4 bits
    out = pick(op == 8, torch.floor(x / q) * q)
    thr = (255.0 * (1.0 - m * 0.8)).view(b, 1, 1, 1)                  # solarize
    out = pick(op == 9, torch.where(x >= thr, 255.0 - x, x))
    lo = x.amin((2, 3), keepdim=True)                                 # autocontrast
    hi = x.amax((2, 3), keepdim=True)
    out = pick(op == 10, (x - lo) * (255.0 / (hi - lo).clamp(min=1.0)))
    if _SHARP_K is None or _SHARP_K.device != dev:
        k = torch.tensor([[1., 1., 1.], [1., 5., 1.], [1., 1., 1.]], device=dev) / 13.
        _SHARP_K = k.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    blur = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), _SHARP_K, groups=3)
    out = pick(op == 11, blur + (x - blur) * f)                       # sharpness
    return out.clamp_(0.0, 255.0)


def to_res(x, res):
    if x.shape[-1] != res:
        x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
    return x


# ----------------------------------------------------------------------- inference
@torch.no_grad()
def logits_of(model, x_u8, device, res, tta="flip", batch=500):
    """Return float32 log-softmax-averaged logits over the TTA views."""
    model.eval()
    outs = []
    for s in range(0, len(x_u8), batch):
        xb = torch.from_numpy(np.ascontiguousarray(x_u8[s:s + batch])).permute(0, 3, 1, 2)
        xb = xb.to(device).float()
        xb = (xb - MEAN.to(device)) / STD.to(device)
        xb = xb.contiguous(memory_format=torch.channels_last)
        views = [(xb, res)]
        if tta in ("flip", "flipms"):
            views.append((torch.flip(xb, dims=[3]), res))
        if tta == "flipms":                        # + a slightly larger test resolution
            big = int(round(res * 7 / 6.))
            views += [(xb, big), (torch.flip(xb, dims=[3]), big)]
        acc = 0.
        for v, r in views:
            vv = to_res(v, r).contiguous(memory_format=torch.channels_last)
            with torch.autocast(device, dtype=torch.bfloat16):
                o = model(vv)
            acc = acc + F.log_softmax(o.float(), dim=1)
        outs.append((acc / len(views)).cpu())
    model.train()
    return torch.cat(outs).numpy().astype(np.float32)


def balanced_acc(pred, y, n=100):
    return float(np.mean([(pred[y == c] == c).mean() for c in range(n) if (y == c).any()]))


# ------------------------------------------------------- post-hoc marginal balancing
def sinkhorn_bias(logp, iters=60):
    """Per-class additive bias b so that softmax(logp + b) has uniform column mass.

    The metric is balanced accuracy and the graded split is 10000 images, so its class
    marginal is almost certainly uniform; a plain argmax under-predicts hard classes.
    """
    b = np.zeros(logp.shape[1], dtype=np.float64)
    lp = logp.astype(np.float64)
    target = np.log(1.0 / logp.shape[1])
    for _ in range(iters):
        z = lp + b
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        b -= (np.log(p.mean(0) + 1e-12) - target)
    return b


def pick_tau(vlog, yv, taus, k=10, draws=80, seed=0):
    """val_public is class-imbalanced, so score the bias on balanced resamples of it.
    Paired across draws: the same subsets are scored at every tau."""
    rng = np.random.default_rng(seed)
    idx_by_c = [np.where(yv == c)[0] for c in range(100)]
    out = {t: [] for t in taus}
    for _ in range(draws):
        sel = np.concatenate([rng.choice(ix, k, replace=len(ix) < k) for ix in idx_by_c])
        sub, ysub = vlog[sel], yv[sel]
        b = sinkhorn_bias(sub)
        for t in taus:
            out[t].append(balanced_acc((sub + t * b).argmax(1), ysub))
    return {t: float(np.mean(v)) for t, v in out.items()}


# ------------------------------------------------------------------------ training
def train(args, device="cuda"):
    x, y = data.load_train(args.instance)
    blocks = tuple(int(v) for v in args.blocks.split(","))
    model = make_model(args.seed, args.width, blocks).to(device)
    model = model.to(memory_format=torch.channels_last)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M  "
          f"width {args.width}  blocks {blocks}  res {args.res}", flush=True)

    decay, no_decay = [], []
    for nm, p in model.named_parameters():
        (no_decay if (p.ndim <= 1 and args.nowd_bn) else decay).append(p)
    opt = torch.optim.SGD([{"params": decay, "weight_decay": args.wd},
                           {"params": no_decay, "weight_decay": 0.0}],
                          lr=args.lr, momentum=0.9, nesterov=True)

    steps = args.steps
    warm = max(1, int(args.warmup * steps))

    def lr_at(t):
        if t < warm:
            return (t + 1) / warm
        p = (t - warm) / max(1, steps - warm)
        return args.lr_end + (1 - args.lr_end) * 0.5 * (1 + np.cos(np.pi * p))

    def mk_ema(decay):
        d = decay ** args.ema_every
        return torch.optim.swa_utils.AveragedModel(
            model, avg_fn=lambda a, c, n, _d=d: _d * a + (1 - _d) * c)

    emas = {}
    if args.ema > 0:
        emas["ema"] = (mk_ema(args.ema), args.ema_start)
    if args.ema2 > 0:
        emas["ema2"] = (mk_ema(args.ema2), args.ema2_start)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    mean_d, std_d = MEAN.to(device), STD.to(device)
    x_gpu = torch.from_numpy(np.ascontiguousarray(x)).to(device)     # uint8 NHWC, 61 MB
    y_gpu = torch.from_numpy(y).to(device)
    presentations = 0
    t0 = time.time()

    # shuffled-epoch sampling (better coverage than iid sampling with replacement)
    order = rng.permutation(len(y))
    ptr = 0
    for t in range(steps):
        if ptr + args.batch > len(order):
            order = rng.permutation(len(y))
            ptr = 0
        i = order[ptr:ptr + args.batch]
        ptr += args.batch
        idx = torch.from_numpy(i).to(device, non_blocking=True)

        xb = x_gpu[idx].permute(0, 3, 1, 2).float()
        yb = y_gpu[idx]
        presentations += len(i)

        xb = random_crop(xb, args.pad)
        fl = torch.rand(len(i), device=device) < 0.5
        xb = torch.where(fl[:, None, None, None], torch.flip(xb, dims=[3]), xb)
        if args.ta > 0:
            xb = trivial_augment(xb, strength=args.ta, p=args.ta_p)
        xb = (xb - mean_d) / std_d
        if args.cutout > 0:
            xb = cutout(xb, args.cutout)
        xb = to_res(xb, args.res).contiguous(memory_format=torch.channels_last)

        lam, perm = 1.0, None
        if args.mixup > 0 and rng.random() < args.mixup_p:
            lam = float(rng.beta(args.mixup, args.mixup))
            perm = torch.randperm(len(i), device=device)
            xb = lam * xb + (1 - lam) * xb[perm]

        for g in opt.param_groups:
            g["lr"] = args.lr * lr_at(t)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device, dtype=torch.bfloat16):
            out = model(xb)
            loss = F.cross_entropy(out.float(), yb, label_smoothing=args.ls)
            if perm is not None:
                loss = lam * loss + (1 - lam) * F.cross_entropy(
                    out.float(), yb[perm], label_smoothing=args.ls)
        loss.backward()
        opt.step()

        if t % args.ema_every == 0:
            for _e, _st in emas.values():
                if t >= _st * steps:
                    _e.update_parameters(model)

        if t % 1000 == 0 or t == steps - 1:
            print(f"  step {t+1}/{steps} loss {loss.item():.3f} "
                  f"lr {opt.param_groups[0]['lr']:.4f} {time.time()-t0:.0f}s", flush=True)

    assert presentations <= BUDGET_PRESENTATIONS, presentations
    assert steps <= BUDGET_STEPS
    print(f"TRAIN DONE steps={steps}/{BUDGET_STEPS} "
          f"presentations={presentations}/{BUDGET_PRESENTATIONS} {time.time()-t0:.1f}s",
          flush=True)

    # BN buffers inside AveragedModel are copies made at construction time; recompute
    # them for the averaged weights with a few forward passes in train mode.
    out = {}
    for name, (e, _) in emas.items():
        em = e.module
        em.train()
        with torch.no_grad():
            for m in em.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.reset_running_stats()
                    m.momentum = None
            for _ in range(args.bn_batches):
                i = rng.choice(len(y), args.batch, replace=False)
                xb = x_gpu[torch.from_numpy(i).to(device)].permute(0, 3, 1, 2).float()
                xb = random_crop(xb, args.pad)
                xb = (xb - mean_d) / std_d
                xb = to_res(xb, args.res).contiguous(memory_format=torch.channels_last)
                with torch.autocast(device, dtype=torch.bfloat16):
                    em(xb)
        out[name] = em
    return model, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="i20k", choices=list(data.INSTANCES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graded", default="../data/graded_images.npz")
    ap.add_argument("--out", default="/submission/preds_i20k.npy")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--blocks", default="3,4,6,3")
    ap.add_argument("--res", type=int, default=48)
    ap.add_argument("--steps", type=int, default=BUDGET_STEPS)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--lr_end", type=float, default=0.0)
    ap.add_argument("--wd", type=float, default=2e-3)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--pad", type=int, default=4)
    ap.add_argument("--cutout", type=int, default=12)
    ap.add_argument("--ta", type=float, default=0.0, help="TrivialAugment strength")
    ap.add_argument("--ta_p", type=float, default=1.0)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--mixup_p", type=float, default=0.5)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--ema_start", type=float, default=0.5)
    ap.add_argument("--ema_every", type=int, default=2)
    ap.add_argument("--ema2", type=float, default=0.0)
    ap.add_argument("--ema2_start", type=float, default=0.5)
    ap.add_argument("--bn_batches", type=int, default=50)
    ap.add_argument("--nowd_bn", type=int, default=0)
    ap.add_argument("--tta", default="flip,flipms")
    ap.add_argument("--taus", default="0.0,0.8,1.2,1.5")
    ap.add_argument("--no_write", action="store_true")
    ap.add_argument("--logitdir", default="/workspace/logits")
    a = ap.parse_args()

    t0 = time.time()
    model, ema_models = train(a)

    xv, yv = data.load_val()
    xg = np.load(a.graded)["x"]
    os.makedirs(a.logitdir, exist_ok=True)

    cands = {"raw": model}
    cands.update(ema_models)

    # all non-empty subsets of the heads, over each TTA scheme.  Every head comes
    # from the same single 8000-step trajectory, so this costs no extra training.
    subsets = [c for r in range(1, len(cands) + 1)
               for c in itertools.combinations(sorted(cands), r)]
    vlog, scores = {}, {}
    for tt in a.tta.split(","):
        base = {k: logits_of(m, xv, "cuda", a.res, tt) for k, m in cands.items()}
        for c in subsets:
            key = ("+".join(c), tt)
            vlog[key] = sum(base[k] for k in c) / len(c)
            scores[key] = balanced_acc(vlog[key].argmax(1), yv)
            print(f"VAL {a.tag} {'+'.join(c):16s} {tt:6s} bal-acc {scores[key]:.4f}",
                  flush=True)

    bk, bt = max(scores, key=scores.get)
    print(f"BEST {a.tag} {bk} {bt} {scores[(bk, bt)]:.4f}", flush=True)

    # tau for the Sinkhorn bias, chosen on balanced resamples of val
    taus = [float(v) for v in a.taus.split(",")]
    tsw = pick_tau(vlog[(bk, bt)], yv, taus)
    btau = max(tsw, key=tsw.get)
    for t in taus:
        print(f"  tau={t:<4} resampled bal-acc {tsw[t]:.4f} "
              f"delta {tsw[t]-tsw[taus[0]]:+.4f}", flush=True)
    print(f"BALANCED {a.tag} tau {btau}", flush=True)

    parts = bk.split("+")
    gl = sum(logits_of(cands[k], xg, "cuda", a.res, bt) for k in parts) / len(parts)
    pred = (gl + btau * sinkhorn_bias(gl)).argmax(1).astype(np.int64)

    np.savez(os.path.join(a.logitdir, f"{a.tag}.npz"),
             **{f"v_{k}_{t}": v for (k, t), v in vlog.items()},
             g_best=gl, yv=yv)
    json.dump({"tag": a.tag, "scores": {f"{k}_{t}": v for (k, t), v in scores.items()},
               "best": f"{bk}_{bt}", "best_score": scores[(bk, bt)],
               "tau": btau, "tau_sweep": {str(k): v for k, v in tsw.items()},
               "args": vars(a), "seconds": time.time() - t0},
              open(os.path.join(a.logitdir, f"{a.tag}.json"), "w"), indent=1)

    if not a.no_write:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        np.save(a.out, pred)
        cnt = np.bincount(pred, minlength=100)
        print(f"wrote {a.out}  class counts min {cnt.min()} max {cnt.max()}", flush=True)
    print(f"TOTAL {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
