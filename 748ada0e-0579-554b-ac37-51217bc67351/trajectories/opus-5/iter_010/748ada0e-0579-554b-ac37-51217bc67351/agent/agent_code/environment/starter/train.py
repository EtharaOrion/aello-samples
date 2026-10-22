"""CIFAR-100 @ 20k images, fixed budget: 8000 optimizer steps / 1,024,000 presentations.

Submitted recipe (one run, batch 128 x 8000 steps = exactly the presentation budget):
  CIFAR-style ResNet-34 (BasicBlock, stages [3,4,6,3]), base width 160 (133M params),
  inputs upsampled 32 -> 48 px, ResNet-D shortcuts (avgpool -> 1x1 conv) at the stage
  transitions.  SGD + Nesterov, lr 0.2, cosine with 3% warmup, wd 2e-3, label smoothing
  0.1.  Augmentation: reflect-pad-4 random crop + horizontal flip + cutout(12).
  Prediction: horizontal-flip TTA, averaging the log-probs of the final weights and of an
  EMA copy of the same trajectory, then per-class Sinkhorn balancing (postproc.py).

MEASURED THIS SESSION (val_public balanced accuracy; sigma between runs is ~0.5-0.6%)
  A_base   no cutout, seed 0        raw 0.7058  ema 0.7086  raw+ema 0.7137
  B_cut    + cutout(12), seed 1     raw 0.7251  ema 0.7173  raw+ema 0.7343   <-- submitted
  C_cutmix + mixup(0.2, p 0.5)      raw 0.7289  ema 0.7082  raw+ema 0.7329
  D_final  B + a 3rd (faster) EMA   raw 0.7251  ema2 0.7140 raw+ema 0.7343 (ema2 adds 0)
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
import argparse, json, os, sys, time
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
    def __init__(self, cin, cout, stride=1, aa=False):
        super().__init__()
        if stride == 1 or not aa:
            self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
            self.pool = nn.Identity()
        else:
            # anti-aliased main path: full-res 3x3 then average-pool (~20% slower)
            self.conv1 = nn.Conv2d(cin, cout, 3, 1, 1, bias=False)
            self.pool = nn.AvgPool2d(2, 2)
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
        o = self.pool(F.relu(self.bn1(self.conv1(x)), inplace=True))
        o = self.bn2(self.conv2(o))
        return F.relu(o + self.short(x), inplace=True)


class ResNet(nn.Module):
    def __init__(self, blocks=(3, 4, 6, 3), width=160, n_classes=100, aa=False):
        super().__init__()
        w = [width, width * 2, width * 4, width * 8]
        self.stem = nn.Sequential(nn.Conv2d(3, w[0], 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(w[0]), nn.ReLU(inplace=True))
        layers, cin = [], w[0]
        for i, n in enumerate(blocks):
            for j in range(n):
                layers.append(BasicBlock(cin, w[i], stride=2 if (j == 0 and i > 0) else 1, aa=aa))
                cin = w[i]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(cin, n_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.body(self.stem(x))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


def make_model(seed, width, blocks, n_classes=100, aa=False):
    torch.manual_seed(seed)
    return ResNet(blocks=blocks, width=width, n_classes=n_classes, aa=aa)


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
        views = [xb]
        if tta in ("flip", "flip5"):
            views.append(torch.flip(xb, dims=[3]))
        if tta == "flip5":
            sh = F.pad(xb, (2, 2, 2, 2), mode="reflect")
            for (dy, dx) in ((0, 2), (4, 2), (2, 0)):
                v = sh[:, :, dy:dy + 32, dx:dx + 32]
                views += [v, torch.flip(v, dims=[3])]
        acc = 0.
        for v in views:
            with torch.autocast(device, dtype=torch.bfloat16):
                o = model(to_res(v, res))
            acc = acc + F.log_softmax(o.float(), dim=1)
        outs.append((acc / len(views)).cpu())
    model.train()
    return torch.cat(outs).numpy().astype(np.float32)


def balanced_acc(pred, y, n=100):
    rec = []
    for c in range(n):
        m = y == c
        if m.sum():
            rec.append((pred[m] == c).mean())
    return float(np.mean(rec))


# ------------------------------------------------------------------------ training
def train(args, device="cuda"):
    x, y = data.load_train(args.instance)
    n_classes = data.N_CLASSES
    blocks = tuple(int(v) for v in args.blocks.split(","))
    model = make_model(args.seed, args.width, blocks, aa=bool(args.aa)).to(device)
    model = model.to(memory_format=torch.channels_last)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"params {nparam/1e6:.1f}M  width {args.width}  blocks {blocks}  res {args.res}",
          flush=True)

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
        return 0.5 * (1 + np.cos(np.pi * p))

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
    presentations = 0
    t0 = time.time()

    # shuffled-epoch sampling (better coverage than iid sampling with replacement)
    order = rng.permutation(len(y))
    ptr = 0
    for t in range(steps):
        if ptr + args.batch > len(order):
            order = rng.permutation(len(y))
            ptr = 0
        i = np.sort(order[ptr:ptr + args.batch])
        ptr += args.batch

        xb = torch.from_numpy(np.ascontiguousarray(x[i])).permute(0, 3, 1, 2)
        xb = xb.to(device, non_blocking=True).float()
        yb = torch.from_numpy(y[i]).to(device, non_blocking=True)
        presentations += len(i)

        xb = random_crop(xb, args.pad)
        fl = torch.rand(len(i), device=device) < 0.5
        xb = torch.where(fl[:, None, None, None], torch.flip(xb, dims=[3]), xb)
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
                i = np.sort(rng.choice(len(y), args.batch, replace=False))
                xb = torch.from_numpy(np.ascontiguousarray(x[i])).permute(0, 3, 1, 2)
                xb = xb.to(device).float()
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
    ap.add_argument("--aa", type=int, default=0)
    ap.add_argument("--steps", type=int, default=BUDGET_STEPS)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--wd", type=float, default=2e-3)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--pad", type=int, default=4)
    ap.add_argument("--cutout", type=int, default=0)
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
    ap.add_argument("--tta", default="flip,flip5")
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
    import itertools
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

    parts = bk.split("+")
    gl = sum(logits_of(cands[k], xg, "cuda", a.res, bt) for k in parts) / len(parts)

    np.savez(os.path.join(a.logitdir, f"{a.tag}.npz"),
             **{f"v_{k}_{t}": v for (k, t), v in vlog.items()},
             g_best=gl, yv=yv)
    json.dump({"tag": a.tag, "scores": {f"{k}_{t}": v for (k, t), v in scores.items()},
               "best": f"{bk}_{bt}", "best_score": scores[(bk, bt)],
               "args": vars(a), "seconds": time.time() - t0},
              open(os.path.join(a.logitdir, f"{a.tag}.json"), "w"), indent=1)

    if not a.no_write:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        np.save(a.out, gl.argmax(1).astype(np.int64))
        print(f"wrote {a.out}", flush=True)
    print(f"TOTAL {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
