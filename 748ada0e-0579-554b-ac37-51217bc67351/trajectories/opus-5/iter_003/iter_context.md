## Refinement Loop - Attempt 3
This is attempt number 3. 2 attempt(s) have been made before this one.

**BEST SO FAR: Iteration 2 (reward 0.6604)** — The agent revised CIFAR-100 defaults, compared regularization and test-time augmentation variants, and generated canonical predictions.. Build on this result; do not discard it.

## Prior Attempts (bounded digest: top 3 scored + last 3)
### Iteration 1 | Reward: 0.4745
- Approach: Attempt 1 complete.
- Measured: reward 0.4745
- What happened: Activity: 29 steps; 28 agent turns; 32 tool calls (Bash x24, Edit x5, Read x2, Write x1). Files written: /workspace/environment/starter/train.py. Agent's final report: Attempt 1 complete. Valid submission on disk. ## Result **Val balanced accuracy: 0.7065** (public val, 2000 images), vs. the ~0.43 starter floor → reward ≈ 0.71 on the ramp. ## What changed The starter is a 4-layer plain convnet with per-*batch* flips and 8000 steps of a 200-epoch OneCycle schedule (so it only ever traverses ~20% of the LR curve). Replacements: | | starter | submitted | |---|---|---| | model | 4-conv plain net, 64ch (0.6M) | CIFAR ResNet-18, width 96, zero-init residual BN (25.2M) | | sampling | with replacement | shuffled epochs (even coverage) | | aug | batch-level flip, random crop | **per-sample** flip, random crop, cutout-8, all GPU-resident | | schedule | truncated OneCycle | warmup 4% → cosine to 0, matched to the 8000-step budget | | reg | wd 5e-4 on everything | wd **2e-3** on weights only (none on BN/bias), label smoothing 0.1 | | predict | single view | hflip TTA | ## Measured, not assumed Sweeps over ~20 full-budget runs on the public val split. Seed-to-seed noise is ~±0.7%, so I only trusted repeated trends: - **Capacity helps**: w64 0.668 → w96 0.691 → but w128 0.687 and r34/r26 ≈ 0.688–0.695, so width 96 at depth 18 is the plateau. - **Weight decay i… (heuristic summary from atif trajectory; no LLM pass)
### Iteration 2 | Reward: 0.6604
- Approach: The agent revised CIFAR-100 defaults, compared regularization and test-time augmentation variants, and generated canonical predictions.
- Measured: reward 0.6604
- What happened: The grader reported verbatim: “this submission earned reward 0.6604 (66% of the maximum achievable 1.00); it was not a valid scoreable success.” The agent inspected the prior training script, reran its old defaults, edited `train.py` to bake selected command-line values into `argparse`, conducted repeated and paired validation comparisons, fixed inference to flip-based TTA instead of validation argmax, ran the bare canonical command, and reported saving predictions and memory notes. It tested stronger regularization, cutmix, mixup, cutout, SE blocks, balanced sampling, logit adjustment, EMA, multiple resolutions, learning rates, and TTA variants. All implementation and performance figures that follow are unvalidated agent claims: the final run used 8,000 steps, batch size 128, 1,024,000 presentations, width 96, weight decay `2e-3`, and 92 seconds of a 432-second allowance. Prior defaults were claimed to be width 64 and weight decay `5e-4`; other tested settings included weight decay `4e-3`, learning rate `0.4`, cutout size 16, and resolutions 48 and 32. The agent claimed local balanced accuracy of 0.7051, training accuracy of 0.998, same-seed variation around ±0.6%, and a flip5 gain of 0.56% across 10 paired runs. It also claimed `/submission/preds_i20k.npy` was an `int64` array with shape `(10000,)`, range 0–99, and all 100 classes represented.

### Best solution so far — code from iteration 2 (reward 0.6604)
This is YOUR OWN prior best source. Build on it (write your chosen code to /workspace); do not restart from scratch unless you have a reason. It will be re-trained and re-graded from scratch.
```python
# ===== agent_src/environment/starter/train.py =====
"""CIFAR-100 pipeline under a fixed step budget.

Budget accounting (honest, hard-capped in code):
  * At most BUDGET_STEPS optimiser updates.
  * At most BUDGET_PRESENT training-sample presentations.  Every step draws exactly
    `batch` images from the training set (sampled without replacement inside an epoch),
    and each drawn image is forwarded exactly once, so a step costs `batch`
    presentations.  The loop stops as soon as either cap would be exceeded.
    Mixing augmentations (mixup / cutmix), when enabled, mix a batch with a *permutation
    of itself* -- no additional images are read, the batch is still `batch` images wide.

IMPORTANT: every tuned hyper-parameter is baked into the argparse DEFAULTS, so the
canonical command reproduces the configuration that was measured:

  python3 train.py --instance i20k --seed 0 \
      --graded ../data/graded_images.npz --out /submission/preds_i20k.npy

Final config: CIFAR ResNet-18, width 96 (25.2M), zero-init residual BN, inputs bilinearly
upscaled to 48px, per-sample random crop + hflip + cutout, SGD-nesterov lr 0.2 with 4%
warmup and cosine-to-zero, wd 2e-3 on weights only, label smoothing 0.1, flip5 TTA.
Runs in ~92s of the 432s wall-clock allowance; 8000 steps x batch 128 = 1,024,000
presentations, exactly the cap.

Measured on the 2000-image public val split (balanced accuracy), ~20 full-budget runs.
Same-seed run-to-run noise is ~+-0.6%, so only repeated, paired effects are trusted:
  * flip5 TTA vs single-view flip: +0.56% mean over 10 paired res-48 runs (positive in
    7/10).  This is the one robust gain; it is forced via --pick rather than chosen by
    val-argmax, since argmax over near-tied variants on 2000 images only adds noise.
  * train_acc reaches 0.998 -- the net fully fits the training set -- yet EVERY added
    regulariser lost: cutmix ~0, cutout 16 ~0, wd 4e-3 -1.0%, heavy combined reg -3.3%
    (train_acc 0.85).  At 8000 steps the budget, not overfitting, is the binding
    constraint, so extra regularisation costs more convergence than it buys.
  * No gain from: SE blocks, mixup, balanced sampling, logit adjustment (the 1.39x class
    imbalance is too mild), flip9 TTA, multi-scale TTA, lr 0.4, EMA.
  * res 48 vs res 32 is a wash (0.7052 n=3 vs 0.7043 n=2); kept because it is neutral and
    well inside the time budget.
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
class SE(nn.Module):
    """Squeeze-and-excitation channel gate."""

    def __init__(self, c, r=4):
        super().__init__()
        self.fc1 = nn.Conv2d(c, max(8, c // r), 1)
        self.fc2 = nn.Conv2d(max(8, c // r), c, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s), inplace=True)))
        return x * s


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride, se=0):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        nn.init.zeros_(self.b2.weight)          # identity-init residual branch
        self.se = SE(cout, se) if se > 0 else nn.Identity()
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                       nn.BatchNorm2d(cout))

    def forward(self, x):
        r = x if self.short is None else self.short(x)
        x = F.relu(self.b1(self.c1(x)), inplace=True)
        x = self.se(self.b2(self.c2(x)))
        return F.relu(x + r, inplace=True)


class ResNet(nn.Module):
    """CIFAR ResNet: 3x3 stem (no downsample), 3 or 4 stages of BasicBlocks."""

    def __init__(self, n_classes, width=64, blocks=(2, 2, 2, 2), stem=None, dropout=0.0,
                 se=0):
        super().__init__()
        stem = stem or width
        self.stem = nn.Sequential(nn.Conv2d(3, stem, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(stem), nn.ReLU(inplace=True))
        layers, cin = [], stem
        for i, n in enumerate(blocks):
            cout = width * (2 ** i)
            for j in range(n):
                layers.append(BasicBlock(cin, cout,
                                         stride=(2 if (j == 0 and i > 0) else 1), se=se))
                cin = cout
        self.body = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(cin, n_classes)

    def forward(self, x):
        x = self.body(self.stem(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(self.drop(x))


def make_model(seed, n_classes, arch="r18", width=96, dropout=0.0, se=0):
    torch.manual_seed(seed)
    cfg = {
        "r18":  dict(width=width, blocks=(2, 2, 2, 2)),
        "r18t": dict(width=width, blocks=(2, 2, 2), stem=width),      # 3 stages
        "r26":  dict(width=width, blocks=(3, 3, 3, 3)),
        "r34":  dict(width=width, blocks=(3, 4, 6, 3)),
        "r10":  dict(width=width, blocks=(1, 1, 1, 1)),
    }[arch]
    return ResNet(n_classes, dropout=dropout, se=se, **cfg)


# --------------------------------------------------------------------------------------
# GPU-resident data + augmentation
# --------------------------------------------------------------------------------------
class GPUData:
    def __init__(self, x, y, device, pad=4, res=32):
        self.n = len(y)
        self.res = res
        # `pad` is specified at 32px scale; scale it with the resolution
        self.pad = pad = max(1, int(round(pad * res / 32.0)))
        t = torch.from_numpy(np.ascontiguousarray(x)).to(device)         # N,H,W,C uint8
        t = t.permute(0, 3, 1, 2).float().div_(255.)
        self.mean = t.mean(dim=(0, 2, 3), keepdim=True)
        self.std = t.std(dim=(0, 2, 3), keepdim=True)
        t = (t - self.mean) / self.std
        if res != t.shape[-1]:
            t = F.interpolate(t, size=(res, res), mode="bilinear", align_corners=False)
        self.xp = F.pad(t, (pad,) * 4, mode="reflect").contiguous()      # N,C,H+2p,W+2p
        self.y = torch.from_numpy(y).to(device)
        self.h = res
        self.device = device

    def norm(self, x_uint8_nhwc):
        t = torch.from_numpy(np.ascontiguousarray(x_uint8_nhwc)).to(self.device)
        t = t.permute(0, 3, 1, 2).float().div_(255.)
        t = (t - self.mean) / self.std
        if self.res != t.shape[-1]:
            t = F.interpolate(t, size=(self.res, self.res), mode="bilinear",
                              align_corners=False)
        return t.contiguous(memory_format=torch.channels_last)

    def norm_at(self, x_uint8_nhwc, res):
        """Normalise and resize to an arbitrary resolution (multi-scale TTA)."""
        t = torch.from_numpy(np.ascontiguousarray(x_uint8_nhwc)).to(self.device)
        t = t.permute(0, 3, 1, 2).float().div_(255.)
        t = (t - self.mean) / self.std
        if res != t.shape[-1]:
            t = F.interpolate(t, size=(res, res), mode="bilinear", align_corners=False)
        return t.contiguous(memory_format=torch.channels_last)

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


def balanced_index_stream(y_np, batch, gen, device, n_classes=100):
    """Class-balanced sampling: each drawn image is one presentation, exactly as in the
    uniform stream, so the budget accounting is unchanged."""
    cnt = np.bincount(y_np, minlength=n_classes).astype(np.float64)
    w = torch.tensor((1.0 / cnt)[y_np], device=device, dtype=torch.float)
    w = w / w.sum()
    while True:
        yield torch.multinomial(w, batch, replacement=True, generator=gen)


# --------------------------------------------------------------------------------------
# train / predict
# --------------------------------------------------------------------------------------
def crops5(p):
    """Centre crop + 4 half-way corner crops, in units of the reflect pad `p`."""
    a, b = max(1, p // 2), p + max(1, p // 2)
    return [(p, p), (a, a), (a, b), (b, a), (b, b)]


def crops9(p):
    """3x3 grid of crop offsets spanning the reflect pad."""
    a = max(1, p // 2)
    offs = [p - a, p, p + a]
    return [(oy, ox) for oy in offs for ox in offs]


TTA_CROPS = {"flip5": crops5, "flip9": crops9}


@torch.no_grad()
def logits_for(model, xb_norm, tta="flip", bs=1000, pad=4):
    """tta: 'none' | 'flip' | 'flip5' (5 crops x {id, hflip} = 10 views)."""
    model.eval()
    out = []
    h = xb_norm.shape[-1]
    offs = TTA_CROPS[tta](pad) if tta in TTA_CROPS else None
    for s in range(0, xb_norm.shape[0], bs):
        xb = xb_norm[s:s + bs]
        views = [xb]
        if offs is not None:
            xp = F.pad(xb, (pad,) * 4, mode="reflect")
            views = [xp[:, :, oy:oy + h, ox:ox + h].contiguous(
                memory_format=torch.channels_last) for (oy, ox) in offs]
        acc, nv = None, 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for v in views:
                lg = F.log_softmax(model(v).float(), 1)
                nv += 1
                if tta != "none":
                    lg = lg + F.log_softmax(model(v.flip(3)).float(), 1)
                    nv += 1
                acc = lg if acc is None else acc + lg
        out.append((acc / nv).float())   # mean log-prob: comparable scale across TTA modes
    model.train()
    return torch.cat(out)


def logits_multiscale(model, x_uint8, ds, scales, base_tta="flip5"):
    """Average log-probs over several input resolutions (free at prediction time)."""
    acc = None
    for r in scales:
        pad = max(1, int(round(4 * r / 32.0)))
        lg = logits_for(model, ds.norm_at(x_uint8, r), tta=base_tta, pad=pad)
        acc = lg if acc is None else acc + lg
    return acc / len(scales)


def balanced_acc(pred, y, n_classes=100):
    rec = []
    for c in range(n_classes):
        m = y == c
        if m.sum() > 0:
            rec.append((pred[m] == c).mean())
    return float(np.mean(rec))


def class_log_prior(y, n_classes=100):
    """log p(class) estimated from the *training* labels only."""
    cnt = np.bincount(y, minlength=n_classes).astype(np.float64)
    return np.log(cnt / cnt.sum())


def train(args, device="cuda"):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    x, y = data.load_train(args.instance)
    ds = GPUData(x, y, device, pad=args.pad, res=args.res)

    model = make_model(args.seed, data.N_CLASSES, args.arch, args.width,
                       args.dropout, args.se).to(device).to(memory_format=torch.channels_last)
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
    stream = (balanced_index_stream(y, args.batch, gen, device, data.N_CLASSES)
              if args.bal_sample else epoch_index_stream(ds.n, args.batch, gen, device))

    # log class prior from the TRAINING labels (used for logit adjustment)
    logprior = torch.tensor(class_log_prior(y, data.N_CLASSES), device=device,
                            dtype=torch.float)
    la = args.la_train * logprior if args.la_train > 0 else None

    emas = []
    for d in args.emas:
        emas.append((d, torch.optim.swa_utils.AveragedModel(
            model, avg_fn=(lambda dd: (lambda a, c, n: dd * a + (1 - dd) * c))(d))))

    cutout_px = int(round(args.cutout * ds.res / 32.0))
    presented = 0
    t0 = time.time()
    for t in range(steps):
        if presented + args.batch > BUDGET_PRESENT:
            steps = t
            break
        idx = next(stream)
        xb, yb = ds.batch(idx, gen, cutout=cutout_px)
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
            if la is not None:                 # logit-adjusted loss (Menon et al.)
                out = out + la
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
    return model, outs, ds, presented, steps, logprior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="i20k", choices=list(data.INSTANCES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graded", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--logits-out", default=None)
    # ---- tuned defaults (measured on val_public; see notes) --------------------------
    ap.add_argument("--arch", default="r18")
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=BUDGET_STEPS)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--wd", type=float, default=2e-3)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--warmup", type=float, default=0.04)
    ap.add_argument("--final-lr-frac", type=float, default=0.0)
    ap.add_argument("--pad", type=int, default=4, help="crop pad, at 32px scale")
    ap.add_argument("--res", type=int, default=48, help="input resolution")
    ap.add_argument("--se", type=int, default=0, help="SE reduction ratio (0=off)")
    ap.add_argument("--cutout", type=int, default=8)
    ap.add_argument("--mix-p", type=float, default=0.0)
    ap.add_argument("--mix-a", type=float, default=1.0)
    ap.add_argument("--mix-mode", default="cutmix", choices=["cutmix", "mixup"])
    ap.add_argument("--emas", type=float, nargs="*", default=[])
    ap.add_argument("--ema-start", type=float, default=0.5)
    ap.add_argument("--tta", default="flip5", choices=["none", "flip", "flip5", "flip9"])
    ap.add_argument("--bal-sample", action="store_true",
                    help="class-balanced sampling (same presentation cost per step)")
    ap.add_argument("--la-train", type=float, default=0.0,
                    help="logit-adjusted training loss strength")
    ap.add_argument("--la-taus", type=float, nargs="*", default=[0.0],
                    help="post-hoc logit-adjustment strengths to evaluate on val")
    ap.add_argument("--ms", type=int, nargs="*", default=[],
                    help="multi-scale TTA resolutions, e.g. --ms 40 48 56")
    # Forced, NOT val-argmax: flip5 beat plain flip in 4/4 paired res-48 runs (+1.2%
    # mean).  Picking by argmax over near-tied variants on a 2000-image val split only
    # adds selection noise, so the winning view is fixed ahead of time.
    ap.add_argument("--pick", default="raw/flip5/la0",
                    help="force a variant name instead of val argmax")
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()

    t_all = time.time()
    model, emas, ds, presented, steps, logprior = train(a)

    xv, yv = data.load_val()
    xvn = ds.norm(xv)
    variants = [("raw", model)] + emas
    res, cache = {}, {}
    for name, m in variants:
        for tta in (["flip", "flip5", "flip9"] if a.tta in TTA_CROPS else [a.tta]):
            lg = logits_for(m, xvn, tta=tta, pad=ds.pad)
            for tau in a.la_taus:
                key = f"{name}/{tta}/la{tau:g}"
                cache[key] = (m, tta, tau)
                res[key] = balanced_acc((lg - tau * logprior[None, :]).argmax(1).cpu().numpy(), yv)
    ntr = 4000
    tr_lg = logits_for(model, ds.xp[:ntr, :, ds.pad:ds.pad + ds.h,
                                   ds.pad:ds.pad + ds.h].contiguous(
                       memory_format=torch.channels_last), tta="none", pad=ds.pad)
    tr_acc = float((tr_lg.argmax(1) == ds.y[:ntr]).float().mean())
    print(f"[fit] tag={a.tag} train_acc={tr_acc:.4f}", flush=True)
    if a.ms:
        lg = logits_multiscale(model, xv, ds, a.ms)
        cache["raw/ms/la0"] = (model, "ms", 0.0)
        res["raw/ms/la0"] = balanced_acc(lg.argmax(1).cpu().numpy(), yv)
    pick = a.pick if (a.pick in res) else max(res, key=res.get)
    best_model, best_tta, best_tau = cache[pick]
    print(f"[val] tag={a.tag} " + " ".join(f"{k}={v:.4f}" for k, v in sorted(res.items())) +
          f" pick={pick} total={time.time()-t_all:.1f}s", flush=True)

    if a.graded and a.out:
        xg = np.load(a.graded)["x"]
        lg = (logits_multiscale(best_model, xg, ds, a.ms) if best_tta == "ms"
              else logits_for(best_model, ds.norm(xg), tta=best_tta, pad=ds.pad))
        pred = (lg - best_tau * logprior[None, :]).argmax(1).cpu().numpy().astype(np.int64)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        np.save(a.out, pred)
        print("wrote %s  shape=%s range=(%d,%d)" % (a.out, pred.shape, pred.min(), pred.max()),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ===== agent_src/environment/aello_data.py =====
"""Shipped data access. Read-only helper; the graded pipeline is starter/train.py."""
import os
import numpy as np

DATA = os.path.dirname(os.path.abspath(__file__))
INSTANCES = ("i20k",)
SIZES = {"i20k": 20000}
N_CLASSES = 100


def load_train(instance):
    """Training images and labels."""
    if instance not in INSTANCES:
        raise ValueError("unknown instance %r; expected one of %s" % (instance, list(INSTANCES)))
    d = np.load(os.path.join(DATA, "data", "train_%s.npz" % instance))
    return d["x"], d["y"].astype(np.int64)


def load_val():
    """Public validation split, disjoint from the training set.

    Use it to measure locally. It is NOT the graded split, and instruction.md rule 4
    forbids training on it.
    """
    d = np.load(os.path.join(DATA, "data", "val_public.npz"))
    return d["x"], d["y"].astype(np.int64)


def load_graded_images():
    """The graded images, labels withheld. Predict on these and write
    /submission/preds_i20k.npy (int array, shape (10000,), values 0..99)."""
    return np.load(os.path.join(DATA, "data", "graded_images.npz"))["x"]
```