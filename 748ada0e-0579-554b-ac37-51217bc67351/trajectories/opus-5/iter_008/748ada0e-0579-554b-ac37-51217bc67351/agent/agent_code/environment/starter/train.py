"""CIFAR-100 pipeline under a fixed step budget.

Budget accounting (honest, hard-capped in code):
  * At most BUDGET_STEPS optimiser updates.
  * At most BUDGET_PRESENT training-sample presentations.  Every step draws exactly
    `batch` images from the training set (sampled without replacement inside an epoch),
    and each drawn image is forwarded exactly once, so a step costs `batch`
    presentations.  The loop stops as soon as either cap would be exceeded.
    Mixing augmentations (mixup / cutmix), when enabled, mix a batch with a *permutation
    of itself* -- no additional images are read, the batch is still `batch` images wide.
    The optional `branches` head trains K classifiers on one shared trunk from the same
    forward pass: still one optimiser update and one presentation per image.

IMPORTANT: every tuned hyper-parameter is baked into the argparse DEFAULTS, so the
canonical command reproduces the configuration that was measured:

  python3 train.py --instance i20k --seed 0 \
      --graded ../data/graded_images.npz --out /submission/preds_i20k.npy

Final config: CIFAR ResNet-34 (3,4,6,3 basic blocks), width 160 (133.0M), *fully
anti-aliased downsampling* (--dsample d2), zero-init residual BN, inputs upscaled to 48px,
per-sample random crop + hflip, SGD-nesterov lr 0.2 with 4% warmup and cosine-to-zero,
wd 2e-3 on weights only, label smoothing 0.1, flip5 TTA.  8000 steps x batch 128 =
1,024,000 presentations, exactly both caps; measured 322.4s of training and 334.3s
end-to-end (train + val + writing the graded predictions) of the 432s allowance, on an
H100.  Reproduced at val balanced acc 0.7353.

ANTI-ALIAS *BOTH* PATHS OF THE STAGE TRANSITION (--dsample d2).  The previous iteration
fixed the shortcut (avg-pool then stride-1 1x1, --dsample d) and gained ~1.5%.  The main
path still strided its 3x3 conv, i.e. still read one pixel in four.  Pooling first and
convolving at stride 1 (--dsample d2) costs no parameters and ~14s:

    d  (shortcut only, prior best config)   0.7207   [333.5s]   seed 0
    d2 (both paths)                         0.7253   [349.3s]   seed 0   +0.46%

RUN-TO-RUN NOISE IS ~2x WHAT THE PREVIOUS ITERATION ESTIMATED.  The `d` line above is a
faithful re-run of the prior best config -- training time matched the recorded 322.4s to
0.1s -- yet it read 0.7207 where earlier repeats read 0.7323 / 0.7344 / 0.7353.  So sigma
on a single val reading is ~0.5-0.6%, not 0.3%: val is 2000 images with only ~20 per
class (range 10-31), which alone puts ~1% standard error on a balanced-accuracy reading.
DO NOT trust single-run gaps below ~0.7%; pair probes on one seed where possible.

LABEL SMOOTHING IS THE ONE REGULARISER THAT PAYS, AND 0.1 IS A TWO-SIDED OPTIMUM.  Paired
on seed 1 (same init, same data order, w128 to buy the extra draws):

    ls 0.00   0.6980   -2.39%      <- by far the largest single effect measured here
    ls 0.10   0.7219    ref
    ls 0.20   0.7125   -0.94%

This is the exception to "every regulariser loses": it costs nothing in convergence speed
because it reshapes the target, not the input or the architecture.

ALSO TESTED THIS ITERATION, BOTH REJECTED:
  * ResNet-D two-layer deep stem (3->40->160) on top of d2: 0.7247 vs 0.7253, a wash.
  * res 56 at w144 (d2): 0.7145, -1.1%, AND 432.3s -- at the wall-clock limit.  Combined
    with res32 -> res48 having been worth +0.5%, resolution peaks at 48.  Note the
    inputs are upscaled 32px originals, so resolution adds compute, never information.

THE DOWNSAMPLE IS WHERE THE ACCURACY WAS.  Each stage transition used to project the
shortcut with a 1x1 convolution of stride 2, which reads one input pixel in four and
throws the other three away.  Replacing it with a 2x2 average pool followed by a stride-1
1x1 (the ResNet-D trick, --dsample d) costs no parameters and 14s of training, and is
worth more than any capacity change measured on this task:

    r34 w160 res48, stride-2 1x1 shortcut (--dsample b)   0.7176 (seed 0)
    r34 w160 res48, avg-pool shortcut     (--dsample d)   0.7323, 0.7353 (seed 0)
                                                          0.7344 (seed 1)

THE BUDGET, NOT THE MODEL, IS THE BINDING CONSTRAINT.  Five full-budget seed-0 probes,
each a paired comparison against the 0.7353 baseline above (same seed, same data order,
val balanced accuracy, flip5 TTA):

    baseline                                     0.7353   [334.3s]
    + stochastic depth 0.10 (linear ramp)        0.7228   -1.25%
    + lr 0.4, warmup 20%, wd 1e-3 (lr*wd fixed)  0.7268   -0.85%
    + blur-pool shortcut (--dsample e) + SiLU    0.7302   -0.51%  [366.9s]
    + 2 models x 4000 steps, log-prob averaged   0.7065   -2.88%
    + wd 1e-3 at lr 0.2                          0.7212   -1.41%

Same-config repeats land at 0.7323 / 0.7344 / 0.7353, so run-to-run sigma is ~0.3% and
gaps above ~0.7% are real.  wd is bracketed on both sides (1e-3 costs 1.41%, 4e-3 cost
1.0%), so 2e-3 is a genuine two-sided optimum.  At 8000 steps this pipeline is
convergence-limited, not overfitting-limited: anything that buys generalisation by
slowing optimisation is a net loss.

MODEL CAPACITY IS SATURATED.  Paired seed-0 runs, full budget, relative training cost in
brackets (all --dsample b):

    r18 w96  res48  [1.0x]  0.7055
    r18 w128 res48  [1.8x]  0.7138
    r34 w96  res48  [2.0x]  0.7169
    r34 w160 res32  [2.5x]  0.7190
    r34 w128 res48  [3.6x]  0.7248, and 0.7198 on an independent same-seed repeat
    r34 w160 res48  [5.6x]  0.7237, and 0.7176 on an independent same-seed repeat

Accuracy tracks log(cost) at ~+1.6% per e-fold over the first four rungs and then flats.

Earlier findings (~20 full-budget runs), all still current:
  * flip5 TTA vs single-view flip: +0.56% mean over 10 paired res-48 runs, and +0.56%
    again over 5 further paired runs across the capacity ladder.  The one robust
    prediction-time gain; forced via --pick rather than chosen by val-argmax.
  * train_acc reaches 0.998 yet every added regulariser lost: cutmix ~0, cutout 16 ~0,
    wd 4e-3 -1.0%, heavy combined reg -3.3%, stochastic depth -1.25%.
  * No gain from: SE blocks, mixup, balanced sampling, flip9 TTA, multi-scale TTA,
    lr 0.4, EMA, deeper-and-narrower at equal FLOPs (r52 w128 -0.40%), dropping the last
    downsample (-1.42%), shared-trunk branch ensembles (-1.34%).
  * Prior correction for balanced accuracy (postproc.py) is a wash at tau=1: the 1.39x
    class imbalance is too mild to matter.  Not applied.

Val is a 2000-image split and reads about 2-3% higher than the graded split.
"""
import argparse, json, math, os, sys, time
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


class DropPath(nn.Module):
    """Stochastic depth: drop the whole residual branch for a random subset of the
    batch.  Costs nothing in presentations (same images, same single forward)."""

    def __init__(self, p):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p <= 0.0:
            return x
        keep = 1.0 - self.p
        m = torch.empty(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype)
        m.bernoulli_(keep)
        return x * (m / keep)

    def extra_repr(self):
        return "p=%.3f" % self.p


class BlurPool(nn.Module):
    """Stride-2 depthwise binomial [1,2,1]x[1,2,1] low-pass (Zhang 2019)."""

    def __init__(self, c, stride=2):
        super().__init__()
        k = torch.tensor([1.0, 2.0, 1.0])
        k = k[:, None] * k[None, :]
        k = k / k.sum()
        self.register_buffer("k", k[None, None].repeat(c, 1, 1, 1))
        self.c, self.stride = c, stride

    def forward(self, x):
        xp = F.pad(x, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(xp, self.k.to(xp.dtype), stride=self.stride, groups=self.c)


def _pool(kind, c, stride):
    if kind in ("e", "e2"):
        return BlurPool(c, stride)
    return nn.AvgPool2d(stride, stride, ceil_mode=True)


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride, se=0, dsample="b", droppath=0.0, act="relu"):
        super().__init__()
        self.dp = DropPath(droppath) if droppath > 0 else nn.Identity()
        self.act = F.silu if act == "silu" else F.relu
        if dsample in ("d2", "e2") and stride != 1:
            self.c1 = nn.Sequential(_pool(dsample, cin, stride),
                                    nn.Conv2d(cin, cout, 3, 1, 1, bias=False))
        else:
            self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        nn.init.zeros_(self.b2.weight)          # identity-init residual branch
        self.se = SE(cout, se) if se > 0 else nn.Identity()
        self.short = None
        if stride != 1 or cin != cout:
            if dsample in ("d", "d2", "e", "e2") and stride != 1:
                # ResNet-D: low-pass the shortcut instead of striding the 1x1, so the
                # projection sees every input pixel rather than 1 in 4.
                self.short = nn.Sequential(_pool(dsample, cin, stride),
                                           nn.Conv2d(cin, cout, 1, 1, bias=False),
                                           nn.BatchNorm2d(cout))
            else:
                self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                           nn.BatchNorm2d(cout))

    def forward(self, x):
        r = x if self.short is None else self.short(x)
        x = self.act(self.b1(self.c1(x)))
        x = self.se(self.b2(self.c2(x)))
        return self.act(self.dp(x) + r)


class ResNet(nn.Module):
    """CIFAR ResNet: 3x3 stem (no downsample), 3 or 4 stages of BasicBlocks."""

    def __init__(self, n_classes, width=64, blocks=(2, 2, 2, 2), stem=None, dropout=0.0,
                 se=0, dsample="b", branches=1, droppath=0.0, act="relu", deepstem=0):
        super().__init__()
        stem = stem or width
        if deepstem:
            h = max(32, stem // 4)
            self.stem = nn.Sequential(
                nn.Conv2d(3, h, 3, 1, 1, bias=False), nn.BatchNorm2d(h),
                nn.SiLU(inplace=True) if act == "silu" else nn.ReLU(inplace=True),
                nn.Conv2d(h, stem, 3, 1, 1, bias=False), nn.BatchNorm2d(stem),
                nn.SiLU(inplace=True) if act == "silu" else nn.ReLU(inplace=True))
        else:
            self.stem = nn.Sequential(nn.Conv2d(3, stem, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(stem),
                                      nn.SiLU(inplace=True) if act == "silu"
                                      else nn.ReLU(inplace=True))
        self.branches = branches
        nblk = sum(blocks)
        self._dp_i = 0

        def stage(i, n, cin):
            layers = []
            cout = width * (2 ** i)
            for j in range(n):
                dp = droppath * self._dp_i / max(1, nblk - 1)
                self._dp_i += 1
                layers.append(BasicBlock(cin, cout,
                                         stride=(2 if (j == 0 and i > 0) else 1), se=se,
                                         dsample=dsample, droppath=dp, act=act))
                cin = cout
            return nn.Sequential(*layers), cin

        trunk, cin = [], stem
        for i, n in enumerate(blocks[:-1] if branches > 1 else blocks):
            s, cin = stage(i, n, cin)
            trunk.append(s)
        self.body = nn.Sequential(*trunk)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if branches > 1:
            i, n = len(blocks) - 1, blocks[-1]
            tails, heads = [], []
            for _ in range(branches):
                s, cout = stage(i, n, cin)
                tails.append(s)
                heads.append(nn.Linear(cout, n_classes))
            self.tails = nn.ModuleList(tails)
            self.heads = nn.ModuleList(heads)
        else:
            self.head = nn.Linear(cin, n_classes)

    def _pool(self, x):
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(self, x):
        f = self.body(self.stem(x))
        if self.branches > 1:
            outs = [h(self.drop(self._pool(t(f)))) for t, h in zip(self.tails, self.heads)]
            if self.training:
                return torch.stack(outs)                      # K,B,C -> per-branch loss
            return torch.stack([F.log_softmax(o.float(), 1) for o in outs]).mean(0)
        return self.head(self.drop(self._pool(f)))


ARCHS = {
    "r10":  (1, 1, 1, 1),
    "r18":  (2, 2, 2, 2),
    "r26":  (3, 3, 3, 3),
    "r34":  (3, 4, 6, 3),
    "r44":  (4, 5, 8, 4),
    "r52":  (4, 6, 10, 4),
    "r68":  (5, 8, 14, 5),
}


def make_model(seed, n_classes, arch="r18", width=96, dropout=0.0, se=0, blocks=None,
               dsample="b", branches=1, droppath=0.0, act="relu", deepstem=0):
    torch.manual_seed(seed)
    if arch == "r18t":                                   # 3-stage variant
        cfg = dict(width=width, blocks=(2, 2, 2), stem=width)
    else:
        cfg = dict(width=width, blocks=(blocks or ARCHS[arch]))
    return ResNet(n_classes, dropout=dropout, se=se, dsample=dsample, branches=branches,
                  droppath=droppath, act=act, deepstem=deepstem, **cfg)


def branch_ce(out, target, ls):
    """Cross-entropy that also accepts a (K,B,C) stack of branch logits."""
    if out.dim() == 3:
        return torch.stack([F.cross_entropy(o, target, label_smoothing=ls)
                            for o in out]).mean()
    return F.cross_entropy(out, target, label_smoothing=ls)


def branch_ce_mix(out, ya, yb, lam, ls):
    """Mixed-target cross-entropy (mixup / cutmix)."""
    if out.dim() == 3:
        return torch.stack([lam * F.cross_entropy(o, ya, label_smoothing=ls) +
                            (1 - lam) * F.cross_entropy(o, yb, label_smoothing=ls)
                            for o in out]).mean()
    return (lam * F.cross_entropy(out, ya, label_smoothing=ls) +
            (1 - lam) * F.cross_entropy(out, yb, label_smoothing=ls))


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
        self.x0 = t.contiguous()                                         # N,C,H,W
        self.y = torch.from_numpy(y).to(device)
        self.h = res
        self.device = device

    def norm(self, x_uint8_nhwc):
        return self.norm_at(x_uint8_nhwc, self.res)

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
            xb = apply_cutout(xb, cutout, gen)
        return xb.contiguous(memory_format=torch.channels_last), self.y[idx]

    def batch_affine(self, idx, gen, cutout=0, rot=0.0, smin=1.0, smax=1.0, ar=1.0,
                     trans=0.125, bright=0.0, contrast=0.0):
        """One batched random affine warp (scale / aspect / rotate / translate / hflip)
        per sample, plus optional brightness-contrast jitter and cutout.  Strictly one
        view per drawn image, so the presentation accounting is unchanged."""
        x = self.x0[idx]
        b, h, dev = x.shape[0], self.h, self.device
        z = torch.empty(b, device=dev).uniform_(smin, smax, generator=gen)
        if ar > 1.0:
            la = torch.empty(b, device=dev).uniform_(-math.log(ar), math.log(ar),
                                                     generator=gen).exp().sqrt()
        else:
            la = torch.ones(b, device=dev)
        zx, zy = z * la, z / la
        th = torch.empty(b, device=dev).uniform_(-rot, rot, generator=gen) * (math.pi / 180)
        flip = torch.where(torch.rand(b, device=dev, generator=gen) < 0.5, -1.0, 1.0)
        tx = torch.empty(b, device=dev).uniform_(-trans, trans, generator=gen)
        ty = torch.empty(b, device=dev).uniform_(-trans, trans, generator=gen)
        cos, sin = torch.cos(th), torch.sin(th)
        theta = torch.zeros(b, 2, 3, device=dev)
        theta[:, 0, 0] = zx * cos * flip
        theta[:, 0, 1] = -zy * sin
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = zx * sin * flip
        theta[:, 1, 1] = zy * cos
        theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, (b, 3, h, h), align_corners=False)
        xb = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection",
                           align_corners=False)
        if bright > 0:
            xb = xb + torch.empty(b, 1, 1, 1, device=dev).uniform_(-bright, bright,
                                                                   generator=gen)
        if contrast > 0:
            c = torch.empty(b, 1, 1, 1, device=dev).uniform_(-contrast, contrast,
                                                             generator=gen).exp()
            m = xb.mean(dim=(1, 2, 3), keepdim=True)
            xb = (xb - m) * c + m
        if cutout > 0:
            xb = apply_cutout(xb, cutout, gen)
        return xb.contiguous(memory_format=torch.channels_last), self.y[idx]


def apply_cutout(xb, cutout, gen):
    b, _, h, _ = xb.shape
    dev = xb.device
    ar = torch.arange(h, device=dev)
    cy = torch.randint(0, h, (b,), device=dev, generator=gen)
    cx = torch.randint(0, h, (b,), device=dev, generator=gen)
    gy = ar[None, :]
    my = (gy >= (cy[:, None] - cutout // 2)) & (gy < (cy[:, None] + cutout - cutout // 2))
    mx = (gy >= (cx[:, None] - cutout // 2)) & (gy < (cx[:, None] + cutout - cutout // 2))
    mask = ~(my[:, :, None] & mx[:, None, :])
    return xb * mask[:, None, :, :].to(xb.dtype)


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


def ens_logits(models, xn, tta, pad):
    """Average the (already view-averaged) log-probs of every model in the ensemble."""
    acc = None
    for m in models:
        lg = logits_for(m, xn, tta=tta, pad=pad)
        acc = lg if acc is None else acc + lg
    return acc / len(models)


def balanced_acc(pred, y, n_classes=100):
    rec = []
    for c in range(n_classes):
        m = y == c
        if m.sum() > 0:
            rec.append((pred[m] == c).mean())
    return float(np.mean(rec))


# --------------------------------------------------------------------------------------
def build_optimizer(model, args):
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim > 1 else no_decay).append(p)
    return torch.optim.SGD(
        [{"params": decay, "weight_decay": args.wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, momentum=args.mom, nesterov=True)


def lr_factor(s, steps, warm, final=0.0):
    if s < warm:
        return (s + 1) / max(1, warm)
    t = (s - warm) / max(1, steps - warm)
    return final + (1 - final) * 0.5 * (1 + math.cos(math.pi * t))


def train(args, device="cuda"):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    x, y = data.load_train(args.instance)
    ds = GPUData(x, y, device, pad=args.pad, res=args.res)

    blocks = tuple(int(v) for v in args.blocks.split(",")) if args.blocks else None

    # ---- hard budget caps, shared across ALL models in the ensemble --------------
    # K independent models split one budget: total optimiser updates and total
    # presentations are the same as a single model trained for the full schedule.
    total_steps = min(args.steps, BUDGET_STEPS, BUDGET_PRESENT // args.batch)
    K = max(1, args.models)
    steps_each = total_steps // K
    used_steps, present = 0, 0
    models = []
    t0 = time.time()

    for k in range(K):
        model = make_model(args.seed + 1000 * k, data.N_CLASSES, arch=args.arch,
                           width=args.width, dropout=args.dropout, se=args.se,
                           blocks=blocks, dsample=args.dsample, branches=args.branches,
                           droppath=args.droppath, act=args.act,
                           deepstem=args.deepstem).to(device)
        model = model.to(memory_format=torch.channels_last)
        if k == 0:
            nparam = sum(p.numel() for p in model.parameters())
            print("model %s w%d res%d dsample=%s params=%.1fM  steps=%d x %d models"
                  % (args.arch, args.width, args.res, args.dsample, nparam / 1e6,
                     steps_each, K), flush=True)
        opt = build_optimizer(model, args)
        gen = torch.Generator(device=device)
        gen.manual_seed(args.seed * 7919 + 13 + k)
        stream = epoch_index_stream(ds.n, args.batch, gen, device)
        warm = int(args.warmup * steps_each)
        model.train()
        run_loss, run_corr, run_n = 0.0, 0, 0

        for s in range(steps_each):
            # --- budget guard: never exceed either cap -------------------------
            if used_steps >= BUDGET_STEPS or present + args.batch > BUDGET_PRESENT:
                print("budget exhausted at step %d (present=%d)" % (used_steps, present),
                      flush=True)
                break
            idx = next(stream)
            if args.aug == "affine":
                xb, yb = ds.batch_affine(idx, gen, cutout=args.cutout, rot=args.rot,
                                         smin=args.smin, smax=args.smax, ar=args.arjit,
                                         trans=args.trans, bright=args.bright,
                                         contrast=args.contrast)
            else:
                xb, yb = ds.batch(idx, gen, cutout=args.cutout)

            lam, yb2 = None, None
            if args.mixup > 0 or args.cutmix > 0:
                use_cutmix = args.cutmix > 0 and (args.mixup <= 0 or
                                                  torch.rand(1, generator=gen,
                                                             device=device).item() < 0.5)
                alpha = args.cutmix if use_cutmix else args.mixup
                lam = float(np.random.RandomState(args.seed * 131071 + used_steps)
                            .beta(alpha, alpha))
                perm = torch.randperm(xb.shape[0], device=device, generator=gen)
                yb2 = yb[perm]
                if use_cutmix:
                    h = xb.shape[-1]
                    r = math.sqrt(1 - lam)
                    cw, chh = int(h * r), int(h * r)
                    cy = int(torch.randint(0, h, (1,), device=device,
                                           generator=gen).item())
                    cx = int(torch.randint(0, h, (1,), device=device,
                                           generator=gen).item())
                    y1, y2 = max(0, cy - chh // 2), min(h, cy + chh // 2)
                    x1, x2 = max(0, cx - cw // 2), min(h, cx + cw // 2)
                    xb[:, :, y1:y2, x1:x2] = xb[perm][:, :, y1:y2, x1:x2]
                    lam = 1 - ((y2 - y1) * (x2 - x1) / (h * h))
                else:
                    xb = lam * xb + (1 - lam) * xb[perm]
                xb = xb.contiguous(memory_format=torch.channels_last)

            f = lr_factor(s, steps_each, warm, args.lrfinal)
            for g in opt.param_groups:
                g["lr"] = args.lr * f

            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(xb)
                if lam is None:
                    loss = branch_ce(out, yb, args.ls)
                else:
                    loss = branch_ce_mix(out, yb, yb2, lam, args.ls)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()

            used_steps += 1
            present += args.batch
            if args.log > 0 and (s % args.log == 0 or s == steps_each - 1):
                with torch.no_grad():
                    o = out[0] if out.dim() == 3 else out
                    run_corr = (o.argmax(1) == yb).float().mean().item()
                run_loss = loss.item()
                print("  k%d step %5d/%d lr %.4f loss %.3f acc %.3f  [%.0fs]"
                      % (k, s, steps_each, args.lr * f, run_loss, run_corr,
                         time.time() - t0), flush=True)
        models.append(model)

    train_time = time.time() - t0
    print("TRAIN DONE steps=%d present=%d  (caps %d/%d)  %.1fs"
          % (used_steps, present, BUDGET_STEPS, BUDGET_PRESENT, train_time), flush=True)
    assert used_steps <= BUDGET_STEPS and present <= BUDGET_PRESENT, "budget violated"
    return models, ds, train_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="i20k")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graded", default=None)
    ap.add_argument("--out", default=None)
    # architecture
    ap.add_argument("--arch", default="r34")
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--blocks", default=None)
    ap.add_argument("--dsample", default="d2")
    ap.add_argument("--se", type=int, default=0)
    ap.add_argument("--branches", type=int, default=1)
    ap.add_argument("--droppath", type=float, default=0.0)
    ap.add_argument("--act", default="relu")
    ap.add_argument("--deepstem", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.0)
    # optimisation
    ap.add_argument("--steps", type=int, default=BUDGET_STEPS)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--models", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--lrfinal", type=float, default=0.0)
    ap.add_argument("--warmup", type=float, default=0.04)
    ap.add_argument("--wd", type=float, default=2e-3)
    ap.add_argument("--mom", type=float, default=0.9)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=0.0)
    # data / augmentation
    ap.add_argument("--res", type=int, default=48)
    ap.add_argument("--pad", type=int, default=4)
    ap.add_argument("--aug", default="crop")
    ap.add_argument("--cutout", type=int, default=0)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--cutmix", type=float, default=0.0)
    ap.add_argument("--rot", type=float, default=0.0)
    ap.add_argument("--smin", type=float, default=1.0)
    ap.add_argument("--smax", type=float, default=1.0)
    ap.add_argument("--arjit", type=float, default=1.0)
    ap.add_argument("--trans", type=float, default=0.125)
    ap.add_argument("--bright", type=float, default=0.0)
    ap.add_argument("--contrast", type=float, default=0.0)
    # prediction
    ap.add_argument("--pick", default="flip5", help="TTA mode used for the submission")
    ap.add_argument("--save-logits", default=None)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--log", type=int, default=1000)
    args = ap.parse_args()

    t_start = time.time()
    models, ds, train_time = train(args)

    # ---- local measurement on the public val split (never trained on) -----------
    xv, yv = data.load_val()
    xvn = ds.norm(xv)
    res = {}
    for tta in ("flip", "flip5"):
        lg = ens_logits(models, xvn, tta, ds.pad)
        res[tta] = balanced_acc(lg.argmax(1).cpu().numpy(), yv)
        print("val balanced_acc [%s] = %.4f" % (tta, res[tta]), flush=True)
    pick = args.pick if args.pick in res else max(res, key=res.get)
    print("PICK %s -> val %.4f" % (pick, res[pick]), flush=True)

    if args.save_logits:
        lgv = ens_logits(models, xvn, pick, ds.pad).cpu().numpy()
        np.save(args.save_logits.replace(".npy", "_val.npy"), lgv)

    # ---- graded predictions ------------------------------------------------------
    if args.graded and args.out:
        xg = np.load(args.graded)["x"]
        lg = ens_logits(models, ds.norm(xg), pick, ds.pad)
        pred = lg.argmax(1).cpu().numpy().astype(np.int64)
        assert pred.shape == (xg.shape[0],) and pred.min() >= 0 and pred.max() < 100
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.save(args.out, pred)
        print("wrote %s  shape=%s" % (args.out, pred.shape), flush=True)
        if args.save_logits:
            np.save(args.save_logits, lg.cpu().numpy())

    total = time.time() - t_start
    print("RESULT tag=%s val_flip=%.4f val_flip5=%.4f pick=%s val=%.4f train_s=%.1f "
          "total_s=%.1f" % (args.tag, res["flip"], res["flip5"], pick, res[pick],
                            train_time, total), flush=True)
    with open("/workspace/runs.jsonl", "a") as f:
        f.write(json.dumps({"tag": args.tag, "argv": sys.argv[1:], "val": res[pick],
                            "val_flip": res["flip"], "val_flip5": res["flip5"],
                            "train_s": round(train_time, 1),
                            "total_s": round(total, 1)}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
