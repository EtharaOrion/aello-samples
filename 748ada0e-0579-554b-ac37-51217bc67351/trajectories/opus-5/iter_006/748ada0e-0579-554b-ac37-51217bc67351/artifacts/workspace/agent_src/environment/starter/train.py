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

Final config: CIFAR ResNet-34 (3,4,6,3 basic blocks), width 160 (133.0M), *ResNet-D
downsampling* (--dsample d), zero-init residual BN, inputs bilinearly upscaled to 48px,
per-sample random crop + hflip, SGD-nesterov lr 0.2 with 4% warmup and cosine-to-zero,
wd 2e-3 on weights only, label smoothing 0.1, flip5 TTA.  8000 steps x batch 128 =
1,024,000 presentations, exactly both caps; measured 322.4s of training and 334.3s
end-to-end (train + val + writing the graded predictions) of the 432s allowance, on an
H100.  Reproduced this iteration at val balanced acc 0.7353.

THE DOWNSAMPLE IS WHERE THE ACCURACY WAS.  Each stage transition used to project the
shortcut with a 1x1 convolution of stride 2, which reads one input pixel in four and
throws the other three away.  Replacing it with a 2x2 average pool followed by a stride-1
1x1 (the ResNet-D trick, --dsample d) costs no parameters and 14s of training, and is
worth more than any capacity change measured on this task:

    r34 w160 res48, stride-2 1x1 shortcut (--dsample b)   0.7176 (seed 0)
    r34 w160 res48, avg-pool shortcut     (--dsample d)   0.7323, 0.7353 (seed 0)
                                                          0.7344 (seed 1)

THE BUDGET, NOT THE MODEL, IS THE BINDING CONSTRAINT -- and this iteration nailed that
down.  Five full-budget seed-0 probes, each a paired comparison against the 0.7353
baseline above (same seed, same data order, val balanced accuracy, flip5 TTA):

    baseline                                     0.7353   [334.3s]
    + stochastic depth 0.10 (linear ramp)        0.7228   -1.25%
    + lr 0.4, warmup 20%, wd 1e-3 (lr*wd fixed)  0.7268   -0.85%
    + blur-pool shortcut (--dsample e) + SiLU    0.7302   -0.51%  [366.9s]
    + 2 models x 4000 steps, log-prob averaged   0.7065   -2.88%
    + wd 1e-3 at lr 0.2                          0.7212   -1.41%

Same-config repeats land at 0.7323 / 0.7344 / 0.7353, so run-to-run sigma is ~0.3% and
gaps above ~0.7% are real.  All five probes are on the losing side and four clear that
bar.  wd is now bracketed on both sides -- 1e-3 costs 1.41% and 4e-3 cost 1.0% in an
earlier iteration -- so 2e-3 is a genuine two-sided optimum rather than an untested edge
of the search, which was the one way the "every regulariser loses" story could have been
an artefact of only ever having searched upwards.

The ensemble result is the decisive one.  --models K trains K independent models on ONE
shared budget (K x 4000 updates = 8000 updates, K x 4000 x 128 = 1,024,000 presentations,
both caps still exactly met) and averages their log-probs at prediction time, which the
rules explicitly allow.  Halving each model's schedule costs ~2.9% and two-model
averaging does not come close to repaying it.  Combined with stochastic depth losing
1.25% -- a regulariser that touches the architecture rather than the input, so it is an
independent probe of the same question -- the picture is unambiguous: at 8000 steps this
pipeline is convergence-limited, not overfitting-limited, and anything that buys
generalisation by slowing optimisation is a net loss.  That also explains why raising the
peak lr fails even with the effective decay lr*wd held constant and a 5x longer warmup:
lr 0.2 is not a compromise, it is the optimum.

MODEL CAPACITY IS SATURATED.  Paired seed-0 runs, full budget, relative training cost in
brackets (all --dsample b):

    r18 w96  res48  [1.0x]  0.7055
    r18 w128 res48  [1.8x]  0.7138
    r34 w96  res48  [2.0x]  0.7169
    r34 w160 res32  [2.5x]  0.7190
    r34 w128 res48  [3.6x]  0.7248, and 0.7198 on an independent same-seed repeat
    r34 w160 res48  [5.6x]  0.7237, and 0.7176 on an independent same-seed repeat

Accuracy tracks log(cost) at ~+1.6% per e-fold over the first four rungs and then flats.
A FLOP audit says the ladder cannot be climbed further anyway: ~34 GFLOP per image
forward, ~13 TFLOP per step of forward+backward, 8000 steps in 322s is ~325 TFLOP/s
sustained -- already compute-bound near H100 peak for bf16 convolutions, so there is no
throughput to unlock and buy capacity with.

Earlier findings (~20 full-budget runs), all still current:
  * flip5 TTA vs single-view flip: +0.56% mean over 10 paired res-48 runs, and +0.56%
    again over 5 further paired runs across the capacity ladder.  The one robust
    prediction-time gain; forced via --pick rather than chosen by val-argmax.
  * train_acc reaches 0.998 yet every added regulariser lost: cutmix ~0, cutout 16 ~0,
    wd 4e-3 -1.0%, heavy combined reg -3.3%, and now stochastic depth -1.25%.
  * No gain from: SE blocks, mixup, balanced sampling, flip9 TTA, multi-scale TTA,
    lr 0.4, EMA, deeper-and-narrower at equal FLOPs (r52 w128 -0.40%), dropping the last
    downsample (-1.42%), shared-trunk branch ensembles (-1.34%).
  * Prior correction for balanced accuracy (postproc.py, subtract tau*log p_train(y)) is
    a wash: +0.18% and -0.06% at tau=1 on two runs' val logits.  The 1.39x class
    imbalance is too mild to matter.  Not applied.

Val is a 2000-image split and reads about 2-3% higher than the graded split.
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
    """Stride-2 depthwise binomial [1,2,1]x[1,2,1] low-pass (Zhang 2019).

    A strictly better anti-alias filter than the 2x2 box average that --dsample d uses:
    same job, wider support, no learnable parameters.
    """

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
                 se=0, dsample="b", branches=1, droppath=0.0, act="relu"):
        super().__init__()
        stem = stem or width
        self.stem = nn.Sequential(nn.Conv2d(3, stem, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(stem),
                                  nn.SiLU(inplace=True) if act == "silu"
                                  else nn.ReLU(inplace=True))
        self.branches = branches
        # stochastic depth: rate ramps linearly with block index (deep blocks drop most)
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
               dsample="b", branches=1, droppath=0.0, act="relu"):
    torch.manual_seed(seed)
    if arch == "r18t":                                   # 3-stage variant
        cfg = dict(width=width, blocks=(2, 2, 2), stem=width)
    else:
        cfg = dict(width=width, blocks=(blocks or ARCHS[arch]))
    return ResNet(n_classes, dropout=dropout, se=se, dsample=dsample, branches=branches,
                  droppath=droppath, act=act, **cfg)


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

    for j in range(K):
        model = make_model(args.seed + 1000 * j, data.N_CLASSES, args.arch, args.width,
                           args.dropout, args.se, blocks, args.dsample, args.branches,
                           args.droppath, args.act)
        model = model.to(device).to(memory_format=torch.channels_last)
        nparam = sum(p.numel() for p in model.parameters())
        opt = build_optimizer(model, args)
        steps = steps_each
        warm = max(1, int(round(args.warmup * steps)))

        gen = torch.Generator(device=device)
        gen.manual_seed((args.seed + 1000 * j) * 9973 + 12345)
        torch.manual_seed(args.seed + 1000 * j)
        stream = epoch_index_stream(ds.n, args.batch, gen, device)
        model.train()
        _train_one(model, opt, ds, stream, gen, args, steps, warm, device,
                   lambda: (used_steps, present),
                   lambda ds_, pr_: None)
        models.append(model)
        used_steps += steps
        present += steps * args.batch

    torch.cuda.synchronize()
    train_s = time.time() - t0
    print("[train] arch=%s w=%d res=%d dsample=%s act=%s params=%.1fM models=%d "
          "steps=%d present=%d time=%.1fs"
          % (args.arch, args.width, args.res, args.dsample, args.act, nparam / 1e6, K,
             used_steps, present, train_s), flush=True)
    assert used_steps <= BUDGET_STEPS and present <= BUDGET_PRESENT
    return models, ds


def _train_one(model, opt, ds, stream, gen, args, steps, warm, device, budget_so_far,
               _unused):
    """Train one model for `steps` updates.  The caller has already reserved this
    model's slice of the global budget; we re-check it here so the caps are enforced
    in code rather than by convention."""
    used0, present0 = budget_so_far()
    for s in range(steps):
        if used0 + s + 1 > BUDGET_STEPS or present0 + (s + 1) * args.batch > BUDGET_PRESENT:
            break
        idx = next(stream)
        if args.aug == "affine":
            xb, yb = ds.batch_affine(idx, gen, cutout=args.cutout, rot=args.rot,
                                     smin=args.smin, smax=args.smax, ar=args.aspect,
                                     trans=args.trans, bright=args.bright,
                                     contrast=args.contrast)
        else:
            xb, yb = ds.batch(idx, gen, cutout=args.cutout)

        f = lr_factor(s, steps, warm, args.lr_final)
        for g in opt.param_groups:
            g["lr"] = args.lr * f

        # optional mixing augmentation: batch mixed with a permutation of ITSELF, so no
        # extra images are read and the presentation count is unchanged.
        mixed, ya, yb2, lam = False, None, None, 1.0
        if args.cutmix > 0 and torch.rand(1, generator=gen, device=device).item() < args.cutmix:
            lam = float(np.random.beta(args.mix_alpha, args.mix_alpha))
            perm = torch.randperm(xb.shape[0], device=device, generator=gen)
            h = xb.shape[-1]
            rl = int(h * math.sqrt(1 - lam))
            if rl > 0:
                cy = int(torch.randint(0, h, (1,), device=device, generator=gen).item())
                cx = int(torch.randint(0, h, (1,), device=device, generator=gen).item())
                y1, y2 = max(0, cy - rl // 2), min(h, cy + rl // 2)
                x1, x2 = max(0, cx - rl // 2), min(h, cx + rl // 2)
                xb[:, :, y1:y2, x1:x2] = xb[perm][:, :, y1:y2, x1:x2]
                lam = 1 - ((y2 - y1) * (x2 - x1) / (h * h))
                mixed, ya, yb2 = True, yb, yb[perm]
        elif args.mixup > 0 and torch.rand(1, generator=gen, device=device).item() < args.mixup:
            lam = float(np.random.beta(args.mix_alpha, args.mix_alpha))
            perm = torch.randperm(xb.shape[0], device=device, generator=gen)
            xb = lam * xb + (1 - lam) * xb[perm]
            xb = xb.contiguous(memory_format=torch.channels_last)
            mixed, ya, yb2 = True, yb, yb[perm]

        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(xb)
            loss = (branch_ce_mix(out, ya, yb2, lam, args.ls) if mixed
                    else branch_ce(out, yb, args.ls))
        loss.backward()
        opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True, choices=list(data.INSTANCES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graded", required=True, help="path to the images to predict")
    ap.add_argument("--out", required=True)
    # model
    ap.add_argument("--arch", default="r34")
    ap.add_argument("--blocks", default=None, help="comma-separated block counts")
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--dsample", default="d", choices=["b", "d", "d2", "e", "e2"])
    ap.add_argument("--act", default="relu", choices=["relu", "silu"])
    ap.add_argument("--branches", type=int, default=1)
    ap.add_argument("--models", type=int, default=1,
                    help="train K independent models on ONE shared budget "
                         "(total updates and presentations unchanged) and average "
                         "their log-probs at prediction time")
    ap.add_argument("--se", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--droppath", type=float, default=0.0,
                    help="stochastic-depth rate at the deepest block (linear ramp)")
    # optimisation
    ap.add_argument("--steps", type=int, default=BUDGET_STEPS)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--lr-final", dest="lr_final", type=float, default=0.0)
    ap.add_argument("--mom", type=float, default=0.9)
    ap.add_argument("--wd", type=float, default=2e-3)
    ap.add_argument("--warmup", type=float, default=0.04)
    ap.add_argument("--ls", type=float, default=0.1)
    # data / augmentation
    ap.add_argument("--res", type=int, default=48)
    ap.add_argument("--pad", type=int, default=4, help="crop pad, at 32px scale")
    ap.add_argument("--aug", default="crop", choices=["crop", "affine"])
    ap.add_argument("--cutout", type=int, default=0)
    ap.add_argument("--cutmix", type=float, default=0.0)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--mix-alpha", dest="mix_alpha", type=float, default=1.0)
    ap.add_argument("--rot", type=float, default=0.0)
    ap.add_argument("--smin", type=float, default=1.0)
    ap.add_argument("--smax", type=float, default=1.0)
    ap.add_argument("--aspect", type=float, default=1.0)
    ap.add_argument("--trans", type=float, default=0.125)
    ap.add_argument("--bright", type=float, default=0.0)
    ap.add_argument("--contrast", type=float, default=0.0)
    # prediction
    ap.add_argument("--pick", default="flip5", choices=["none", "flip", "flip5", "flip9"])
    ap.add_argument("--no-val", action="store_true")
    ap.add_argument("--save-logits", default=None)
    a = ap.parse_args()

    t0 = time.time()
    models, ds = train(a)

    if not a.no_val:
        xv, yv = data.load_val()
        lg = ens_logits(models, ds.norm(xv), a.pick, ds.pad)
        pv = lg.argmax(1).cpu().numpy()
        print("[val] tta=%s balanced_acc=%.4f  (top1=%.4f)"
              % (a.pick, balanced_acc(pv, yv), float((pv == yv).mean())), flush=True)
        if a.save_logits:
            np.save(a.save_logits.replace(".npy", "_val.npy"),
                    lg.cpu().numpy().astype(np.float16))

    xg = np.load(a.graded)["x"]
    lgg = ens_logits(models, ds.norm(xg), a.pick, ds.pad)
    pred = lgg.argmax(1).cpu().numpy().astype(np.int64)
    assert pred.shape == (len(xg),) and pred.min() >= 0 and pred.max() < data.N_CLASSES
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.save(a.out, pred)
    if a.save_logits:
        np.save(a.save_logits, lgg.cpu().numpy().astype(np.float16))
    print("wrote %s  total=%.1fs" % (a.out, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
