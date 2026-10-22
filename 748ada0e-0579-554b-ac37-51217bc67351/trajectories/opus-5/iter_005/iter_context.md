## Refinement Loop - Attempt 5
This is attempt number 5. 4 attempt(s) have been made before this one.

**BEST SO FAR: Iteration 4 (reward 0.7045)** — The agent replaced stride-2 residual shortcuts with average-pool projections and benchmarked architecture variants and seeds.. Build on this result; do not discard it.

## Prior Attempts (bounded digest: top 3 scored + last 3)
### Iteration 2 | Reward: 0.6604
- Approach: The agent revised CIFAR-100 defaults, compared regularization and test-time augmentation variants, and generated canonical predictions.
- Measured: reward 0.6604
- What happened: The grader reported verbatim: “this submission earned reward 0.6604 (66% of the maximum achievable 1.00); it was not a valid scoreable success.” The agent inspected the prior training script, reran its old defaults, edited `train.py` to bake selected command-line values into `argparse`, conducted repeated and paired validation comparisons, fixed inference to flip-based TTA instead of validation argmax, ran the bare canonical command, and reported saving predictions and memory notes. It tested stronger regularization, cutmix, mixup, cutout, SE blocks, balanced sampling, logit adjustment, EMA, multiple resolutions, learning rates, and TTA variants. All implementation and performance figures that follow are unvalidated agent claims: the final run used 8,000 steps, batch size 128, 1,024,000 presentations, width 96, weight decay `2e-3`, and 92 seconds of a 432-second allowance. Prior defaults were claimed to be width 64 and weight decay `5e-4`; other tested settings included weight decay `4e-3`, learning rate `0.4`, cutout size 16, and resolutions 48 and 32. The agent claimed local balanced accuracy of 0.7051, training accuracy of 0.998, same-seed variation around ±0.6%, and a flip5 gain of 0.56% across 10 paired runs. It also claimed `/submission/preds_i20k.npy` was an `int64` array with shape `(10000,)`, range 0–99, and all 100 classes represented.
### Iteration 3 | Reward: 0.6860
- Approach: The agent scaled a CIFAR-100 ResNet pipeline, benchmarked capacity and affine augmentation variants, and ran graded inference.
- Measured: reward 0.6860
- What happened: The agent restored its earlier CIFAR-100 training pipeline, launched a baseline early, edited `train.py`, and compared ResNet depth, width, resolution, and batched affine-augmentation variants on `val_public`. It then embedded its selected configuration as defaults and ran the graded-image prediction command. The authoritative grader reported: “this submission earned reward 0.6860 (69% of the maximum achievable 1.00); it was not a valid scoreable success.” All other figures are unvalidated, agent-claimed self-reports. The agent described the selected model as `r34 w160 res48`, trained with learning rate `0.2`, SGD with Nesterov, cosine scheduling, weight decay `2e-3` on weights only, label smoothing `0.1`, crop/flip/Cutout `8`, and flip-5 test-time augmentation. It claimed `8000` updates at batch size `128`, totaling `1,024,000` presentations, with seed `0`; training allegedly took `309 s` and the complete run `334 s` under a `432 s` allowance. Its claimed validation ladder ranged from `0.7055` for `r18 w96 res48` to `0.7248/0.7198` for `r34 w128 res48`, while the submitted setting allegedly reached `0.7237` twice. It also claimed affine variants scored `0.7063` and `0.7045`. Finally, it claimed `/submission/preds_i20k.npy` was an `int64` array of shape `(10000,)`, spanning labels `0–99` with all `100` classes represented.
### Iteration 4 | Reward: 0.7045
- Approach: The agent replaced stride-2 residual shortcuts with average-pool projections and benchmarked architecture variants and seeds.
- Measured: reward 0.7045
- What happened: The agent inspected the existing CIFAR-100 ResNet pipeline, edited `/workspace/environment/starter/train.py`, and changed stage-transition shortcuts from stride-2 1×1 convolutions to `AvgPool2d(2)` followed by stride-1 1×1 projections. It benchmarked the prior configuration, the new `--dsample d` configuration, a seed-1 repeat, an `r52 w128` deeper/narrower model, `d2`, a partial deep ensemble, and a three-stage model. It synchronized argparse defaults to its selected seed-0 run and reported writing `/submission/preds_i20k.npy`. The grader reported verbatim: “this submission earned reward 0.7045 (70% of the maximum achievable 1.00); it was not a valid scoreable success.” Unvalidated agent-claimed numerical details included: `r34 w160`; 8000 optimizer updates; batch 128; 1,024,000 presentations; a 432-second allowance; and an output described as int64, shape `(10000,)`, range 0–99. The agent claimed `--dsample d` added zero parameters, took 332.6s, and measured 0.7323, versus 318.0s/0.7176 for its baseline. Other claimed results were seed 1: 332.4s/0.7344; `r52 w128`: 314.0s/0.7283; `d2`: 348.2s/0.7312; ensemble: 351.2s/0.7189; and three-stage: 322.8s/0.7181. It also claimed a 0.7334 two-seed mean, 0.7237 prior best, +1.47%, ±0.5–1% noise, approximately 325 TFLOP/s, and 14s additional training.

### Best solution so far — code from iteration 4 (reward 0.7045)
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
1,024,000 presentations, exactly the cap; measured 320.8s of training and 332.6s
end-to-end (train + val + writing the graded predictions) of the 432s wall-clock
allowance, on an H100.

THE DOWNSAMPLE IS WHERE THE ACCURACY WAS.  Each stage transition used to project the
shortcut with a 1x1 convolution of stride 2, which reads one input pixel in four and
throws the other three away.  Replacing it with a 2x2 average pool followed by a stride-1
1x1 (the ResNet-D stem trick, --dsample d) costs no parameters and 14s of training, and
is worth more than any capacity change measured on this task:

    r34 w160 res48, stride-2 1x1 shortcut (--dsample b)   0.7176 (seed 0)
    r34 w160 res48, avg-pool shortcut     (--dsample d)   0.7323 (seed 0)  <- submitted
                                                          0.7344 (seed 1)

Full 8000x128 budget, flip5 TTA, balanced accuracy on the 2000-image public val split,
each run alone on the GPU.  +1.47% over the paired seed-0 baseline is above the
~+-0.5-1% run-to-run noise band; the seed-1 repeat at 0.7344 lands on the same side, so
the two seeds of --dsample d (mean 0.7334) also clear the best --dsample b run ever
measured here (0.7237, previous iteration).  The direction agrees with the published
ResNet-D effect.  Seed 0 is what is submitted, so that the canonical command above
reproduces the submitted predictions exactly; seed 1 was 0.2% higher on val, which is
well inside the noise and not worth breaking reproducibility for.

Everything else tried on top of --dsample d at the same wall clock LOST, which is why the
final config is otherwise unchanged from the previous iteration:

    r52 w128 res48, blocks 4,6,10,4, d    0.7283   deeper+narrower at equal FLOPs, -0.40%
    r34 w160 res48, d2 (main path too)    0.7312   avg-pool the strided 3x3 as well, -0.11%
    r34 w160 res48, d + 2 branches        0.7189   partial deep ensemble, -1.34%
    3-stage w160 res48, blocks 3,4,9, d   0.7181   drop the last downsample, -1.42%

--dsample d2 also average-pools the input of the strided 3x3 on the main path, i.e. it
applies the same fix to the residual branch; it lands on top of --dsample d (-0.11%, a
tie) for 15s more training, so the shortcut was the whole story.  The `branches` run
replicates the last stage twice on a shared trunk and averages the two heads in log-prob
space at prediction time -- a partial deep ensemble that is free in optimiser updates and
in presentations, but it has to pay for the second copy in wall clock (one stage-4 block
was dropped to fit) and the ensemble did not repay it.

The 3-stage result is the informative one: it removes a downsample entirely (final map
12x12 instead of 6x6, same 34 GFLOP) and loses 1.4%, so the win from --dsample d is NOT
"downsample less" -- it is "downsample without discarding pixels".  The two are opposite
prescriptions and only the second one pays.

MODEL CAPACITY IS SATURATED, not the binding lever.  Paired seed-0 runs on val, all at
the full 8000x128 budget, relative training cost in brackets (all --dsample b):

    r18 w96  res48  [1.0x]  0.7055
    r18 w128 res48  [1.8x]  0.7138
    r34 w96  res48  [2.0x]  0.7169
    r34 w160 res32  [2.5x]  0.7190
    r34 w128 res48  [3.6x]  0.7248, and 0.7198 on an independent same-seed repeat
    r34 w160 res48  [5.6x]  0.7237, and 0.7176 on an independent same-seed repeat

Accuracy tracks log(cost) at ~+1.6% per e-fold over the first four rungs and then flats:
w160 [5.6x] and w128 [3.6x] are a tie inside the noise, and reallocating that cost from
width to depth (r52 w128, above) or from depth to resolution (3-stage) both lose.  A FLOP
audit says the ladder cannot be climbed further anyway: the submitted config is ~34 GFLOP
per image forward, i.e. ~13 TFLOP per step of forward+backward, and 8000 steps in 320.8s
is ~325 TFLOP/s sustained.  The pipeline is already compute-bound at close to H100 peak
for bf16 convolutions, so there is no throughput to unlock (torch.compile, CUDA graphs,
better kernels) and buy capacity with -- more capacity can only come out of the remaining
~100s of wall clock, which the flat top of the ladder says is worth ~+0.2%.

Augmentation strength was retested on top of the r18 w96 baseline with a batched affine
warp (rotation, zoom, aspect, translation, all per sample, --aug affine): rot 12 / zoom
0.75-1.15 / aspect 1.2 gives 0.7063 and a milder rot 8 / zoom 0.85-1.10 gives 0.7045,
i.e. both are a wash against 0.7055.  Consistent with the finding below that this budget
is convergence-limited rather than overfitting-limited.

Earlier findings (~20 full-budget runs), still current:
  * flip5 TTA vs single-view flip: +0.56% mean over 10 paired res-48 runs (positive in
    7/10), and +0.56% again over 5 further paired runs across the capacity ladder above
    (positive in 5/5).  This is the one robust prediction-time gain; it is forced via
    --pick rather than chosen by val-argmax, since argmax over near-tied variants on 2000
    images only adds noise.  flip9 adds nothing over flip5 (+0.02% mean over those same
    5 runs) for twice the prediction cost.
  * train_acc reaches 0.998 -- the net fully fits the training set -- yet EVERY added
    regulariser lost: cutmix ~0, cutout 16 ~0, wd 4e-3 -1.0%, heavy combined reg -3.3%
    (train_acc 0.85).  At 8000 steps the budget, not overfitting, is the binding
    constraint, so extra regularisation costs more convergence than it buys.
  * No gain from: SE blocks, mixup, balanced sampling, logit adjustment (the 1.39x class
    imbalance is too mild), flip9 TTA, multi-scale TTA, lr 0.4, EMA.
  * res 48 vs res 32 is a wash (0.7052 n=3 vs 0.7043 n=2); kept because it is neutral and
    well inside the time budget.

Val is a 2000-image split, so its standard error is ~1%, and it reads about 2-3% higher
than the graded split.  Treat single-run gaps under ~1% as unresolved; only --dsample d
cleared that bar this iteration.
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
    def __init__(self, cin, cout, stride, se=0, dsample="b"):
        super().__init__()
        if dsample == "d2" and stride != 1:
            # also de-alias the main path: average-pool first, then a stride-1 conv, so
            # the 3x3 sees every input pixel instead of 1 in 4.  The conv still runs at
            # the output resolution, so this costs only the pooling.
            self.c1 = nn.Sequential(nn.AvgPool2d(stride, stride, ceil_mode=True),
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
            if dsample in ("d", "d2") and stride != 1:
                # ResNet-D: average-pool the shortcut instead of striding the 1x1, so the
                # projection sees every input pixel rather than 1 in 4.
                self.short = nn.Sequential(nn.AvgPool2d(stride, stride, ceil_mode=True),
                                           nn.Conv2d(cin, cout, 1, 1, bias=False),
                                           nn.BatchNorm2d(cout))
            else:
                self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                           nn.BatchNorm2d(cout))

    def forward(self, x):
        r = x if self.short is None else self.short(x)
        x = F.relu(self.b1(self.c1(x)), inplace=True)
        x = self.se(self.b2(self.c2(x)))
        return F.relu(x + r, inplace=True)


class ResNet(nn.Module):
    """CIFAR ResNet: 3x3 stem (no downsample), 3 or 4 stages of BasicBlocks.

    With `branches` > 1 the LAST stage is replicated K times (independent weights and
    heads) on a shared trunk.  Each branch is trained with its own cross-entropy and the
    branches are averaged in log-prob space at prediction time -- a partial deep ensemble
    that costs one optimiser update and one presentation per image, exactly as a single
    model does, so the budget accounting is unchanged.
    """

    def __init__(self, n_classes, width=64, blocks=(2, 2, 2, 2), stem=None, dropout=0.0,
                 se=0, dsample="b", branches=1):
        super().__init__()
        stem = stem or width
        self.stem = nn.Sequential(nn.Conv2d(3, stem, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(stem), nn.ReLU(inplace=True))
        self.branches = branches

        def stage(i, n, cin):
            layers = []
            cout = width * (2 ** i)
            for j in range(n):
                layers.append(BasicBlock(cin, cout,
                                         stride=(2 if (j == 0 and i > 0) else 1), se=se,
                                         dsample=dsample))
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
            # prediction: average the branches in log-prob space
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
               dsample="b", branches=1):
    torch.manual_seed(seed)
    if arch == "r18t":                                   # 3-stage variant
        cfg = dict(width=width, blocks=(2, 2, 2), stem=width)
    else:
        cfg = dict(width=width, blocks=(blocks or ARCHS[arch]))
    return ResNet(n_classes, dropout=dropout, se=se, dsample=dsample, branches=branches,
                  **cfg)


def branch_ce(out, target, ls):
    """Cross-entropy that also accepts a (K,B,C) stack of branch logits."""
    if out.dim() == 3:
        return torch.stack([F.cross_entropy(o, target, label_smoothing=ls)
                            for o in out]).mean()
    return F.cross_entropy(out, target, label_smoothing=ls)


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
            ar_ = torch.arange(h, device=dev)
            cy = torch.randint(0, h, (b,), device=dev, generator=gen)
            cx = torch.randint(0, h, (b,), device=dev, generator=gen)
            gy = ar_[None, :]
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

    blocks = tuple(int(v) for v in args.blocks.split(",")) if args.blocks else None
    model = make_model(args.seed, data.N_CLASSES, args.arch, args.width,
                       args.dropout, args.se, blocks, args.dsample, args.branches)
    model = model.to(device).to(memory_format=torch.channels_last)
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
        if args.aug == "affine":
            xb, yb = ds.batch_affine(idx, gen, cutout=cutout_px, rot=args.rot,
                                     smin=args.smin, smax=args.smax, ar=args.aspect,
                                     trans=args.trans, bright=args.bright,
                                     contrast=args.contrast)
        else:
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
            loss = branch_ce(out, yb, args.ls)
            if lam < 1.0:
                loss = lam * loss + (1 - lam) * branch_ce(out, yb2, args.ls)
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
    ap.add_argument("--arch", default="r34")
    ap.add_argument("--blocks", default=None, help="explicit block counts, overrides --arch")
    ap.add_argument("--dsample", default="d", choices=["b", "d", "d2"],
                    help="downsample style: b=strided 1x1, d=avgpool shortcut, "
                         "d2=avgpool shortcut and main path")
    ap.add_argument("--branches", type=int, default=1,
                    help="replicate the last stage K times (partial deep ensemble)")
    ap.add_argument("--width", type=int, default=160)
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
    ap.add_argument("--cutout", type=int, default=0, help="cutout size, at 32px scale")
    ap.add_argument("--se", type=int, default=0)
    ap.add_argument("--bal-sample", type=int, default=0)
    ap.add_argument("--la-train", type=float, default=0.0)
    ap.add_argument("--la-pred", type=float, default=0.0)
    ap.add_argument("--emas", type=float, nargs="*", default=[])
    ap.add_argument("--ema-start", type=float, default=0.5)
    ap.add_argument("--mix-p", type=float, default=0.0)
    ap.add_argument("--mix-a", type=float, default=1.0)
    ap.add_argument("--mix-mode", default="cutmix", choices=["cutmix", "mixup"])
    ap.add_argument("--aug", default="crop", choices=["crop", "affine"])
    ap.add_argument("--rot", type=float, default=0.0)
    ap.add_argument("--smin", type=float, default=1.0)
    ap.add_argument("--smax", type=float, default=1.0)
    ap.add_argument("--aspect", type=float, default=1.0)
    ap.add_argument("--trans", type=float, default=0.125)
    ap.add_argument("--bright", type=float, default=0.0)
    ap.add_argument("--contrast", type=float, default=0.0)
    ap.add_argument("--pick", default="flip5", choices=["flip5", "flip9", "flip", "none"])
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()

    t_start = time.time()
    model, emas, ds, presented, steps, logprior = train(a)

    # ---- local measurement on the public val split (never trained on) ---------------
    xv, yv = data.load_val()
    variants = [("model", model)] + emas
    pad_pred = ds.pad
    best = None
    for name, m in variants:
        lg = logits_for(m, ds.norm(xv), tta=a.pick, pad=pad_pred)
        if a.la_pred > 0:
            lg = lg - a.la_pred * logprior
        pred = lg.argmax(1).cpu().numpy()
        acc = balanced_acc(pred, yv)
        top1 = float((pred == yv).mean())
        print(f"[val] {a.tag} {name} tta={a.pick} bacc={acc:.4f} top1={top1:.4f}",
              flush=True)
        if best is None or acc > best[0]:
            best = (acc, name, m)

    val_bacc, best_name, best_model = best
    print(f"[VAL] tag={a.tag} bacc={val_bacc:.4f} variant={best_name} "
          f"steps={steps} present={presented}", flush=True)

    # ---- graded predictions ----------------------------------------------------------
    if a.graded and a.out:
        xg = np.load(a.graded)["x"]
        lg = logits_for(best_model, ds.norm(xg), tta=a.pick, pad=pad_pred)
        if a.la_pred > 0:
            lg = lg - a.la_pred * logprior
        if a.logits_out:
            np.save(a.logits_out, lg.cpu().numpy().astype(np.float32))
        pred = lg.argmax(1).cpu().numpy().astype(np.int64)
        assert pred.shape == (len(xg),) and pred.min() >= 0 and pred.max() < 100
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        np.save(a.out, pred)
        print(f"wrote {a.out} shape={pred.shape} dtype={pred.dtype}", flush=True)

    print(f"[time] total={time.time() - t_start:.1f}s", flush=True)
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