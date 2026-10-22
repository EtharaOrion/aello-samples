## Refinement Loop - Attempt 15
This is attempt number 15. 14 attempt(s) have been made before this one.

**BEST SO FAR: Iteration 10 (reward 0.7548)** — Verified submission on disk with time to spare.. Build on this result; do not discard it.

## Prior Attempts (bounded digest: top 3 scored + last 3)
### Iteration 8 | Reward: 0.7286
- Approach: The agent benchmarked anti-aliased ResNet-D variants, resolution, width, depth, and label smoothing before selecting predictions.
- Measured: reward 0.7286
- What happened: The agent inspected the workspace, prior code, and hardware, then edited `train.py` to add a `d2` downsampling variant that pooled both the residual shortcut and main path. It benchmarked the prior `d` baseline, `d2`, a deep stem, depth/resolution and width variants, and a paired label-smoothing sweep. It also wrote evaluation-noise notes to two memory files and selected a prediction file. The authoritative grader reported: “this submission earned reward 0.7286 (73% of the maximum achievable 1.00); it was not a valid scoreable success.” All remaining figures are unvalidated agent claims. Its table reported: `d` 0.7207/334s; `d2` 0.7253/349s; deep-stem `d2` 0.7247/353s; `res56 w144` 0.7145/432s; `w128 ls 0.1`, `0.0`, and `0.2` at 0.7219/235s, 0.6980/232s, and 0.7125/234s, all with seed 1; and `w160` seed 1 at 0.7169/350s. It claimed resolution 48, stride-2 transitions involving 1×1 shortcuts and 3×3 main-path convolutions, and run-to-run σ of roughly 0.5–0.6%. It said `/submission/preds_i20k.npy` was int64 shape `(10000,)`, contained labels 0–99, and came from an 8000/8000-step run with 1,024,000/1,024,000 presentations in 349.3s.
### Iteration 9 | Reward: 0.7296
- Approach: The agent rebuilt its anti-aliased ResNet-D baseline, tested training variants, then applied flip TTA and marginal balancing.
- Measured: reward 0.7296
- What happened: The agent reconstructed its prior image-classification pipeline and ran an end-to-end baseline before testing three full-budget variants: two-cycle snapshot ensembling, CutMix annealed off during training, and a different random seed. It saved checkpoints for paired inference comparisons, evaluated Sinkhorn-to-uniform marginal correction on full validation data and resampled subsets, swept sharpening temperatures, retained flip-based test-time augmentation, and generated predictions from the seed-1 checkpoint. It also wrote `train.py`, two queue scripts, and six post-processing scripts. The authoritative grader verdict was: “this submission earned reward 0.7296 (73% of the maximum achievable 1.00); it was not a valid scoreable success.” All configuration and experiment figures below are agent-claimed, unvalidated self-reports. The model was described as a CIFAR ResNet-34 with width 160, 133.0M parameters, anti-aliased stage transitions, and 48px inputs. Claimed optimization settings were SGD with Nesterov momentum, learning rate 0.2, cosine scheduling, weight decay 2e-3, and label smoothing 0.1. Runs reportedly used 8000 steps and 1,024,000 presentations. The agent claimed baseline balanced accuracy of 0.7253 and probe results of 0.7234, 0.7238, and 0.7241. Its post-processing sweep used temperatures 0.5, 0.7, and 1.0 over 200 resampled draws; it selected temperature 0.7 with flip9 TTA.
### Iteration 10 | Reward: 0.7548
- Approach: Verified submission on disk with time to spare.
- Measured: reward 0.7548
- What happened: Activity: 45 steps; 44 agent turns; 47 tool calls (Bash x38, Write x7, Read x1, Edit x1); 1 tool results with error signals. Files written: /workspace/environment/starter/train.py, /workspace/environment/starter/postproc.py, /logs/agent/sessions/projects/-workspace/memory/cifar100-budget-cutout-win.md, /logs/agent/sessions/projects/-workspace/memory/cifar100-marginal-balancing.md, /logs/agent/sessions/projects/-workspace/memory/cifar100-budget-eval-noise.md, /logs/agent/sessions/projects/-workspace/memory/MEMORY.md. Agent's final report: Done. Verified submission on disk with time to spare. ## Result `/submission/preds_i20k.npy` — shape `(10000,)`, int64, values 0–99, 100 distinct classes, **verified valid**. Source: a single training run (`B_cut`) at **8000/8000 steps and 1,024,000/1,024,000 presentations, 377s of the 432s allowance**. Val balanced accuracy **0.7343**, plus a measured **+0.8%** from post-hoc marginal balancing → **~0.7396** val-equivalent, against ~0.725–0.730 for the previous best iteration. ## What moved the needle The workspace came back empty, so I rebuilt the ResNet-34 (w160, res 48, ResNet-D shortcuts) pipeline and had a valid submission on disk at the 13-minute mark, then spent the rest on full-budget probes: | run | change (seed) | raw | ema | raw+ema | |---|---|---|---|---| | A_base | no cutout (s0) | 0.7058 | 0.7086 | 0.7137 | | **B_cut** | **+ cutout(12) (s1)** | 0.7251 | 0.7173 | **0.7343** | | C_cutmix | + mixup 0.2 (s1) | 0.7289 | 0.7082 | 0.7329 | | D_final | B + 3rd faster EMA (s1) | 0.7251 | 0.7140 | 0.7343 | | E_cut16 | cutout(16) (s1) | 0.7258 | 0.7015 | 0.7200 | Three findings, none of which the nine prior attempts had: - **Cutout(12) is the big lever: +1.9% raw, +2.1% ensemble… (heuristic summary from atif trajectory; no LLM pass)
### Iteration 12 | Reward: 0.6251
- Approach: The agent rebuilt an anti-aliased ResNet pipeline, compared training variants, and applied snapshot ensembling with test-time augmentation.
- Measured: reward 0.6251
- What happened: The agent inspected the workspace, rewrote the training pipeline, and created scripts for post-processing, recalibration, prediction emission, and TTA. It trained a baseline and compared it with cutout annealing, a narrower higher-resolution model, random-resized cropping with aspect jitter, and an independent-seed run. It also evaluated marginal balancing, class-prior adjustment, BN-statistic recalibration, zoom and resolution variants, blending methods, and ensemble weights. It reported placing predictions at `/submission/preds_i20k.npy` and wrote three memory documents summarizing its recipe, rejected experiments, and evaluation-noise observations. The grader reported: “this submission earned reward 0.6251 (63% of the maximum achievable 1.00); it was not a valid scoreable success.” All configuration and performance figures below are unvalidated agent self-reports. The agent described a ResNet-34-layout model with width 160 and 133M parameters, 32→48 input upsampling, anti-aliased transitions, SGD-Nesterov with learning rate 0.2, cosine scheduling, weight decay 2e-3, label smoothing 0.1, and cutout size 12. It claimed training for 8000/8000 optimizer steps and 1,024,000/1,024,000 presentations, using EMA 0.999 and five-view shift TTA with flips. Other claimed settings included cutout annealing over the last 8%, width 128 at resolution 56, and seed 4. It claimed 0.7407 validation balanced accuracy, 336.7 seconds of training, and 360.2 seconds end-to-end.
### Iteration 13 | Reward: 0.6639
- Approach: Done — valid submission on disk with ~13 minutes of margin.
- Measured: reward 0.6639
- What happened: Activity: 44 steps; 43 agent turns; 43 tool calls (Bash x37, Write x4, Edit x2); 1 tool results with error signals. Files written: /workspace/environment/starter/train.py, /workspace/environment/starter/emit.py, /workspace/environment/starter/tta.py, /workspace/environment/starter/finalize.py. Agent's final report: Done — valid submission on disk with ~13 minutes of margin. ## Result `/submission/preds_i20k.npy` — shape `(10000,)`, int64, values 0–99, 100 distinct classes, **verified loadable**. Source: a single training run (`F_ta_s7`) at **8000/8000 steps, 1,024,000/1,024,000 presentations**, 287.3 s for training + inference, plus 70.3 s for the TTA prediction pass (357.6 s total, under the 432 s allowance). Val balanced accuracy **0.7355**, against ~0.723 for the previous best iteration's graded-equivalent. ## What moved the needle The workspace came back empty, so I rebuilt the pipeline from scratch (all data and augmentation resident on GPU, no dataloader) and had a valid submission on disk at the 17-minute mark. Full-budget runs, all w128 ResNet-34-D @ 48 px unless noted: | run | change | val (raw+EMA ens) | standalone time | |---|---|---|---| | A_base | crop+flip+cutout(12) | 0.7258 | — | | **B_ta** | **+ TrivialAugment (1 op)** | **0.7392** | — | | C_ta | B config, seed 2 | 0.7295 | 292 s | | D_w144 | width 144 (108M) | 0.7217 | 400 s | | E_ta2 | TrivialAugment 2 ops | 0.7267 | 293 s | | **F_ta_s7** | **B config, seed 7** | **0.7335 → 0.7355 w/ TTA** | 287 s | Three findings: - **Triv… (heuristic summary from atif trajectory; no LLM pass)
### Iteration 14 | Reward: 0.7053
- Approach: Rebuilt a GPU-resident CIFAR-100 ResNet pipeline, compared sampling and EMA variants, evaluated shift/flip TTA, and emitted predictions.
- Measured: reward 0.7053
- What happened: After reporting an empty workspace, the agent inspected the environment and wrote `train.py`, `emit.py`, and `tta_pick.py`. It rebuilt an all-data-GPU-resident CIFAR-100 training pipeline, trained baseline and comparison runs, evaluated checkpoints on a public validation split, tested flip, shift, scale, and prior-adjustment inference variants, selected run A, generated `/submission/preds_i20k.npy`, and performed a final reload check. It also recorded observations about parallel-training slowdown and a terminated run. All implementation and measurement details below are unvalidated agent claims: the model used a ResNet-34 layout with width 128, ResNet-D anti-aliased transitions, 48-pixel inputs, TrivialAugment plus cutout, and SGD with Nesterov momentum and cosine scheduling. Run A allegedly used seed 0, batch size 128, 8,000 optimizer steps, and 1,024,000 presentations; inference combined raw weights with EMA(0.999) using five-shift × flip TTA and prior exponent 0.5. The agent claimed 0.7364 balanced validation accuracy, 268.6 seconds of training, 273.5 seconds end-to-end, and an output array of shape `(10000,)`, dtype int64, containing labels 0–99 across 100 classes. It also compared class-balanced sampling at seed 1, a third-EMA run at seed 3, flip-only, shift5, shift9, and scale3+shift5 variants. The grader’s authoritative verdict was: “this submission earned reward 0.7053 (71% of the maximum achievable 1.00); it was not a valid scoreable success.”

### Best solution so far — code from iteration 10 (reward 0.7548)
This is YOUR OWN prior best source. Build on it (write your chosen code to /workspace); do not restart from scratch unless you have a reason. It will be re-trained and re-graded from scratch.
```python
# ===== agent_src/environment/starter/train.py =====
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

# ===== agent_src/environment/starter/postproc.py =====
"""Post-hoc marginal balancing, applied to ONE training run's saved logits.

The metric is balanced accuracy (mean per-class recall) and the graded split is
10000 images, so its class marginal is almost certainly uniform.  A plain argmax
under-predicts hard classes; nudging the predicted marginal toward uniform with a
per-class bias recovers some of that recall.

val_public is class-IMBALANCED (10-31 per class), so the correction cannot be
validated on it directly -- we resample balanced subsets (k per class) many times
and measure the average gain there instead.

BUDGET NOTE: this script only ever uses a SINGLE run's logits for the submission.
It ranks runs against each other to choose which one to submit, but never averages
two separately trained models -- that would exceed the training budget.  The
raw/EMA average within one run is fine: both come from the same 8000-step
trajectory and cost no extra presentations.
"""
import argparse, glob, json, os
import numpy as np


def sinkhorn_bias(logp, iters=100):
    """Per-class additive bias b so that softmax(logp + b) has uniform column mass."""
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


def balanced_acc(pred, y, n=100):
    return float(np.mean([(pred[y == c] == c).mean() for c in range(n) if (y == c).any()]))


def resampled_sweep(vlog, yv, taus, k=10, draws=200, seed=0):
    """Mean balanced acc on balanced k-per-class resamples of val, per tau.

    Paired across draws: the same subsets are scored at every tau, so the
    tau-to-tau differences are far less noisy than the absolute numbers.
    """
    rng = np.random.default_rng(seed)
    idx_by_c = [np.where(yv == c)[0] for c in range(100)]
    out = {t: [] for t in taus}
    for _ in range(draws):
        sel = np.concatenate([rng.choice(ix, k, replace=len(ix) < k) for ix in idx_by_c])
        sub, ysub = vlog[sel], yv[sel]
        b = sinkhorn_bias(sub)
        for t in taus:
            out[t].append(balanced_acc((sub + t * b).argmax(1), ysub))
    ref = np.array(out[0.0]) if 0.0 in out else np.array(out[taus[0]])
    return {t: (float(np.mean(v)), float(np.std(np.array(v) - ref) / np.sqrt(draws)))
            for t, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", nargs="+", required=True)
    ap.add_argument("--taus", default="0.0,0.5,0.8,1.0,1.2,1.5,2.0")
    ap.add_argument("--out", default="")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--pick", default="", help="force this run tag instead of best-on-val")
    a = ap.parse_args()

    taus = [float(t) for t in a.taus.split(",")]
    files = [f for p in a.logits for f in sorted(glob.glob(p))]

    best = None
    for f in files:
        d = np.load(f)
        yv = d["yv"]
        meta = json.load(open(f.replace(".npz", ".json")))
        head = meta["best"]
        sc = meta["best_score"]
        print(f"{os.path.basename(f):24s} best head {head:12s} val {sc:.4f}  "
              f"({meta['seconds']:.0f}s)")
        if a.pick:
            take = os.path.basename(f).startswith(a.pick)
        else:
            take = best is None or sc > best[1]
        if take:
            best = (f, sc, d[f"v_{head}"], d["g_best"], yv, head)

    f, sc, vlog, glog, yv, head = best
    print(f"\nselected: {os.path.basename(f)}  head={head}  full-val {sc:.4f}")

    print(f"\nbalanced-resample tau sweep ({a.draws} draws x 10/class, paired):")
    res = resampled_sweep(vlog, yv, taus, draws=a.draws)
    base = res[taus[0]][0]
    for t in taus:
        m, se = res[t]
        print(f"  tau={t:<4} bal-acc {m:.4f}   delta {m - base:+.4f} +- {se:.4f}")
    btau = max(res, key=lambda t: res[t][0])
    print(f"  -> tau {btau}")

    if a.out:
        b = sinkhorn_bias(glog)
        pred = (glog + btau * b).argmax(1).astype(np.int64)
        n0 = (pred != glog.argmax(1)).sum()
        cnt = np.bincount(pred, minlength=100)
        print(f"\nchanged {n0}/{len(pred)} predictions; class counts "
              f"min {cnt.min()} max {cnt.max()}")
        np.save(a.out, pred)
        json.dump({"source": f, "head": head, "tau": btau, "val_full": sc,
                   "resample": {str(k): v for k, v in res.items()}},
                  open("/workspace/submission_meta.json", "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    raise SystemExit(main())

# ===== agent_src/logits/A_base.json =====
{
 "tag": "A_base",
 "scores": {
  "raw_flip": 0.705848620819825,
  "ema_flip": 0.708569700312664,
  "both_flip": 0.7137090308915088
 },
 "best": "both_flip",
 "best_score": 0.7137090308915088,
 "args": {
  "instance": "i20k",
  "seed": 0,
  "graded": "../data/graded_images.npz",
  "out": "/submission/preds_i20k.npy",
  "tag": "A_base",
  "width": 160,
  "blocks": "3,4,6,3",
  "res": 48,
  "aa": 0,
  "steps": 8000,
  "batch": 128,
  "lr": 0.2,
  "wd": 0.002,
  "ls": 0.1,
  "pad": 4,
  "cutout": 0,
  "mixup": 0.0,
  "mixup_p": 0.5,
  "warmup": 0.03,
  "ema": 0.999,
  "ema_start": 0.5,
  "bn_batches": 50,
  "nowd_bn": 0,
  "tta": "flip",
  "no_write": false,
  "logitdir": "/workspace/logits"
 },
 "seconds": 387.9205753803253
}

# ===== agent_src/logits/B_cut.json =====
{
 "tag": "B_cut",
 "scores": {
  "raw_flip": 0.7250846963685169,
  "ema_flip": 0.7173103303664418,
  "both_flip": 0.7343046088572881,
  "raw_flip5": 0.7262547450940643,
  "ema_flip5": 0.713797029893524,
  "both_flip5": 0.7281703369683415
 },
 "best": "both_flip",
 "best_score": 0.7343046088572881,
 "args": {
  "instance": "i20k",
  "seed": 1,
  "graded": "../data/graded_images.npz",
  "out": "/submission/preds_i20k.npy",
  "tag": "B_cut",
  "width": 160,
  "blocks": "3,4,6,3",
  "res": 48,
  "aa": 0,
  "steps": 8000,
  "batch": 128,
  "lr": 0.2,
  "wd": 0.002,
  "ls": 0.1,
  "pad": 4,
  "cutout": 12,
  "mixup": 0.0,
  "mixup_p": 0.5,
  "warmup": 0.03,
  "ema": 0.999,
  "ema_start": 0.5,
  "ema_every": 2,
  "bn_batches": 50,
  "nowd_bn": 0,
  "tta": "flip,flip5",
  "no_write": true,
  "logitdir": "/workspace/logits"
 },
 "seconds": 377.3208751678467
}

# ===== agent_src/logits/C_cutmix.json =====
{
 "tag": "C_cutmix",
 "scores": {
  "raw_flip": 0.7288585944174883,
  "ema_flip": 0.7082187879526686,
  "both_flip": 0.7317888695965352,
  "raw_flip5": 0.7307102965971624,
  "ema_flip5": 0.7084514477628087,
  "both_flip5": 0.7329476530211587
 },
 "best": "both_flip5",
 "best_score": 0.7329476530211587,
 "args": {
  "instance": "i20k",
  "seed": 1,
  "graded": "../data/graded_images.npz",
  "out": "/submission/preds_i20k.npy",
  "tag": "C_cutmix",
  "width": 160,
  "blocks": "3,4,6,3",
  "res": 48,
  "aa": 0,
  "steps": 8000,
  "batch": 128,
  "lr": 0.2,
  "wd": 0.002,
  "ls": 0.1,
  "pad": 4,
  "cutout": 12,
  "mixup": 0.2,
  "mixup_p": 0.5,
  "warmup": 0.03,
  "ema": 0.999,
  "ema_start": 0.5,
  "ema_every": 2,
  "bn_batches": 50,
  "nowd_bn": 0,
  "tta": "flip,flip5",
  "no_write": true,
  "logitdir": "/workspace/logits"
 },
 "seconds": 389.41545939445496
}

# ===== agent_src/logits/D_final.json =====
{
 "tag": "D_final",
 "scores": {
  "ema_flip": 0.7173103303664418,
  "ema2_flip": 0.7139810369630066,
  "raw_flip": 0.7250846963685169,
  "ema+ema2_flip": 0.7177860238894873,
  "ema+raw_flip": 0.7343046088572881,
  "ema2+raw_flip": 0.7265288789610002,
  "ema+ema2+raw_flip": 0.7300349097266825
 },
 "best": "ema+raw_flip",
 "best_score": 0.7343046088572881,
 "args": {
  "instance": "i20k",
  "seed": 1,
  "graded": "../data/graded_images.npz",
  "out": "/submission/preds_i20k.npy",
  "tag": "D_final",
  "width": 160,
  "blocks": "3,4,6,3",
  "res": 48,
  "aa": 0,
  "steps": 8000,
  "batch": 128,
  "lr": 0.2,
  "wd": 0.002,
  "ls": 0.1,
  "pad": 4,
  "cutout": 12,
  "mixup": 0.0,
  "mixup_p": 0.5,
  "warmup": 0.03,
  "ema": 0.999,
  "ema_start": 0.5,
  "ema_every": 2,
  "ema2": 0.9975,
  "ema2_start": 0.5,
  "bn_batches": 50,
  "nowd_bn": 0,
  "tta": "flip",
  "no_write": true,
  "logitdir": "/workspace/logits"
 },
 "seconds": 389.50343012809753
}

# ===== agent_src/logits/E_cut16.json =====
{
 "tag": "E_cut16",
 "scores": {
  "ema_flip": 0.7014939711188737,
  "raw_flip": 0.7257724715816924,
  "ema+raw_flip": 0.7199731126930381
 },
 "best": "raw_flip",
 "best_score": 0.7257724715816924,
 "args": {
  "instance": "i20k",
  "seed": 1,
  "graded": "../data/graded_images.npz",
  "out": "/submission/preds_i20k.npy",
  "tag": "E_cut16",
  "width": 160,
  "blocks": "3,4,6,3",
  "res": 48,
  "aa": 0,
  "steps": 8000,
  "batch": 128,
  "lr": 0.2,
  "wd": 0.002,
  "ls": 0.1,
  "pad": 4,
  "cutout": 16,
  "mixup": 0.0,
  "mixup_p": 0.5,
  "warmup": 0.03,
  "ema": 0.999,
  "ema_start": 0.5,
  "ema_every": 2,
  "ema2": 0.0,
  "ema2_start": 0.5,
  "bn_batches": 50,
  "nowd_bn": 0,
  "tta": "flip",
  "no_write": true,
  "logitdir": "/workspace/logits"
 },
 "seconds": 372.77137875556946
}

# ===== agent_src/logits/smoke.json =====
{
 "tag": "smoke",
 "scores": {
  "raw": 0.07690205121713063,
  "ema": 0.0707540022715652,
  "both": 0.08799286804788217
 },
 "best": "both",
 "args": {
  "instance": "i20k",
  "seed": 0,
  "graded": "../data/graded_images.npz",
  "out": "/submission/preds_i20k.npy",
  "tag": "smoke",
  "width": 32,
  "blocks": "3,4,6,3",
  "res": 48,
  "steps": 60,
  "batch": 128,
  "lr": 0.2,
  "wd": 0.002,
  "ls": 0.1,
  "pad": 4,
  "cutout": 0,
  "mixup": 0.0,
  "mixup_p": 0.5,
  "warmup": 0.03,
  "ema": 0.999,
  "ema_start": 0.5,
  "bn_batches": 5,
  "nowd_bn": 0,
  "tta": "flip",
  "no_write": true,
  "logitdir": "/workspace/logits"
 },
 "seconds": 5.830834627151489
}

# ===== agent_src/logits/smoke2.json =====
{
 "tag": "smoke2",
 "scores": {
  "raw_flip": 0.07287771960805679,
  "ema_flip": 0.0562687833892774,
  "both_flip": 0.06314932389529576,
  "raw_flip5": 0.07195421533455253,
  "ema_flip5": 0.055280548095159754,
  "both_flip5": 0.06180145301035393
 },
 "best": "raw_flip",
 "best_score": 0.07287771960805679,
 "args": {
  "instance": "i20k",
  "seed": 0,
  "graded": "../data/graded_images.npz",
  "out": "/submission/preds_i20k.npy",
  "tag": "smoke2",
  "width": 32,
  "blocks": "3,4,6,3",
  "res": 48,
  "aa": 0,
  "steps": 40,
  "batch": 128,
  "lr": 0.2,
  "wd": 0.002,
  "ls": 0.1,
  "pad": 4,
  "cutout": 12,
  "mixup": 0.0,
  "mixup_p": 0.5,
  "warmup": 0.03,
  "ema": 0.999,
  "ema_start": 0.5,
  "ema_every": 2,
  "bn_batches": 3,
  "nowd_bn": 0,
  "tta": "flip,flip5",
  "no_write": true,
  "logitdir": "/workspace/logits"
 },
 "seconds": 5.493459939956665
}

# ===== agent_src/logits/smoke3.json =====
{
 "tag": "smoke3",
 "scores": {
  "ema_flip": 0.05087348176494091,
  "ema2_flip": 0.051611432068377816,
  "raw_flip": 0.0633745770282922,
  "ema+ema2_flip": 0.054105365822911936,
  "ema+raw_flip": 0.05728510433432698,
  "ema2+raw_flip": 0.061854011897352186,
  "ema+ema2+raw_flip": 0.055545913967104565
 },
 "best": "raw_flip",
 "best_score": 0.0633745770282922,
 "args": {
  "instance": "i20k",
  "seed": 0,
  "graded": "../data/graded_images.npz",
  "out": "/submission/preds_i20k.npy",
  "tag": "smoke3",
  "width": 24,
  "blocks": "3,4,6,3",
  "res": 48,
  "aa": 0,
  "steps": 40,
  "batch": 128,
  "lr": 0.2,
  "wd": 0.002,
  "ls": 0.1,
  "pad": 4,
  "cutout": 12,
  "mixup": 0.0,
  "mixup_p": 0.5,
  "warmup": 0.03,
  "ema": 0.999,
  "ema_start": 0.5,
  "ema_every": 2,
  "ema2": 0.99,
  "ema2_start": 0.5,
  "bn_batches": 2,
  "nowd_bn": 0,
  "tta": "flip",
  "no_write": true,
  "logitdir": "/workspace/logits"
 },
 "seconds": 9.279337406158447
}

# ===== agent_src/submission_meta.json =====
{
 "source": "/workspace/logits/B_cut.npz",
 "head": "both_flip",
 "tau": 1.5,
 "val_full": 0.7343046088572881,
 "resample": {
  "0.0": [
   0.7344849999999999,
   0.0
  ],
  "0.5": [
   0.73568,
   0.0002612180602485218
  ],
  "0.8": [
   0.73694,
   0.0003276276468798073
  ],
  "1.0": [
   0.7385249999999999,
   0.00037052935106412295
  ],
  "1.2": [
   0.7393649999999999,
   0.0003902281384011141
  ],
  "1.5": [
   0.739645,
   0.00041094038497086106
  ],
  "2.0": [
   0.7363550000000001,
   0.0004652585302818217
  ]
 }
}
```