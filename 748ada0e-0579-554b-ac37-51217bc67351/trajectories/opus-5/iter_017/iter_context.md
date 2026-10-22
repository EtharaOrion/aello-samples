## Refinement Loop - Attempt 17
This is attempt number 17. 16 attempt(s) have been made before this one.

**BEST SO FAR: Iteration 16 (reward 0.7566)** — The agent rebuilt a GPU-resident CIFAR-100 ResNet pipeline, compared capacity and regularization variants, and generated TTA predictions.. Build on this result; do not discard it.

## Prior Attempts (bounded digest: top 3 scored + last 3)
### Iteration 9 | Reward: 0.7296
- Approach: The agent rebuilt its anti-aliased ResNet-D baseline, tested training variants, then applied flip TTA and marginal balancing.
- Measured: reward 0.7296
- What happened: The agent reconstructed its prior image-classification pipeline and ran an end-to-end baseline before testing three full-budget variants: two-cycle snapshot ensembling, CutMix annealed off during training, and a different random seed. It saved checkpoints for paired inference comparisons, evaluated Sinkhorn-to-uniform marginal correction on full validation data and resampled subsets, swept sharpening temperatures, retained flip-based test-time augmentation, and generated predictions from the seed-1 checkpoint. It also wrote `train.py`, two queue scripts, and six post-processing scripts. The authoritative grader verdict was: “this submission earned reward 0.7296 (73% of the maximum achievable 1.00); it was not a valid scoreable success.” All configuration and experiment figures below are agent-claimed, unvalidated self-reports. The model was described as a CIFAR ResNet-34 with width 160, 133.0M parameters, anti-aliased stage transitions, and 48px inputs. Claimed optimization settings were SGD with Nesterov momentum, learning rate 0.2, cosine scheduling, weight decay 2e-3, and label smoothing 0.1. Runs reportedly used 8000 steps and 1,024,000 presentations. The agent claimed baseline balanced accuracy of 0.7253 and probe results of 0.7234, 0.7238, and 0.7241. Its post-processing sweep used temperatures 0.5, 0.7, and 1.0 over 200 resampled draws; it selected temperature 0.7 with flip9 TTA.
### Iteration 10 | Reward: 0.7548
- Approach: Verified submission on disk with time to spare.
- Measured: reward 0.7548
- What happened: Activity: 45 steps; 44 agent turns; 47 tool calls (Bash x38, Write x7, Read x1, Edit x1); 1 tool results with error signals. Files written: /workspace/environment/starter/train.py, /workspace/environment/starter/postproc.py, /logs/agent/sessions/projects/-workspace/memory/cifar100-budget-cutout-win.md, /logs/agent/sessions/projects/-workspace/memory/cifar100-marginal-balancing.md, /logs/agent/sessions/projects/-workspace/memory/cifar100-budget-eval-noise.md, /logs/agent/sessions/projects/-workspace/memory/MEMORY.md. Agent's final report: Done. Verified submission on disk with time to spare. ## Result `/submission/preds_i20k.npy` — shape `(10000,)`, int64, values 0–99, 100 distinct classes, **verified valid**. Source: a single training run (`B_cut`) at **8000/8000 steps and 1,024,000/1,024,000 presentations, 377s of the 432s allowance**. Val balanced accuracy **0.7343**, plus a measured **+0.8%** from post-hoc marginal balancing → **~0.7396** val-equivalent, against ~0.725–0.730 for the previous best iteration. ## What moved the needle The workspace came back empty, so I rebuilt the ResNet-34 (w160, res 48, ResNet-D shortcuts) pipeline and had a valid submission on disk at the 13-minute mark, then spent the rest on full-budget probes: | run | change (seed) | raw | ema | raw+ema | |---|---|---|---|---| | A_base | no cutout (s0) | 0.7058 | 0.7086 | 0.7137 | | **B_cut** | **+ cutout(12) (s1)** | 0.7251 | 0.7173 | **0.7343** | | C_cutmix | + mixup 0.2 (s1) | 0.7289 | 0.7082 | 0.7329 | | D_final | B + 3rd faster EMA (s1) | 0.7251 | 0.7140 | 0.7343 | | E_cut16 | cutout(16) (s1) | 0.7258 | 0.7015 | 0.7200 | Three findings, none of which the nine prior attempts had: - **Cutout(12) is the big lever: +1.9% raw, +2.1% ensemble… (heuristic summary from atif trajectory; no LLM pass)
### Iteration 14 | Reward: 0.7053
- Approach: Rebuilt a GPU-resident CIFAR-100 ResNet pipeline, compared sampling and EMA variants, evaluated shift/flip TTA, and emitted predictions.
- Measured: reward 0.7053
- What happened: After reporting an empty workspace, the agent inspected the environment and wrote `train.py`, `emit.py`, and `tta_pick.py`. It rebuilt an all-data-GPU-resident CIFAR-100 training pipeline, trained baseline and comparison runs, evaluated checkpoints on a public validation split, tested flip, shift, scale, and prior-adjustment inference variants, selected run A, generated `/submission/preds_i20k.npy`, and performed a final reload check. It also recorded observations about parallel-training slowdown and a terminated run. All implementation and measurement details below are unvalidated agent claims: the model used a ResNet-34 layout with width 128, ResNet-D anti-aliased transitions, 48-pixel inputs, TrivialAugment plus cutout, and SGD with Nesterov momentum and cosine scheduling. Run A allegedly used seed 0, batch size 128, 8,000 optimizer steps, and 1,024,000 presentations; inference combined raw weights with EMA(0.999) using five-shift × flip TTA and prior exponent 0.5. The agent claimed 0.7364 balanced validation accuracy, 268.6 seconds of training, 273.5 seconds end-to-end, and an output array of shape `(10000,)`, dtype int64, containing labels 0–99 across 100 classes. It also compared class-balanced sampling at seed 1, a third-EMA run at seed 3, flip-only, shift5, shift9, and scale3+shift5 variants. The grader’s authoritative verdict was: “this submission earned reward 0.7053 (71% of the maximum achievable 1.00); it was not a valid scoreable success.”
### Iteration 15 | Reward: 0.6864
- Approach: Done, with ~15 minutes of margin.
- Measured: reward 0.6864
- What happened: Activity: 34 steps; 33 agent turns; 39 tool calls (Bash x24, Edit x10, Write x5). Files written: /workspace/environment/starter/train.py, /workspace/environment/starter/postproc.py, /logs/agent/sessions/projects/-workspace/memory/cifar100-budget-recipe.md, /logs/agent/sessions/projects/-workspace/memory/cifar100-budget-val-noise.md, /logs/agent/sessions/projects/-workspace/memory/MEMORY.md. Agent's final report: Done, with ~15 minutes of margin. ## Result `/submission/preds_i20k.npy` — shape `(10000,)`, int64, values 0–99, 100 distinct classes, **verified valid** and byte-for-byte confirmed to be run A's predictions. Source: a **single** training run at **8000/8000 optimizer updates and 1,024,000/1,024,000 presentations**, **252s** standalone wall clock (train + prediction) against the 432s allowance. Val balanced accuracy **0.7384**. ## What I did The workspace came back empty, so I rebuilt the pipeline (GPU-resident data, batched TrivialAugment/cutout, no dataloader) and had a **verified submission on disk at the 13-minute mark**, then spent the rest on full-budget probes. | run | change | val (flip, raw+EMA) | standalone | |---|---|---|---| | **A_base** | **w128, res 48, TA+cutout(12)** | **0.7384** | **252 s** | | B_res56 | resolution 56 | 0.7324 | 314 s | | C_sam | SAM ρ=.05 @ batch 64 | 0.6705 | 512 s | | D_w96 | width 96 | 0.7175 | — | | E_deep | blocks (4,6,8,4), 116M | 0.7349 | 325 s | | F_cifarlayout | 32px, strides 1,1,2,2 | 0.7367 | 298 s | Nothing beat the baseline, so I left A's predictions in place rather than swapping in a within-noise "winner." ## The finding that mattered… (heuristic summary from atif trajectory; no LLM pass)
### Iteration 16 | Reward: 0.7566
- Approach: The agent rebuilt a GPU-resident CIFAR-100 ResNet pipeline, compared capacity and regularization variants, and generated TTA predictions.
- Measured: reward 0.7566
- What happened: The agent checked the workspace, rebuilt training and postprocessing scripts, and created a GPU-resident CIFAR-100 workflow using batched augmentation without a dataloader. It trained multiple ResNet variants, compared model widths, weight decay, TrivialAugment, and EMA settings, replicated the selected configuration with two seeds, evaluated shift-and-flip test-time augmentation and class-prior correction, and wrote a prediction file. It also saved notes about validation noise and the chosen training recipe. The authoritative grader reported verbatim: “this submission earned reward 0.7566 (76% of the maximum achievable 1.00); it was not a valid scoreable success.” Agent-claimed, unvalidated parameters included 8,000 optimizer updates, 1,024,000 presentations, a ResNet-34-D at 48-pixel resolution, widths of 128 and 160, weight decay values of 5e-4 and 2e-3, and EMA decays of 0.99, 0.999, and 0.9997. The agent also claimed a 133M-parameter selected architecture, an approximately 51-epoch budget, a 2,000-image validation split, five-shift-by-flip TTA, and class-prior correction using an exponent of 0.75. Its unvalidated report stated that the selected run took 357.8 seconds against a 432-second cap and that the output was an int64 array of shape `(10000,)` containing labels 0–99 across 100 classes.

### Best solution so far — code from iteration 16 (reward 0.7566)
This is YOUR OWN prior best source. Build on it (write your chosen code to /workspace); do not restart from scratch unless you have a reason. It will be re-trained and re-graded from scratch.
```python
# ===== agent_src/environment/starter/train.py =====
"""CIFAR-100 @ 20k images, fixed budget: 8000 steps x batch 128 = 1,024,000 presentations.

Single training run; all data GPU-resident; batched augmentation (no dataloader).
Saves val + graded probability tensors for raw and EMA weights so post-processing
(TTA combination, prior/logit adjustment) can be chosen without retraining.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BUDGET_STEPS = 8000
BATCH = 128
N_CLASSES = 100
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        nn.init.zeros_(self.b2.weight)
        if stride != 1 or cin != cout:
            pool = [nn.AvgPool2d(2, 2)] if stride != 1 else []          # ResNet-D shortcut
            self.sc = nn.Sequential(*pool, nn.Conv2d(cin, cout, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(cout))
        else:
            self.sc = None

    def forward(self, x):
        o = F.relu(self.b1(self.c1(x)), inplace=True)
        o = self.b2(self.c2(o))
        return F.relu(o + (x if self.sc is None else self.sc(x)), inplace=True)


class Net(nn.Module):
    def __init__(self, w=128, blocks=(3, 4, 6, 3), n_classes=N_CLASSES):
        super().__init__()
        ws = [w, 2 * w, 4 * w, 8 * w]
        self.stem = nn.Sequential(nn.Conv2d(3, w, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(w), nn.ReLU(inplace=True))
        layers, cin = [], w
        for i, (n, c) in enumerate(zip(blocks, ws)):
            for j in range(n):
                layers.append(Block(cin, c, 2 if (j == 0 and i > 0) else 1))
                cin = c
        self.body = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, n_classes)

    def forward(self, x):
        x = self.body(self.stem(x))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


# ---------------------------------------------------------------- augmentation (GPU, batched)
def trivial_aug(x, gen):
    """Batched TrivialAugment-Wide: one random op per sample, magnitude ~ U[0,1]. x in [0,1]."""
    b, c, h, w = x.shape
    dev = x.device
    op = torch.randint(0, 12, (b,), device=dev, generator=gen)
    m = torch.rand(b, device=dev, generator=gen)
    sgn = torch.where(torch.rand(b, device=dev, generator=gen) < 0.5, -1.0, 1.0)
    ms = m * sgn

    th = torch.zeros(b, 2, 3, device=dev)
    th[:, 0, 0] = 1.0
    th[:, 1, 1] = 1.0
    sh = 0.99 * ms
    s1 = op == 1
    th[:, 0, 1] = torch.where(s1, sh, th[:, 0, 1])
    s2 = op == 2
    th[:, 1, 0] = torch.where(s2, sh, th[:, 1, 0])
    tr = 0.64 * ms                                   # +-0.32 of image size (grid units are 2x)
    th[:, 0, 2] = torch.where(op == 3, tr, th[:, 0, 2])
    th[:, 1, 2] = torch.where(op == 4, tr, th[:, 1, 2])
    ang = (135.0 * np.pi / 180.0) * ms
    ca, sa = torch.cos(ang), torch.sin(ang)
    s5 = op == 5
    th[:, 0, 0] = torch.where(s5, ca, th[:, 0, 0])
    th[:, 0, 1] = torch.where(s5, -sa, th[:, 0, 1])
    th[:, 1, 0] = torch.where(s5, sa, th[:, 1, 0])
    th[:, 1, 1] = torch.where(s5, ca, th[:, 1, 1])
    grid = F.affine_grid(th, (b, c, h, w), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    f = (1.0 + 0.99 * ms).view(b, 1, 1, 1)
    gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    o = op.view(b, 1, 1, 1)
    x = torch.where(o == 6, x * f, x)                                    # brightness
    x = torch.where(o == 7, gray + (x - gray) * f, x)                    # colour / saturation
    x = torch.where(o == 8, gray.mean((2, 3), keepdim=True) + (x - gray.mean((2, 3), keepdim=True)) * f, x)
    bits = (8 - (m * 6).round()).view(b, 1, 1, 1)                        # posterize 8..2 bits
    lv = torch.pow(2.0, bits)
    x = torch.where(o == 9, (x.clamp(0, 1) * lv).floor() / lv, x)
    thr = (1.0 - m).view(b, 1, 1, 1)                                     # solarize
    x = torch.where(o == 10, torch.where(x >= thr, 1.0 - x, x), x)
    lo = x.amin((2, 3), keepdim=True)
    hi = x.amax((2, 3), keepdim=True)
    x = torch.where(o == 11, (x - lo) / (hi - lo).clamp_min(1e-3), x)    # autocontrast
    return x.clamp_(0, 1)


def augment(x, pad, cut, res, gen, mean=None, std=None, ta=0):
    """x: (B,3,32,32) float in [0,1]. Random crop + hflip (+TA) + normalise + cutout + resize."""
    b, c, h, w = x.shape
    xp = F.pad(x, (pad,) * 4, mode="reflect")
    oy = torch.randint(0, 2 * pad + 1, (b,), device=x.device, generator=gen)
    ox = torch.randint(0, 2 * pad + 1, (b,), device=x.device, generator=gen)
    ar = torch.arange(h, device=x.device)
    rows = (oy[:, None] + ar)[:, None, :, None].expand(b, c, h, xp.shape[3])
    x = xp.gather(2, rows)
    cols = (ox[:, None] + ar)[:, None, None, :].expand(b, c, h, w)
    x = x.gather(3, cols)

    flip = torch.rand(b, 1, 1, 1, device=x.device, generator=gen) < 0.5
    x = torch.where(flip, x.flip(3), x)

    if ta:
        x = trivial_aug(x, gen)
    if mean is not None:
        x = (x - mean) / std

    if cut > 0:
        cy = torch.randint(0, h, (b, 1, 1), device=x.device, generator=gen)
        cx = torch.randint(0, w, (b, 1, 1), device=x.device, generator=gen)
        yy = ar.view(1, h, 1)
        xx = ar.view(1, 1, w)
        m = ((yy - cy).abs() < cut // 2) & ((xx - cx).abs() < cut // 2)
        x = x * (~m).unsqueeze(1).to(x.dtype)

    if res != h:
        x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
    return x.contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def probs(model, x, res, mean, std, flip_tta=True, bs=500, shift=0):
    """x: uint8 GPU tensor (N,32,32,3) -> softmax probs (N,100). Optional flip/shift TTA."""
    model.eval()
    offs = [(0, 0)] if shift == 0 else [(0, 0), (-shift, 0), (shift, 0), (0, -shift), (0, shift)]
    out = []
    for s in range(0, len(x), bs):
        x0 = x[s:s + bs].permute(0, 3, 1, 2).float().div_(255.).sub_(mean).div_(std)
        xp = F.pad(x0, (shift,) * 4, mode="reflect") if shift else x0
        h = x0.shape[-1]
        acc, n = 0.0, 0
        for dy, dx in offs:
            xb = xp[:, :, shift + dy:shift + dy + h, shift + dx:shift + dx + h] if shift else x0
            if res != h:
                xb = F.interpolate(xb, size=(res, res), mode="bilinear", align_corners=False)
            xb = xb.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                acc = acc + model(xb).float().softmax(1)
                n += 1
                if flip_tta:
                    acc = acc + model(xb.flip(3)).float().softmax(1)
                    n += 1
        out.append((acc / n).cpu())
    model.train()
    return torch.cat(out).numpy()


def bal_acc(p, y):
    pred = p.argmax(1)
    return float(np.mean([(pred[y == c] == c).mean() for c in range(N_CLASSES) if (y == c).any()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="i20k")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graded", default=os.path.join(DATA, "graded_images.npz"))
    ap.add_argument("--out", default="/submission/preds_i20k.npy")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--blocks", default="3,4,6,3")
    ap.add_argument("--res", type=int, default=48)
    ap.add_argument("--steps", type=int, default=BUDGET_STEPS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--cut", type=int, default=12)
    ap.add_argument("--ta", type=int, default=0)
    ap.add_argument("--shift", type=int, default=0)
    ap.add_argument("--pad", type=int, default=4)
    ap.add_argument("--warm", type=int, default=300)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--ema2", type=float, default=0.9997)
    ap.add_argument("--probdir", default="/workspace/probs")
    ap.add_argument("--write", type=int, default=1)
    a = ap.parse_args()

    t0 = time.time()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    d = np.load(os.path.join(DATA, "train_i20k.npz"))
    xtr = torch.from_numpy(d["x"]).to(dev)                       # (N,32,32,3) uint8
    ytr = torch.from_numpy(d["y"]).to(dev)
    dv = np.load(os.path.join(DATA, "val_public.npz"))
    xva = torch.from_numpy(dv["x"]).to(dev)
    yva = dv["y"]
    xgr = torch.from_numpy(np.load(a.graded)["x"]).to(dev)

    xf = xtr.permute(0, 3, 1, 2).float().div_(255.)
    mean = xf.mean((0, 2, 3)).view(1, 3, 1, 1)
    std = xf.std((0, 2, 3)).view(1, 3, 1, 1)
    xf = xf.contiguous()                                         # [0,1] train, GPU-resident
    del d

    blocks = tuple(int(v) for v in a.blocks.split(","))
    model = Net(a.width, blocks).to(dev).to(memory_format=torch.channels_last)
    nparam = sum(p.numel() for p in model.parameters())

    decay, nodecay = [], []
    for n_, p_ in model.named_parameters():
        (nodecay if p_.ndim <= 1 else decay).append(p_)
    opt = torch.optim.SGD([{"params": decay, "weight_decay": a.wd},
                           {"params": nodecay, "weight_decay": 0.0}],
                          lr=a.lr, momentum=0.9, nesterov=True)

    emas = []
    for dcy in (a.ema, a.ema2):
        if dcy > 0:
            m = [p.detach().clone().float() for p in model.parameters()]
            emas.append([dcy, m])
    mparams = list(model.parameters())

    gen = torch.Generator(device=dev); gen.manual_seed(a.seed + 1234)
    N = xf.shape[0]
    perm = torch.randperm(N, device=dev, generator=gen)
    ptr = 0
    presentations = 0
    model.train()
    for step in range(a.steps):
        if ptr + a.batch > N:
            perm = torch.randperm(N, device=dev, generator=gen)
            ptr = 0
        idx = perm[ptr:ptr + a.batch]; ptr += a.batch
        xb = augment(xf[idx], a.pad, a.cut, a.res, gen, mean, std, a.ta)
        yb = ytr[idx]
        presentations += xb.shape[0]

        lr = a.lr * (step + 1) / a.warm if step < a.warm else \
            0.5 * a.lr * (1 + np.cos(np.pi * (step - a.warm) / max(1, a.steps - a.warm)))
        for g in opt.param_groups:
            g["lr"] = lr

        with torch.autocast("cuda", dtype=torch.bfloat16):
            if a.mixup > 0:
                lam = float(np.random.beta(a.mixup, a.mixup))
                pi = torch.randperm(xb.shape[0], device=dev, generator=gen)
                xb = lam * xb + (1 - lam) * xb[pi]
                o = model(xb)
                loss = lam * F.cross_entropy(o, yb, label_smoothing=a.ls) + \
                    (1 - lam) * F.cross_entropy(o, yb[pi], label_smoothing=a.ls)
            else:
                loss = F.cross_entropy(model(xb), yb, label_smoothing=a.ls)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            for dcy, mv in emas:
                d_ = dcy if step >= 200 else min(dcy, (step + 1) / (step + 10))
                torch._foreach_mul_(mv, d_)
                torch._foreach_add_(mv, [p.detach() for p in mparams], alpha=1 - d_)

    ttrain = time.time() - t0

    os.makedirs(a.probdir, exist_ok=True)
    res = {"tag": a.tag, "params": nparam, "steps": a.steps,
           "presentations": presentations, "train_s": ttrain, "cfg": vars(a)}
    store = {}
    store["val_raw"] = probs(model, xva, a.res, mean, std)
    store["gr_raw"] = probs(model, xgr, a.res, mean, std)
    if a.shift:
        store["val_rawS"] = probs(model, xva, a.res, mean, std, shift=a.shift)
        store["gr_rawS"] = probs(model, xgr, a.res, mean, std, shift=a.shift)
    res["val_raw"] = bal_acc(store["val_raw"], yva)

    bk = [p.detach().clone() for p in mparams]
    for k, (dcy, mv) in enumerate(emas):
        with torch.no_grad():
            for p, m in zip(mparams, mv):
                p.copy_(m)
        # BN buffers are shared with the raw model (running stats already smoothed)
        store[f"val_ema{k}"] = probs(model, xva, a.res, mean, std)
        store[f"gr_ema{k}"] = probs(model, xgr, a.res, mean, std)
        if a.shift:
            store[f"val_ema{k}S"] = probs(model, xva, a.res, mean, std, shift=a.shift)
            store[f"gr_ema{k}S"] = probs(model, xgr, a.res, mean, std, shift=a.shift)
        res[f"val_ema{k}"] = bal_acc(store[f"val_ema{k}"], yva)
    with torch.no_grad():
        for p, b in zip(mparams, bk):
            p.copy_(b)

    ens = store["val_raw"].copy()
    for k in range(len(emas)):
        ens += store[f"val_ema{k}"]
    for k_ in [k_ for k_ in store if k_.startswith("val_") and k_.endswith("S")]:
        res[k_] = bal_acc(store[k_], yva)
    res["val_ens"] = bal_acc(ens, yva)
    if len(emas) > 0:
        res["val_raw_ema0"] = bal_acc(store["val_raw"] + store["val_ema0"], yva)

    prior = np.bincount(ytr.cpu().numpy(), minlength=N_CLASSES).astype(np.float64)
    prior /= prior.sum()
    for tau in (0.5, 1.0):
        res[f"val_ens_la{tau}"] = bal_acc(ens / prior[None, :] ** tau, yva)

    np.savez(os.path.join(a.probdir, f"{a.tag}.npz"), yva=yva, prior=prior, **store)
    res["total_s"] = time.time() - t0
    print(json.dumps(res, default=str))
    with open(os.path.join(a.probdir, f"{a.tag}.json"), "w") as f:
        json.dump(res, f, default=str)

    if a.write:
        g = store["gr_raw"].copy()
        for k in range(len(emas)):
            g += store[f"gr_ema{k}"]
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        np.save(a.out, (g / prior[None, :] ** 0.5).argmax(1).astype(np.int64))
        print("wrote", a.out, "in", round(time.time() - t0, 1), "s")
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
"""Pick an inference combination on val_public and (optionally) write graded predictions.

Uses only probability tensors already produced by a single budgeted training run.
"""
import argparse, itertools, json, os
import numpy as np

N_CLASSES = 100


def bal_acc(p, y):
    pred = p.argmax(1)
    return float(np.mean([(pred[y == c] == c).mean() for c in range(N_CLASSES) if (y == c).any()]))


def boot(p, y, tau, prior, n=200, seed=0):
    """Bootstrap sd of balanced accuracy (how much val can be trusted)."""
    rng = np.random.default_rng(seed)
    q = p / prior[None, :] ** tau
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        vals.append(bal_acc(q[i], y[i]))
    return float(np.std(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--probdir", default="/workspace/probs")
    ap.add_argument("--write", default="")
    ap.add_argument("--combo", default="")          # e.g. raw+ema0
    ap.add_argument("--tau", type=float, default=-1)
    a = ap.parse_args()

    for tag in a.tags:
        z = np.load(os.path.join(a.probdir, f"{tag}.npz"))
        yva, prior = z["yva"], z["prior"]
        keys = [k[4:] for k in z.files if k.startswith("val_")]
        combos = []
        for r in range(1, len(keys) + 1):
            combos += [c for c in itertools.combinations(sorted(keys), r)]
        print(f"== {tag}")
        best = None
        for c in combos:
            v = sum(z["val_" + k] for k in c)
            v = v / v.sum(1, keepdims=True)
            for tau in (0.0, 0.25, 0.5, 0.75, 1.0):
                s = bal_acc(v / prior[None, :] ** tau, yva)
                print(f"   {'+'.join(c):24s} tau={tau:<5} {s:.4f}")
                if best is None or s > best[0]:
                    best = (s, c, tau)
        print(f"   BEST {best[1]} tau={best[2]} {best[0]:.4f}")
        vb = sum(z["val_" + k] for k in best[1])
        print(f"   bootstrap sd of val bal-acc ~ {boot(vb, yva, best[2], prior):.4f}")

        if a.write:
            combo = tuple(a.combo.split("+")) if a.combo else best[1]
            tau = a.tau if a.tau >= 0 else best[2]
            g = sum(z["gr_" + k] for k in combo)
            g = g / g.sum(1, keepdims=True)
            pred = (g / prior[None, :] ** tau).argmax(1).astype(np.int64)
            os.makedirs(os.path.dirname(a.write), exist_ok=True)
            np.save(a.write, pred)
            print(f"   wrote {a.write} from {combo} tau={tau} "
                  f"(val {bal_acc(sum(z['val_' + k] for k in combo) / prior[None, :] ** tau, yva):.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ===== agent_src/probs/A_w128.json =====
{"tag": "A_w128", "params": 85172964, "steps": 8000, "presentations": 1024000, "train_s": 226.01177525520325, "cfg": {"instance": "i20k", "seed": 0, "graded": "/workspace/environment/data/graded_images.npz", "out": "/submission/preds_i20k.npy", "tag": "A_w128", "width": 128, "blocks": "3,4,6,3", "res": 48, "steps": 8000, "batch": 128, "lr": 0.2, "wd": 0.0005, "ls": 0.1, "cut": 12, "pad": 4, "warm": 300, "mixup": 0.0, "ema": 0.999, "ema2": 0.9997, "probdir": "/workspace/probs", "write": 1}, "val_raw": 0.7155465948227154, "val_ema0": 0.7040594114315772, "val_ema1": 0.7040594114315772, "val_ens": 0.7097322470339626, "val_raw_ema0": 0.7120959175833719, "val_ens_la0.5": 0.7102453038696003, "val_ens_la1.0": 0.7109193822710548, "total_s": 229.92209267616272}

# ===== agent_src/probs/B_w160.json =====
{"tag": "B_w160", "params": 133038980, "steps": 8000, "presentations": 1024000, "train_s": 887.8003823757172, "cfg": {"instance": "i20k", "seed": 1, "graded": "/workspace/environment/data/graded_images.npz", "out": "/submission/preds_i20k.npy", "tag": "B_w160", "width": 160, "blocks": "3,4,6,3", "res": 48, "steps": 8000, "batch": 128, "lr": 0.2, "wd": 0.0005, "ls": 0.1, "cut": 12, "pad": 4, "warm": 300, "mixup": 0.0, "ema": 0.999, "ema2": 0.9997, "probdir": "/workspace/probs", "write": 0}, "val_raw": 0.7238195373377705, "val_ema0": 0.677863508228839, "val_ema1": 0.3079403436845826, "val_ens": 0.685289065050513, "val_raw_ema0": 0.7131989121832436, "val_ens_la0.5": 0.6853761098684401, "val_ens_la1.0": 0.6836806211466357, "total_s": 893.9666590690613}

# ===== agent_src/probs/C_wd2e3.json =====
{"tag": "C_wd2e3", "params": 85172964, "steps": 8000, "presentations": 1024000, "train_s": 710.37535572052, "cfg": {"instance": "i20k", "seed": 1, "graded": "/workspace/environment/data/graded_images.npz", "out": "/submission/preds_i20k.npy", "tag": "C_wd2e3", "width": 128, "blocks": "3,4,6,3", "res": 48, "steps": 8000, "batch": 128, "lr": 0.2, "wd": 0.002, "ls": 0.1, "cut": 12, "pad": 4, "warm": 300, "mixup": 0.0, "ema": 0.99, "ema2": 0.999, "probdir": "/workspace/probs", "write": 0}, "val_raw": 0.7268812802813711, "val_ema0": 0.7267516506517415, "val_ema1": 0.5521829615360078, "val_ens": 0.7147821591926177, "val_raw_ema0": 0.7267516506517415, "val_ens_la0.5": 0.7138451628556213, "val_ens_la1.0": 0.7147569275615037, "total_s": 724.0098843574524}

# ===== agent_src/probs/D_ta.json =====
{"tag": "D_ta", "params": 85172964, "steps": 8000, "presentations": 1024000, "train_s": 713.9755764007568, "cfg": {"instance": "i20k", "seed": 1, "graded": "/workspace/environment/data/graded_images.npz", "out": "/submission/preds_i20k.npy", "tag": "D_ta", "width": 128, "blocks": "3,4,6,3", "res": 48, "steps": 8000, "batch": 128, "lr": 0.2, "wd": 0.0005, "ls": 0.1, "cut": 12, "ta": 1, "pad": 4, "warm": 300, "mixup": 0.0, "ema": 0.99, "ema2": 0.999, "probdir": "/workspace/probs", "write": 0}, "val_raw": 0.7298562507565872, "val_ema0": 0.7298716353719717, "val_ema1": 0.7049645418633566, "val_ens": 0.726202895133717, "val_raw_ema0": 0.7298716353719717, "val_ens_la0.5": 0.7248510432818651, "val_ens_la1.0": 0.726233406535722, "total_s": 722.9210572242737}

# ===== agent_src/probs/E_ta_wd2e3.json =====
{"tag": "E_ta_wd2e3", "params": 85172964, "steps": 8000, "presentations": 1024000, "train_s": 489.166974067688, "cfg": {"instance": "i20k", "seed": 2, "graded": "/workspace/environment/data/graded_images.npz", "out": "/submission/preds_i20k.npy", "tag": "E_ta_wd2e3", "width": 128, "blocks": "3,4,6,3", "res": 48, "steps": 8000, "batch": 128, "lr": 0.2, "wd": 0.002, "ls": 0.1, "cut": 12, "ta": 1, "shift": 0, "pad": 4, "warm": 300, "mixup": 0.0, "ema": 0.99, "ema2": 0.0, "probdir": "/workspace/probs", "write": 0}, "val_raw": 0.7167607210709006, "val_ema0": 0.7168923417893633, "val_ens": 0.7173162766264564, "val_raw_ema0": 0.7173162766264564, "val_ens_la0.5": 0.7166534676432138, "val_ens_la1.0": 0.7173740661911883, "total_s": 494.9528138637543}

# ===== agent_src/probs/FINAL1.json =====
{"tag": "FINAL1", "params": 85172964, "steps": 8000, "presentations": 1024000, "train_s": 221.37687849998474, "cfg": {"instance": "i20k", "seed": 5, "graded": "/workspace/environment/data/graded_images.npz", "out": "/submission/preds_i20k.npy", "tag": "FINAL1", "width": 128, "blocks": "3,4,6,3", "res": 48, "steps": 8000, "batch": 128, "lr": 0.2, "wd": 0.0005, "ls": 0.1, "cut": 12, "ta": 1, "shift": 2, "pad": 4, "warm": 300, "mixup": 0.0, "ema": 0.99, "ema2": 0.0, "probdir": "/workspace/probs", "write": 0}, "val_raw": 0.7164940344773862, "val_ema0": 0.7169438672533058, "val_rawS": 0.7203714431765175, "val_ema0S": 0.7208977589659912, "val_ens": 0.7169940344773862, "val_raw_ema0": 0.7169940344773862, "val_ens_la0.5": 0.7164316652897716, "val_ens_la1.0": 0.7176471030777901, "total_s": 236.45577812194824}

# ===== agent_src/probs/FINAL2.json =====
{"tag": "FINAL2", "params": 133038980, "steps": 8000, "presentations": 1024000, "train_s": 334.1600728034973, "cfg": {"instance": "i20k", "seed": 6, "graded": "/workspace/environment/data/graded_images.npz", "out": "/submission/preds_i20k.npy", "tag": "FINAL2", "width": 160, "blocks": "3,4,6,3", "res": 48, "steps": 8000, "batch": 128, "lr": 0.2, "wd": 0.0005, "ls": 0.1, "cut": 12, "ta": 1, "shift": 2, "pad": 4, "warm": 300, "mixup": 0.0, "ema": 0.99, "ema2": 0.0, "probdir": "/workspace/probs", "write": 0}, "val_raw": 0.7239390321741145, "val_ema0": 0.7240311248640332, "val_rawS": 0.7361529513623776, "val_ema0S": 0.736097395806822, "val_ens": 0.7231615596466419, "val_raw_ema0": 0.7231615596466419, "val_ens_la0.5": 0.7275183970405236, "val_ens_la1.0": 0.7287241776570593, "total_s": 357.76145696640015}

# ===== agent_src/probs/FINAL3.json =====
{"tag": "FINAL3", "params": 133038980, "steps": 8000, "presentations": 1024000, "train_s": 334.26240062713623, "cfg": {"instance": "i20k", "seed": 7, "graded": "/workspace/environment/data/graded_images.npz", "out": "/submission/preds_i20k.npy", "tag": "FINAL3", "width": 160, "blocks": "3,4,6,3", "res": 48, "steps": 8000, "batch": 128, "lr": 0.2, "wd": 0.0005, "ls": 0.1, "cut": 12, "ta": 1, "shift": 2, "pad": 4, "warm": 300, "mixup": 0.0, "ema": 0.99, "ema2": 0.0, "probdir": "/workspace/probs", "write": 0}, "val_raw": 0.7313350668219464, "val_ema0": 0.7317795112663908, "val_rawS": 0.7349108784480642, "val_ema0S": 0.7354577058746871, "val_ens": 0.7323350668219463, "val_raw_ema0": 0.7323350668219463, "val_ens_la0.5": 0.7308287155857996, "val_ens_la1.0": 0.7301435304006145, "total_s": 357.8730721473694}

# ===== agent_src/probs/F_ta_s3.json =====
{"tag": "F_ta_s3", "params": 85172964, "steps": 8000, "presentations": 1024000, "train_s": 491.9966685771942, "cfg": {"instance": "i20k", "seed": 3, "graded": "/workspace/environment/data/graded_images.npz", "out": "/submission/preds_i20k.npy", "tag": "F_ta_s3", "width": 128, "blocks": "3,4,6,3", "res": 48, "steps": 8000, "batch": 128, "lr": 0.2, "wd": 0.0005, "ls": 0.1, "cut": 12, "ta": 1, "shift": 0, "pad": 4, "warm": 300, "mixup": 0.0, "ema": 0.99, "ema2": 0.0, "probdir": "/workspace/probs", "write": 0}, "val_raw": 0.7260370620542956, "val_ema0": 0.7255608715781051, "val_ens": 0.7265370620542956, "val_raw_ema0": 0.7265370620542956, "val_ens_la0.5": 0.7255935513166673, "val_ens_la1.0": 0.7265817866107848, "total_s": 495.4251010417938}

# ===== agent_src/probs/smoke.json =====
{"tag": "smoke", "params": 85172964, "steps": 200, "presentations": 25600, "train_s": 7.925849676132202, "cfg": {"instance": "i20k", "seed": 0, "graded": "/workspace/environment/data/graded_images.npz", "out": "/submission/preds_i20k.npy", "tag": "smoke", "width": 128, "blocks": "3,4,6,3", "res": 48, "steps": 200, "batch": 128, "lr": 0.2, "wd": 0.0005, "ls": 0.1, "cut": 12, "pad": 4, "warm": 300, "mixup": 0.0, "ema": 0.999, "ema2": 0.9997, "probdir": "/workspace/probs", "write": 0}, "val_raw": 0.08754856987042048, "val_ema0": 0.11405329180100347, "val_ema1": 0.11405329180100347, "val_ens": 0.11838396888292901, "val_raw_ema0": 0.10826555313057577, "val_ens_la0.5": 0.1203750602657187, "val_ens_la1.0": 0.12082731796205075, "total_s": 11.89630675315857}

# ===== agent_src/probs/smoke2.json =====
{"tag": "smoke2", "params": 85172964, "steps": 120, "presentations": 15360, "train_s": 17.886139631271362, "cfg": {"instance": "i20k", "seed": 0, "graded": "/workspace/environment/data/graded_images.npz", "out": "/submission/preds_i20k.npy", "tag": "smoke2", "width": 128, "blocks": "3,4,6,3", "res": 48, "steps": 120, "batch": 128, "lr": 0.2, "wd": 0.0005, "ls": 0.1, "cut": 12, "ta": 1, "pad": 4, "warm": 300, "mixup": 0.0, "ema": 0.999, "ema2": 0.9997, "probdir": "/workspace/probs", "write": 0}, "val_raw": 0.057694583972908775, "val_ema0": 0.08452378964705995, "val_ema1": 0.08452378964705995, "val_ens": 0.08229817014583807, "val_raw_ema0": 0.07561882242384328, "val_ens_la0.5": 0.08092458568300875, "val_ens_la1.0": 0.0832032971676026, "total_s": 31.543294191360474}
```