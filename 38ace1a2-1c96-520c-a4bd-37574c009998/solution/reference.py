"""Reference solution for AELLO-C8-S1 (CIFAR-100 at a fixed step budget).

Four changes over the starter, one of which is a defect repair:

  1. REPAIR. The starter builds its one-cycle schedule from steps_per_epoch * EPOCHS, which
     is 156 * 200 = 31200 steps, while the training loop takes the graded 8000. Training
     therefore stops 25.6 percent into the cycle and the learning rate NEVER ANNEALS. It is
     silent in every way that matters: no error, no warning, a scheduler that is genuinely
     being stepped, and an entirely plausible loss curve. The repair is total_steps =
     BUDGET_STEPS. Measured worth: the reference with this defect left in scores 0.46633
     against 0.70263 with it repaired.
  2. Residual connections, so depth is usable at this budget.
  3. Width 64 -> 128.
  4. Learning rate 0.1 -> 0.2, with label smoothing 0.1.
  5. Cutout at a strength derived by sweep on the shipped shard at the graded budget. The
     sweep chose 0.0 here, so cutout is off; the value is recorded because it was measured
     rather than assumed, which is why cutout() ships despite being inactive.
"""
import argparse, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# Environment resolution that works BOTH under Harbor (solution mounted at /solution,
# environment baked at /workspace/environment) and on the authoring host (bundle-relative).
# The earlier revision resolved only bundle-relatively, which inside the container pointed
# at a nonexistent /environment and made every oracle run die on import.
_ENV_CANDIDATES = [os.path.join("/workspace", "environment"),
                   os.path.join(os.path.dirname(HERE), "environment")]
ENV_DIR = next((p for p in _ENV_CANDIDATES if os.path.isdir(p)), None)
if ENV_DIR is None:
    raise SystemExit("reference.py: no environment tree at any of %s" % _ENV_CANDIDATES)
sys.path.insert(0, ENV_DIR)
import aello_data as data

BUDGET_STEPS = 8000
BATCH = 128
LR = 0.2
WIDTH = 128
CROP_PAD = 4
LABEL_SMOOTHING = 0.1

# Derived by sweep at the graded budget, never authored. The sweep chose 0.0, so cutout
# is off for the shipped instance.
CUTOUT_FRAC = {"i20k": 0.00}


class Block(torch.nn.Module):
    def __init__(self, a, b, stride=1):
        super().__init__()
        self.c1 = torch.nn.Conv2d(a, b, 3, stride, 1, bias=False); self.n1 = torch.nn.BatchNorm2d(b)
        self.c2 = torch.nn.Conv2d(b, b, 3, 1, 1, bias=False); self.n2 = torch.nn.BatchNorm2d(b)
        self.sc = (torch.nn.Sequential(torch.nn.Conv2d(a, b, 1, stride, bias=False),
                                       torch.nn.BatchNorm2d(b))
                   if stride != 1 or a != b else torch.nn.Identity())

    def forward(self, x):
        y = torch.nn.functional.relu(self.n1(self.c1(x)), inplace=True)
        return torch.nn.functional.relu(self.n2(self.c2(y)) + self.sc(x), inplace=True)


def make_model(seed, k, width=WIDTH):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, width, 3, 1, 1, bias=False), torch.nn.BatchNorm2d(width),
        torch.nn.ReLU(inplace=True),
        Block(width, width), Block(width, width),
        Block(width, width * 2, 2), Block(width * 2, width * 2),
        Block(width * 2, width * 4, 2), Block(width * 4, width * 4),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(width * 4, k))


def random_crop(x, pad):
    """Per-sample random crop, identical to the starter's. An earlier docstring here
    claimed the starter cropped per-batch; it does not, and that candidate defect was
    measured at 0.000 worth and removed (see TRUTH.md)."""
    b, c, h, w = x.shape
    xp = torch.nn.functional.pad(x, (pad,) * 4, mode="reflect")
    oy = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    ox = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    ar = torch.arange(b, device=x.device)
    rows = (oy[:, None] + torch.arange(h, device=x.device)[None, :])[:, None, :, None]
    cols = (ox[:, None] + torch.arange(w, device=x.device)[None, :])[:, None, None, :]
    return xp[ar[:, None, None, None], torch.arange(c, device=x.device)[None, :, None, None],
              rows, cols]


def cutout(x, size):
    if size < 2:
        return x
    b, _, h, w = x.shape
    cy = torch.randint(0, h, (b,), device=x.device); cx = torch.randint(0, w, (b,), device=x.device)
    ys = torch.arange(h, device=x.device)[None, :, None]
    xs = torch.arange(w, device=x.device)[None, None, :]
    mask = (((ys - cy[:, None, None]).abs() < size // 2) &
            ((xs - cx[:, None, None]).abs() < size // 2))
    return x * (~mask)[:, None, :, :]


def predict(model, x, device, batch=512):
    model.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(x), batch):
            xb = torch.from_numpy(np.ascontiguousarray(x[s:s + batch])).permute(0, 3, 1, 2)
            xb = xb.to(device).float().div_(255.)
            with torch.autocast(device, dtype=torch.bfloat16):
                out.append(model(xb).argmax(1).cpu().numpy())
    model.train()
    return np.concatenate(out)


def train(instance, seed, device="cuda", cutout_frac=None):
    x, y = data.load_train(instance)
    frac = CUTOUT_FRAC[instance] if cutout_frac is None else cutout_frac
    cut = int(round(frac * x.shape[1]))
    model = make_model(seed, data.N_CLASSES).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, nesterov=True,
                          weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=BUDGET_STEPS,
                                                pct_start=0.15)
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    for _ in range(BUDGET_STEPS):
        i = np.sort(rng.integers(0, len(y), BATCH))
        xb = torch.from_numpy(np.ascontiguousarray(x[i])).permute(0, 3, 1, 2)
        xb = xb.contiguous(memory_format=torch.channels_last).to(device).float().div_(255.)
        xb = random_crop(xb, CROP_PAD)
        if rng.random() < 0.5:
            xb = torch.flip(xb, dims=[3])
        xb = cutout(xb, cut)
        yb = torch.from_numpy(y[i]).to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device, dtype=torch.bfloat16):
            torch.nn.functional.cross_entropy(model(xb), yb,
                                              label_smoothing=LABEL_SMOOTHING).backward()
        opt.step(); sched.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True, choices=list(data.INSTANCES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graded", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cutout-frac", type=float, default=None)
    a = ap.parse_args()
    model = train(a.instance, a.seed, cutout_frac=a.cutout_frac)
    xg = np.load(a.graded)["x"]
    np.save(a.out, predict(model, xg, "cuda").astype(np.int64))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-bea18ed252d9784fdf61fe87fa7f4998
#   slot1: AELLO-CANARY-c1d6ab4e4a9586741a987c0f788e87f9
#   slot2: AELLO-CANARY-55f6e11e7b352cbdd5a41275b962be86
#   slot3: AELLO-CANARY-5c9e29f9f3d650b9fee3b9cb4577a4d5
# AELLO-CANARY-END
