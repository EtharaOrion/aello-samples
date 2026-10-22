"""Budgeted CIFAR-100 trainer: one 8,000 update run, from random initialization."""
import argparse
import copy
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aello_data as data

STEPS = 8000
BATCH = 128                 # exactly 1,024,000 sample presentations
MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


class WideBasic(nn.Module):
    def __init__(self, cin, cout, stride, drop=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.c1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.drop = drop
        self.short = (nn.Conv2d(cin, cout, 1, stride=stride, bias=False)
                      if cin != cout or stride != 1 else None)

    def forward(self, x):
        z = F.relu(self.bn1(x), inplace=True)
        shortcut = self.short(z) if self.short is not None else x
        z = self.c1(z)
        z = F.relu(self.bn2(z), inplace=True)
        if self.drop:
            z = F.dropout(z, self.drop, self.training)
        return shortcut + self.c2(z)


class WRN(nn.Module):
    def __init__(self, depth=40, widen=10, classes=100):
        super().__init__()
        n = (depth - 4) // 6
        widths = [16, 16*widen, 32*widen, 64*widen]
        self.c0 = nn.Conv2d(3, widths[0], 3, padding=1, bias=False)
        self.g1 = self._group(n, widths[0], widths[1], 1)
        self.g2 = self._group(n, widths[1], widths[2], 2)
        self.g3 = self._group(n, widths[2], widths[3], 2)
        self.bn = nn.BatchNorm2d(widths[3])
        self.fc = nn.Linear(widths[3], classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    @staticmethod
    def _group(n, cin, cout, stride):
        return nn.Sequential(WideBasic(cin, cout, stride),
                             *[WideBasic(cout, cout, 1) for _ in range(n-1)])

    def forward(self, x):
        x = self.g3(self.g2(self.g1(self.c0(x))))
        return self.fc(F.adaptive_avg_pool2d(F.relu(self.bn(x), inplace=True), 1).flatten(1))


def tensor_images(a, device):
    return (torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2)
            .to(device=device, dtype=torch.float32, memory_format=torch.channels_last).div_(255.0))


def normalize(x):
    mean = x.new_tensor(MEAN)[None, :, None, None]
    std = x.new_tensor(STD)[None, :, None, None]
    return (x - mean) / std


def augment(x):
    """Per-image reflect crop, flip, and occasional 8x8 cutout."""
    b, c, h, w = x.shape
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    oy = torch.randint(0, 9, (b,), device=x.device)
    ox = torch.randint(0, 9, (b,), device=x.device)
    yy = oy[:, None] + torch.arange(h, device=x.device)[None]
    xx = ox[:, None] + torch.arange(w, device=x.device)[None]
    x = xp[torch.arange(b, device=x.device)[:, None, None, None],
           torch.arange(c, device=x.device)[None, :, None, None],
           yy[:, None, :, None], xx[:, None, None, :]]
    flip = torch.rand(b, device=x.device) < 0.5
    x[flip] = x[flip].flip(3)
    use = torch.rand(b, device=x.device) < 0.5
    cy = torch.randint(0, h, (b,), device=x.device)
    cx = torch.randint(0, w, (b,), device=x.device)
    gy = torch.arange(h, device=x.device)[None, :, None]
    gx = torch.arange(w, device=x.device)[None, None, :]
    mask = use[:, None, None] & (abs(gy-cy[:, None, None]) < 4) & (abs(gx-cx[:, None, None]) < 4)
    return x.masked_fill(mask[:, None], 0.0)


@torch.no_grad()
def ema_update(avg, model, decay):
    for pa, p in zip(avg.parameters(), model.parameters()):
        pa.lerp_(p, 1.0-decay)
    for ba, b in zip(avg.buffers(), model.buffers()):
        ba.copy_(b)


def soft_ce(logit, ya, yb, lam, smooth=0.04):
    return (lam*F.cross_entropy(logit, ya, label_smoothing=smooth) +
            (1.0-lam)*F.cross_entropy(logit, yb, label_smoothing=smooth))


def logits(model, a, device, tta=True, batch=512):
    model.eval(); ans = []
    with torch.inference_mode():
        for s in range(0, len(a), batch):
            x = normalize(tensor_images(a[s:s+batch], device))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(x).float()
                if tta:
                    z.add_(model(x.flip(3)).float()).mul_(0.5)
            ans.append(z.cpu())
    return torch.cat(ans).numpy()


def balanced(y, p):
    return float(np.mean([np.mean(p[y == k] == k) for k in range(100)]))


def train(instance, seed, device, val_path):
    x, y = data.load_train(instance)
    val = np.load(val_path)
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    model = WRN().to(device=device, memory_format=torch.channels_last)
    avg = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.SGD(model.parameters(), lr=0.12, momentum=0.9,
                          nesterov=True, weight_decay=5e-4)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y)); pos = 0
    start = time.time(); model.train()
    for step in range(STEPS):
        if pos + BATCH > len(y):
            order = rng.permutation(len(y)); pos = 0
        idx = order[pos:pos+BATCH]; pos += BATCH
        xb = augment(tensor_images(x[idx], device))
        ya = torch.from_numpy(y[idx]).to(device)
        perm = torch.randperm(BATCH, device=device)
        lam = float(rng.beta(0.2, 0.2)) if rng.random() < 0.5 else 1.0
        xb = normalize(xb.mul(lam).add_(xb[perm], alpha=1.0-lam))
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = soft_ce(model(xb), ya, ya[perm], lam)
        loss.backward(); opt.step()
        if step < 400:
            mult = (step + 1) / 400
        else:
            mult = 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi*(step-399)/(STEPS-400)))
        for g in opt.param_groups: g["lr"] = 0.12 * mult
        ema_update(avg, model, 0.999 if step >= 1000 else 0.99)
        if (step+1) % 1000 == 0:
            print(f"step {step+1}/{STEPS} loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.5f} elapsed {time.time()-start:.1f}s", flush=True)
    pv = logits(avg, val["x"], device)
    pred = pv.argmax(1)
    print(f"public accuracy {(pred == val['y']).mean():.5f} balanced {balanced(val['y'], pred):.5f}", flush=True)
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True, choices=list(data.INSTANCES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graded", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    val_path = os.path.join(os.path.dirname(a.graded), "val_public.npz")
    model = train(a.instance, a.seed, "cuda", val_path)
    xg = np.load(a.graded)["x"]
    pred = logits(model, xg, "cuda").argmax(1).astype(np.int64)
    tmp = a.out + ".candidate.npy"
    np.save(tmp, pred)
    check = np.load(tmp)
    assert check.shape == (10000,) and check.dtype.kind in "iu" and check.min() >= 0 and check.max() < 100
    os.replace(tmp, a.out)
    print(f"wrote {a.out}: shape={pred.shape} range={pred.min()}..{pred.max()} classes={len(np.unique(pred))}")


if __name__ == "__main__":
    main()
