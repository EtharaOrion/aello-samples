"""Hyperparameter sweep on a held-out slice of the TRAIN split.

One process, one compiled model reused across configs (weights reset in place),
so the ~13 s compile cost is paid once instead of per config. Selection uses a
5000-image validation split carved out of train -- the test split is never used
for tuning.
"""
import copy
import json
import sys
import time

import torch
import torch.nn.functional as F

from model import Net, init_whitening_conv
from train import Augment, load_split, standardize

DEV = torch.device("cuda")

BASE = dict(epochs=18, bs=1000, lr=4.0, wd=1.2e-3, bias_lr_scale=16.0, momentum=0.85,
            label_smoothing=0.2, warmup_frac=0.20, final_lr_frac=0.02, cutout=0)


def train_one(cfg, model, fwd, init_sd, xtr_pad, ytr, xva, yva, aug_cache):
    model.load_state_dict(init_sd)
    w = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    b = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    opt = torch.optim.SGD(
        [dict(params=w, lr=cfg["lr"], weight_decay=cfg["wd"]),
         dict(params=b, lr=cfg["lr"] * cfg["bias_lr_scale"], weight_decay=0.0)],
        momentum=cfg["momentum"], nesterov=True)

    aug = aug_cache[cfg["cutout"]]
    ntr, bs = xtr_pad.shape[0], cfg["bs"]
    spe = ntr // bs
    total = spe * cfg["epochs"]
    warm = max(1, int(cfg["warmup_frac"] * total))
    g = torch.Generator(device=DEV).manual_seed(cfg.get("seed", 0) + 7)

    step = 0
    model.train()
    t0 = time.time()
    for _ in range(cfg["epochs"]):
        order = torch.randperm(ntr, device=DEV, generator=g)[:spe * bs].view(spe, bs)
        for i in range(spe):
            idx = order[i]
            f = (0.10 + 0.90 * step / warm) if step < warm else \
                1.0 + (cfg["final_lr_frac"] - 1.0) * (step - warm) / max(1, total - warm)
            opt.param_groups[0]["lr"] = cfg["lr"] * f
            opt.param_groups[1]["lr"] = cfg["lr"] * cfg["bias_lr_scale"] * f
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(fwd(aug(xtr_pad[idx], g)), ytr[idx],
                                       label_smoothing=cfg["label_smoothing"])
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
    torch.cuda.synchronize()
    dt = time.time() - t0

    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(yva), 1000):
            correct += (model(xva[i:i + 1000]).argmax(1) == yva[i:i + 1000]).sum().item()
    model.train()
    return correct / len(yva), dt


def main(configs):
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    xtr_u8, ytr = load_split("train", DEV)
    g0 = torch.Generator(device=DEV).manual_seed(1234)
    perm = torch.randperm(len(ytr), device=DEV, generator=g0)
    vi, ti = perm[:5000], perm[5000:]
    xva, yva = standardize(xtr_u8[vi]), ytr[vi]
    xtr = standardize(xtr_u8[ti])
    ytr = ytr[ti]
    xtr_pad = F.pad(xtr, (2,) * 4, mode="reflect").half().contiguous(memory_format=torch.channels_last)

    model = Net().to(DEV).to(memory_format=torch.channels_last)
    init_whitening_conv(model.body[0], xtr[:10000])
    del xtr, xtr_u8
    init_sd = copy.deepcopy(model.state_dict())
    fwd = torch.compile(model, dynamic=False)
    aug_cache = {c: Augment(2, 32, c, DEV) for c in {cfg.get("cutout", 0) for cfg in configs}}

    results = []
    for cfg_over in configs:
        cfg = {**BASE, **cfg_over}
        acc, dt = train_one(cfg, model, fwd, init_sd, xtr_pad, ytr, xva, yva, aug_cache)
        name = cfg_over.get("name", json.dumps(cfg_over))
        results.append((acc, name, dt))
        print(f"val_acc={acc:.4f}  train={dt:.2f}s  {name}", flush=True)
    print("--- ranked ---")
    for acc, name, dt in sorted(results, reverse=True):
        print(f"{acc:.4f}  {name}")


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
