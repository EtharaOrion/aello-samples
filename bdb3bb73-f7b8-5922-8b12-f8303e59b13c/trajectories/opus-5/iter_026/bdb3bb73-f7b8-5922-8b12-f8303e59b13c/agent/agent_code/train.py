"""Train the CIFAR-10 whitening convnet and emit a per-epoch checkpoint ladder.

Timing contract
---------------
The verifier reconstructs the run's clock from the *save times* of the periodic
checkpoints, so this script does exactly one thing between consecutive saves:
one genuine pass over the training split.  No evaluation, no logging that
syncs the GPU, no sleeping.  Accuracy is measured afterwards, out of band, by
verify_ladder.py.

cuDNN autotuning (`benchmark=True`) is warmed up *before* the timer-relevant
region using a throwaway model fed random noise, so that the algorithm search
is not charged to any epoch and every epoch costs the same.
"""
import argparse, json, os, shutil, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (Net, BatchNorm, load_split, standardize, init_whitening,
                    rand_crop_flip)

DEFAULTS = dict(
    epochs=16,
    batch_size=1000,
    lr=11.5,             # per-1024-example units, airbench parameterisation
    momentum=0.85,
    weight_decay=0.0153,
    bias_scaler=64.0,
    label_smoothing=0.2,
    whiten_bias_epochs=3,
    widths=(128, 384, 512),
    bn_momentum=0.6,
    scale=1.0 / 9,
    pad=2,
    peak_frac=0.23,      # fraction of steps at which the triangular LR peaks
    tail_epochs=0,       # final epochs held at lr_end (ladder robustness)
    lr_start=0.2,
    lr_end=0.07,
    seed=0,
)


_DATA_CACHE = {}


def prepare_data(holdout, pad, dev="cuda"):
    """Standardize + reflect-pad the train split once and keep it resident.

    Cached across runs inside one process so a sweep pays the ~1 s data prep a
    single time; a fresh `python train.py` (i.e. the graded run) still pays it,
    and it is included in the reported wall clock.
    """
    key = (holdout, pad)
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]
    xtr_u8, ytr = load_split("train", dev)
    if holdout > 0:
        # deterministic split of the TRAIN set; the test split is never read
        g = torch.Generator(device="cpu"); g.manual_seed(1234567)
        perm = torch.randperm(xtr_u8.size(0), generator=g).to(dev)
        hold_idx, tr_idx = perm[:holdout], perm[holdout:]
        xho, yho = standardize(xtr_u8[hold_idx], torch.half), ytr[hold_idx]
        xtr_u8, ytr = xtr_u8[tr_idx], ytr[tr_idx]
    else:
        xho = yho = None
    xtr = standardize(xtr_u8, torch.float32)
    whiten_src = xtr[:10000].clone()
    xpad = F.pad(xtr, (pad,) * 4, mode="reflect").half()
    xpad = xpad.contiguous(memory_format=torch.channels_last)
    del xtr, xtr_u8
    out = (xpad, ytr, xho, yho, whiten_src)
    _DATA_CACHE[key] = out
    return out


# The configuration that produced the graded ladder. DEFAULTS is deliberately
# left at the *screening* baseline so that re-running sweep.py reproduces the
# logged holdout numbers; SHIPPED is what train.py's CLI uses.
SHIPPED = dict(DEFAULTS, epochs=24, lr_end=0.02, tail_epochs=3, seed=0)


def build_optimizer(model, cfg, total_steps):
    bs = cfg["batch_size"]
    mom = cfg["momentum"]
    # airbench's "kilostep" parameterisation: the loss uses reduction='sum', so
    # the raw LR must be divided by the batch size, and by (1 + 1/(1-momentum))
    # to keep the *effective* step size invariant to the momentum setting.
    kilostep = 1024 * (1 + 1 / (1 - mom))
    lr = cfg["lr"] / kilostep
    wd = cfg["weight_decay"] * bs / kilostep
    lr_bias = lr * cfg["bias_scaler"]

    norm_biases = [p for n, p in model.named_parameters()
                   if p.requires_grad and p.ndim < 2]
    other = [p for n, p in model.named_parameters()
             if p.requires_grad and p.ndim >= 2]
    groups = [
        dict(params=norm_biases, lr=lr_bias, weight_decay=wd / lr_bias),
        dict(params=other, lr=lr, weight_decay=wd / lr),
    ]
    opt = torch.optim.SGD(groups, momentum=mom, nesterov=True)
    base = [g["lr"] for g in opt.param_groups]
    # Triangular warmup/decay, then optionally hold at `lr_end` for the last
    # `tail_epochs` epochs. The hold exists for ladder robustness, not for
    # peak accuracy: with a schedule that is still decaying at the final step,
    # only the very last checkpoint sits at the converged accuracy and the
    # ladder has exactly one checkpoint above the target. Holding a ~2%-of-peak
    # LR for a few epochs makes the last several checkpoints equivalent.
    tail_steps = int(cfg.get("tail_epochs", 0)) * cfg["steps_per_epoch"]
    knee = total_steps - tail_steps
    xs = [0, int(cfg["peak_frac"] * knee), knee, total_steps]
    ys = [cfg["lr_start"], 1.0, cfg["lr_end"], cfg["lr_end"]]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, [lambda s: float(np.interp(s, xs, ys))] * len(groups))
    return opt, sched, base


def make_model(cfg, whiten_source, device="cuda"):
    torch.manual_seed(cfg["seed"])
    m = Net(widths=tuple(cfg["widths"]), bn_mom=cfg["bn_momentum"],
            scale=cfg["scale"], tta=True).to(device)
    init_whitening(m.whiten, whiten_source)
    m = m.to(memory_format=torch.channels_last).half()
    for mod in m.modules():
        if isinstance(mod, BatchNorm):
            mod.float()
    return m


def warmup_cudnn(cfg, whiten_source, steps=3):
    """Pay the cuDNN algorithm search on a throwaway model + random noise."""
    m = make_model(dict(cfg, seed=cfg["seed"] + 991), whiten_source)
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.0)
    bs = cfg["batch_size"]
    x = torch.randn(bs, 3, 32, 32, device="cuda",
                    dtype=torch.half).contiguous(memory_format=torch.channels_last)
    y = torch.randint(0, 10, (bs,), device="cuda")
    m.train()
    for _ in range(steps):
        loss = F.cross_entropy(m.logits(x).float(), y, reduction="sum")
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    m.eval()
    with torch.no_grad():
        m(torch.randn(bs, 3, 32, 32, device="cuda", dtype=torch.half))
    del m, opt, x, y
    torch.cuda.synchronize()


def run(cfg, ckpt_dir, holdout=0, save_ladder=True, log_path=None, verbose=True,
        per_epoch_eval=False):
    t_run_start = time.time()
    dev = "cuda"
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator(device=dev); gen.manual_seed(cfg["seed"] + 1234)

    xpad, ytr, xho, yho, whiten_src = prepare_data(holdout, cfg["pad"], dev)
    n = xpad.size(0)

    warmup_cudnn(cfg, whiten_src)

    model = make_model(cfg, whiten_src)
    scripted = torch.jit.script(model)   # shares parameter storage with `model`

    bs = cfg["batch_size"]
    steps_per_epoch = (n + bs - 1) // bs
    total_steps = steps_per_epoch * cfg["epochs"]
    cfg = dict(cfg, steps_per_epoch=steps_per_epoch)
    opt, sched, _ = build_optimizer(model, cfg, total_steps)
    ls = cfg["label_smoothing"]

    if save_ladder:
        os.makedirs(ckpt_dir, exist_ok=True)

    model.train()
    epoch_marks = []
    curve = []
    for epoch in range(1, cfg["epochs"] + 1):
        if epoch == cfg["whiten_bias_epochs"] + 1:
            model.whiten.bias.requires_grad = False
        order = torch.randperm(n, device=dev, generator=gen)
        for s in range(steps_per_epoch):
            idx = order[s * bs:(s + 1) * bs]
            xb = rand_crop_flip(xpad[idx], 32, gen)
            yb = ytr[idx]
            out = model.logits(xb)
            loss = F.cross_entropy(out.float(), yb, label_smoothing=ls,
                                   reduction="sum")
            loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True); sched.step()
        if save_ladder:
            scripted.save(os.path.join(ckpt_dir, f"ckpt_epoch_{epoch:04d}.pt"))
        epoch_marks.append(time.time())
        if per_epoch_eval and xho is not None:
            # sweep-only diagnostic; never enabled for the graded run, because
            # it would land between two checkpoint saves and inflate t*.
            from common import evaluate
            curve.append(evaluate(model, xho, yho, tta=True))
            model.train()

    torch.cuda.synchronize()
    wall = time.time() - t_run_start
    span = epoch_marks[-1] - epoch_marks[0] if len(epoch_marks) > 1 else 0.0
    E = cfg["epochs"]
    t_star_full = span * E / (E - 1) if E > 1 else float("nan")

    res = dict(cfg={k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in cfg.items()},
               wall_clock_sec=wall, ckpt_span_sec=span,
               t_star_if_last_epoch_crosses=t_star_full,
               mean_epoch_sec=span / (E - 1) if E > 1 else float("nan"),
               holdout=holdout)
    if curve:
        res["holdout_curve"] = curve
    if xho is not None:
        from common import evaluate
        res["holdout_acc_tta"] = evaluate(model, xho, yho, tta=True)
        res["holdout_acc_plain"] = evaluate(model, xho, yho, tta=False)
    if verbose:
        print(json.dumps(res), flush=True)
    if log_path:
        with open(log_path, "a") as f:
            f.write(json.dumps(res) + "\n")
    return model, res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=SHIPPED["epochs"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=SHIPPED["lr"])
    ap.add_argument("--wd", type=float, default=SHIPPED["weight_decay"])
    ap.add_argument("--momentum", type=float, default=SHIPPED["momentum"])
    ap.add_argument("--ls", type=float, default=SHIPPED["label_smoothing"])
    ap.add_argument("--widths", type=str, default="128,384,512")
    ap.add_argument("--holdout", type=int, default=0)
    ap.add_argument("--ckpt-dir", type=str, default="/workspace/scratch_ckpt")
    ap.add_argument("--no-ladder", action="store_true")
    ap.add_argument("--log", type=str, default=None)
    a = ap.parse_args()
    cfg = dict(SHIPPED)
    cfg.update(epochs=a.epochs, seed=a.seed, lr=a.lr, weight_decay=a.wd,
               momentum=a.momentum, label_smoothing=a.ls,
               widths=tuple(int(v) for v in a.widths.split(",")))
    run(cfg, a.ckpt_dir, holdout=a.holdout, save_ladder=not a.no_ladder,
        log_path=a.log)
