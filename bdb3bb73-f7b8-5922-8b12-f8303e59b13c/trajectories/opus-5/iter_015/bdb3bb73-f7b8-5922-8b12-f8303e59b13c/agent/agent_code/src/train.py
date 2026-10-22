"""CIFAR-10 -> 94% speedrun trainer.

Writes one scripted checkpoint per epoch to `--ckpt-dir` as
`ckpt_epoch_{NNNN}.pt` (contiguous, 1-indexed) so the grading ladder is complete
and every rung is independently loadable with `torch.jit.load`.
"""
import argparse
import json
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import Augmenter, load_split, train_val_split
from net import CifarNet, PlainModel, TTAModel

# Stamped at import so we can report an end-to-end number that includes data loading,
# whitening estimation, kernel warmup and scripting -- not just the epoch loop.
T_PROCESS_START = time.perf_counter()

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# --------------------------------------------------------------------------------------
# frozen ZCA whitening
# --------------------------------------------------------------------------------------
def init_whitening(conv, images, eps=5e-4, max_images=10000, seed=0):
    """Set `conv.weight` to the ZCA filters of 2x2 image patches (then freeze it).

    The covariance is estimated over *every* 2x2 patch of a random subset of training
    images (all patch positions, not a strided subsample) so no spatial bias creeps in.
    Output channels are [E/sqrt(lam+eps), -E/sqrt(lam+eps)] so that the following GELU
    sees both signs of every projection.
    """
    k = conv.weight.shape[-1]
    g = torch.Generator(device="cpu").manual_seed(seed)
    sel = torch.randperm(len(images), generator=g)[:max_images].to(images.device)
    imgs = images[sel].float()
    # (N, C, H', W', k, k) -> (N*H'*W', C*k*k)
    patches = imgs.unfold(2, k, 1).unfold(3, k, 1)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, imgs.size(1) * k * k)
    patches = patches - patches.mean(0, keepdim=True)
    cov = (patches.T @ patches) / patches.size(0)
    eigval, eigvec = torch.linalg.eigh(cov.double())
    eigval = eigval.flip(0).float()
    eigvec = eigvec.T.reshape(-1, imgs.size(1), k, k).flip(0).float()
    filters = eigvec / torch.sqrt(eigval.clamp_min(0).view(-1, 1, 1, 1) + eps)
    with torch.no_grad():
        conv.weight.copy_(torch.cat([filters, -filters]).to(conv.weight.dtype))
        conv.bias.zero_()


# --------------------------------------------------------------------------------------
# schedule / optimizer
# --------------------------------------------------------------------------------------
def make_optimizer(model, cfg, batch_size):
    momentum = cfg["momentum"]
    # airbench-style parameterisation: `lr`/`wd` are quoted per 1024 examples and per
    # unit of (1 + 1/(1-momentum)) so that changing batch size or momentum keeps the
    # effective step size fixed.
    kilostep = 1024 * (1 + 1 / (1 - momentum))
    lr = cfg["lr"] / kilostep
    wd = cfg["weight_decay"] * batch_size / kilostep
    lr_bias = lr * cfg["bias_scaler"]

    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    other = [p for n, p in model.named_parameters() if "norm" not in n and p.requires_grad]
    groups = [
        dict(params=other, lr=lr, weight_decay=wd / lr),
        dict(params=norm_biases, lr=lr_bias, weight_decay=wd / lr_bias),
    ]
    return torch.optim.SGD(groups, momentum=momentum, nesterov=True)


def lr_lambda_factory(total_steps, warmup_frac=0.23, lo=0.2, hi=1.0, end=0.07):
    warmup = max(1, int(total_steps * warmup_frac))
    warmdown = max(1, total_steps - warmup)

    def f(step):
        if step < warmup:
            t = step / warmup
            return lo * (1 - t) + hi * t
        t = min(1.0, (step - warmup) / warmdown)
        return hi * (1 - t) + end * t

    return f


# --------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(module, x, y, batch=1000):
    module.eval()
    correct = 0
    for i in range(0, len(x), batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = module(x[i : i + batch].float())
        correct += (out.float().argmax(1) == y[i : i + batch]).sum().item()
    module.train()
    return correct / len(x)


DEFAULT_CFG = dict(
    epochs=20,
    batch_size=1024,
    lr=11.5,
    momentum=0.85,
    weight_decay=0.0153,
    bias_scaler=64.0,
    label_smoothing=0.2,
    whiten_bias_frac=0.30,   # fraction of training during which the whiten bias is trained
    w1=64, w2=256, w3=256,
    bn_momentum=0.6,
    scale=1.0 / 9.0,
    pad=2, flip=True, cutout=0,
    warmup_frac=0.23, lr_lo=0.2, lr_end=0.07,
    tta=True,
)


def run(cfg, ckpt_dir, seed=0, holdout=0, save_ckpts=True, log_path=None, quiet=False,
        track_val=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    xtr, ytr = load_split("train")
    if holdout > 0:
        xtr, ytr, xva, yva = train_val_split(xtr, ytr, n_val=holdout, seed=0)
    else:
        xva = yva = None

    from net import Conv as _Conv
    _Conv.dirac = cfg.get("dirac", True)
    model = CifarNet(cfg["w1"], cfg["w2"], cfg["w3"], cfg["bn_momentum"], cfg["scale"]).cuda()
    _Conv.dirac = True
    model = model.to(memory_format=torch.channels_last)
    if cfg.get("whiten", True):
        init_whitening(model.whiten, xtr, seed=seed)
    else:
        # Ablation: keep the same layer shape but let it train from a standard init.
        model.whiten.weight.requires_grad = True

    bs = cfg["batch_size"]
    steps_per_epoch = len(xtr) // bs
    total_steps = steps_per_epoch * cfg["epochs"]
    opt = make_optimizer(model, cfg, bs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda_factory(total_steps, cfg["warmup_frac"], cfg["lr_lo"], 1.0, cfg["lr_end"])
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"], reduction="sum")
    aug = Augmenter(xtr, pad=cfg["pad"], flip=cfg["flip"], cutout=cfg["cutout"])

    wrapper = (TTAModel(model) if cfg["tta"] else PlainModel(model)).cuda()
    if save_ckpts:
        os.makedirs(ckpt_dir, exist_ok=True)
        for f in os.listdir(ckpt_dir):
            if f.endswith(".pt"):
                os.remove(os.path.join(ckpt_dir, f))

    # ---- warm up kernels *before* the timed region (cudnn.benchmark autotune, lazy
    # ---- init of the scripter).  This is one-off setup, not training work.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(aug(torch.arange(bs, device=xtr.device))).sum().backward()
    opt.zero_grad(set_to_none=True)
    model.head.weight.data.zero_()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
    scripted = torch.jit.script(wrapper.eval())
    wrapper.train()
    torch.cuda.synchronize()

    whiten_bias_steps = int(total_steps * cfg["whiten_bias_frac"])
    history = []
    val_curve = []
    t0 = time.perf_counter()
    step = 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        perm = torch.randperm(len(xtr), device=xtr.device)
        for i in range(steps_per_epoch):
            idx = perm[i * bs : (i + 1) * bs]
            model.whiten.bias.requires_grad = step < whiten_bias_steps
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(aug(idx))
                loss = loss_fn(out.float(), ytr[idx])
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
        if save_ckpts:
            scripted.load_state_dict(wrapper.state_dict())
            scripted.eval()
            scripted.save(os.path.join(ckpt_dir, f"ckpt_epoch_{epoch:04d}.pt"))
        history.append(time.perf_counter() - t0)
        # Only ever enabled for holdout sweeps -- it would otherwise sit inside the
        # graded run's inter-checkpoint clock.
        if track_val and xva is not None:
            val_curve.append(round(evaluate(wrapper, xva, yva), 4))
            t0 += 0.0  # (evaluation time is deliberately not excluded; sweeps are untimed)
    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0
    end_to_end = time.perf_counter() - T_PROCESS_START

    result = dict(cfg=cfg, seed=seed, holdout=holdout, total_time=total_time,
                  end_to_end=end_to_end, epoch_times=history)
    if save_ckpts:
        names = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))
        mt = [os.stat(os.path.join(ckpt_dir, n)).st_mtime for n in names]
        result["ckpt_span"] = mt[-1] - mt[0]
        with open("/workspace/logs/run_manifest.json", "w") as f:
            json.dump({k: v for k, v in result.items() if k != "epoch_times"}, f, indent=2)
    if xva is not None:
        result["val_acc"] = evaluate(wrapper, xva, yva)
        if track_val:
            result["val_curve"] = val_curve
            cross = [i + 1 for i, a in enumerate(val_curve) if a >= 0.94]
            result["val_cross_epoch"] = cross[0] if cross else None
    if not quiet:
        print(json.dumps({k: v for k, v in result.items() if k != "epoch_times"}), flush=True)
    if log_path:
        with open(log_path, "a") as f:
            f.write(json.dumps(result) + "\n")
    return result, wrapper, model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="/workspace/scratch_ckpt")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=0)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--cfg", default=None, help="JSON dict of config overrides")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    cfg = dict(DEFAULT_CFG)
    if args.cfg:
        cfg.update(json.loads(args.cfg))
    if args.epochs:
        cfg["epochs"] = args.epochs

    res, _, _ = run(cfg, args.ckpt_dir, seed=args.seed, holdout=args.holdout,
                    save_ckpts=not args.no_save, log_path=args.log)
