"""Training loop for the CIFAR-10 speedrun.

One epoch == one genuine pass over all 50,000 training images (50 steps of
batch 1000).  A TorchScript checkpoint with TTA folded in is written after
every epoch.
"""
import json
import math
import os
import time
from dataclasses import dataclass, asdict, field

import torch
import torch.nn as nn

from .data import TrainData, load_split
from .net import TTAModel, init_whitening_conv, make_net


@dataclass
class Config:
    epochs: int = 20
    batch_size: int = 1000
    lr: float = 11.5                # per-1024-examples, rescaled below
    momentum: float = 0.85
    weight_decay: float = 0.0153
    bias_scaler: float = 64.0
    label_smoothing: float = 0.2
    whiten_bias_epochs: int = 3
    warmup_frac: float = 0.23
    lr_init_frac: float = 0.2       # LR multiplier at step 0
    lr_final_frac: float = 0.0      # LR multiplier at the last step
    widths: tuple = (128, 384, 512)
    bn_momentum: float = 0.6
    scaling_factor: float = 1 / 9
    tta: bool = True
    seed: int = 0
    precision: str = "fp16"


def _dtype(cfg):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[cfg.precision]


def build_model(cfg, train_images):
    torch.manual_seed(cfg.seed)
    model = make_net(cfg.widths, cfg.bn_momentum, cfg.scaling_factor).cuda()
    init_whitening_conv(model[0], train_images[:5000].float())
    model = model.to(memory_format=torch.channels_last).to(_dtype(cfg))
    return model


def make_optimizer(cfg, model, steps_per_epoch):
    # Nesterov momentum inflates the effective step by 1 + 1/(1-m) relative to
    # the raw per-example gradient; dividing it out makes `lr` and `momentum`
    # approximately independent knobs.
    kilostep_scale = 1024 * (1 + 1 / (1 - cfg.momentum))
    lr = cfg.lr / kilostep_scale
    wd = cfg.weight_decay * cfg.batch_size / kilostep_scale
    lr_biases = lr * cfg.bias_scaler

    whiten_bias = model[0].bias
    norm_biases, weights = [], []
    for name, p in model.named_parameters():
        if p is whiten_bias or not p.requires_grad:
            continue
        (weights if p.ndim >= 2 else norm_biases).append(p)

    groups = [
        dict(params=norm_biases, lr=lr_biases, weight_decay=wd / lr_biases),
        dict(params=weights, lr=lr, weight_decay=wd / lr),
        dict(params=[whiten_bias], lr=lr, weight_decay=wd / lr),
    ]
    opt = torch.optim.SGD(groups, momentum=cfg.momentum, nesterov=True)

    total = cfg.epochs * steps_per_epoch
    warm = max(1, int(total * cfg.warmup_frac))
    whiten_steps = cfg.whiten_bias_epochs * steps_per_epoch

    def base(step):
        if step < warm:
            f = step / warm
            return cfg.lr_init_frac * (1 - f) + f
        f = (step - warm) / max(1, total - warm)
        return (1 - f) + cfg.lr_final_frac * f

    lambdas = [base, base, lambda s: base(s) if s < whiten_steps else 0.0]
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambdas)


def evaluate(model, images, labels, tta_module, batch=1000):
    """Accuracy through the same TTA wrapper that gets checkpointed."""
    was_training = model.training
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(images), batch):
            out = tta_module(images[i:i + batch].float())
            correct += (out.argmax(1) == labels[i:i + batch]).sum().item()
    if was_training:
        model.train()
    return correct / len(images)


def train(cfg, train_images, train_labels, outdir=None, eval_sets=None,
          log=print, save_every=1, eval_epochs=None, compile_mode=None):
    """Run one training job.  Returns a dict of per-epoch records."""
    torch.manual_seed(cfg.seed)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(cfg.seed + 1234)

    data = TrainData(train_images, train_labels)
    model = build_model(cfg, train_images)
    net = torch.compile(model, mode=compile_mode) if compile_mode else model
    steps_per_epoch = math.ceil(data.n / cfg.batch_size)
    opt, sched = make_optimizer(cfg, model, steps_per_epoch)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing, reduction="sum")

    tta_module = TTAModel(model, tta=cfg.tta).to(_dtype(cfg)).cuda().eval()
    scripted = None
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        scripted = torch.jit.script(tta_module)
        scripted.eval()

    records = []
    torch.cuda.synchronize()
    t_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(data.n, device="cuda", generator=gen)
        for s in range(steps_per_epoch):
            idx = perm[s * cfg.batch_size:(s + 1) * cfg.batch_size]
            x, y = data.batch(idx, gen)
            loss = loss_fn(net(x), y)
            model.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

        rec = {"epoch": epoch}
        if outdir is not None and epoch % save_every == 0:
            scripted.load_state_dict(tta_module.state_dict())
            scripted.save(os.path.join(outdir, f"ckpt_epoch_{epoch:04d}.pt"))
        torch.cuda.synchronize()
        rec["t"] = time.time() - t_start
        if eval_sets and (eval_epochs is None or epoch in eval_epochs):
            for name, (xs, ys) in eval_sets.items():
                rec[name] = evaluate(model, xs, ys, tta_module)
        records.append(rec)
        log(f"  epoch {epoch:3d}  t={rec['t']:7.2f}s  " +
            "  ".join(f"{k}={rec[k]:.4f}" for k in rec if k not in ("epoch", "t")))

    return {"config": asdict(cfg), "records": records,
            "wall_clock_sec": time.time() - t_start}
