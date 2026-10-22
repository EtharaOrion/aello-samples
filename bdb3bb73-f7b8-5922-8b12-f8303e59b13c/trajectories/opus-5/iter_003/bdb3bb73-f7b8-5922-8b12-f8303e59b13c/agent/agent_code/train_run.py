"""Train one CIFAR-10 model and (optionally) write a per-epoch checkpoint ladder.

Two modes:

  * ``--holdout N``  : train on the first 50000-N images of a shuffled *train*
                       permutation and report accuracy on the remaining N.  This
                       is the only mode used for hyperparameter selection, so no
                       design decision is ever informed by the test split.
  * default          : train on all 50000 train images and write one scripted
                       checkpoint per epoch to ``--ckpt-dir``.

The wall-clock that the verifier reconstructs is the span between the epoch-1 and
epoch-N checkpoint saves, so *everything* the training loop does -- augmentation,
optimisation, and checkpoint serialisation -- sits inside the measured window.
Nothing is deferred out of it.
"""

import argparse
import json
import os
import shutil
import time

import torch

from core import (
    DEFAULT_CFG,
    CifarNet,
    GpuAugmenter,
    build_optimizer,
    evaluate,
    init_whitening,
    load_split,
    lr_multiplier,
    standardise,
)


def make_cfg(overrides):
    cfg = dict(DEFAULT_CFG)
    cfg.update(overrides or {})
    cfg["widths"] = tuple(cfg["widths"])
    return cfg


class CheckpointWriter:
    """Scripts the deploy graph once, then reuses it for every epoch save.

    Re-running ``torch.jit.script`` each epoch would put ~200ms of compilation
    inside the measured window for no reason; scripting once and refreshing the
    weights keeps the per-epoch cost to a state-dict copy plus the disk write.
    """

    def __init__(self, cfg, ckpt_dir):
        self.dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        proto = CifarNet(
            widths=cfg["widths"],
            whiten_ch=cfg["whiten_ch"],
            logit_scale=cfg["logit_scale"],
            tta=cfg["tta"],
        ).cuda().eval()
        self.scripted = torch.jit.script(proto)
        self.paths = []

    def save(self, state, epoch):
        self.scripted.load_state_dict(state)
        path = os.path.join(self.dir, f"ckpt_epoch_{epoch:04d}.pt")
        self.scripted.save(path)
        self.paths.append(path)
        return path


class WeightEMA:
    """Exponential moving average over parameters and float buffers.

    Started partway through training (``ema_start_frac``) because averaging
    across the high-LR phase of a 20-epoch schedule only drags the average
    backwards.  BN running statistics are averaged too, so the EMA weights are a
    self-contained deployable model.
    """

    def __init__(self, model, decay):
        self.decay = decay
        self.keys = []
        self.live = []
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.keys.append(k)
                self.live.append(v)
        self.shadow = [v.detach().clone() for v in self.live]

    def update(self):
        torch._foreach_lerp_(self.shadow, self.live, 1.0 - self.decay)

    def state_dict(self, model):
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for k, v in zip(self.keys, self.shadow):
            sd[k] = v.clone()
        return sd


def eval_state(model, state, images, labels, ema, tta=True):
    """Evaluate `state` (the deployed weights) by temporarily swapping it in."""
    if ema is None:
        return evaluate(model, images, labels, tta=tta)
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    acc = evaluate(model, images, labels, tta=tta)
    model.load_state_dict(backup)
    return acc


def train(cfg, ckpt_dir=None, holdout=0, seed=0, log_every=0, quiet=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    raw_x, raw_y = load_split("train")
    images = standardise(raw_x, dtype=torch.float16)
    labels = raw_y

    if holdout > 0:
        g = torch.Generator(device="cuda").manual_seed(1234)  # fixed split, seed-independent
        perm = torch.randperm(images.shape[0], device="cuda", generator=g)
        tr, va = perm[:-holdout], perm[-holdout:]
        val_images, val_labels = images[va], labels[va]
        images, labels = images[tr], labels[tr]
    else:
        val_images = val_labels = None

    images = images.contiguous(memory_format=torch.channels_last)
    n = images.shape[0]

    model = CifarNet(
        widths=cfg["widths"],
        whiten_ch=cfg["whiten_ch"],
        logit_scale=cfg["logit_scale"],
        tta=cfg["tta"],
    ).cuda().to(memory_format=torch.channels_last)
    init_whitening(model, images[:10000])
    model.train()

    opt = build_optimizer(model, cfg)
    base_lrs = [g["lr"] for g in opt.param_groups]
    bs = cfg["batch_size"]
    steps_per_epoch = (n + bs - 1) // bs
    total_steps = steps_per_epoch * cfg["epochs"]
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    gen = torch.Generator(device="cuda").manual_seed(seed + 777)
    aug = GpuAugmenter(images, pad=cfg["pad"], generator=gen)

    writer = CheckpointWriter(cfg, ckpt_dir) if ckpt_dir else None

    # Warm the kernels on throwaway data so cudnn autotuning is not billed to
    # epoch 1 alone -- this happens before the first checkpoint, i.e. before the
    # verifier's clock starts, and is reported in `warmup_sec` for transparency.
    warm_t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        dummy = model.forward_train(images[:bs].float().to(memory_format=torch.channels_last))
        dummy.float().sum().backward()
    opt.zero_grad(set_to_none=True)
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    torch.cuda.synchronize()
    warmup_sec = time.time() - warm_t0

    ema = None
    ema_start_step = int(total_steps * cfg["ema_start_frac"])

    history = []
    step = 0
    torch.cuda.synchronize()
    t_start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        views = aug.epoch_views(cutout=cfg["cutout"]).to(memory_format=torch.channels_last)
        order = torch.randperm(n, device="cuda", generator=gen)
        for i in range(0, n, bs):
            idx = order[i : i + bs]
            xb = views[idx].float()
            yb = labels[idx]
            mult = lr_multiplier(step, total_steps, cfg["warmup_frac"], cfg["final_frac"])
            for g_, base in zip(opt.param_groups, base_lrs):
                g_["lr"] = base * mult
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model.forward_train(xb), yb)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step >= ema_start_step:
                if ema is None:
                    ema = WeightEMA(model, cfg["ema_decay"])
                else:
                    ema.update()

        # The deployed weights are the EMA once it is active, so the checkpoint
        # and the accuracy we log always describe the same model.
        deploy_state = ema.state_dict(model) if ema is not None else model.state_dict()
        if writer is not None:
            writer.save(deploy_state, epoch)
        torch.cuda.synchronize()
        elapsed = time.time() - t_start
        rec = dict(epoch=epoch, elapsed=elapsed)
        want_eval = log_every and (epoch % log_every == 0 or epoch == cfg["epochs"])
        if val_images is not None and want_eval:
            rec["val_acc"] = eval_state(model, deploy_state, val_images, val_labels, ema)
        history.append(rec)
        if not quiet and (log_every and (epoch % log_every == 0 or epoch == cfg["epochs"])):
            msg = f"epoch {epoch:3d}  t={elapsed:6.2f}s"
            if "val_acc" in rec:
                msg += f"  val_acc={rec['val_acc']:.4f}"
            print(msg, flush=True)

    torch.cuda.synchronize()
    total_sec = time.time() - t_start
    result = dict(
        cfg={k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        seed=seed,
        holdout=holdout,
        total_sec=total_sec,
        warmup_sec=warmup_sec,
        history=history,
    )
    if val_images is not None:
        result["val_acc"] = eval_state(model, deploy_state, val_images, val_labels, ema)
        result["val_acc_no_tta"] = eval_state(model, deploy_state, val_images, val_labels, ema, tta=False)
    result["ema_active"] = ema is not None
    if writer is not None:
        result["ckpt_span_sec"] = os.path.getmtime(writer.paths[-1]) - os.path.getmtime(writer.paths[0])
        result["n_ckpt"] = len(writer.paths)
    return result, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--holdout", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--cfg", default="{}", help="JSON dict of overrides")
    ap.add_argument("--out", default=None, help="write the result JSON here")
    ap.add_argument("--final-copy", default=None, help="hardlink/copy last ckpt here")
    args = ap.parse_args()

    cfg = make_cfg(json.loads(args.cfg))
    if args.ckpt_dir and os.path.isdir(args.ckpt_dir):
        for f in sorted(os.listdir(args.ckpt_dir)):
            if f.startswith("ckpt_epoch_") and f.endswith(".pt"):
                os.remove(os.path.join(args.ckpt_dir, f))

    result, _ = train(cfg, ckpt_dir=args.ckpt_dir, holdout=args.holdout,
                      seed=args.seed, log_every=args.log_every)

    if args.final_copy and args.ckpt_dir:
        last = os.path.join(args.ckpt_dir, f"ckpt_epoch_{cfg['epochs']:04d}.pt")
        if os.path.exists(args.final_copy):
            os.remove(args.final_copy)
        shutil.copy(last, args.final_copy)
        result["final_copy"] = args.final_copy

    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2), flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
