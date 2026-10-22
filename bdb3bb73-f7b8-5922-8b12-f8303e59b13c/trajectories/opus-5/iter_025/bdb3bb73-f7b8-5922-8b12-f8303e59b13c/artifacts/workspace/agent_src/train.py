"""
Training driver.

Usage:
  python train.py --out DIR [--epochs N] [--holdout 5000] [--seed 0] [--save-ckpts]

Two modes:
  * holdout > 0  : train on (50000 - holdout) images, report accuracy on the held-out
                   train images. Used for *all* hyper-parameter / schedule decisions.
  * holdout == 0 : train on the full 50000-image train split and write the graded
                   checkpoint ladder.

The checkpoint ladder is written one file per epoch as a TorchScript module.
The scripted module shares parameter storage with the live model, so it is built once
before the timing loop and re-serialised each epoch (`.save()` only).
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

T_PROCESS_START = time.time()

from airbench import (DEFAULT_CFG, Augmenter, Net, TTAModel, init_whitening,
                      load_split, lr_lambda_factory, make_optimizer, standardize)


def evaluate(model, x_std, y, batch=1000):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, x_std.shape[0], batch):
            out = model(x_std[i:i + batch])
            correct += (out.argmax(1) == y[i:i + batch]).sum().item()
    model.train()
    return correct / x_std.shape[0]


def train(cfg, out_dir, holdout=0, seed=0, save_ckpts=True, eval_every=0,
          eval_data=None, log=print, warmup_iters=6):
    dev = "cuda"
    torch.manual_seed(seed)

    x_u8, y_all = load_split("train", dev)
    n_total = x_u8.shape[0]

    # deterministic holdout carve-out (same permutation for every seed so that runs are
    # compared on identical data; the *training* shuffle is what the seed controls)
    g = torch.Generator(device="cpu").manual_seed(1234)
    perm = torch.randperm(n_total, generator=g).to(dev)
    if holdout > 0:
        tr_idx, va_idx = perm[holdout:], perm[:holdout]
    else:
        tr_idx, va_idx = perm, perm[:0]

    x_std_all = standardize(x_u8, torch.float32)
    x_tr = x_std_all[tr_idx].contiguous()
    y_tr = y_all[tr_idx].contiguous()
    x_va = x_std_all[va_idx].contiguous() if holdout > 0 else None
    y_va = y_all[va_idx].contiguous() if holdout > 0 else None
    del x_std_all

    n_train = x_tr.shape[0]
    bs = cfg["batch_size"]
    steps_per_epoch = n_train // bs          # one genuine pass over the train split
    total_steps = steps_per_epoch * cfg["epochs"]

    net = Net(widths=cfg["widths"], bn_momentum=cfg["bn_momentum"],
              scaling_factor=cfg["scaling_factor"],
              logit_scale=cfg["logit_scale"]).to(dev).to(memory_format=torch.channels_last)
    init_whitening(net.whiten, x_tr)

    aug = Augmenter(x_tr.to(torch.bfloat16), pad=cfg["pad"], cutout=cfg.get("cutout", 0))

    opt = make_optimizer(net, cfg, bs)
    lams = lr_lambda_factory(total_steps, steps_per_epoch * cfg["whiten_bias_epochs"], cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lams)

    scripted = torch.jit.script(TTAModel(net)) if save_ckpts else None
    if save_ckpts:
        os.makedirs(out_dir, exist_ok=True)

    # `torch.compile` fuses the memory-bound BN/GELU/pool chains. It shares parameter
    # storage with `net`, so `scripted` (built from the eager module) still serialises
    # the live weights. All compilation happens in the warmup below, i.e. before the
    # first checkpoint, so it costs session time but not epoch time.
    fwd = torch.compile(net, mode=cfg.get("compile_mode", "default")) if cfg.get("compile") else net

    # ---- pre-timing warmup: cuDNN algorithm selection + kernel autotune on synthetic
    # noise (never train data, never test data). Weights/optimizer state are restored,
    # so this only pays down one-time kernel-selection cost.
    state0 = {k: v.detach().clone() for k, v in net.state_dict().items()}
    dummy_x = torch.randn(bs, 3, 32, 32, device=dev).to(memory_format=torch.channels_last)
    dummy_y = torch.randint(0, 10, (bs,), device=dev)
    net.train()
    for _ in range(warmup_iters):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(fwd(dummy_x), dummy_y,
                                   label_smoothing=cfg["label_smoothing"], reduction="sum")
        loss.backward()
        opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        for _ in range(2):
            net.eval()
            scripted(dummy_x.float()) if scripted is not None else net(dummy_x)
    net.load_state_dict(state0)
    opt.zero_grad(set_to_none=True)
    for gr in opt.param_groups:
        for p in gr["params"]:
            opt.state.pop(p, None)
    del state0, dummy_x, dummy_y
    torch.cuda.synchronize()

    net.train()
    history = []
    t_start = time.time()
    step = 0
    for epoch in range(1, cfg["epochs"] + 1):
        order = torch.randperm(n_train, device=dev)
        for i in range(steps_per_epoch):
            idx = order[i * bs:(i + 1) * bs]
            xb = aug(idx).float().to(memory_format=torch.channels_last)
            yb = y_tr[idx]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = fwd(xb)
                loss = F.cross_entropy(out, yb, label_smoothing=cfg["label_smoothing"],
                                       reduction="sum")
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1

        if save_ckpts:
            net.eval()
            scripted.save(os.path.join(out_dir, f"ckpt_epoch_{epoch:04d}.pt"))
            net.train()
        rec = dict(epoch=epoch, t=time.time() - t_start,
                       t_run=time.time() - T_PROCESS_START)
        if eval_every and (epoch % eval_every == 0 or epoch == cfg["epochs"]):
            ev_x, ev_y = (x_va, y_va) if eval_data is None else eval_data
            if ev_x is not None:
                net.eval()
                tta = TTAModel(net)
                rec["acc"] = evaluate(tta, ev_x, ev_y)
                net.train()
        history.append(rec)
        log(f"epoch {epoch:3d}  t={rec['t']:7.3f}s" +
            (f"  acc={rec['acc']*100:.2f}%" if "acc" in rec else ""))

    torch.cuda.synchronize()
    total = time.time() - t_start
    final = {}
    net.eval()
    tta = TTAModel(net)
    if holdout > 0:
        final["val_acc"] = evaluate(tta, x_va, y_va)
    return dict(history=history, total_sec=total, final=final, net=net, tta=tta,
                setup_sec=t_start - T_PROCESS_START,
                run_wall_clock_sec=history[-1]["t_run"] if history else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/scratch_ckpt")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--holdout", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-ckpts", action="store_true")
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    cfg = dict(DEFAULT_CFG)
    cfg.update(json.loads(args.cfg))
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    r = train(cfg, args.out, holdout=args.holdout, seed=args.seed,
              save_ckpts=args.save_ckpts, eval_every=args.eval_every)
    print(json.dumps({k: v for k, v in r.items() if k not in ("net", "tta")}, default=str))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"cfg": {k: v for k, v in cfg.items()},
                       "history": r["history"], "total_sec": r["total_sec"],
                       "setup_sec": r["setup_sec"],
                       "run_wall_clock_sec": r["run_wall_clock_sec"],
                       "final": r["final"], "seed": args.seed}, f, default=str)


if __name__ == "__main__":
    main()
