"""Post-hoc BatchNorm statistic recalibration for weight-averaged (EMA) snapshots.

EMA'd running_mean/var do not match EMA'd weights, which is the usual reason an EMA
snapshot scores below the raw model. This re-estimates the BN statistics with
forward-only passes (no gradients, no optimizer updates, no parameter changes) and
reports val balanced accuracy for each recipe.
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aello_data as data
from train import Net, Aug, bal_acc, sinkhorn_balance, logits_tta


def recal_bn(model, X, mean, std, aug, gen, nbatch, batch, use_aug):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None          # cumulative average over the passes
    model.train()
    n = X.shape[0]
    with torch.no_grad():
        for _ in range(nbatch):
            idx = torch.randint(0, n, (batch,), device=X.device, generator=gen)
            xb = X[idx].float().div_(255.)
            if use_aug:
                xb = aug(xb, gen)
            else:
                if aug.res != xb.shape[-1]:
                    xb = F.interpolate(xb, size=(aug.res, aug.res), mode="bilinear",
                                       align_corners=False)
                xb = xb.contiguous(memory_format=torch.channels_last)
            xb = ((xb - mean) / std).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(xb)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--nbatch", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--tta", default="flip")
    a = ap.parse_args()
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    ck = torch.load(a.ckpt, map_location="cpu")
    cfg = ck["cfg"]
    mean, std = ck["mean"].to(dev), ck["std"].to(dev)
    depths = tuple(int(v) for v in cfg["depths"].split(","))
    x, y = data.load_train(cfg["instance"])
    xv, yv = data.load_val()
    X = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).contiguous().to(dev)
    XV = torch.from_numpy(np.ascontiguousarray(xv)).permute(0, 3, 1, 2).contiguous().to(dev)
    aug = Aug(cfg["res"], cfg["pad"], cfg["cut"], dev)
    gen = torch.Generator(device=dev)

    store = {}
    for nm, sd in ck["models"].items():
        for mode in ("asis", "aug", "clean"):
            if nm == "raw" and mode != "asis":
                pass
            m = Net(data.N_CLASSES, cfg["width"], depths, cfg["stem"]).to(dev)
            m = m.to(memory_format=torch.channels_last)
            m.load_state_dict(sd)
            if mode != "asis":
                gen.manual_seed(7)
                recal_bn(m, X, mean, std, aug, gen, a.nbatch, a.batch, mode == "aug")
            lv = logits_tta(m, XV, cfg["res"], mean, std, dev, views=a.tta).cpu().numpy()
            store[(nm, mode)] = lv
            print("  %-10s %-6s val %.4f   +sink0.7 %.4f" %
                  (nm, mode, bal_acc(lv.argmax(1), yv),
                   bal_acc(sinkhorn_balance(lv, temp=0.7).argmax(1), yv)), flush=True)
            del m
    # ensembles across snapshots, per recal mode
    names = list(ck["models"].keys())
    print("\n snapshot ensembles:")
    for mode in ("asis", "aug", "clean"):
        for combo in [["raw", names[1]], names]:
            if any((c, mode if c != "raw" else mode) not in store for c in combo):
                continue
            lv = sum(store[(c, mode)] for c in combo) / len(combo)
            print("  %-6s %-26s val %.4f   +sink0.7 %.4f" %
                  (mode, "+".join(combo), bal_acc(lv.argmax(1), yv),
                   bal_acc(sinkhorn_balance(lv, temp=0.7).argmax(1), yv)), flush=True)
    # cross-mode: raw(asis) + ema(best recal)
    print("\n mixed:")
    for mode in ("aug", "clean"):
        lv = (store[("raw", "asis")] + sum(store[(n, mode)] for n in names[1:])) / len(names)
        print("  raw_asis+emas_%-6s val %.4f   +sink0.7 %.4f" %
              (mode, bal_acc(lv.argmax(1), yv),
               bal_acc(sinkhorn_balance(lv, temp=0.7).argmax(1), yv)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
