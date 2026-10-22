"""Offline test-time augmentation on the weights of an already-budgeted training run.

Inference only: no optimiser steps, no training-sample presentations. Computes each
TTA *atom* (shift x scale x flip) once, then scores a handful of pre-stated symmetric
view sets on val_public and writes the graded predictions for the chosen rule.
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from train import Net, bal_acc, N_CLASSES, DATA


@torch.no_grad()
def atom(model, x, mean, std, dy, dx, r, flip, bs=2000):
    out = []
    for s in range(0, len(x), bs):
        z = x[s:s + bs].permute(0, 3, 1, 2).float().div_(255.).sub_(mean).div_(std)
        p = max(abs(dy), abs(dx))
        if p:
            z = F.pad(z, (p,) * 4, mode="reflect")
            h = z.shape[-1] - 2 * p
            z = z[:, :, p + dy:p + dy + h, p + dx:p + dx + h]
        if r != z.shape[-1]:
            z = F.interpolate(z, size=(r, r), mode="bilinear", align_corners=False)
        if flip:
            z = z.flip(3)
        z = z.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(z).float().softmax(1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/probs/R5.pt")
    ap.add_argument("--graded", default=os.path.join(DATA, "graded_images.npz"))
    ap.add_argument("--out", default="")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--rule", default="B")
    ap.add_argument("--shift", type=int, default=2)
    a = ap.parse_args()

    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    ck = torch.load(a.ckpt, map_location=dev)
    cfg = ck["cfg"]
    model = Net(cfg["width"], tuple(int(v) for v in cfg["blocks"].split(","))).to(dev)
    model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(ck["sd"])
    model.eval()
    mean, std = ck["mean"].to(dev), ck["std"].to(dev)
    base = cfg["res"]

    dv = np.load(os.path.join(DATA, "val_public.npz"))
    xva = torch.from_numpy(dv["x"]).to(dev)
    yva = dv["y"]
    xgr = torch.from_numpy(np.load(a.graded)["x"]).to(dev)
    d = np.load(os.path.join(DATA, "train_i20k.npz"))
    prior = np.bincount(d["y"], minlength=N_CLASSES).astype(np.float64)
    prior /= prior.sum()

    s = a.shift
    offs = [(dy, dx) for dy in (-s, 0, s) for dx in (-s, 0, s)]
    scales = {"a": base, "b": int(round(base * 1.17 / 8)) * 8, "c": int(round(base * 1.33 / 8)) * 8}
    V, G = {}, {}
    t0 = __import__("time").time()
    for sk, r in scales.items():
        for i, (dy, dx) in enumerate(offs):
            for fl in (0, 1):
                k = f"{sk}{i}{fl}"
                V[k] = atom(model, xva, mean, std, dy, dx, r, fl)
                G[k] = atom(model, xgr, mean, std, dy, dx, r, fl)
    print(f"atoms: {len(V)} in {__import__('time').time() - t0:.0f}s")

    def sel(scale_keys, use_offs):
        return [f"{sk}{i}{fl}" for sk in scale_keys for i in use_offs for fl in (0, 1)]

    rules = {                                   # pre-stated symmetric view sets
        "centre": sel("a", [4]),
        "A": sel("a", range(9)),                              # 3x3 grid @ base res
        "B": sel("ab", range(9)),                             # 3x3 grid @ base + 1.17x
        "C": sel("abc", range(9)),                            # 3x3 grid @ 3 scales
        "D": sel("ab", [1, 3, 4, 5, 7]),                      # plus-shape shifts @ 2 scales
        "E": sel("b", range(9)),
    }
    rep = {}
    for name, keys in rules.items():
        v = sum(V[k] for k in keys)
        v /= v.sum(1, keepdims=True)
        for tau in (0.0, 0.5, 1.0):
            rep[f"{name}_t{tau}"] = round(bal_acc(v / prior[None, :] ** tau, yva), 4)
    print(json.dumps(rep, indent=0))

    keys = rules[a.rule]
    v = sum(V[k] for k in keys)
    v /= v.sum(1, keepdims=True)
    vs = bal_acc(v / prior[None, :] ** a.tau, yva)
    g = sum(G[k] for k in keys)
    g /= g.sum(1, keepdims=True)
    pred = (g / prior[None, :] ** a.tau).argmax(1).astype(np.int64)
    print(f"rule {a.rule} tau {a.tau}: val {vs:.4f}")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        np.save(a.out, pred)
        print("wrote", a.out, pred.shape, pred.dtype, pred.min(), pred.max())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
