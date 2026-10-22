"""Prediction-time exploration on an already-trained checkpoint.

Training is what the budget constrains; TTA and calibration at prediction time are
explicitly allowed.  This script reloads a checkpoint written by train.py --save-model
and sweeps prediction-time choices (TTA view sets, prob- vs log-prob averaging,
temperature, and the class-marginal correction), scoring each on the public val split.
No training happens here.
"""
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aello_data as data
from train import (GPUData, make_model, balance_logits, balanced_acc, logits_for,
                   TTA_CROPS)


@torch.no_grad()
def views_logits(model, xb_norm, offs, pad, bs=1000, mode="log", flip=True):
    """Return a (V, N, C) stack of per-view log-probs (or probs)."""
    model.eval()
    h = xb_norm.shape[-1]
    chunks = []
    for s in range(0, xb_norm.shape[0], bs):
        xb = xb_norm[s:s + bs]
        xp = F.pad(xb, (pad,) * 4, mode="reflect")
        vs = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for (oy, ox) in offs:
                v = xp[:, :, oy:oy + h, ox:ox + h].contiguous(
                    memory_format=torch.channels_last)
                for vv in ((v, v.flip(3)) if flip else (v,)):
                    lg = F.log_softmax(model(vv).float(), 1)
                    vs.append(lg if mode == "log" else lg.exp())
        chunks.append(torch.stack(vs))
    return torch.cat(chunks, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--graded", default="../data/graded_images.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--tta", default=None)
    a = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    ck = torch.load(a.ckpt, map_location="cuda")
    ta = ck["args"]
    x, y = data.load_train(ta["instance"])
    ds = GPUData(x, y, "cuda", pad=ta["pad"], res=ta["res"])
    blocks = tuple(int(v) for v in ta["blocks"].split(",")) if ta["blocks"] else None
    model = make_model(ta["seed"], data.N_CLASSES, ta["arch"], ta["width"], ta["dropout"],
                       ta["se"], blocks, ta["dsample"], ta["branches"],
                       ta.get("droppath", 0.0))
    model = model.to("cuda").to(memory_format=torch.channels_last)
    model.load_state_dict(ck["sd"])
    model.eval()

    xv, yv = data.load_val()
    nv = ds.norm(xv)
    p = ds.pad
    t0 = time.time()

    # ---- view sets to compare (all are crops of the reflect-padded image) ----
    a2, b2 = max(1, p // 2), p + max(1, p // 2)
    a3, b3 = max(1, p - p // 3), p + p // 3
    SETS = {
        "center":  [(p, p)],
        "flip5":   [(p, p), (a2, a2), (a2, b2), (b2, a2), (b2, b2)],
        "flip5s":  [(p, p), (a3, a3), (a3, b3), (b3, a3), (b3, b3)],
        "flip9":   [(oy, ox) for oy in (p - a2, p, p + a2) for ox in (p - a2, p, p + a2)],
        "flip5p":  [(p, p), (p, a2), (p, b2), (a2, p), (b2, p)],
    }
    cache = {}
    for name, offs in SETS.items():
        V = views_logits(model, nv, offs, p, mode="log")
        cache[name] = V
        for agg in ("log", "prob"):
            L = V.mean(0) if agg == "log" else V.exp().mean(0).clamp_min(1e-12).log()
            for al in (0.0, 0.25, 0.5, 0.75, 1.0):
                pv = balance_logits(L, al).argmax(1).cpu().numpy()
                print("%-8s %-4s alpha=%.2f  bal_acc %.4f"
                      % (name, agg, al, balanced_acc(pv, yv)), flush=True)
    print("val sweep %.1fs" % (time.time() - t0), flush=True)

    if a.out:
        offs = SETS[a.tta]
        xg = np.load(a.graded)["x"]
        V = views_logits(model, ds.norm(xg), offs, p, mode="log")
        L = V.mean(0)
        pg = balance_logits(L, a.alpha).argmax(1).cpu().numpy().astype(np.int64)
        np.save(a.out, pg)
        print("wrote %s tta=%s alpha=%.2f" % (a.out, a.tta, a.alpha), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
