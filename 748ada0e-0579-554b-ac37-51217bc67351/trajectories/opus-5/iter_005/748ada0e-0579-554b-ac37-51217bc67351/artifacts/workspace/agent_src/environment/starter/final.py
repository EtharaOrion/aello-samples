"""Write the graded predictions from a trained checkpoint using the chosen
prediction-time recipe (flip5s TTA, probability-space averaging, class-marginal
correction).  Prediction only -- no training happens here."""
import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aello_data as data
from train import GPUData, make_model, balance_logits, balanced_acc
from post import views_logits


def offsets(p, kind="flip5s"):
    a2, b2 = max(1, p // 2), p + max(1, p // 2)
    a3, b3 = max(1, p - p // 3), p + p // 3
    if kind == "flip5":
        return [(p, p), (a2, a2), (a2, b2), (b2, a2), (b2, b2)]
    if kind == "flip9":
        return [(oy, ox) for oy in (p - a2, p, p + a2) for ox in (p - a2, p, p + a2)]
    return [(p, p), (a3, a3), (a3, b3), (b3, a3), (b3, b3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--graded", default="../data/graded_images.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tta", default="flip5s")
    ap.add_argument("--alpha", type=float, default=1.0)
    a = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    t0 = time.time()
    ck = torch.load(a.ckpt, map_location="cuda")
    ta = ck["args"]
    x, y = data.load_train(ta["instance"])
    ds = GPUData(x, y, "cuda", pad=ta["pad"], res=ta["res"])
    blocks = tuple(int(v) for v in ta["blocks"].split(",")) if ta["blocks"] else None
    m = make_model(ta["seed"], data.N_CLASSES, ta["arch"], ta["width"], ta["dropout"],
                   ta["se"], blocks, ta["dsample"], ta["branches"],
                   ta.get("droppath", 0.0)).cuda().to(memory_format=torch.channels_last)
    m.load_state_dict(ck["sd"])
    m.eval()
    offs = offsets(ds.pad, a.tta)

    xv, yv = data.load_val()
    V = views_logits(m, ds.norm(xv), offs, ds.pad, mode="log").cpu()
    Lv = V.exp().mean(0).clamp_min(1e-12).log()           # probability-space average
    rng = np.random.default_rng(0)
    subs = [np.concatenate([rng.choice(np.where(yv == c)[0], 10, replace=False)
                            for c in range(100)]) for _ in range(40)]
    for al in (0.0, a.alpha):
        full = balanced_acc(balance_logits(Lv, al).argmax(1).numpy(), yv)
        bs = np.mean([balanced_acc(balance_logits(Lv[i], al).argmax(1).numpy(), yv[i])
                      for i in subs])
        print("%s tta=%s alpha=%.2f  full-val %.4f  bal-sub %.4f"
              % (os.path.basename(a.ckpt), a.tta, al, full, bs), flush=True)

    if a.out:
        xg = np.load(a.graded)["x"]
        Vg = views_logits(m, ds.norm(xg), offs, ds.pad, mode="log")
        Lg = Vg.exp().mean(0).clamp_min(1e-12).log()
        pg = balance_logits(Lg, a.alpha).argmax(1).cpu().numpy().astype(np.int64)
        np.save(a.out, pg)
        c = np.bincount(pg, minlength=100)
        print("wrote %s  counts min %d max %d  predict %.1fs"
              % (a.out, c.min(), c.max(), time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
