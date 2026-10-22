"""Prediction-time search: TTA view sets, raw/EMA snapshot combos, marginal balancing.

Loads a checkpoint written by train.py (raw + EMA snapshots of ONE training run),
caches per-view log-softmax for val and graded, then grid-searches on the public val
split. Nothing here trains; it only changes how the already-trained weights predict.
"""
import argparse, itertools, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aello_data as data
from train import Net, bal_acc, sinkhorn_balance

OFF9 = [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2), (2, 2), (2, -2), (-2, 2), (-2, -2)]
VIEWSETS = {"flip1": OFF9[:1], "flip5": OFF9[:5], "flip9": OFF9}


@torch.no_grad()
def per_view(model, x_u8, res, mean, std, offs, batch=1000):
    """Returns dict (dy,dx,flip) -> [N,100] log-softmax, float16 on cpu."""
    model.eval()
    outs = {k: [] for k in itertools.product(offs, (False, True))}
    for s in range(0, x_u8.shape[0], batch):
        xb0 = x_u8[s:s + batch].float().div_(255.)
        pad = F.pad(xb0, (4, 4, 4, 4), mode="reflect")
        for off in offs:
            dy, dx = off
            xs = xb0 if (dy == 0 and dx == 0) else \
                torch.roll(pad, shifts=(dy, dx), dims=(2, 3))[:, :, 4:-4, 4:-4]
            for f in (False, True):
                xf = xs.flip(3) if f else xs
                if res != xf.shape[-1]:
                    xf = F.interpolate(xf, size=(res, res), mode="bilinear",
                                       align_corners=False)
                xf = ((xf - mean) / std).contiguous(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = model(xf)
                outs[(off, f)].append(F.log_softmax(o.float(), 1).half().cpu())
    return {k: torch.cat(v).float().numpy() for k, v in outs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--graded", default="../data/graded_images.npz")
    ap.add_argument("--out", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--views", default="flip9")
    a = ap.parse_args()

    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    ck = torch.load(a.ckpt, map_location="cpu")
    cfg = ck["cfg"]
    res = cfg["res"]
    mean, std = ck["mean"].to(dev), ck["std"].to(dev)
    depths = tuple(int(v) for v in cfg["depths"].split(","))

    xv, yv = data.load_val()
    xg = np.load(a.graded)["x"]
    XV = torch.from_numpy(np.ascontiguousarray(xv)).permute(0, 3, 1, 2).contiguous().to(dev)
    XG = torch.from_numpy(np.ascontiguousarray(xg)).permute(0, 3, 1, 2).contiguous().to(dev)

    offs = VIEWSETS[a.views]
    cache = {}
    if a.cache and os.path.exists(a.cache):
        z = np.load(a.cache, allow_pickle=True)
        cache = z["c"].item()
        print("loaded cache", a.cache)
    else:
        for nm, sd in ck["models"].items():
            m = Net(data.N_CLASSES, cfg["width"], depths, cfg["stem"]).to(dev)
            m = m.to(memory_format=torch.channels_last)
            m.load_state_dict(sd)
            cache[("val", nm)] = per_view(m, XV, res, mean, std, offs)
            cache[("graded", nm)] = per_view(m, XG, res, mean, std, offs)
            print("views done for", nm, flush=True)
            del m
        if a.cache:
            np.savez_compressed(a.cache, c=np.array(cache, dtype=object))

    names = list(ck["models"].keys())
    model_combos = []
    for r in range(1, len(names) + 1):
        model_combos += [list(c) for c in itertools.combinations(names, r)]

    def stack(split, mods, vs):
        tot = None
        n = 0
        for nm in mods:
            d = cache[(split, nm)]
            for off in VIEWSETS[vs]:
                for f in (False, True):
                    v = d[(off, f)]
                    tot = v.copy() if tot is None else tot + v
                    n += 1
        return tot / n

    # class-prior correction: the training split is imbalanced (166..231 per class),
    # which biases predictions toward frequent classes and hurts balanced accuracy.
    ytr = data.load_train(cfg["instance"])[1]
    logprior = np.log(np.bincount(ytr, minlength=100) / len(ytr)).astype(np.float32)

    def finalize(lv, tau, T):
        z = lv - tau * logprior[None, :]
        return z.argmax(1) if T == 0.0 else sinkhorn_balance(z, temp=T).argmax(1)

    rows = []
    for mods in model_combos:
        for vs in ("flip1", "flip5", "flip9"):
            if len(VIEWSETS[vs]) > len(offs):
                continue
            lv = stack("val", mods, vs)
            for tau in (0.0, 0.5, 1.0):
                for T in (0.0, 1.0, 0.7, 0.5):
                    rows.append((bal_acc(finalize(lv, tau, T), yv), "+".join(mods), vs,
                                 tau, T))
    rows.sort(reverse=True)
    print("\n top 20 (val balanced accuracy):")
    for r in rows[:20]:
        print("  %.4f  %-26s %-6s tau=%.2f sink_T=%s" % r)
    print("\n reference points:")
    for key in [("raw", "flip1", 0.0, 0.0), ("raw", "flip9", 0.0, 0.0),
                ("raw+ema0.999", "flip1", 0.0, 0.0), ("raw+ema0.999", "flip9", 0.0, 0.7)]:
        for r in rows:
            if r[1:] == key:
                print("  %.4f  %-26s %-6s tau=%.2f sink_T=%s" % r)
                break

    best = rows[0]
    print("\n BEST val %.4f : models=%s views=%s tau=%s sink_T=%s" % best)
    if a.out:
        lg = stack("graded", best[1].split("+"), best[2])
        pg = finalize(lg, best[3], best[4]).astype(np.int64)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        np.save(a.out, pg)
        print(" wrote %s  shape %s  distinct %d" % (a.out, pg.shape, len(np.unique(pg))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
