"""Prediction-time TTA search over shifts x flips x zoom scales, plus snapshot combos.

One grid_sample per view handles shift, hflip and zoom together, so the test-time view
distribution can be matched to whatever the training augmentation was. Selection is done
on the public val split only; nothing here trains.
"""
import argparse, itertools, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aello_data as data
from train import Net, bal_acc, sinkhorn_balance

OFF9 = [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2), (2, 2), (2, -2), (-2, 2), (-2, -2)]


@torch.no_grad()
def per_view(model, x_u8, res, mean, std, views, batch=1000):
    """views: list of (dy, dx, flip, scale). Returns {view: [N,100] log-softmax}."""
    model.eval()
    out = {v: [] for v in views}
    n = x_u8.shape[0]
    for s in range(0, n, batch):
        x0 = x_u8[s:s + batch].float().div_(255.)
        b, c, h, _ = x0.shape
        for (dy, dx, fl, sc) in views:
            th = torch.zeros(b, 2, 3, device=x0.device)
            th[:, 0, 0] = sc * (-1.0 if fl else 1.0)
            th[:, 1, 1] = sc
            th[:, 0, 2] = 2.0 * dx / h
            th[:, 1, 2] = 2.0 * dy / h
            grid = F.affine_grid(th, (b, c, res, res), align_corners=False)
            xf = F.grid_sample(x0, grid, mode="bilinear", padding_mode="reflection",
                               align_corners=False)
            xf = ((xf - mean) / std).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(xf)
            out[(dy, dx, fl, sc)].append(F.log_softmax(o.float(), 1).half().cpu())
    return {k: torch.cat(v).float().numpy() for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--graded", default="../data/graded_images.npz")
    ap.add_argument("--out", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--scales", default="1.0,0.88,1.12")
    ap.add_argument("--models", default="")
    a = ap.parse_args()

    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    ck = torch.load(a.ckpt, map_location="cpu")
    cfg = ck["cfg"]
    res = cfg["res"]
    mean, std = ck["mean"].to(dev), ck["std"].to(dev)
    depths = tuple(int(v) for v in cfg["depths"].split(","))
    scales = [float(s) for s in a.scales.split(",")]
    views = [(dy, dx, fl, sc) for sc in scales for (dy, dx) in OFF9 for fl in (False, True)]

    xv, yv = data.load_val()
    xg = np.load(a.graded)["x"]
    XV = torch.from_numpy(np.ascontiguousarray(xv)).permute(0, 3, 1, 2).contiguous().to(dev)
    XG = torch.from_numpy(np.ascontiguousarray(xg)).permute(0, 3, 1, 2).contiguous().to(dev)

    names = [n for n in (a.models.split(",") if a.models else list(ck["models"].keys()))]
    cache = {}
    if a.cache and os.path.exists(a.cache):
        cache = np.load(a.cache, allow_pickle=True)["c"].item()
        print("loaded cache", a.cache, flush=True)
    else:
        for nm in names:
            m = Net(data.N_CLASSES, cfg["width"], depths, cfg["stem"]).to(dev)
            m = m.to(memory_format=torch.channels_last)
            m.load_state_dict(ck["models"][nm])
            cache[("val", nm)] = per_view(m, XV, res, mean, std, views)
            cache[("graded", nm)] = per_view(m, XG, res, mean, std, views)
            print("views done for", nm, flush=True)
            del m
        if a.cache:
            np.savez(a.cache, c=np.array(cache, dtype=object))

    def stack(split, mods, offs, scs):
        tot, n = None, 0
        for nm in mods:
            d = cache[(split, nm)]
            for sc in scs:
                for (dy, dx) in offs:
                    for fl in (False, True):
                        v = d[(dy, dx, fl, sc)]
                        tot = v.copy() if tot is None else tot + v
                        n += 1
        return tot / n

    off_sets = {"o1": OFF9[:1], "o5": OFF9[:5], "o9": OFF9}
    sc_sets = {}
    for r in range(1, len(scales) + 1):
        for c in itertools.combinations(scales, r):
            sc_sets["s" + "/".join("%g" % v for v in c)] = list(c)
    mod_sets = []
    for r in range(1, len(names) + 1):
        mod_sets += [list(c) for c in itertools.combinations(names, r)]

    rows = []
    for mods in mod_sets:
        for on, offs in off_sets.items():
            for sn, scs in sc_sets.items():
                lv = stack("val", mods, offs, scs)
                rows.append((bal_acc(lv.argmax(1), yv), "+".join(mods), on, sn))
    rows.sort(reverse=True)
    print("\n top 15 val balanced accuracy:")
    for r in rows[:15]:
        print("  %.4f  %-24s %-3s %s" % r)
    for key in [("raw+ema0.999", "o1", "s1"), ("raw+ema0.999", "o9", "s1")]:
        for r in rows:
            if r[1:] == key:
                print("  ref %.4f  %-24s %-3s %s" % r)
    best = rows[0]
    print("\n BEST val %.4f models=%s offs=%s scales=%s" % best)
    if a.out:
        lg = stack("graded", best[1].split("+"), off_sets[best[2]], sc_sets[best[3]])
        pg = lg.argmax(1).astype(np.int64)
        np.save(a.out, pg)
        print(" wrote %s shape %s distinct %d" % (a.out, pg.shape, len(np.unique(pg))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
