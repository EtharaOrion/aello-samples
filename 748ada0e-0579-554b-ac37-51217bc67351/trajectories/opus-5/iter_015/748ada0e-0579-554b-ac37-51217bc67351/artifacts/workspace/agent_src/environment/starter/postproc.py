"""Offline selection + emission from saved probability files (no retraining).

Evaluates {raw, ema, raw+ema} x {flip, shift5} x train-prior logit adjustment on the
public val split, with a bootstrap std so we do not chase 2000-image noise.
"""
import argparse, itertools, os
import numpy as np

N = 100


def bacc(pred, y):
    return float(np.mean([(pred[y == c] == c).mean() for c in range(N) if (y == c).any()]))


def boot_std(pred, y, n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    return float(np.std([bacc(pred[i], y[i]) for i in
                         (rng.choice(idx, len(idx), replace=True) for _ in range(n))]))


def adjust(p, prior, tau):
    if tau == 0:
        return p
    return p / (prior ** tau)[None, :]


def combos(z):
    yva, ytr = z["yva"], z["ytr"]
    prior = np.bincount(ytr, minlength=N).astype(np.float64)
    prior /= prior.sum()
    out = {}
    for tta in ("flip", "shift5"):
        for mix, w in (("raw", (1, 0)), ("ema", (0, 1)), ("ens", (1, 1))):
            pv = w[0] * z[f"{tta}_vr"] + w[1] * z[f"{tta}_ve"]
            pg = w[0] * z[f"{tta}_gr"] + w[1] * z[f"{tta}_ge"]
            for tau in (0.0, 0.25, 0.5, 1.0):
                out[(tta, mix, tau)] = (adjust(pv, prior, tau), adjust(pg, prior, tau))
    return out, yva


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--emit", default="")
    ap.add_argument("--pick", default="", help="tta,mix,tau e.g. flip,ens,0.5")
    ap.add_argument("--boot", type=int, default=0)
    a = ap.parse_args()

    for f in a.files:
        z = np.load(f)
        cs, yva = combos(z)
        print(f"== {os.path.basename(f)}  steps={z['steps']} pres={z['presentations']} "
              f"train_s={float(z['train_s']):.1f}")
        rows = []
        for k, (pv, pg) in cs.items():
            s = bacc(pv.argmax(1), yva)
            rows.append((s, k))
        for s, k in sorted(rows, reverse=True):
            sd = f" +-{boot_std(cs[k][0].argmax(1), yva):.4f}" if a.boot else ""
            print(f"   {k[0]:6s} {k[1]:3s} tau={k[2]:<4} {s:.4f}{sd}")

        if a.emit and a.pick:
            tta, mix, tau = a.pick.split(",")
            key = (tta, mix, float(tau))
            pg = cs[key][1]
            pred = pg.argmax(1).astype(np.int64)
            os.makedirs(os.path.dirname(a.emit), exist_ok=True)
            np.save(a.emit, pred)
            print(f"   EMIT {key} val={bacc(cs[key][0].argmax(1), yva):.4f} -> {a.emit} "
                  f"shape={pred.shape} classes={len(np.unique(pred))}")


if __name__ == "__main__":
    raise SystemExit(main())
