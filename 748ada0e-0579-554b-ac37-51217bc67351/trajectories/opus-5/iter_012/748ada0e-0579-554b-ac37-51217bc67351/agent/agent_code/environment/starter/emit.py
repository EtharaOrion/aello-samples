"""Write predictions from a cached per-view logit dump with an explicit recipe."""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aello_data as data
from train import bal_acc, sinkhorn_balance
from postproc import VIEWSETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--models", default="raw,ema0.999")
    ap.add_argument("--views", default="flip9")
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--sink", type=float, default=0.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    cache = np.load(a.cache, allow_pickle=True)["c"].item()
    mods = a.models.split(",")
    ytr = data.load_train("i20k")[1]
    logprior = np.log(np.bincount(ytr, minlength=100) / len(ytr)).astype(np.float32)
    _, yv = data.load_val()

    def stack(split):
        tot, n = None, 0
        for nm in mods:
            d = cache[(split, nm)]
            for off in VIEWSETS[a.views]:
                for f in (False, True):
                    v = d[(off, f)]
                    tot = v.copy() if tot is None else tot + v
                    n += 1
        return tot / n

    def finalize(lv):
        z = lv - a.tau * logprior[None, :]
        return (z.argmax(1) if a.sink == 0.0
                else sinkhorn_balance(z, temp=a.sink).argmax(1)).astype(np.int64)

    pv = finalize(stack("val"))
    print("recipe models=%s views=%s tau=%s sink=%s -> val bal-acc %.4f  acc %.4f"
          % (a.models, a.views, a.tau, a.sink, bal_acc(pv, yv), (pv == yv).mean()))
    if a.out:
        pg = finalize(stack("graded"))
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        np.save(a.out, pg)
        print("wrote %s shape %s dtype %s distinct %d"
              % (a.out, pg.shape, pg.dtype, len(np.unique(pg))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
