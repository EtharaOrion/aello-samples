"""Pick an inference combination on val_public and (optionally) write graded predictions.

Uses only probability tensors already produced by a single budgeted training run.
"""
import argparse, itertools, json, os
import numpy as np

N_CLASSES = 100


def bal_acc(p, y):
    pred = p.argmax(1)
    return float(np.mean([(pred[y == c] == c).mean() for c in range(N_CLASSES) if (y == c).any()]))


def boot(p, y, tau, prior, n=200, seed=0):
    """Bootstrap sd of balanced accuracy (how much val can be trusted)."""
    rng = np.random.default_rng(seed)
    q = p / prior[None, :] ** tau
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        vals.append(bal_acc(q[i], y[i]))
    return float(np.std(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--probdir", default="/workspace/probs")
    ap.add_argument("--write", default="")
    ap.add_argument("--combo", default="")          # e.g. raw+ema0
    ap.add_argument("--tau", type=float, default=-1)
    a = ap.parse_args()

    for tag in a.tags:
        z = np.load(os.path.join(a.probdir, f"{tag}.npz"))
        yva, prior = z["yva"], z["prior"]
        keys = [k[4:] for k in z.files if k.startswith("val_")]
        combos = []
        for r in range(1, len(keys) + 1):
            combos += [c for c in itertools.combinations(sorted(keys), r)]
        print(f"== {tag}")
        best = None
        for c in combos:
            v = sum(z["val_" + k] for k in c)
            v = v / v.sum(1, keepdims=True)
            for tau in (0.0, 0.25, 0.5, 0.75, 1.0):
                s = bal_acc(v / prior[None, :] ** tau, yva)
                print(f"   {'+'.join(c):24s} tau={tau:<5} {s:.4f}")
                if best is None or s > best[0]:
                    best = (s, c, tau)
        print(f"   BEST {best[1]} tau={best[2]} {best[0]:.4f}")
        vb = sum(z["val_" + k] for k in best[1])
        print(f"   bootstrap sd of val bal-acc ~ {boot(vb, yva, best[2], prior):.4f}")

        if a.write:
            combo = tuple(a.combo.split("+")) if a.combo else best[1]
            tau = a.tau if a.tau >= 0 else best[2]
            g = sum(z["gr_" + k] for k in combo)
            g = g / g.sum(1, keepdims=True)
            pred = (g / prior[None, :] ** tau).argmax(1).astype(np.int64)
            os.makedirs(os.path.dirname(a.write), exist_ok=True)
            np.save(a.write, pred)
            print(f"   wrote {a.write} from {combo} tau={tau} "
                  f"(val {bal_acc(sum(z['val_' + k] for k in combo) / prior[None, :] ** tau, yva):.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
