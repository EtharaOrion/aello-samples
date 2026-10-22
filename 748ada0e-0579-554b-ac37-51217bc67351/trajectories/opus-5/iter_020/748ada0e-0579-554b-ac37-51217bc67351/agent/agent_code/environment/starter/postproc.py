"""Compare inference combinations on val_public with bootstrap CIs.

Only reads probability tensors produced by ONE already-budgeted training run (a single
`--tag`), so the submitted predictions always come from a single budget-compliant run.
The point is to pick a *robust* combination rather than the argmax over thousands of
highly-correlated candidates, which would overfit a 2000-image val split.
"""
import argparse, os
import numpy as np

N_CLASSES = 100


def bal_acc(p, y):
    pred = p.argmax(1)
    return float(np.mean([(pred[y == c] == c).mean() for c in range(N_CLASSES) if (y == c).any()]))


def boot_ci(p, y, n=200, seed=0):
    rng = np.random.default_rng(seed)
    v = [bal_acc(p[i], y[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return float(np.mean(v)), float(np.std(v))


def norm(z, pre, keys):
    v = sum(z[pre + k] for k in keys)
    return v / v.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--probdir", default="/workspace/probs")
    ap.add_argument("--write", default="")
    ap.add_argument("--combo", default="")
    ap.add_argument("--tau", type=float, default=-1.0)
    ap.add_argument("--boot", type=int, default=1)
    a = ap.parse_args()

    for tag in a.tags:
        z = np.load(os.path.join(a.probdir, f"{tag}.npz"))
        yva, prior = z["yva"], z["prior"]
        keys = sorted(k[4:] for k in z.files if k.startswith("val_"))
        print(f"== {tag}   views={keys}")

        # principled candidates only (chosen a priori, not searched)
        cands = {k: (k,) for k in keys}
        pref = sorted({k.rstrip("STZ") for k in keys})            # per weight-set groups
        for p in pref:
            sel = [k for k in keys if k.rstrip("STZ") == p]
            if len(sel) > 1:
                cands["all:" + p] = tuple(sel)
        cands["allviews"] = tuple(keys)
        shifted = [k for k in keys if k.endswith(("S", "T"))]
        if shifted:
            cands["shifted"] = tuple(shifted)

        rows = []
        for name, c in cands.items():
            v = norm(z, "val_", c)
            for tau in (0.0, 0.5, 1.0):
                rows.append((bal_acc(v / prior[None, :] ** tau, yva), name, tau, c))
        rows.sort(key=lambda t: -t[0])
        for s, name, tau, _ in rows[:12]:
            print(f"   {name:14s} tau={tau:<4} {s:.4f}")
        s, name, tau, c = rows[0]
        if a.boot:
            m, sd = boot_ci(norm(z, "val_", c) / prior[None, :] ** tau, yva)
            print(f"   BEST {name} tau={tau} {s:.4f}  (bootstrap sd {sd:.4f})")

        if a.write:
            combo = tuple(a.combo.split("+")) if a.combo else c
            tau_w = a.tau if a.tau >= 0 else tau
            g = norm(z, "gr_", combo)
            pred = (g / prior[None, :] ** tau_w).argmax(1).astype(np.int64)
            os.makedirs(os.path.dirname(a.write), exist_ok=True)
            np.save(a.write, pred)
            vs = bal_acc(norm(z, "val_", combo) / prior[None, :] ** tau_w, yva)
            print(f"   wrote {a.write} from {tag}:{combo} tau={tau_w} (val {vs:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
