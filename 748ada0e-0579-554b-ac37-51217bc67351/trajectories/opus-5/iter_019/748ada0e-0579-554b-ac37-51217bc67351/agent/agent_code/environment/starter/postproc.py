"""Compare inference combinations on val_public with bootstrap CIs.

Only reads probability tensors produced by an already-budgeted training run: this
chooses how to *combine* views/weight-sets at prediction time, it does not train.
The point is to pick a robust combination rather than the argmax over hundreds of
highly-correlated candidates, which would overfit a 2000-image val split.
"""
import argparse, itertools, os
import numpy as np

N_CLASSES = 100


def bal_acc(p, y):
    pred = p.argmax(1)
    return float(np.mean([(pred[y == c] == c).mean() for c in range(N_CLASSES) if (y == c).any()]))


def boot(p, y, n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(n)]
    v = [bal_acc(p[i], y[i]) for i in idx]
    return float(np.mean(v)), float(np.std(v))


def norm(z, pre, keys):
    v = sum(z[pre + k] for k in keys)
    return v / v.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--probdir", default="/workspace/probs")
    ap.add_argument("--write", default="")
    ap.add_argument("--combo", default="")          # e.g. "rawS+rawZ" or "tag:key+tag:key"
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--full", type=int, default=0)  # exhaustive subset search (reporting only)
    a = ap.parse_args()

    Z = {t: np.load(os.path.join(a.probdir, f"{t}.npz")) for t in a.tags}
    t0 = a.tags[0]
    yva, prior = Z[t0]["yva"], Z[t0]["prior"]

    # ---- per-tag report over principled candidates
    for t in a.tags:
        z = Z[t]
        keys = sorted(k[4:] for k in z.files if k.startswith("val_"))
        sets = sorted({k.rstrip("SZT") or "raw" for k in keys})
        cands = {k: (k,) for k in keys}
        cands["allviews"] = tuple(keys)
        for s in {k[:3] for k in keys}:
            sel = [k for k in keys if k.startswith(s)]
            if len(sel) > 1:
                cands[s + "_views"] = tuple(sel)
        rows = []
        for name, c in cands.items():
            v = norm(z, "val_", c)
            for tau in (0.0, 0.5, 1.0):
                rows.append((bal_acc(v / prior[None, :] ** tau, yva), name, tau, c))
        rows.sort(key=lambda r: -r[0])
        print(f"== {t}  sets={sets}")
        for s, name, tau, _ in rows[:8]:
            print(f"   {name:12s} tau={tau:<4} {s:.4f}")
        if a.full:
            best = max(((bal_acc(norm(z, 'val_', c) / prior[None, :] ** tau, yva), c, tau)
                        for r in range(1, len(keys) + 1)
                        for c in itertools.combinations(keys, r)
                        for tau in (0.0, 0.5, 1.0)), key=lambda r: r[0])
            print(f"   exhaustive best {'+'.join(best[1])} tau={best[2]} {best[0]:.4f}")
        s, name, tau, c = rows[0]
        m, sd = boot(norm(z, "val_", c) / prior[None, :] ** tau, yva)
        print(f"   BEST {name} tau={tau} {s:.4f}  (bootstrap sd {sd:.4f})")

    # ---- cross-tag average of every view (only meaningful for same-budget variants)
    if len(a.tags) > 1:
        acc = 0.0
        for t in a.tags:
            z = Z[t]
            acc = acc + norm(z, "val_", sorted(k[4:] for k in z.files if k.startswith("val_")))
        for tau in (0.0, 0.5, 1.0):
            print(f"   XTAG mean tau={tau} {bal_acc(acc / prior[None, :] ** tau, yva):.4f}")

    if a.write:
        t, combo = t0, a.combo
        if ":" in combo:
            t, combo = combo.split(":", 1)
        keys = tuple(combo.split("+")) if combo else tuple(
            sorted(k[4:] for k in Z[t].files if k.startswith("val_")))
        vs = bal_acc(norm(Z[t], "val_", keys) / prior[None, :] ** a.tau, yva)
        pred = (norm(Z[t], "gr_", keys) / prior[None, :] ** a.tau).argmax(1).astype(np.int64)
        os.makedirs(os.path.dirname(a.write), exist_ok=True)
        np.save(a.write, pred)
        print(f"   wrote {a.write} from {t}:{'+'.join(keys)} tau={a.tau} (val {vs:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
