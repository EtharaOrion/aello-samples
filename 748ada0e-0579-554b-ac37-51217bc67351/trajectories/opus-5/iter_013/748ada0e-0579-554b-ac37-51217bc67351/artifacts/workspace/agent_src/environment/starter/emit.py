"""Emit predictions for the graded split from ONE run's saved probabilities.

Only a single training run's probability file is ever used (TTA / raw+EMA views of that
same run), so the submitted predictions stay inside the one-run training budget.
"""
import sys, numpy as np

N = 100


def bal_acc(pred, y):
    return float(np.mean([(pred[y == c] == c).mean() for c in range(N) if (y == c).sum()]))


def main():
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else ""
    d = np.load(path)
    yva = d["yva"]
    vk = [k for k in d.files if k.startswith("v_")]
    print("views:", vk)
    for k in vk:
        print(f"  {k:10s} {bal_acc(d[k].astype(np.float32).argmax(1), yva):.4f}")
    ev = sum(d[k].astype(np.float32) for k in vk) / len(vk)
    eg = sum(d["g_" + k[2:]].astype(np.float32) for k in vk) / len(vk)
    print(f"  {'ENS':10s} {bal_acc(ev.argmax(1), yva):.4f}")
    if out:
        p = eg.argmax(1).astype(np.int64)
        np.save(out, p)
        print("wrote", out, p.shape, p.dtype, p.min(), p.max(), len(np.unique(p)))


if __name__ == "__main__":
    main()
