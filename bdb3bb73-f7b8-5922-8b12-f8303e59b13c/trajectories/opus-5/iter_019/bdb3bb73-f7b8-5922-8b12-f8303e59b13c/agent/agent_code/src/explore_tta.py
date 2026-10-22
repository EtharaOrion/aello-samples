"""EXPLORATORY (not confirmatory) stage: does a richer TTA view set help?

Declared as exploratory because it was NOT on the pre-registered candidate list
in PREREGISTRATION.md -- it was added after the screen showed TTA is the only
axis with an effect above the noise floor.  It is reported separately and is
held to the same promotion rule (>= 1 sigma on >= 3 shared seeds).

TTA is evaluated by calling ``net.features`` directly, so all schemes are scored
on the SAME trained weights per seed -- a perfectly paired comparison with zero
extra training runs beyond one per seed.
"""

import json
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/src")
import train as T

SEEDS = [0, 1, 2]


def mirror(net, x):
    return net.features(x) + net.features(x.flip(3))


@torch.no_grad()
def score(net, x, y, shifts, w, bs=1000):
    """(1-w) * mirror(x) + w * mean_i mirror(shift_i(x)), all mirror-normalized."""
    correct = 0
    for i in range(0, len(x), bs):
        xb = x[i:i + bs]
        base = mirror(net, xb) / 2.0
        if shifts:
            p = F.pad(xb, (1, 1, 1, 1), mode="reflect")
            acc = None
            for (a, b) in shifts:
                v = mirror(net, p[:, :, a:a + 32, b:b + 32]) / 2.0
                acc = v if acc is None else acc + v
            out = (1.0 - w) * base + w * (acc / len(shifts))
        else:
            out = base
        correct += (out.argmax(1) == y[i:i + bs]).sum().item()
    return correct / len(x)


# (name, shift offsets into the 34x34 reflect-padded image, weight)
SCHEMES = [
    ("mirror_only",      [],                                 0.0),
    ("diag2_w0.5",       [(0, 0), (2, 2)],                   0.5),   # incumbent
    ("diag4_w0.5",       [(0, 0), (2, 2), (0, 2), (2, 0)],   0.5),
    ("diag4_w0.6",       [(0, 0), (2, 2), (0, 2), (2, 0)],   0.6),
    ("cross4_w0.5",      [(1, 0), (1, 2), (0, 1), (2, 1)],   0.5),
    ("all8_w0.6",        [(0, 0), (2, 2), (0, 2), (2, 0),
                          (1, 0), (1, 2), (0, 1), (2, 1)],   0.6),
]


def main():
    out = []
    for s in SEEDS:
        res = T.main(["--val", "holdout", "--quiet", "--epochs", "20", "--seed", str(s)])
        net, (xv, yv) = res["_net"], res["_val"]
        net.eval()
        row = {"seed": s}
        for name, sh, w in SCHEMES:
            row[name] = score(net, xv, yv, sh, w)
        out.append(row)
        print(json.dumps(row), flush=True)
        del net, res
        torch.cuda.empty_cache()
    with open("/workspace/scratch_ckpt/explore_tta.json", "w") as f:
        json.dump(out, f, indent=2)
    import statistics as st
    print("\nmean over seeds (pp):")
    inc = st.mean(r["diag2_w0.5"] for r in out) * 100
    for name, _, _ in SCHEMES:
        m = st.mean(r[name] for r in out) * 100
        print(f"  {name:14s} {m:6.2f}   delta vs incumbent {m - inc:+.2f}")


if __name__ == "__main__":
    main()
