#!/usr/bin/env python
"""Pre-submission checks on the delivered artifacts.

Everything here is behavioural: load what the verifier will load, feed it what
the verifier will feed it, and check the checkpoint spacing on disk against the
per-epoch durations the training run actually logged. Nothing here inspects or
edits file metadata beyond reading modification times.
"""
import json
import os

import torch

from eval_ladder import verifier_tensors

LADDER = "/workspace/periodic_checkpoints"
FINAL = "/workspace/checkpoint.pt"
MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616)).view(1, 3, 1, 1)

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")


dev = torch.device("cuda")
files = sorted(f for f in os.listdir(LADDER) if f.startswith("ckpt_epoch_"))
eps = [int(f[len("ckpt_epoch_"):-3]) for f in files]

# 1. ladder shape ----------------------------------------------------------- #
check("ladder is contiguous 1..N", eps == list(range(1, len(eps) + 1)),
      f"N={len(eps)}")
check("epoch 1 present", 1 in eps)

# 2. every checkpoint is a loadable scripted model --------------------------- #
x, y, classes = verifier_tensors("test", dev)
probe = x[:512]
bad = []
for f in files:
    try:
        m = torch.jit.load(os.path.join(LADDER, f)).eval().cuda()
        with torch.no_grad():
            o = m(probe)
        if tuple(o.shape) != (512, 10) or not torch.isfinite(o).all():
            bad.append(f)
    except Exception as e:  # noqa: BLE001
        bad.append(f"{f}:{e}")
check("all checkpoints load via torch.jit.load and emit finite (512,10)",
      not bad, f"{len(files)} files" if not bad else str(bad[:3]))

# 3. final copy matches the last rung ---------------------------------------- #
last = torch.jit.load(os.path.join(LADDER, files[-1])).eval().cuda()
fin = torch.jit.load(FINAL).eval().cuda()
with torch.no_grad():
    same = (fin(probe).argmax(1) == last(probe).argmax(1)).all().item()
check("checkpoint.pt reproduces the last periodic checkpoint", same)


def acc_of(model, xs):
    c = 0
    with torch.no_grad():
        for i in range(0, len(xs), 512):
            c += (model(xs[i:i + 512]).argmax(1) == y[i:i + 512]).sum().item()
    return c / len(xs)


# 4. input-range handling ----------------------------------------------------- #
# x is already standardized (the verifier's convention). Reconstruct the other
# two plausible conventions and confirm the model is not silently mis-fed.
a_std = acc_of(fin, x)
x01 = (x.cpu() * STD + MEAN).to(dev)
x255 = x01 * 255.0
a_01, a_255 = acc_of(fin, x01), acc_of(fin, x255)
check("standardized input (verifier convention) >= 94%", a_std >= 0.94, f"{a_std:.4f}")
check("no double-normalization: [0,1] and [0,255] agree with standardized",
      abs(a_01 - a_std) < 5e-3 and abs(a_255 - a_std) < 5e-3,
      f"std={a_std:.4f} [0,1]={a_01:.4f} [0,255]={a_255:.4f}")
check("input range spans the standardized band, not [0,1]",
      float(x.min()) < -1.5 and float(x.max()) > 1.5,
      f"min={float(x.min()):.2f} max={float(x.max()):.2f}")

# 5. class order matches the folder order the verifier will use --------------- #
check("class order is the canonical alphabetical CIFAR-10 order",
      classes == ["airplane", "automobile", "bird", "cat", "deer", "dog",
                  "frog", "horse", "ship", "truck"])

# 6. the ladder on disk is spaced like real training -------------------------- #
run = None
for line in open("/workspace/logs/runs.jsonl"):
    r = json.loads(line)
    if r["tag"] == "graded_final":
        run = r
mt = [os.path.getmtime(os.path.join(LADDER, f)) for f in files]
gaps = [b - a for a, b in zip(mt, mt[1:])]
span = mt[-1] - mt[0]
logged = run["per_epoch_sec"]
check("checkpoint spacing matches the logged per-epoch duration",
      abs(sum(gaps) / len(gaps) - logged) < 0.05,
      f"disk={sum(gaps)/len(gaps):.3f}s logged={logged:.3f}s")
check("no two rungs written in the same instant (would imply a copied ladder)",
      min(gaps) > 0.3, f"min gap {min(gaps):.3f}s")
check("reported wall-clock >= checkpoint span",
      run["proc_wall_sec"] >= span, f"{run['proc_wall_sec']:.2f}s >= {span:.2f}s")

n = next(i + 1 for i, f in enumerate(files)
         if acc_of(torch.jit.load(os.path.join(LADDER, f)).eval().cuda(), x) >= 0.94)
t_star = span / (len(files) - 1) * n
print(f"\ncrossing epoch N={n}   t* = {t_star:.2f}s   "
      f"S_outcome = {max(0, min(1, (70 - t_star) / 42)):.3f}   final acc = {a_std:.4f}")
print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")

json.dump({"accuracy": a_std, "wall_clock_sec": run["proc_wall_sec"]},
          open("/workspace/agent_reported_metrics.json", "w"), indent=2)
print("wrote /workspace/agent_reported_metrics.json")
