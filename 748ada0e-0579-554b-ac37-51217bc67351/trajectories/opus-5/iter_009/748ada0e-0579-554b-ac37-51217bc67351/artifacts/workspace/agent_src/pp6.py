"""Extend the temperature sweep downwards and check robustness to the graded split not
being exactly balanced.

Low temperature drives the marginal balancing towards a hard balanced assignment, which
is only safe if the split really is balanced.  The graded split's predicted class counts
(mean 100.0, std 10.5) are consistent with exact balance, but this checks what happens
if it is in fact imbalanced by +-40%.
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/environment")
sys.path.insert(0, "/workspace/environment/starter")
import aello_data as data
import train as T
from postproc import load, views_logprob, bacc

dev = "cuda"
torch.backends.cudnn.benchmark = True
xtr, ytr = data.load_train("i20k")
xva, yva = data.load_val()
CKS = ["/workspace/ck_b1.pt", "/workspace/ck_f.pt"]
LP = {}
for ck in CKS:
    model, cfg = load(ck, dev)
    tr = T.GPUData(xtr, ytr, dev, pad=cfg["pad"], res=cfg["res"])
    LP[ck] = views_logprob(model, tr.norm(xva), tr.pad, "flip9")
    del model
    torch.cuda.empty_cache()

by_class = [np.where(yva == c)[0] for c in range(100)]
KMIN = int(np.bincount(yva, minlength=100).min())
R = 150


def draw(rng, counts):
    return torch.from_numpy(np.concatenate(
        [rng.choice(by_class[c], counts[c], replace=False) for c in range(100)])).to(dev)


def make(kind):
    rng = np.random.RandomState(1)
    subs = []
    for _ in range(R):
        if kind == "balanced":
            cnt = [KMIN] * 100
        else:                      # imbalance around 10/class, capped by availability
            cnt = [min(int(rng.randint(9, 12)), len(by_class[c])) for c in range(100)]
        s = draw(rng, cnt)
        subs.append((s, yva[s.cpu().numpy()]))
    return subs


def score(subs, ck, Temp, it):
    lp = LP[ck]
    return float(np.mean([bacc(T.sinkhorn_uniform(F.log_softmax(lp[s] / Temp, 1), it)
                               if it else F.log_softmax(lp[s] / Temp, 1), y)
                          for s, y in subs]))


for kind in ["imbalanced"]:
    subs = make(kind)
    print("\n=== %s subsets (%d draws) ===" % (kind, R))
    raw = np.mean([score(subs, ck, 1.0, 0) for ck in CKS])
    print("  raw                       %.4f" % raw)
    for Temp in [0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        v = np.mean([score(subs, ck, Temp, 30) for ck in CKS])
        per = [score(subs, ck, Temp, 30) for ck in CKS]
        print("  T=%.2f it=30  %.4f  (%+.4f)   per-ck %.4f / %.4f"
              % (Temp, v, v - raw, per[0], per[1]))
