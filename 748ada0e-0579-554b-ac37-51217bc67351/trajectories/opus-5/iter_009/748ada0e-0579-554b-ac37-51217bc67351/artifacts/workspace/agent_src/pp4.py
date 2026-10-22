"""Does prediction-time marginal balancing help on a BALANCED split?

Val is imbalanced (10-31 images/class), so forcing a uniform marginal there is fighting
the true prior and understates the benefit for the graded split, whose predicted
marginal (mean 100.0/class, std 10.5) indicates it is balanced.

This draws many balanced subsets of val (k images per class), applies each method to the
subset's own log-probs, and averages -- the same situation the graded split presents.
Paired across checkpoints and resamples, so the comparison is low-variance.
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
cnt = np.bincount(yva, minlength=100)
print("val counts min %d max %d" % (cnt.min(), cnt.max()))

CKS = ["/workspace/ck_b1.pt", "/workspace/ck_f.pt"]
LP = {}
for ck in CKS:
    model, cfg = load(ck, dev)
    tr = T.GPUData(xtr, ytr, dev, pad=cfg["pad"], res=cfg["res"])
    xvn = tr.norm(xva)
    for m in ["flip5", "flip9"]:
        LP[(ck, m)] = views_logprob(model, xvn, tr.pad, m)
    del model
    torch.cuda.empty_cache()

by_class = [np.where(yva == c)[0] for c in range(100)]
K = int(cnt.min())            # 10 per class -> a perfectly balanced 1000-image split
R = 200
rng = np.random.RandomState(0)
subsets = []
for r in range(R):
    idx = np.concatenate([rng.choice(b, K, replace=False) for b in by_class])
    subsets.append(idx)

print("\n=== balanced subsets: %d draws of %d images (%d/class) ===" % (R, K * 100, K))
for m in ["flip5", "flip9"]:
    for ck in CKS:
        lp = LP[(ck, m)]
        raw, snk = [], []
        for idx in subsets:
            ti = torch.from_numpy(idx).to(dev)
            sub, ys = lp[ti], yva[idx]
            raw.append(bacc(sub, ys))
            snk.append(bacc(T.sinkhorn_uniform(sub, 10), ys))
        raw, snk = np.array(raw), np.array(snk)
        d = snk - raw
        print("  %-6s %-22s raw %.4f  sink %.4f  delta %+.4f  (sem %.4f, %d/%d draws +)"
              % (m, ck.split("/")[-1], raw.mean(), snk.mean(), d.mean(),
                 d.std() / np.sqrt(R), (d > 0).sum(), R))

print("\n=== full (imbalanced) val, for contrast ===")
for m in ["flip5", "flip9"]:
    for ck in CKS:
        lp = LP[(ck, m)]
        print("  %-6s %-22s raw %.4f  sink %.4f  delta %+.4f"
              % (m, ck.split("/")[-1], bacc(lp, yva),
                 bacc(T.sinkhorn_uniform(lp, 10), yva),
                 bacc(T.sinkhorn_uniform(lp, 10), yva) - bacc(lp, yva)))

print("\n=== flip9 vs flip5 (balanced subsets, no sinkhorn) ===")
for ck in CKS:
    d = []
    for idx in subsets:
        ti = torch.from_numpy(idx).to(dev)
        d.append(bacc(LP[(ck, "flip9")][ti], yva[idx]) -
                 bacc(LP[(ck, "flip5")][ti], yva[idx]))
    d = np.array(d)
    print("  %-22s flip9-flip5 %+.4f (sem %.4f)" % (ck.split("/")[-1], d.mean(),
                                                    d.std() / np.sqrt(R)))
