"""Tune the prediction-time marginal balancing in the setting that matches the graded
split: balanced subsets of val, pooled over both checkpoints, 200 resamples each.

Sweeps temperature (sharpness of the distribution the balancing acts on) x number of IPF
iterations.  Low temperature + many iterations approaches a hard balanced assignment;
T=1 with a few iterations is the soft version.  Only a broad plateau gets adopted.
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
K = int(np.bincount(yva, minlength=100).min())
R = 200
rng = np.random.RandomState(0)
subsets = [torch.from_numpy(
    np.concatenate([rng.choice(b, K, replace=False) for b in by_class])).to(dev)
    for _ in range(R)]
ys = [yva[s.cpu().numpy()] for s in subsets]

TEMPS = [0.5, 0.75, 1.0, 1.5, 2.0]
ITERS = [0, 3, 10, 30, 100]


def score(ck, Temp, it):
    lp = LP[ck]
    out = []
    for s, y in zip(subsets, ys):
        l = F.log_softmax(lp[s] / Temp, 1)
        if it:
            l = T.sinkhorn_uniform(l, it)
        out.append(bacc(l, y))
    return float(np.mean(out))


print("balanced subsets: %d draws x %d images (%d/class), flip9" % (R, K * 100, K))
print("mean over both checkpoints; raw (no balancing) baseline in the 'iters 0' row\n")
print("        T:" + "".join("%9.2f" % t for t in TEMPS))
best = None
for it in ITERS:
    row = "iters %3d" % it
    for Temp in TEMPS:
        v = np.mean([score(ck, Temp, it) for ck in CKS])
        row += "%9.4f" % v
        if best is None or v > best[0]:
            best = (v, Temp, it)
    print(row)
print("\nbest: %.4f at T=%.2f iters=%d" % best)

print("\nper-checkpoint at the adopted setting vs raw:")
for ck in CKS:
    print("  %-12s raw %.4f   T=1,it=10 %.4f   best(T=%.2f,it=%d) %.4f"
          % (ck.split("/")[-1], score(ck, 1.0, 0), score(ck, 1.0, 10),
             best[1], best[2], score(ck, best[1], best[2])))
