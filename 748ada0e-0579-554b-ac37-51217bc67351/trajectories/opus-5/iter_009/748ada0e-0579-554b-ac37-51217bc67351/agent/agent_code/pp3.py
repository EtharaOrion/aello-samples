"""Second prediction-time sweep: temperature x sinkhorn, and richer view sets.

All paired on one fixed checkpoint (no training noise).  Only settings that show a broad
plateau on val are adopted, so the choice is not fitted to 2000 images.
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

CK = sys.argv[1] if len(sys.argv) > 1 else "/workspace/ck_f.pt"
dev = "cuda"
torch.backends.cudnn.benchmark = True
model, cfg = load(CK, dev)
xtr, ytr = data.load_train(cfg["instance"])
xva, yva = data.load_val()
tr = T.GPUData(xtr, ytr, dev, pad=cfg["pad"], res=cfg["res"])
pad = tr.pad
xvn = tr.norm(xva)

lp = {m: views_logprob(model, xvn, pad, m) for m in ["flip5", "flip9", "flip13"]}
for m, v in lp.items():
    print("[tta %-7s] raw %.4f  sink %.4f" % (m, bacc(v, yva),
                                              bacc(T.sinkhorn_uniform(v, 10), yva)))

base = lp["flip9"]
print("\n=== temperature x sinkhorn iters (flip9) ===")
print("      T:  " + "".join("%8.2f" % t for t in [0.5, 0.75, 1.0, 1.5, 2.0]))
for it in [0, 3, 10, 30]:
    row = "iters %2d" % it
    for Temp in [0.5, 0.75, 1.0, 1.5, 2.0]:
        l = base / Temp
        l = F.log_softmax(l, 1)
        if it:
            l = T.sinkhorn_uniform(l, it)
        row += "%8.4f" % bacc(l, yva)
    print(row)

print("\n=== sinkhorn partial step tau (flip9, 10 iters) ===")
for tau in [0.5, 0.75, 1.0, 1.25]:
    print("  tau %.2f  %.4f" % (tau, bacc(T.sinkhorn_uniform(base, 10, tau), yva)))

# a check that the gain is not an artefact of val's own imbalance: restrict to the
# classes that are well represented in val
cnt = np.bincount(yva, minlength=100)
keep = np.where(cnt >= 20)[0]
mask = np.isin(yva, keep)
print("\n=== restricted to the %d val classes with >=20 examples (%d images) ==="
      % (len(keep), mask.sum()))
print("  raw %.4f -> sink %.4f"
      % (bacc(base[torch.from_numpy(mask).to(dev)], yva[mask]),
         bacc(T.sinkhorn_uniform(base, 10)[torch.from_numpy(mask).to(dev)], yva[mask])))
