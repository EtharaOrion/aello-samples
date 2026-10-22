"""Robustness check for the prediction-time post-processing found by postproc.py.

Paired on one fixed checkpoint, so training noise is removed.  Split-half analysis to
check the gain is not a couple of lucky flips on 2000 images.
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
from postproc import load, views_logprob, sinkhorn, adjust, bacc

dev = "cuda"
torch.backends.cudnn.benchmark = True
model, cfg = load("/workspace/ck_b1.pt", dev)
xtr, ytr = data.load_train(cfg["instance"])
xva, yva = data.load_val()
tr = T.GPUData(xtr, ytr, dev, pad=cfg["pad"], res=cfg["res"])
pad = tr.pad
xvn = tr.norm(xva)

lp = {m: views_logprob(model, xvn, pad, m) for m in ["flip5", "flip9"]}

print("=== full val ===")
for m in ["flip5", "flip9"]:
    b0 = bacc(lp[m], yva)
    bs = bacc(sinkhorn(lp[m], 10), yva)
    ba = bacc(adjust(lp[m], 1.0), yva)
    print("  %-6s raw %.4f | adjust(t=1) %.4f | sinkhorn10 %.4f" % (m, b0, ba, bs))

base = lp["flip9"]
sk = sinkhorn(base, 10)
p0 = base.argmax(1).cpu().numpy()
p1 = sk.argmax(1).cpu().numpy()
ch = p0 != p1
print("\nsinkhorn changes %d/%d predictions; of those, %d wrong->right, %d right->wrong"
      % (ch.sum(), len(p0), ((p0 != yva) & (p1 == yva)).sum(),
         ((p0 == yva) & (p1 != yva)).sum()))

print("\n=== split-half (sinkhorn fitted independently on each half) ===")
rng = np.random.RandomState(0)
for rep in range(3):
    perm = rng.permutation(len(yva))
    for half, idx in enumerate([perm[:1000], perm[1000:]]):
        sub = base[torch.from_numpy(idx).to(dev)]
        ys = yva[idx]
        print("  rep%d half%d  raw %.4f -> sinkhorn %.4f  (%+.4f)"
              % (rep, half, bacc(sub, ys), bacc(sinkhorn(sub, 10), ys),
                 bacc(sinkhorn(sub, 10), ys) - bacc(sub, ys)))

print("\n=== sinkhorn step size / partial application (flip9, full val) ===")
for tau in [0.25, 0.5, 0.75, 1.0]:
    print("  tau %.2f  %.4f" % (tau, bacc(sinkhorn(base, 10, tau=tau), yva)))

print("\n=== target = train prior instead of uniform (flip9) ===")
cnt = np.bincount(ytr, minlength=100).astype(np.float64)
tgt = torch.tensor(cnt / cnt.sum(), device=dev, dtype=base.dtype)
print("  train-prior target %.4f | uniform target %.4f"
      % (bacc(sinkhorn(base, 10, target=tgt), yva), bacc(sinkhorn(base, 10), yva)))

print("\n=== what the graded split's predicted marginal looks like ===")
xg = data.load_graded_images()
lpg = views_logprob(model, tr.norm(xg), pad, "flip9")
q = lpg.exp().mean(0)
c = np.bincount(lpg.argmax(1).cpu().numpy(), minlength=100)
print("  graded argmax counts: min %d max %d mean %.1f std %.1f"
      % (c.min(), c.max(), c.mean(), c.std()))
cv = np.bincount(base.argmax(1).cpu().numpy(), minlength=100)
print("  val    argmax counts: min %d max %d mean %.1f std %.1f (true min %d max %d)"
      % (cv.min(), cv.max(), cv.mean(), cv.std(),
         np.bincount(yva, minlength=100).min(), np.bincount(yva, minlength=100).max()))
np.save("/workspace/lg_b1_flip9.graded.npy", lpg.cpu().numpy().astype(np.float32))
