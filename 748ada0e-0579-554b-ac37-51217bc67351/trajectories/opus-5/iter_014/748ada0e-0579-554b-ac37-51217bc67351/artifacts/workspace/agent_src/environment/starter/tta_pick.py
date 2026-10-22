"""Evaluate several TTA view-sets on val for one checkpoint, then emit the best."""
import os, sys, time
import numpy as np
import torch
from train import bal_acc, DATA, N_CLASSES
from emit import build, probs_views, VIEWSETS, DEV

ck_path, out_path = sys.argv[1], sys.argv[2]
t0 = time.time()
ck = torch.load(ck_path, map_location=DEV)
mean, std = ck["mean"].to(DEV), ck["std"].to(DEV)
va = np.load(os.path.join(DATA, "val_public.npz"))
xva = torch.from_numpy(va["x"]).to(DEV); yva = va["y"]
tr = np.load(os.path.join(DATA, "train_i20k.npz"))
prior = torch.bincount(torch.from_numpy(tr["y"]), minlength=N_CLASSES).float()
prior = (prior / prior.sum()).to(DEV)

models = {k: build(ck, k) for k in ("raw", "ema1")}
best = (-1, None)
val_cache = {}
for vname, vs in VIEWSETS.items():
    pv = torch.zeros(len(yva), N_CLASSES, device=DEV)
    for k, m in models.items():
        v = probs_views(m, xva, mean, std, vs)
        val_cache[(vname, k)] = v
        pv += v / v.sum(1, keepdim=True)
    for pp in (0.0, 0.5, 1.0):
        b = bal_acc((pv / prior.pow(pp).unsqueeze(0)).argmax(1).cpu().numpy(), yva)
        print(f"  views={vname:14s} prior_pow={pp}  bal_acc {b:.4f}", flush=True)
        if b > best[0]:
            best = (b, (vname, pp))
print(f"BEST {best} ({time.time()-t0:.0f}s)", flush=True)

vname, pp = best[1]
xg = torch.from_numpy(np.load(os.path.join(DATA, "graded_images.npz"))["x"]).to(DEV)
pg = torch.zeros(xg.shape[0], N_CLASSES, device=DEV)
for k, m in models.items():
    g = probs_views(m, xg, mean, std, VIEWSETS[vname])
    pg += g / g.sum(1, keepdim=True)
p = (pg / prior.pow(pp).unsqueeze(0)).argmax(1).cpu().numpy().astype(np.int64)
np.save(out_path, p)
print(f"wrote {out_path} shape {p.shape} dtype {p.dtype} uniq {len(np.unique(p))} "
      f"range {p.min()}-{p.max()} ({time.time()-t0:.0f}s)", flush=True)
