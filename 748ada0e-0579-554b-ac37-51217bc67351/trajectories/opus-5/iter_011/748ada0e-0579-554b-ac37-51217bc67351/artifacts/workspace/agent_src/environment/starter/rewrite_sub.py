"""Rewrite /submission/preds_i20k.npy from ONE run's saved logits.

Budget note: this only ever selects a single training run's logits (the best on
val_public) and re-applies the per-class Sinkhorn bias.  It never averages two
separately trained models -- that would exceed the 8000-step training budget.
The raw/EMA average inside one run is fine: both heads come from the same
8000-step trajectory and cost no extra presentations.
"""
import argparse, glob, json, os
import numpy as np
from train import sinkhorn_bias, pick_tau, balanced_acc

ap = argparse.ArgumentParser()
ap.add_argument("--logits", default="/workspace/logits/*.npz")
ap.add_argument("--pick", default="", help="force this run tag")
ap.add_argument("--out", default="/submission/preds_i20k.npy")
ap.add_argument("--dry", action="store_true")
a = ap.parse_args()

best = None
for f in sorted(glob.glob(a.logits)):
    j = f.replace(".npz", ".json")
    if not os.path.exists(j):
        continue
    meta = json.load(open(j))
    if meta["args"]["steps"] < 8000:
        continue
    sc = meta["best_score"]
    print(f"{os.path.basename(f):22s} head {meta['best']:16s} val {sc:.4f} "
          f"tau {meta.get('tau')}  ({meta['seconds']:.0f}s)")
    take = os.path.basename(f).startswith(a.pick) if a.pick else (best is None or sc > best[1])
    if take:
        best = (f, sc, meta)

f, sc, meta = best
d = np.load(f)
head = meta["best"]
gl, yv, vlog = d["g_best"], d["yv"], d[f"v_{head}"]
taus = [float(t) for t in meta["args"]["taus"].split(",")]
tsw = pick_tau(vlog, yv, taus)
btau = max(tsw, key=tsw.get)
print(f"\nselected {os.path.basename(f)}  head {head}  val {sc:.4f}  tau {btau}")
for t in taus:
    print(f"  tau={t:<4} resampled {tsw[t]:.4f}  delta {tsw[t]-tsw[taus[0]]:+.4f}")

pred = (gl + btau * sinkhorn_bias(gl)).argmax(1).astype(np.int64)
cnt = np.bincount(pred, minlength=100)
print(f"changed {(pred != gl.argmax(1)).sum()}/{len(pred)}; counts {cnt.min()}..{cnt.max()}")
if not a.dry:
    np.save(a.out, pred)
    json.dump({"source": f, "head": head, "tau": btau, "val": sc},
              open("/workspace/submission_meta.json", "w"), indent=1)
    print(f"wrote {a.out}")
