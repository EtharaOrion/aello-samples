"""Sweep the *inference* resolution (FixRes effect) for a trained checkpoint, on val."""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aello_data as data
from train import Net, bal_acc, logits_tta

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--models", default="raw,ema0.999")
ap.add_argument("--res", default="40,44,48,52,56,64")
ap.add_argument("--views", default="flip9")
a = ap.parse_args()
dev = "cuda"; torch.backends.cudnn.benchmark = True
ck = torch.load(a.ckpt, map_location="cpu"); cfg = ck["cfg"]
mean, std = ck["mean"].to(dev), ck["std"].to(dev)
depths = tuple(int(v) for v in cfg["depths"].split(","))
xv, yv = data.load_val()
XV = torch.from_numpy(np.ascontiguousarray(xv)).permute(0, 3, 1, 2).contiguous().to(dev)
names = a.models.split(",")
per = {}
for nm in names:
    m = Net(data.N_CLASSES, cfg["width"], depths, cfg["stem"]).to(dev).to(memory_format=torch.channels_last)
    m.load_state_dict(ck["models"][nm])
    for r in [int(v) for v in a.res.split(",")]:
        per[(nm, r)] = logits_tta(m, XV, r, mean, std, dev, views=a.views).cpu().numpy()
        print("  %-10s res %2d  val %.4f" % (nm, r, bal_acc(per[(nm, r)].argmax(1), yv)), flush=True)
    del m
print(" ensembles (%s, %s):" % ("+".join(names), a.views))
rs = [int(v) for v in a.res.split(",")]
for r in rs:
    lv = sum(per[(nm, r)] for nm in names) / len(names)
    print("  res %2d        val %.4f" % (r, bal_acc(lv.argmax(1), yv)))
for combo in [(48, 56), (44, 48, 52), (48, 52), (48, 52, 56)]:
    if all(r in rs for r in combo):
        lv = sum(per[(nm, r)] for nm in names for r in combo) / (len(names) * len(combo))
        print("  res %-12s val %.4f" % ("+".join(map(str, combo)), bal_acc(lv.argmax(1), yv)))
