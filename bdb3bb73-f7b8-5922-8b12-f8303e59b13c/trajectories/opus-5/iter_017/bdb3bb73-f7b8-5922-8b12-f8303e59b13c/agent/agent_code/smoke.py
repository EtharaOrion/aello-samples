import time, torch, os, sys
sys.path.insert(0, "/workspace")
from train import *

dev = "cuda"
xtr_u8, ytr = load_split("train", dev)
tr_idx, va_idx = stratified_holdout(ytr, 500, seed=0)
print("split", len(tr_idx), len(va_idx), "val class counts", torch.bincount(ytr[va_idx]).tolist())
xa_u8, ya = xtr_u8[tr_idx], ytr[tr_idx]
xv, yv = standardize(xtr_u8[va_idx]).contiguous(memory_format=torch.channels_last), ytr[va_idx]

cfg = dict(DEFAULT_CFG); cfg["epochs"] = 4
t0 = time.time()
r = train_run(cfg, seed=0, ckpt_dir="/workspace/scratch_ckpt/smoke",
              val_data=(xv, yv), x_tr=xa_u8, y_tr=ya, log="smoke")
print("params", r["nparam"], "wall", f"{r['wall']:.2f}s")
ct = r["ckpt_times"]
if len(ct) > 1:
    d = [ct[i+1]-ct[i] for i in range(len(ct)-1)]
    print("inter-ckpt gaps (incl. val eval):", [f"{x:.3f}" for x in d])

# scripted checkpoint must load and match the in-memory model
m = torch.jit.load("/workspace/scratch_ckpt/smoke/ckpt_epoch_0004.pt").eval().cuda()
with torch.no_grad():
    a = m(xv[:512]).float().argmax(1)
acc = (a == yv[:512]).float().mean().item()
print("loaded-scripted val acc on first 512:", acc)
