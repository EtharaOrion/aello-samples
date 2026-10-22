import time, torch, sys
sys.path.insert(0,"/workspace"); from train import *
dev="cuda"
xtr_u8, ytr = load_split("train", dev)
tr_idx, va_idx = stratified_holdout(ytr, 500, seed=0)
xa_u8, ya = xtr_u8[tr_idx], ytr[tr_idx]
xv = standardize(xtr_u8[va_idx]).contiguous(memory_format=torch.channels_last); yv = ytr[va_idx]
for E in [12, 20]:
    cfg = dict(DEFAULT_CFG); cfg["epochs"]=E
    r = train_run(cfg, seed=0, val_data=(xv,yv), x_tr=xa_u8, y_tr=ya, log=f"E{E}", eval_every=2)
    print(f"E={E} wall={r['wall']:.2f}s final={r['hist'][-1]['val']}")
