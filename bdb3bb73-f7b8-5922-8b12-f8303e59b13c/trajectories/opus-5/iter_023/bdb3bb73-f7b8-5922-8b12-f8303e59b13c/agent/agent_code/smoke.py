"""Fast smoke test: shapes, scripting round-trip, per-epoch cost, holdout acc."""
import torch

import airbench as ab

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

x, y = ab.load_split_standardized("train")
tx, ty, hx, hy = ab.stratified_holdout(x, y)
print("train", tx.shape, "holdout", hx.shape,
      "holdout class counts", torch.bincount(hy).tolist())

cfg = dict(ab.DEFAULTS)
cfg["epochs"] = 5
hist, model, tot = ab.train(cfg, tx, ty, ckpt_dir="/workspace/scratch_ckpt/smoke",
                            eval_x=hx, eval_y=hy,
                            log=lambda r: print(r, flush=True))
print("total", tot, "params", sum(p.numel() for p in model.parameters()))

m = torch.jit.load("/workspace/scratch_ckpt/smoke/ckpt_epoch_0005.pt").eval().cuda()
with torch.no_grad():
    acc = (m(hx).argmax(1) == hy).float().mean().item()
print("scripted holdout acc", acc)
