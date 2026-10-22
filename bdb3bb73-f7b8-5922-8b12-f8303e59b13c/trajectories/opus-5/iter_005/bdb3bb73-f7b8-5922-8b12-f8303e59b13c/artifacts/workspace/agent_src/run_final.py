"""Train on the FULL 50k train split and write the per-epoch checkpoint ladder."""
import argparse
import json
import os
import shutil
import time
from dataclasses import replace, asdict

from train import Config, train

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/workspace/periodic_checkpoints")
ap.add_argument("--cfg", default="{}")
ap.add_argument("--final", action="store_true", help="also write /workspace/checkpoint.pt")
a = ap.parse_args()

cfg = replace(Config(epochs=32, holdout=0), **json.loads(a.cfg))
print(json.dumps({k: str(v) for k, v in asdict(cfg).items()}), flush=True)

# Fresh ladder: the verifier scores only what is left here, and it must come
# from exactly one run with contiguous epochs 1..N.
if os.path.isdir(a.out):
    for f in os.listdir(a.out):
        if f.startswith("ckpt_epoch_") and f.endswith(".pt"):
            os.remove(os.path.join(a.out, f))
os.makedirs(a.out, exist_ok=True)

t0 = time.time()
r = train(cfg, ckpt_dir=a.out)
wall = time.time() - t0

n = cfg.epochs
last = os.path.join(a.out, f"ckpt_epoch_{n:04d}.pt")
if a.final:
    dst = "/workspace/checkpoint.pt"
    if os.path.exists(dst):
        os.remove(dst)
    shutil.copy(last, dst)

print(f"train_wall={r['wall']:.2f}s total_wall={wall:.2f}s "
      f"steps/ep={r['steps_per_epoch']} ckpts={len(os.listdir(a.out))}", flush=True)
with open(os.path.join(a.out, "..", "last_run.json"), "w") as f:
    json.dump({"cfg": {k: str(v) for k, v in asdict(cfg).items()},
               "train_wall": r["wall"], "total_wall": wall, "out": a.out}, f, indent=1)
