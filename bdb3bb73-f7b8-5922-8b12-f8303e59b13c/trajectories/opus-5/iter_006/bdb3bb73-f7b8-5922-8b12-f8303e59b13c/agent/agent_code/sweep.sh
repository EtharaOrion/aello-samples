#!/bin/bash
# Holdout sweep: every configuration decision is made on 5,000 held-out TRAIN
# images. The test split is not read by train.py at all.
# Usage: ./sweep.sh <tag> <extra train.py args...>   -- runs seeds 0,1,2
set -e
TAG=$1; shift
for SEED in 0 1 2; do
  python train.py --holdout 5000 --seed "$SEED" \
    --out-dir /workspace/scratch_ckpt --no-final-copy \
    --log "/workspace/sweeplogs/${TAG}_s${SEED}.json" "$@" > /dev/null
done
python - "$TAG" <<'PY'
import json, sys, glob, numpy as np
tag = sys.argv[1]
accs, cross, ep_t = [], [], []
for f in sorted(glob.glob(f"/workspace/sweeplogs/{tag}_s*.json")):
    d = json.load(open(f))
    va = [e["val_acc"] for e in d["epochs"]]
    accs.append(va[-1])
    c = next((e["epoch"] for e in d["epochs"] if e["val_acc"] >= 0.94), None)
    cross.append(c if c else 999)
    t = d["epochs"]
    ep_t.append((t[-1]["t_since_start"] - t[0]["t_since_start"]) / (len(t) - 1))
print(f"{tag:28s} final={np.mean(accs)*100:6.2f}% +-{np.std(accs)*100:4.2f} "
      f"runs={[round(a*100,2) for a in accs]} cross={cross} ep={np.mean(ep_t):.3f}s")
PY
