#!/usr/bin/env bash
# Reproduce the graded checkpoint ladder end to end on a single H100.
set -euo pipefail
cd /workspace
python cache_data.py                      # decode CIFAR-10 PNGs -> uint8 .npy caches
python run_ladder.py \
  --out   /workspace/periodic_checkpoints \
  --final /workspace/checkpoint.pt \
  --epochs 32 --seed 0 \
  --cfg '{"scale":0.03125,"lr":11.5,"wd":0.0051,"cutout":16}' \
  --meta  /workspace/artifacts/ladder_final_meta.json
python verify_ladder.py --dir /workspace/periodic_checkpoints   # exact verifier convention
