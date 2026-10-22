#!/bin/bash
# Reproducibility run (scratch, different seed) -> then the graded run in place.
set -e
cd /workspace
CFG=$(cat final_cfg.json)
echo "=== repro run (seed 1, scratch) ==="
python finalize.py --cfg "$CFG" --seed 1 --ckpt-dir /workspace/scratch_ckpt
python verify_ladder.py --ckpt-dir /workspace/scratch_ckpt --start 14 --out /workspace/verify_scratch.json
echo "=== graded run (seed 0, in place) ==="
python finalize.py --cfg "$CFG" --seed 0 --ckpt-dir /workspace/periodic_checkpoints
python verify_ladder.py --ckpt-dir /workspace/periodic_checkpoints --out /workspace/verify_final.json
echo "=== validation ==="
python validate_checkpoints.py
