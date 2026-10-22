#!/bin/bash
# Round 1: baseline solo (writes the submission), then two variants in parallel.
cd /workspace/environment/starter
S=/workspace/environment/data/graded_images.npz

echo "=== A_base solo $(date +%s)"
python3 train.py --seed 1 --tag A_base --graded $S \
  --out /submission/preds_i20k.npy \
  --logits /workspace/lg_A.npz --ckpt /workspace/ck_A.pt > /workspace/log_A.txt 2>&1
echo "=== A done $(date +%s)"
tail -14 /workspace/log_A.txt

echo "=== B,C parallel $(date +%s)"
# B: cutout annealed off over the final 8% of steps
python3 train.py --seed 2 --tag B_augoff --aug-off 0.08 --graded $S \
  --logits /workspace/lg_B.npz --ckpt /workspace/ck_B.pt > /workspace/log_B.txt 2>&1 &
# C: narrower but higher-resolution at ~equal FLOPs
python3 train.py --seed 3 --tag C_w128r56 --width 128 --res 56 --cut 12 --graded $S \
  --logits /workspace/lg_C.npz --ckpt /workspace/ck_C.pt > /workspace/log_C.txt 2>&1 &
wait
echo "=== B,C done $(date +%s)"
tail -14 /workspace/log_B.txt
tail -14 /workspace/log_C.txt
