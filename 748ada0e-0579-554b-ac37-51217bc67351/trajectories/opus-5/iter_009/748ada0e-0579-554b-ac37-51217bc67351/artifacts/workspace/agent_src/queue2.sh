#!/bin/bash
# P3: faithful baseline repeat, but saving the checkpoint and the val/graded log-probs.
# Buys (a) a second independent noise sample of the baseline, (b) a checkpoint that lets
# prediction-time work be explored offline without spending further training runs.
cd /workspace/environment/starter
while pgrep -f "tag cmx07_s0" > /dev/null; do sleep 5; done

echo "=== P3 baseline+ckpt (seed 1) ==="
python3 train.py --instance i20k --seed 1 --graded ../data/graded_images.npz \
    --out /workspace/preds_b1.npy --save /workspace/ck_b1.pt \
    --savelogits /workspace/lg_b1 \
    --tag base_s1_ckpt --log /workspace/runs.jsonl > /workspace/log_b1.txt 2>&1
echo "=== queue2 done ==="
