#!/bin/bash
# Run probes back-to-back once the baseline finishes. Probes write their graded
# predictions to /workspace (never directly to /submission) so the live submission is
# only overwritten after a probe has been measured and shown to be better.
cd /workspace/environment/starter
while pgrep -f "tag base_d2_s0" > /dev/null; do sleep 5; done

echo "=== P1 cyc2 ==="
python3 train.py --instance i20k --seed 0 --graded ../data/graded_images.npz \
    --out /workspace/preds_cyc2.npy --cycles 2 --cyc1 0.65 --cycmul 0.25 --ens 1 \
    --tag cyc2_s0 --log /workspace/runs.jsonl > /workspace/log_cyc2.txt 2>&1

echo "=== P2 cutmix-anneal ==="
python3 train.py --instance i20k --seed 0 --graded ../data/graded_images.npz \
    --out /workspace/preds_cmx.npy --cutmix 1.0 --mixstop 0.7 \
    --tag cmx07_s0 --log /workspace/runs.jsonl > /workspace/log_cmx.txt 2>&1

echo "=== queue1 done ==="
