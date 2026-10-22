#!/bin/bash
cd /workspace
W="--lr_final 0.02 --cutout 8 --dirac 1 --bn_weight_grad 0 --epochs 24 --lr_hold_epochs 6 --holdout 0"
for s in 1 2 3; do ( timeout 500 python train.py $W --seed $s --ckpt_dir /workspace/scratch_ckpt/s$s --tag seed$s >/dev/null 2>&1 ) & done
wait
for s in 1 2 3; do ( python eval_ladder.py --dir /workspace/scratch_ckpt/s$s --out /workspace/lad_s$s.json > /workspace/ev_s$s.log 2>&1 ) & done
wait
echo SEEDCHECK_DONE
