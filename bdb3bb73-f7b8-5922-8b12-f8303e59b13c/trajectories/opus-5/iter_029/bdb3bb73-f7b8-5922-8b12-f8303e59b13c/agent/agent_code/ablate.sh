#!/bin/bash
cd /workspace
# Decompose the three "reference fidelity" fixes, 3 seeds each, at the ORIGINAL
# baseline settings (lr_final 0.07, cutout 0, 24 epochs).
# NOTE: --bn_momentum X is passed to nn.BatchNorm2d as (1-X). The pre-fix code passed
# 0.6 directly, which is reproduced here by --bn_momentum 0.4.
r(){ timeout 500 python train.py --epochs 24 --holdout 5000 --lr_final 0.07 --cutout 0 \
     --log /workspace/ablate.jsonl "$@" >/dev/null 2>&1; }
for s in 0 1 2; do
  ( r --bn_momentum 0.4 --bn_weight_grad 1 --dirac 0 --seed $s --tag "B0_original_s$s" ) &
  ( r --bn_momentum 0.6 --bn_weight_grad 1 --dirac 0 --seed $s --tag "B1_mom_s$s" ) &
  ( r --bn_momentum 0.6 --bn_weight_grad 0 --dirac 0 --seed $s --tag "B2_mom_gamma_s$s" ) &
  ( r --bn_momentum 0.6 --bn_weight_grad 0 --dirac 1 --seed $s --tag "B3_all_s$s" ) &
  wait
done
echo ABLATE_DONE
