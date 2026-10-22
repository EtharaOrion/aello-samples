---
name: cifar100-short-budget-regularisation-hurts
description: "At an 8000-step CIFAR-100 budget, train_acc hits 0.998 yet every added regulariser loses — the step budget, not overfitting, is binding."
metadata: 
  node_type: memory
  type: project
  originSessionId: 8226c946-6e6b-4702-84cd-cc85201a32e3
  modified: 2026-09-03T05:42:33.229Z
---

CIFAR-100, 20k train images, 8000 optimiser steps at batch 128 (~51 epochs), ResNet-18
width 96. The net reaches `train_acc = 0.998`, which looks like textbook overfitting, but
adding regularisation consistently **lost** on val balanced accuracy:

| change | val delta |
|---|---|
| cutmix p=0.5 / mixup | ~0 |
| cutout 8 -> 16 | ~0 |
| wd 2e-3 -> 4e-3 | -1.0% |
| cutout16 + cutmix + wd 3e-3 | -3.3% (train_acc fell to 0.85) |

Also no gain: SE blocks, balanced sampling, logit adjustment (class imbalance is only
1.39x), flip9 and multi-scale TTA, lr 0.4, EMA (its BN running stats are wrong unless
recomputed). Input res 48 vs 32 was a wash. The single robust win was **flip5 TTA**,
+0.56% mean over 10 paired runs.

**Why:** at ~51 epochs the budget is the binding constraint. Regularisers trade
convergence speed for asymptotic generalisation, and there is no room to collect the
asymptotic half. High train accuracy does NOT imply regularisation will help at short
schedules.

**How to apply:** on a fixed-step budget, spend effort on optimisation efficiency and on
free prediction-time gains (TTA), not on augmentation strength. Measure the noise floor
first — same-seed run-to-run spread here was ~+-0.6%, so single-run deltas under ~1% are
meaningless; prefer *paired* comparisons (TTA modes on one trained model) which cancel
training noise. Related: [[cifar100-budget-grader-reruns-defaults]]
