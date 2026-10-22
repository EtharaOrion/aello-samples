---
name: cifar100-shift-tta-win
description: On the CIFAR-100 fixed-step-budget task, 5-view shift TTA is a reliable +0.6% over flip-only TTA; extra scales/diagonal shifts add nothing.
metadata:
  type: project
---
CIFAR-100 8000-step budget task: at prediction time, averaging softmax over 5 views
(center + 2px up/down/left/right, each also hflipped) beat flip-only TTA by ~+0.6%
balanced accuracy (0.7302 -> 0.7364 on val_public), measured 2026-09-03.
Adding scales (44/52px) or diagonal shifts did NOT help (0.733-0.735, i.e. noise).
Train-prior logit adjustment (divide probs by prior^p) is worth <=0.2% and its best
p flips between 0 and 1 run to run -- it is noise, do not tune it.
**Why:** train split is mildly class-imbalanced (166-231/class) but the metric is
balanced accuracy, so prior correction *sounds* principled yet is below noise.
**How to apply:** always ship shift5+flip TTA; do not spend runs tuning prior_pow.
See [[cifar100-budget-what-fails]].
