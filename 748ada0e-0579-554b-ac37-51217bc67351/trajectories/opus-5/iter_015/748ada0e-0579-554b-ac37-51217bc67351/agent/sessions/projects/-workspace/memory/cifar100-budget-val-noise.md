---
name: cifar100-budget-val-noise
description: The 2000-image val split in the CIFAR-100 budget task is too noisy to select post-processing; uniform marginal balancing actively hurts
metadata: 
  node_type: memory
  type: project
  originSessionId: 96dfae2f-ce99-4f3e-963d-48ae3bb72736
  modified: 2026-09-03T19:12:07.602Z
---

In the AELLO CIFAR-100 budget task the local split is `val_public.npz`, only 2000 images
and itself class-imbalanced (10-31 per class). Measured bootstrap std of balanced accuracy
there is **+-0.0095**, while the entire spread across 24 post-processing choices
({raw, EMA, raw+EMA} x {flip, shift5 TTA} x prior-adjust tau in {0,.25,.5,1}) was only
0.7343-0.7399. So **every post-processing knob is inside the noise** — tuning them on val
is pure noise-fitting, and doing so is what produced the erratic val->graded gaps in
earlier attempts (claimed val ~0.74 landing at graded 0.667 and 0.688).

Policy: fix the simple default (raw+EMA average, flip TTA, no prior adjustment) and only
accept a *training* change that wins by more than ~1.5% on val.

Also: train labels are imbalanced (166-231 per class) and val is too, so the graded split
is probably NOT uniform. Forcing output marginals to uniform (Sinkhorn / marginal
balancing) is unjustified here and coincided with the worst graded scores.

Config that these apply to: [[cifar100-budget-recipe]].
