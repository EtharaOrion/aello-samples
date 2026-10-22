---
name: cifar100-budget-eval-noise
description: val_public (2000 imgs) on the CIFAR-100 budget task is ~1% noisy; avoid post-hoc prior correction
metadata:
  type: project
---

`val_public.npz` has only 2000 images over 100 classes (10-31 per class), so balanced
accuracy measured on it swings ~1% between seeds of an identical config (0.7392 vs 0.7295,
measured 2026-09-03). Differences under ~1% between two configs are not resolvable with one
run each.

**Why:** per-class recall estimated from ~20 samples is very high variance, and the graded
split is a different 10000 images.
**How to apply:** prefer changes with a strong prior (more augmentation, more capacity within
the time cap) over chasing small val deltas. Do NOT fit post-hoc corrections on val — an
earlier iteration of this task applied Sinkhorn/marginal balancing plus class-prior and BN
recalibration tuned on val, reported val 0.7407, and scored the equivalent of 0.672 on the
graded split (its worst result). Plain flip+shift TTA over the run's raw and EMA weights is
the safe inference-side gain (+0.3-0.4%). See [[cifar100-budget-trivialaugment-win]].
