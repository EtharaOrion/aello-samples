---
name: cifar100-budget-eval-noise
description: "The AELLO CIFAR-100 val split is tiny and imbalanced, so single-run probes under ~1% are noise"
metadata: 
  node_type: memory
  type: project
  originSessionId: 658eafb4-1478-4fed-94c9-489ed15da7fa
  modified: 2026-09-03T12:59:00.141Z
---

`val_public.npz` in AELLO-C8-S1 is only **2000 images and class-imbalanced (10-31 per
class)**. Balanced-accuracy run-to-run sigma from paired reruns is **~0.5-0.6%**.

**Why:** classes with 10 examples move the per-class recall average in 10% jumps, so the
mean over 100 classes is far noisier than a 2000-image accuracy would suggest.

**How to apply:** treat any single-run delta under ~1% as unmeasured. Do not spend budget
confirming a small win or chasing a small loss — spend it on changes with a large prior
(augmentation, ensembling) and accept them only when the delta clears ~1%. When comparing
post-processing choices rather than models, use *paired* evaluation on the same resampled
subsets; that removes almost all the variance. Related: [[cifar100-budget-cutout-win]],
[[cifar100-marginal-balancing]].
