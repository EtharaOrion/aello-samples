---
name: cifar100-budget-trivialaugment-win
description: GPU TrivialAugment is the biggest single lever on the CIFAR-100 fixed-step-budget task (+1.3% balanced acc)
metadata:
  type: project
---

On the AELLO CIFAR-100 task (20k train images, <=8000 steps, <=1,024,000 presentations),
adding a TrivialAugment-style random-op augmentation on top of crop+flip+cutout(12) moved
val balanced accuracy from 0.7258 to 0.7392 (same seed, same net). Measured 2026-09-03.

13 ops: identity, autocontrast, rotate(0.35rad), shearX/Y(0.25), translateX/Y(0.15),
brightness/contrast/colour/sharpness(+-0.7), posterize(8->4 bits), solarize. One op per
sample, magnitude ~U(0,1), random sign.

**Why:** 20k images / 100 classes at ~51 epochs is squarely in the overfitting regime, so
extra input diversity beats extra capacity.
**How to apply:** implement it batched on GPU with no data-dependent branching — build one
affine matrix per sample (flip + pixel translate + the geometry op composed) for a single
`grid_sample`, then apply every colour op to the whole batch masked by `where(idx==k, ...)`.
Cost is negligible; no `.any()`/`.item()` calls, so no host syncs. See
[[cifar100-budget-timing-h100]] and [[cifar100-budget-eval-noise]].
