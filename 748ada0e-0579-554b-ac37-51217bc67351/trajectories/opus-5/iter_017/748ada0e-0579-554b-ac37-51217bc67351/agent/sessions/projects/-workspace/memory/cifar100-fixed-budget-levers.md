---
name: cifar100-fixed-budget-levers
description: Measured results of accuracy levers for the CIFAR-100 20k-image / 8000-step budget task — what helped and what did not
metadata: 
  node_type: memory
  type: project
  originSessionId: c1cbbf34-0767-426d-8f42-7915285a38e7
  modified: 2026-09-03T21:03:00.537Z
---

Task: CIFAR-100, 20k train images, budget 8000 optimizer updates x batch 128 =
1,024,000 presentations, 432 s wall clock per attempt, scored on balanced accuracy.
Baseline that scores well: ResNet-34-ish (stem stride 1, blocks 3,4,6,3, width 160,
ResNet-D shortcut), inputs upsampled 32 -> 48, SGD lr 0.2 cosine + 300 warmup,
wd 5e-4, label smoothing 0.1, TrivialAugment + cutout 12, EMA 0.99, plus-shaped
shift TTA (5 offsets x hflip). Val balanced accuracy ~0.736 on the 2000-image
val_public split; graded came in ~1.3 points lower than val.

Measured on 2026-09-03 (all seed 0, so data order and augmentation are identical
across runs; val noise sd ~0.01 on 2000 images):

- Constant res 48 baseline: **0.7359**
- Progressive resolution (32 for first 42%, then 48): 0.7310 — saved 61 s of train
  time but cost ~0.5 pt. Bad trade.
- CutMix alpha 1.0, p 0.5: 0.7282 — clearly negative. At only ~51 epochs, stacking
  CutMix on top of TrivialAugment + cutout over-regularizes.
- SE blocks (ratio 16): 0.7349 — no gain, and +18% wall clock (not the ~6% the FLOP
  count suggests; the pooling + Linear ops are latency-bound, not FLOP-bound).
- BN recalibration on clean training views after training: 0.7234 vs 0.7265 without.
  Slightly negative — the augmented-vs-clean BN stat mismatch is not a real problem here.
- Grid TTA (3x3 = 9 offsets) vs plus-shaped (5 offsets): 0.7315 vs 0.7349. The corner
  shifts hurt; keep the plus shape.
- Multi-resolution TTA (test at 40 / 48 / 56 when trained at 48): no gain at all,
  and it cost ~60 s of the 432 s budget. Test resolution should match train resolution.
- Width scaling is the one lever with positive evidence: w128 -> w160 gave about
  +1.4 pt. Going past w160 at res 48 does not fit in 432 s (w176 ~= 405 s train alone).

Balanced accuracy with a mildly imbalanced train set (166-231 per class) wants logit
adjustment: divide probabilities by prior^tau. tau=1.0 is the principled choice and
val agreed.

**Why:** several of these are the "obvious" tricks that a fresh session would burn
attempts re-testing; all of them are neutral-to-negative in this specific
short-schedule, small-data regime.

**How to apply:** start from the constant-res-48 w160 baseline and do not re-spend
attempts on CutMix, SE, BN recalibration, progressive resizing, grid TTA, or
multi-res TTA. Spend attempts on capacity only if wall-clock headroom appears.
See [[cifar100-budget-val-selection]].
