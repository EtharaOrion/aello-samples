---
name: cifar100-budget-recipe
description: "Measured levers for CIFAR-100 20k @ 8000 steps x batch 128 — TrivialAugment helps, stacking it with high weight decay hurts"
metadata: 
  node_type: memory
  type: project
  originSessionId: 36a45a92-908c-4f5a-ad4e-40ce35dcd6ea
  modified: 2026-09-03T20:07:23.371Z
---

Full-budget runs (8000 steps, 1,024,000 presentations, ResNet-34-D w128 @ 48px, SGD lr 0.2 cosine,
label smoothing 0.1, cutout 12, flip TTA), val balanced accuracy, 2026-09-03:

| config | val |
|---|---|
| wd 5e-4 (baseline) | 0.7155 |
| width 160, wd 5e-4 | 0.7238 |
| wd 2e-3 | 0.7269 |
| **+ TrivialAugment, wd 5e-4** | **0.7299 / 0.7260 / 0.7165** (3 seeds) |
| TrivialAugment + wd 2e-3 | 0.7168 (combination over-regularises) |

- Batched GPU TrivialAugment (affine ops via one `grid_sample`, colour ops elementwise) costs ~7%
  wall clock; the whole run is ~236 s of the 432 s allowance, so capacity/aug is not time-limited.
- EMA of weights **hurts** here unless fast: decay 0.99 ≈ raw weights, 0.999 loses ~2%, and a
  0.9997 EMA collapsed to 0.55 (BN running stats belong to the raw weights and stop matching).
- 5-shift x flip TTA is worth ~+0.4%; logit adjustment by the *train* class prior
  (`p / prior**tau`, tau 0.5–1.0) is worth ~+0.2% and is principled because the metric is
  balanced accuracy and the train split is not class-balanced (166–231 images per class).

**Why:** 85M params on 20k images overfits hard, so the gains come from augmentation, not capacity;
but two strong regularisers stacked underfit within a 51-epoch budget.

**How to apply:** start from TrivialAugment + wd 5e-4 + cutout; judge changes against
[[cifar100-budget-val-noise]] before believing them.
