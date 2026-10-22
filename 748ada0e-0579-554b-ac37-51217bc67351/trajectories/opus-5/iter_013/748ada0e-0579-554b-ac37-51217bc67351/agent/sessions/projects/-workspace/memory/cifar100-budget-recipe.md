---
name: cifar100-budget-recipe
description: Best measured recipe + the tried-and-rejected knobs for the CIFAR-100 fixed-step-budget task
metadata:
  type: project
---

Best config measured 2026-09-03 (val balanced acc 0.7355, 287s train+infer standalone,
8000/8000 steps, 1,024,000/1,024,000 presentations):

ResNet-34 layout (3,4,6,3) BasicBlocks, base width 128 (85M params), ResNet-D transitions
(avgpool2 on BOTH the shortcut and the main path), zero-init of the second BN gamma;
32px inputs bilinearly upsampled to 48px. SGD-Nesterov lr 0.2 (250-step warmup + cosine),
wd 2e-3 on weights only, label smoothing 0.1, batch 128, class-balanced sampling.
Aug: pad-4 pixel translate + hflip + one TrivialAugment op + cutout(12).
Inference: average softmax over {raw, EMA 0.999, EMA 0.9995} x {flip} x {5 shifts} x
{48px, 56px}.

Tried and NOT better at this budget (each a full 8000-step run):
- width 144 (108M, 400s): 0.7217 — extra capacity does not convert; the run is data-limited.
- two TrivialAugment ops per sample (RandAugment N=2): 0.7267 — N=1 is the sweet spot.
- no cutout: 0.7258 vs 0.7392 with cutout.

**Why:** the budget caps presentations, so ~51 epochs on 20k images; regularisation and input
diversity pay, raw capacity does not.
**How to apply:** start from this config and spend attempts on augmentation/inference, not on
scaling the net. See [[cifar100-budget-trivialaugment-win]], [[cifar100-budget-timing-h100]],
[[cifar100-budget-eval-noise]].
