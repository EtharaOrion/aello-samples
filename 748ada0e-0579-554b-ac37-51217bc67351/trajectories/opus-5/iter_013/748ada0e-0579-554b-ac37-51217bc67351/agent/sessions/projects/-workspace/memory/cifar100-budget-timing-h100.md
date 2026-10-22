---
name: cifar100-budget-timing-h100
description: Measured H100 wall-clock for the CIFAR-100 fixed-budget task; parallel runs give no speedup
metadata:
  type: project
---

Measured 2026-09-03 on one H100 80GB, bf16 autocast + channels_last, batch 128, 8000 steps,
all data resident on GPU (no dataloader). ResNet-34 layout (3,4,6,3), ResNet-D shortcuts
with avgpool on both paths, 32px inputs bilinearly upsampled to 48px:

- width 128 (85M params): **292s** end-to-end including inference. Fits the 432s cap easily.
- width 160 (133M params): ~445s extrapolated — **over** the 432s cap.
- width 144: ~370s.

**Why:** the cap is wall-clock, not FLOPs, so knowing the width/time curve tells you how much
capacity is actually affordable.
**How to apply:** do NOT run two training jobs concurrently on the one GPU to "save time" —
two full runs in parallel took 607s each, i.e. exactly the same total as running them
sequentially (the GPU is already saturated). Run probes sequentially so each one's timing is
also a valid standalone measurement. See [[cifar100-budget-trivialaugment-win]].
