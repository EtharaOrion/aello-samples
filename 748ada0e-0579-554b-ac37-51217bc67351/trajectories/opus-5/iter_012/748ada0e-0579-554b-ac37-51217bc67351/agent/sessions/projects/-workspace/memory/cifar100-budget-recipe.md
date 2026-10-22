---
name: cifar100-budget-recipe
description: "Best-known config for the CIFAR-100 8000-step/1.024M-presentation task (AELLO i20k), rebuilt from scratch each attempt because /workspace resets."
metadata: 
  node_type: memory
  type: project
  originSessionId: d0f17760-f084-4cd6-9566-20476c55ef0e
  modified: 2026-09-03T16:53:59.114Z
---

The `/workspace` for this task is wiped between attempts — the starter is stock every
time, so the pipeline must be rewritten from scratch. Recipe that reproduces ~0.732 val
balanced accuracy (single H100, ~337s train):

ResNet-34 layout (basic blocks, depths 3,4,6,3), base width 160 (133M params), CIFAR stem
(3x3, no maxpool), anti-aliased stage transitions (AvgPool2d(2) applied to *both* the
shortcut and the main path, then stride-1 convs), zero-init the second BN gamma per block.
Inputs bilinearly upsampled 32 -> 48. SGD Nesterov lr 0.2 (cosine to 0, 3% warmup),
momentum 0.9, wd 2e-3 on conv/fc weights only, label smoothing 0.1, batch 128, 8000 steps.
Augmentation: reflect-pad-4 random crop, hflip, cutout 12 (at 32px scale). Keep all data
on the GPU as uint8 and augment on-device — CPU indexing is the bottleneck otherwise.

Measured on val (2000 imgs): raw 0.718, EMA(0.999) 0.729, raw+EMA 0.732, and with
9-offset x flip TTA 0.738. See [[cifar100-budget-what-doesnt-help]] and
[[cifar100-budget-eval-noise]].
