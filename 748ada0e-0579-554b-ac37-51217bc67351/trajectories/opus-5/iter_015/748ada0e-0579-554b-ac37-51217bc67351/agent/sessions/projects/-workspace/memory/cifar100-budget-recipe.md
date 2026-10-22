---
name: cifar100-budget-recipe
description: Winning config + measured dead ends for the CIFAR-100 8000-step budget task (workspace resets every attempt)
metadata: 
  node_type: memory
  type: project
  originSessionId: 96dfae2f-ce99-4f3e-963d-48ae3bb72736
  modified: 2026-09-03T19:25:19.764Z
---

AELLO CIFAR-100 fixed-budget task (20k train imgs, 8000 updates, 1.024M presentations,
432s/attempt). The `/workspace` dir is EMPTY at the start of every attempt — the pipeline
must be rebuilt from scratch each time. Budget so far in the session, so build first,
probe second: a valid submission was on disk at the 13-minute mark.

Best measured config (val balanced acc 0.7384, 252s standalone incl. inference):
ResNet-34-D (basic blocks 3,4,6,3), **width 128** (85.2M params), 32->**48px** bilinear
upsample, GPU-resident data with batched TrivialAugment + cutout(12) + pad4-crop + flip,
SGD-Nesterov lr 0.2 / wd 2e-3 (none on BN+bias) / cosine + 300-step warmup, label
smoothing 0.1, EMA 0.999. Predict with raw+EMA softmax average and flip TTA.

Measured dead ends (all at 8000/8000 steps, same seed):
- res 56: 0.7324 (worse than 48). res 60 also crashes: avgpool-vs-stride shape mismatch
  at odd feature sizes, so any res must be divisible by 8.
- width 96: 0.7175. Width 144 was worse in an earlier attempt too -> **128 is a real optimum**.
- SAM (rho .05) at batch 64 to stay presentation-legal: 0.6705. The halved batch costs far
  more than SAM buys.
- Deeper blocks (4,6,8,4), 116M params: 0.7349. Depth does not help either.
- Classic CIFAR layout (32px input, strides 1,1,2,2, so 32->32->16->8 instead of
  48->24->12->6): 0.7367 — statistically identical to the 48px-upsample net at the same
  width, and no cheaper. Not worth switching.
Net conclusion: every capacity axis (width up/down, depth, resolution, downsampling
schedule) is already at its optimum; the recipe is saturated near val ~0.74. Further gains
need a different idea, not more tuning of this family.

Speed: fuse the EMA update and SAM perturbation with `torch._foreach_*`; the naive
per-tensor Python loop over ~200 tensors dominated step time (30.5ms/step after fusing).

See [[cifar100-budget-val-noise]] for why not to tune post-processing on val.
