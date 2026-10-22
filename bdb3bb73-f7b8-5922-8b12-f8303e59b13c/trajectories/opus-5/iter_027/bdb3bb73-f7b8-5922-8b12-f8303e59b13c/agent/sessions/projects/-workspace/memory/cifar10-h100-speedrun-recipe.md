---
name: cifar10-h100-speedrun-recipe
description: "Working CIFAR-10 recipe and the two bugs that silently destroy it (shared-module .eval(), double normalization)"
metadata: 
  node_type: memory
  type: project
  originSessionId: dc2165dd-63a2-4a47-9077-99ea01c3d7b4
  modified: 2026-08-19T07:55:54.786Z
---

Recipe that reaches 94.95% test in 24 epochs / ~20 s on a (shared) H100:
widened airbench net — frozen 2x2 ZCA whitening conv 3->24, then ConvGroups
128/384/512, MaxPool2d(3), `Linear(512,10,bias=False)`, `Mul(1/9)`; pure fp16 +
channels_last, fp32 BatchNorm with frozen unit scale; nesterov SGD on a
*summed* loss (lr in per-example units, 64x LR on biases), label smoothing 0.2,
LR ramp to peak at 23% then decay; 6-way TTA folded into the scripted forward.

Measured on a 5k held-out slice of train (sigma = 0.157 pp over 5 seeds):
- **`end_mult` 0.07 -> 0.02 is worth +0.62 pp** — the single largest effect.
- Schedule length alone does nothing (16 vs 30 epochs: 94.38 vs 94.42) *unless*
  paired with stronger augmentation; pad-4 + cutout-12 at 40 epochs gives 95.33.
- wd above ~0.02 hurts; lr/label-smoothing/whiten-bias tweaks were all noise.

**Why these two bugs matter:** both leave the training loop looking healthy.
1. `Deploy(net).eval()` on a wrapper that holds the *live* net propagates into
   `net`, freezing BatchNorm for the rest of training — loss keeps falling
   while test accuracy sits at exactly 10%.
2. The verifier feeds inputs that are *already* standardized; re-normalizing
   inside `forward` collapses accuracy under the verifier while a [0,1] harness
   still shows 94%.

**How to apply:** build the checkpoint wrapper so it never mutates the training
module's mode, and make `forward` branchlessly detect the input convention
(`max > 20` -> /255 then standardize; `min >= -0.02` -> standardize; else pass
through). Verify with `ImageFolder` + `ToTensor` + `Normalize` at batch 512
before trusting any number. See [[cifar10-h100-speedrun-scoring]].
