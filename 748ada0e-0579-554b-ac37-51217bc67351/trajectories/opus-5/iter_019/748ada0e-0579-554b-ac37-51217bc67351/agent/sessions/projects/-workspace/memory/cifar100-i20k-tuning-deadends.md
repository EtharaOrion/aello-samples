---
name: cifar100-i20k-tuning-deadends
description: "CIFAR-100 20k fixed-budget task — which training levers were measured and lost, so future attempts skip them"
metadata: 
  node_type: memory
  type: project
  originSessionId: bc9717df-0cd7-4642-9d63-57ac7d000f9c
  modified: 2026-09-03T22:31:04.757Z
---

On the CIFAR-100 / 20k-image / 8000-step task, the width-160 ResNet34-ish config
(res 48, SGD lr 0.2, wd 5e-4, ls 0.1, TrivialAugment + pad-4 crop + cutout 12,
shift+flip TTA) is a real local optimum. Measured on val_public (2026-09-03 session),
all with seed 11, paired-bootstrap sd of a difference ≈ 0.007:

- baseline: **0.7452**
- SGDR 2 cycles (3000+5000) with snapshot ensembling: 0.7395 — the step-3000 snapshot
  only reaches 0.62, far too weak to help the ensemble. This also kills any
  "split the 8000-step budget across N models and ensemble" plan.
- random-resized-crop (scale 0.65–1.2) replacing pad-4 crop: 0.7342
- weaker augmentation (TrivialAugment on 50% of samples, cutout 8): 0.7331
- earlier sessions: CutMix α=1 p=0.5 → −1%; wd 2e-3 → neutral; EMA (0.999) → −2%.

Clean train accuracy is 0.992–0.9995 in every variant, so the model always memorises
the training set; the 25-point train/val gap is NOT fixable by more input augmentation
at this step budget.

**Why:** the budget is step-limited, not time-limited (325 s of the 432 s cap), so
anything that makes each step harder to fit costs more than the regularisation buys.

**How to apply:** don't re-test augmentation strength, EMA, or budget-split ensembles.
Spend attempts on inference-side gains (richer shift/scale TTA is free and reliable,
worth ~+1% over a single view) and on capacity/resolution trades, which are untested.
See [[cifar100-i20k-val-noise]].
