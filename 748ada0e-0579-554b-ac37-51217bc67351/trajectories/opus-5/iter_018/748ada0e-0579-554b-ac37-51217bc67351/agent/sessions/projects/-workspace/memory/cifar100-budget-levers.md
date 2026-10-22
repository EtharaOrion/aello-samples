---
name: cifar100-budget-levers
description: Measured what helps/hurts on the CIFAR-100 20k-image 8000-step budget task
metadata: 
  node_type: memory
  type: project
  originSessionId: e82b5997-bd49-43e0-9609-1abbc9f4bd21
  modified: 2026-09-03T21:52:28.815Z
---

On the CIFAR-100 @20k / 8000-step / 1.024M-presentation task (ResNet-34-shape, width 160,
inputs upsampled 32->48, TrivialAugment + cutout, SGD lr 0.2 cosine), measured on
`val_public` in the 2026-09-03 session:

- **Shift-TTA is the biggest free win: +1.1%** (raw 0.7294 -> 0.7407). Confirmed on 5+ runs.
  An earlier session computed it but never used it in the write path — always check that the
  submission actually uses the best *measured* inference combination.
- **CutMix (alpha 1.0, p 0.5) HURTS: -1.0%** — 51 epochs is too short for it to pay off.
- **Weight decay 2e-3 is not better than 5e-4** at width 160.
- **EMA is consistently worse than the raw weights** (0.709 vs 0.729); cosine-to-zero already
  averages. Not worth the code path.
- Width 128 -> 160 gave +1.2%. Width 160 @ res 48 trains in ~326s solo on an H100
  (~360s incl. TTA), so ~432s wall clock leaves little room for more capacity.
- val_public is only 2000 images: bootstrap sd of balanced accuracy is **~0.0095**. Seed-to-seed
  spread is ~0.005. Do not chase differences below ~0.01, and prefer a-priori view averaging
  over argmax across many correlated TTA combos.
- Running 3 jobs concurrently on one H100 slows each ~3.4x (326s -> 1095s). Fine for
  experiments, but measure the graded run solo. See [[cifar100-budget-workflow]].
