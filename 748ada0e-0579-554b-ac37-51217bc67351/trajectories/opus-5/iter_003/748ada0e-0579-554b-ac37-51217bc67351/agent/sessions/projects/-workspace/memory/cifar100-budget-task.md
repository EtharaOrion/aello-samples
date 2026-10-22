---
name: cifar100-budget-task
description: "CIFAR-100 fixed-step-budget task — how local val maps to the graded score, and what actually moves the needle"
metadata: 
  node_type: memory
  type: project
  originSessionId: d7df8c3c-6799-4def-b31c-33ab013ffdd7
  modified: 2026-09-03T06:26:10.791Z
---

The `/workspace` CIFAR-100 task (8000 updates x batch 128, 432s per attempt, graded on
balanced accuracy over 10000 private images) has a stable offset between the local
`val_public` split and the graded score: iteration 2 measured val 0.7055 and earned
reward 0.6604, i.e. graded balanced accuracy 0.686 — about **1.9% below val**. Use that
offset when predicting reward from a local number; 2000-image val also carries ~±0.5%
same-seed run-to-run noise, so gains under ~1% are not measurable in one run.

As of 2026-09-03 (iteration 3) the binding lever is **model capacity**, not augmentation
or regularisation: scaling the CIFAR ResNet from r18-w96 to r34-w160 at 48px took val
0.7055 -> 0.7237, and accuracy tracked log(training cost) at ~+1.6% per e-fold until it
flattened near the wall-clock ceiling. Batched affine augmentation (rotation/zoom/aspect)
was a wash, matching the earlier finding that this budget is convergence-limited. The
full ladder and every negative result live in the module docstring of
`environment/starter/train.py` — read it before re-deriving anything.

**Why:** each iteration starts from the previous best source, so the cheapest way to gain
is to extend that ladder rather than re-test the knobs already ruled out.
**How to apply:** next iteration, the untested direction is capacity *shape* at fixed
cost (deeper-and-narrower than r34-w160, or bottleneck blocks) plus anything that makes a
step cheaper, since ~5.6x baseline cost is all the 432s allows.
