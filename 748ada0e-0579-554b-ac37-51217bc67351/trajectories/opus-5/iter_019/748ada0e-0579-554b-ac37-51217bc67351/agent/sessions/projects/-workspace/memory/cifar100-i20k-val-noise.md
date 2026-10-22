---
name: cifar100-i20k-val-noise
description: val_public on the CIFAR-100 20k task is too noisy to resolve sub-1.5% differences — budget attempts accordingly
metadata: 
  node_type: memory
  type: project
  originSessionId: bc9717df-0cd7-4642-9d63-57ac7d000f9c
  modified: 2026-09-03T22:31:11.077Z
---

val_public.npz is 2000 images but class counts run 10–31, so balanced accuracy on it
has a bootstrap sd of ~0.010 per run, and ~0.007 on a *paired* difference between two
runs scored on the same split. A config must win by ≳1.5% before the result means
anything.

**Why:** balanced accuracy averages per-class recall, and the classes with 10 samples
each contribute ~0.15 sd apiece.

**How to apply:** only test changes with a large expected effect; don't run
one-variable-at-a-time sweeps hoping to stack +0.3% wins, and don't pick the argmax
over hundreds of correlated TTA/weight-set combinations — that selection bias is
comparable to the differences being chased. Prefer a stated rule (e.g. average every
TTA view within 2% of the best one) over an exhaustive search.
See [[cifar100-i20k-tuning-deadends]].
