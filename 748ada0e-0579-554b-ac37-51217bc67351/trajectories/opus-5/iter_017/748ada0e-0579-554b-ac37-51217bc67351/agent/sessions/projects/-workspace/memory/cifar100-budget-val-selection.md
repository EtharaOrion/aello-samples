---
name: cifar100-budget-val-selection
description: Guard against overfitting the 2000-image val split when picking configs/TTA in the CIFAR-100 budget task
metadata: 
  node_type: memory
  type: project
  originSessionId: c1cbbf34-0767-426d-8f42-7915285a38e7
  modified: 2026-09-03T21:03:10.218Z
---

The local eval split (`val_public.npz`) is only 2000 images, so balanced accuracy on
it has a bootstrap sd of roughly 0.01. Scoring ~90 inference combinations
(weights x resolution x TTA x logit-adjustment tau) and taking the argmax reliably
overfits that noise.

Two guards that worked:

1. Inside the run, rank all candidates but then pick, among everything within 0.005
   of the top, the candidate that averages the *most* variants. Averaging is the more
   robust estimator and the top few candidates are statistically tied anyway.
2. Across runs, only promote a new config over the incumbent if it wins by more than
   about 0.008 val. Otherwise keep the incumbent — a config that wins by noise on val
   will not carry that win to the graded split.

Also worth structuring runs this way: compute val probabilities for every inference
variant (cheap — 2000 images), pick the combination, then compute graded
probabilities only for the chosen combination (10000 images, the expensive part).
Cap the chosen combination's size so graded inference cost stays bounded.

**Why:** with every lever landing within ~1 point of the baseline, selection noise
was the dominant error source, not modelling.

**How to apply:** treat sub-0.008 val differences as ties and prefer the simpler or
already-measured option. See [[cifar100-fixed-budget-levers]].
