---
name: cifar100-budget-what-fails
description: On the CIFAR-100 fixed-step-budget task, val_public (2000 imgs) has ~1% noise, so heavy post-hoc tuning overfits it and loses graded reward.
metadata:
  type: feedback
---
CIFAR-100 8000-step budget task. val_public is only 2000 images -> sigma ~1% on
balanced accuracy. Iterations that stacked post-hoc knobs (marginal/Sinkhorn balancing,
class-prior adjustment, BN recalibration, blend weights) reported val 0.735-0.741 but
scored graded 0.625-0.664; the iteration that shipped a plain single-run + simple TTA
reported val 0.734 and scored 0.755.
**Why:** each knob is selected on the same 2000 images, so the reported val gain is
selection noise that does not transfer to the 10000 graded images.
**How to apply:** pick the recipe a priori, commit to a decision rule before looking
(e.g. "only swap runs if val beats by >0.5%"), and ship. Also: class-balanced sampling
gave no gain (0.7243 vs 0.7302); running 3 trainings in parallel on one H100 is ~2.9x
slower each, not free -- run at most 2. See [[cifar100-shift-tta-win]].
