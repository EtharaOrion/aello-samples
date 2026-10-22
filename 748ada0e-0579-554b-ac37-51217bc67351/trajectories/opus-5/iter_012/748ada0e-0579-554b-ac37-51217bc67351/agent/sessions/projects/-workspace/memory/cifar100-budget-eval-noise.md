---
name: cifar100-budget-eval-noise
description: "The 2000-image public val split for the CIFAR-100 budget task has ~1% noise on balanced accuracy — don't chase sub-1% differences, including TTA differences."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d0f17760-f084-4cd6-9566-20476c55ef0e
  modified: 2026-09-03T17:16:55.529Z
---

`val_public.npz` has only 2000 images over 100 classes (10-31 per class, not uniform), so
per-class recall is estimated from ~20 samples and the standard error of *balanced*
accuracy is ~1.0%. A difference of <1% between two training configs is not evidence.

Concrete instance: two runs of the *identical* config (seeds 1 and 4) measured 0.7324 and
0.7336 with flip-only TTA. Adding 4 shifted views moved seed 1 to 0.7407 (+0.8%) but
seed 4 to 0.7316 (-0.2%). The same "improvement" was worth +0.8% on one checkpoint and
-0.2% on another, so even *paired* prediction-time comparisons need to be read as
~±0.5%, not as ranked results.

**Why:** several attempts burned full-budget runs (~350s each) ranking configs whose true
difference was smaller than the measurement noise, and picked "winners" that did not
transfer to the graded split.

**How to apply:** spend runs only on changes with a large expected effect. Prefer
*paired* comparisons — TTA view sets, snapshot combos, post-hoc corrections on the *same*
trained weights — but still treat sub-0.5% paired gaps as ties and prefer the cheaper or
more averaged option. Never pick the best of N seeds by val; that buys ~+0.7% of val
overfitting and none of it transfers. Val also reads ~1-2% optimistic relative to the
graded split. Related: [[cifar100-budget-recipe]], [[cifar100-budget-what-doesnt-help]].
