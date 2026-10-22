---
name: cifar100-budget-what-doesnt-help
description: "CIFAR-100 fixed-budget task — levers measured as gains vs. measured as flat/negative, so later attempts don't re-test them."
metadata: 
  node_type: memory
  type: project
  originSessionId: d0f17760-f084-4cd6-9566-20476c55ef0e
  modified: 2026-09-03T16:54:07.857Z
---

Gains (measured, val balanced accuracy, single H100 runs at the full 8000-step budget):
- Multi-crop TTA is the most reliable cheap win: raw flip-only 0.7179 -> 9 shifts x flip
  0.7267; raw+EMA 0.7324 -> 0.7379. ~+0.6-0.9%, costs ~50s of inference.
- EMA(0.999) snapshot + raw logit averaging: ~+1% over raw alone.
- Cutout: ~+1.9% over no cutout (measured in an earlier attempt).

Flat or negative (do not spend a run re-testing):
- Annealing cutout off over the final 8% of steps: 0.7209 vs 0.7324 baseline.
- Narrower + higher resolution at equal FLOPs (w128 @ res56): 0.7287 vs 0.7324.
- EMA decay 0.9995 over an 8000-step run: collapses to ~0.35 (stale BN buffers).
- Sinkhorn-to-uniform balancing and class-prior logit adjustment: help a *weak* single
  model (+0.7%) but are flat-to-negative once raw+EMA+TTA is in place.
- BN recalibration of EMA snapshots: only +0.1-0.3% when EMA decay is already 0.999. It
  *does* rescue over-long EMAs (0.9995: 0.35 -> 0.71). Recalibrate on *augmented* training
  batches; recalibrating on clean images is much worse (raw 0.718 -> 0.705).
- Two-cycle snapshot ensembling, CutMix/mixup: neutral in earlier attempts.

Related: [[cifar100-budget-recipe]], [[cifar100-budget-eval-noise]].
