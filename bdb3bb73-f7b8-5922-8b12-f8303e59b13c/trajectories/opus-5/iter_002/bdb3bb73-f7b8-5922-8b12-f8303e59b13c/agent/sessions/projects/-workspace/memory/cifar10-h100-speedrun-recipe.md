---
name: cifar10-h100-speedrun-recipe
description: Tuned recipe and scoring insight for the CIFAR-10 94%-on-1xH100 speedrun task in /workspace
metadata: 
  node_type: memory
  type: project
  originSessionId: 1ad7b116-c08b-4980-bf84-b953f0aa2262
  modified: 2026-08-17T20:19:40.047Z
---

For the CIFAR-10 → 94% top-1 on 1×H100 task (attempt 2, 2026-08-17): an
airbench-lineage convnet reached t\* = 8.92 s (first checkpoint ≥94% at epoch
21, final 94.93%) with widths 128/384/512, batch 1000 (= exactly 50 steps/epoch
over 50k), 24 epochs, Nesterov SGD momentum 0.85, lr 11.5 and wd 0.008 quoted
per-1024-examples with sum-reduced loss, label smoothing 0.2, frozen 2×2
whitening conv from patch-covariance eigenvectors, BatchNorm with frozen unit
scale, and mirror+1px-translate TTA folded into `forward`.

**Why:** the scored quantity saturates long before this — `S_outcome` clamps to
1.0 for any t\* ≤ 28 s (baseline 70 s), and attempts 1 and 2 both hit 1.0 at
~8–9 s. Once a run is under ~15 s, more speed buys literally nothing and the
only remaining headroom is the process-rubric factor (validation discipline,
documentation, honest metrics).

**How to apply:** get a verified ladder on disk first, then spend the session on
(a) held-out tuning with *repeats* — same config/seed measured 94.76% in one
sweep process and 94.06% in another, so only compare configs inside one process
— and (b) verifiable process quality. Two levers dominate accuracy: width
(+1.3%) and TTA level (+0.7%); epochs past ~24 add nothing. Do not enable
`torch.compile` for the graded run: it moves ~15 s of compile before epoch 1,
outside the measured window, shrinking t\* while making the job slower overall.
