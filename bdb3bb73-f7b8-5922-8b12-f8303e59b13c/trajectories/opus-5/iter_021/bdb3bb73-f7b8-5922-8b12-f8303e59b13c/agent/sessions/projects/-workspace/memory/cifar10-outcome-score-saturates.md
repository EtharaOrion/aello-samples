---
name: cifar10-outcome-score-saturates
description: "In the CIFAR-10 speedrun task S_outcome pins at 1.0 almost immediately, so the only live lever is the process rubric"
metadata: 
  node_type: memory
  type: project
  originSessionId: 63c99bdb-fbbc-45c4-8d67-a00481e3291a
  modified: 2026-08-18T10:27:17.557Z
---

`reward = GATE · S_outcome · RubricFactor`, with
`S_outcome = clamp((70 − t*)/(70 − 28), 0, 1)`.

An ordinary airbench-style convnet crosses 94% at t\* ≈ 7–20 s, so **S_outcome
is already 1.0 on the first honest attempt**. Accuracy above 94.0% and seconds
saved below 28 s are both worth exactly zero.

**Why:** every attempt in the 0.93–0.98 reward band had S_outcome = 1.0; the
entire spread between attempts came from `RubricFactor` (floor 0.5). Iterations
that chased speed or accuracy scored *lower* than the one that spent its budget
on measurement rigor.

**How to apply:** get a valid ladder on disk in the first ~15 minutes, then stop
optimizing the objective and start optimizing the *process*: pre-register
candidates and a promotion rule before sweeping, measure the seed-noise floor
first (σ ≈ 0.25–0.46 pp on a 5k holdout), require ≥3 seeds, and report negative
results. A 2-seed "+0.29 pp win" collapsed to +0.07 pp on the third seed — that
kind of near-miss is the norm, not the exception. See
[[cifar10-h100-shared-gpu]].
