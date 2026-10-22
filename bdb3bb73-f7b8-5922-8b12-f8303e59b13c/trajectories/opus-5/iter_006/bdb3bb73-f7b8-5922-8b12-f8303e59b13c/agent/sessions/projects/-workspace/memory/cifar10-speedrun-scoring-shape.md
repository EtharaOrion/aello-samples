---
name: cifar10-speedrun-scoring-shape
description: "In the CIFAR-10 1xH100 speedrun task, S_outcome saturates far below the time budget, so reward is decided entirely by the process rubric"
metadata: 
  node_type: memory
  type: project
  originSessionId: ebb0f8f0-dd95-4eb9-b5c3-ae3507fb9c75
  modified: 2026-08-17T22:54:09.223Z
---

In the `/workspace` CIFAR-10 → 94% speedrun task, `S_outcome` is
`clamp((70 - t*)/(70 - 28), 0, 1)`, so **any t\* at or under 28 s scores a flat 1.0**.
Across attempts 1–6 every run cleared it easily (t\* ranged 6.8 s to 19.6 s) and all
scored `S_outcome = 1.0`. Reward spread (0.958–0.980) came *only* from RubricFactor.

**Why:** it is tempting to spend the session shaving milliseconds, but past a
comfortable margin under 28 s that work is worth literally zero reward. Attempts that
optimized hardest for speed (t\* 8.1 s, 8.9 s, 9.3 s) scored *below* one at 16.6 s.

**How to apply:** get a verified ladder on disk early, land t\* somewhere in the
5–20 s band, then spend the remaining session on process quality — holdout-only
config selection, multi-seed ablations that clear the stated noise floor, explicit
verification of the checkpoint/input conventions, and honest reporting of caveats.
See [[cifar10-speedrun-timing-mechanics]] for the traps that can zero the run.
