---
name: cifar10-ladder-timestamps-are-the-clock
description: "The CIFAR-10 speedrun verifier derives t* from checkpoint save times, so the final run must train directly into the graded directory"
metadata: 
  node_type: memory
  type: project
  originSessionId: 67707c3d-342e-4e24-94a8-e0f10d5ca4a1
  modified: 2026-08-18T12:52:50.606Z
---

The verifier rebuilds `t_star` as
`(save_time[ckpt_N] - save_time[ckpt_0001]) * N/(N-1)` from the graded
checkpoints' own save times.

**Why:** copying a finished ladder in from scratch space stamps every file
within a fraction of a second of every other, which would report a training
speed that never happened — a dishonest timer, and a knockout. It also means a
verified ladder **cannot be restored** once deleted, so "re-run and roll back if
worse" is not available as a strategy.

**How to apply:** train the final run *directly into*
`/workspace/periodic_checkpoints/` after clearing it; only
`/workspace/checkpoint.pt` is a copy. Treat any decision to re-run as
irreversible and price it accordingly. See
[[cifar10-speedrun-reward-shape]].
