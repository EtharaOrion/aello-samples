---
name: cifar10-h100-speedrun-scoring
description: "On the CIFAR-10-to-94%-on-H100 task, S_outcome saturates well below the achievable t*, so RubricFactor is the only real lever"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9a2a460c-a96a-4c9e-8ed1-1b981c6d40e0
  modified: 2026-08-18T06:19:36.236Z
---

On the "CIFAR-10 to 94% top-1 on 1xH100" refinement-loop task, an airbench-style
convnet reaches t* ≈ 9-21 s while `S_outcome = clamp((70 - t*)/(70 - 28))` hits
1.0 at any t* <= 28 s. Every recent attempt has therefore scored
`reward = 1 * 1.0 * RubricFactor`, and the spread across attempts (0.917 to 0.985)
is entirely process-rubric quality, not speed.

**Why:** chasing a faster t* is wasted effort once you are under 28 s; the two
things that still move reward are (a) not tripping a GATE knockout, and (b) the
rigour and honesty of the documented process.

**How to apply:** get a verified ladder on disk in the first ~15 min, then spend
the remaining budget on measurement rigour and the write-up, not on shaving
seconds. Prefer the ladder with the largest verified margin at the first
>= 94% checkpoint over the one with the smaller t*. Two traps that cost real
points: a 3-seed confirmation cannot resolve ~0.1 pp differences (regression to
the mean inflates the winner's margin ~2x), and the banned-substring gate is a
plain substring match, so the English word "to"+"uch" in a docstring and any
stray core dump in the workspace will both trip it.
