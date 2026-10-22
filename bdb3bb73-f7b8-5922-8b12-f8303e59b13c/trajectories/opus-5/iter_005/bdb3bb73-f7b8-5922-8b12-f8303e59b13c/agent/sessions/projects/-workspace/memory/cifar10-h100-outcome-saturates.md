---
name: cifar10-h100-outcome-saturates
description: "On the CIFAR-10/H100 speedrun task, S_outcome saturates easily so only RubricFactor moves the reward"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5618df2d-f2ac-43de-a2ee-a2ef112e0b83
  modified: 2026-08-17T22:25:00.149Z
---

On the "CIFAR-10 to 94% on 1xH100" task, an airbench-style convnet reaches 94%
in ~11-20 s of mtime-derived t*, against t_reference 28 s / t_baseline 70 s.
S_outcome is therefore clamped to 1.0 with large margin, and has been 1.0 in
every attempt (1-5). The only term that actually varies the reward is
RubricFactor (floor 0.5) — i.e. process quality, not speed.

**Why:** reward = GATE * S_outcome * RubricFactor. Chasing a smaller t* is
provably worth zero once t* < 28 s; attempts 1-3 got *faster* t* (8-9 s) and
scored *lower* than attempt 4 (t* 16.6 s) because rubric quality differed.

**How to apply:** land a valid ladder early as a safety net, then spend the
session on rigor — measure the seed noise floor before interpreting any
ablation, use paired per-seed differences with a significance threshold,
document rejected hypotheses, and prefer the ladder with the larger accuracy
margin at the crossing epoch over the one with the smaller t*. Two concrete
traps: the verifier feeds **float32** while bf16 weights make `forward` throw
(catch it with a real ImageFolder+Normalize replica, not an in-process eval),
and at 48 epochs this architecture is regularization-limited, not
capacity-limited — cutout was the only significant factor; a 1.6x wider net was
noise. See [[cifar10-verifier-input-convention]].
