---
name: cifar10-gpu-contention
description: "The H100 in the CIFAR-10 speedrun container is shared, which sets the noise floor"
metadata: 
  node_type: memory
  type: project
  originSessionId: 13ca8317-1c66-4ecc-a0cd-9b65df12c1c1
  modified: 2026-08-19T08:29:50.584Z
---

The H100 in the CIFAR-10 speedrun container is **shared with an external workload**:
`nvidia-smi` shows 100% utilization and ~6 GB used while reporting "No running
processes found". Epoch time is ~0.80 s rather than the ~0.41 s earlier attempts logged.

**Why:** timing-based `cudnn.benchmark` autotuning picks different algorithms under
contention, so two identical-command full-train runs (same seed, same config) differed by
~0.13 pp final accuracy and ±1 crossing epoch — while an in-process held-out sweep
reproduced its per-seed values bit-exactly across separate processes.

**How to apply:** treat held-out deltas below ~0.2 pp on 2 seeds as noise and refuse to
promote them. Budget schedule length so the crossing epoch keeps ≥10 s of t\* headroom
under the 28 s saturation point, and aim for several checkpoints above 94% rather than
one, so a ±1 epoch shift cannot change the score. Also benchmark epoch time early — do
not assume timings quoted by previous attempts still hold.

Related: [[cifar10-h100-speedrun-task]].
