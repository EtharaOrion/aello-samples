---
name: cifar10-h100-shared-gpu
description: The H100 in the CIFAR-10 speedrun task is shared with an outside tenant; identical runs vary ~3x in wall clock
metadata: 
  node_type: memory
  type: project
  originSessionId: 63c99bdb-fbbc-45c4-8d67-a00481e3291a
  modified: 2026-08-18T10:27:10.814Z
---

In the CIFAR-10 → 94% speedrun task (as of 2026-08-18), the "1×H100" is **not
exclusive**. `nvidia-smi` showed 100% utilization and 543 W drawn while my own
container had zero compute processes, at full 1980 MHz with no throttle flags.

Identical 32-epoch runs measured 11.4 s, 15.1 s, 24.4 s, 25.0 s, and later 45 s.

**Why:** `t*` is reconstructed by the verifier from *checkpoint save times*, so
foreign contention inflates the graded metric directly. A config validated at
t\* = 12 s can ship at t\* = 41 s and drop S_outcome from 1.0 to 0.69.

**How to apply:** never assume a timing measurement reproduces. Put a hard
validity gate in the promotion script (refuse the swap unless the new ladder is
contiguous, contains a ≥94% checkpoint, and reconstructs to t\* ≤ 26 s), and
retry rather than accept a slow ladder. Prefer configs whose *crossing epoch* is
early, since t\* scales with it. See [[cifar10-outcome-score-saturates]].
