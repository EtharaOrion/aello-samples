---
name: shared-gpu-check-tenancy-first
description: The /workspace H100 is shared with another tenant and runs ~4x below spec; microbenchmark before planning any time budget
metadata: 
  node_type: memory
  type: project
  originSessionId: 67707c3d-342e-4e24-94a8-e0f10d5ca4a1
  modified: 2026-08-18T12:53:05.847Z
---

The H100 80GB in this environment is **not** exclusive. Measured 2026-08-18:
221 TFLOPS fp16 matmul (spec ~700–990) and 764 GB/s HBM copy (spec ~3350).
Another tenant held 57–76 GB of the 80 GB card and drove it to 97–100%
utilisation; six consecutive 144 MB allocations failed while this process held
only 3.6 GB.

**Why:** it makes absolute timings from other sessions non-comparable, and it
silently breaks long sweeps with OOM. A CIFAR-10 epoch that reportedly takes
0.4 s on a quiet card took 1.3 s here, and epoch times drifted 35% between
sweep rounds with no code change.

**How to apply:** run a 30-second matmul/bandwidth microbenchmark before
committing to a time budget, and branch the plan on the measured number rather
than the sticker spec. Add OOM retry loops around long sweeps, free
intermediate GPU tensors aggressively (`del` + `empty_cache()`), and prefer
plans that keep a verified artefact on disk while the card is contended.
