---
name: noise-floor-across-processes
description: "With cudnn.benchmark on a shared GPU, seed-to-seed variance inside one process understates the real noise floor by ~2x"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 67707c3d-342e-4e24-94a8-e0f10d5ca4a1
  modified: 2026-08-18T12:53:00.324Z
---

When measuring a hyper-parameter noise floor on GPU, vary the **process**, not
just the seed. Same config + same seed reproduced bit-exactly within a process
but differed by 0.14–0.24 pp across processes.

**Why:** with `cudnn.benchmark=True` the autotuner picks convolution algorithms
based on free GPU memory at startup. On a shared card, a different co-tenant
footprint selects a different algorithm, changing the floating-point reduction
order and the whole trajectory. In the CIFAR-10 speedrun this made a within-
process sigma of 0.101 pp look authoritative when the true cross-process sigma
was 0.17–0.20 pp — and a `momentum=0.80` candidate cleared a Bonferroni-
corrected bar at +0.255 pp, then replicated at **-0.053 pp** over 9 runs.

**How to apply:** budget one run per process for noise estimation (or set
`cudnn.benchmark=False` and accept the slowdown), and always replicate a
"winner" on fresh runs before shipping it. A bar built from an underestimated
sigma manufactures false positives at exactly the effect sizes worth chasing.
See [[shared-gpu-check-tenancy-first]].
