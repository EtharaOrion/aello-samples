---
name: cifar10-speedrun-timing-mechanics
description: "Traps in the CIFAR-10 speedrun verifier — mtime-derived clock, holdout optimism, and the script-once/save-per-epoch pattern"
metadata: 
  node_type: memory
  type: project
  originSessionId: ebb0f8f0-dd95-4eb9-b5c3-ae3507fb9c75
  modified: 2026-08-17T22:54:23.490Z
---

Mechanics of the `/workspace` CIFAR-10 speedrun that are not obvious from the spec:

- **The ladder must be trained in place.** t\* is rebuilt from checkpoint mtimes.
  Copying a finished ladder in from a scratch directory stamps all files with
  near-identical mtimes and destroys the timing signal. Delete the old ladder, then
  point `--out-dir` straight at `periodic_checkpoints/`. To keep a restorable
  backup without disturbing mtimes, hardlink (`cp -l`) rather than copy.
- **Skip `torch.compile`.** Its ~15 s warmup sits *before* the epoch-1 save, so it
  shrinks measured t\* without speeding anything up — a free-looking win that is
  really timer gaming. cuDNN autotune inside epoch 1 is fine (the verifier's
  `(N-1)`-interval formula already excludes epoch 1).
- **`torch.jit.script(net)` shares parameter storage with the eager module**, so you
  can script once before training and call `.save()` each epoch to snapshot live
  weights — far cheaper than re-scripting 20 times inside the timed window. Assert it
  by checking no two consecutive checkpoints hold identical weights.
- **Held-out train images read ~0.35 pp optimistic vs the real test split**
  (95.11% holdout vs 94.76% test at the same config). The gap is consistent across
  configs so rankings still transfer, but never treat a holdout number as a test
  prediction — leave real margin above 94%.
- The container's `/dev/shm` is too small for `DataLoader` workers; use
  `num_workers=0` or it dies with a bus error and dumps ~700 MB core files.

**Why:** each of these either silently zeroes the reward or wastes scarce session
time on a rediscovery.

**How to apply:** ladder in place, no compile, script once, aim for ≥94.3% at the
crossing epoch. See [[cifar10-speedrun-scoring-shape]] for where to spend the
time saved.
