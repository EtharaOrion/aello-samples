---
name: workspace-dev-shm-64mb
description: "/dev/shm in the /workspace container is only 64 MB, so PyTorch DataLoader workers crash"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3730e192-6a22-4174-a49f-b64e144d7fc1
  modified: 2026-08-19T09:59:15.044Z
---

In the `/workspace` CIFAR-10 container, `/dev/shm` is capped at 64 MB while the root
overlay has ~550 GB free. Any `torch.utils.data.DataLoader` with `num_workers > 0` dies
during collation with a misleading `No space left on device` followed by
`DataLoader worker ... killed by signal: Bus error`.

**Why:** worker collation allocates shared-memory storage, which lands in the tiny
`/dev/shm`, not on the roomy overlay — so free disk space is no guide.

**How to apply:** use `num_workers=0` and iterate the dataset directly (10k 32×32 PNGs
decode in ~2 s single-process), or pre-decode to `.npy` once and load onto the GPU.
Do not spend time hunting for a disk-space problem; check `df -h /dev/shm` first.
