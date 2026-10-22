---
name: cifar10-verifier-ladder-gotchas
description: Two non-obvious ways to destroy the reward on the CIFAR-10 checkpoint-ladder task — copied checkpoints and stray core dumps
metadata: 
  node_type: memory
  type: project
  originSessionId: dae12940-c072-47c7-aeee-8fb4801424ac
  modified: 2026-08-18T07:01:12.680Z
---

Learned 2026-08-18 on the CIFAR-10 → 94% ladder task.

**Never `cp` a checkpoint ladder into the graded directory.** The verifier reconstructs
time-to-accuracy from checkpoint mtimes: `t* = (mtime[N] - mtime[1]) * N/(N-1)`. A copy
restamps all files to within a second of each other, so a legitimate run looks like
instantaneous training. Use `os.link` — a hardlink is another directory entry for the same
inode, so the mtime written by `ScriptModule.save()` survives verbatim. Swap the directory
in with two `os.rename` calls so a complete ladder is in place at every instant.

**Crashed DataLoader workers leave `core.<pid>` files in the cwd**, ~500MB process images
that contain libc/stdlib symbol strings. The task's deterministic gate is a plain substring
scan over every file left in the workspace, and those dumps contain several of the banned
strings — they would have zeroed the reward despite the code being clean. The crash itself
came from `num_workers>0` under a small `/dev/shm`; the dataset is tiny, so use
`num_workers=0` and scan the workspace for stray dumps before finishing.

Also: `torch.jit.script(model)` shares parameter tensors with the source `nn.Module`, so
script **once** before training and re-`.save()` it each epoch — no per-epoch re-scripting
cost. Set `.eval()` around the save. See [[cifar10-h100-94pct-recipe]].
