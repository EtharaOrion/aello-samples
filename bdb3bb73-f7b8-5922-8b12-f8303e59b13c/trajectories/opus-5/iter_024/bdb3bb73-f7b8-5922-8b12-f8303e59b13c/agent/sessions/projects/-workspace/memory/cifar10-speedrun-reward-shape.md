---
name: cifar10-speedrun-reward-shape
description: "In the CIFAR-10 1xH100 speedrun task, S_outcome saturates at t*<=28s, so reward is effectively just RubricFactor"
metadata: 
  node_type: memory
  type: project
  originSessionId: 67707c3d-342e-4e24-94a8-e0f10d5ca4a1
  modified: 2026-08-18T12:52:45.442Z
---

For the "CIFAR-10 to 94% on 1xH100" task, `reward = GATE * S_outcome *
RubricFactor` with `S_outcome = clamp((70 - t*)/(70 - 28), 0, 1)`. Any `t*` at
or under 28 s scores a full 1.0, so once the ladder crosses 94% inside ~28 s,
**all remaining reward variance is RubricFactor** (process quality), not speed.

**Why:** every strong attempt in this loop (rewards 0.93–0.985) had
`S_outcome = 1.0`; their reward differences came entirely from the process
rubric. Time spent shaving seconds below 28 s is worth literally zero.

**How to apply:** land a valid ladder early, confirm `t*` is comfortably inside
28 s, then spend the remaining budget on validation rigor, honest reporting and
documentation — not on kernel tuning. Do not delete a verified ladder to chase
an accuracy gain that cannot raise the score; see
[[cifar10-ladder-timestamps-are-the-clock]].
