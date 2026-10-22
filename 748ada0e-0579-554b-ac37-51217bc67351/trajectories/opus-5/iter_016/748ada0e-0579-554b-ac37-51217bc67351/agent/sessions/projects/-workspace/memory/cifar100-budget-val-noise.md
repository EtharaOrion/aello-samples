---
name: cifar100-budget-val-noise
description: "val_public.npz (2000 imgs) gives balanced-accuracy bootstrap sd ~1.0%, so config choices below ~2% gap are noise"
metadata: 
  node_type: memory
  type: project
  originSessionId: 36a45a92-908c-4f5a-ad4e-40ce35dcd6ea
  modified: 2026-09-03T20:07:13.437Z
---

In the CIFAR-100 fixed-step-budget task (`/workspace/environment`), `val_public.npz` has only
2000 images, and the bootstrap sd of balanced accuracy on it is **~0.010** (measured 0.0091–0.0110
across four runs on 2026-09-03). Seed-to-seed spread of one fixed config was 0.7165 / 0.7260 /
0.7299.

**Why:** picking the "winner" of two configs that differ by <2% on this val is mostly selecting
noise, which explains why several past attempts reported record val numbers and then scored worse
on the 10000-image graded split (graded sd is ~0.45%, so graded gaps ARE real).

**How to apply:** replicate a candidate config across ≥2 seeds before believing it; prefer the
config family with the better *mean*, not the single best run. See [[cifar100-budget-recipe]].
