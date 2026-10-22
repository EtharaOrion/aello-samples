---
name: cifar100-budget-cutout-win
description: "On the AELLO CIFAR-100 8000-step task, cutout is a +2% win and mixup/flip5-TTA are not"
metadata: 
  node_type: memory
  type: project
  originSessionId: 658eafb4-1478-4fed-94c9-489ed15da7fa
  modified: 2026-09-03T12:58:45.306Z
---

AELLO-C8-S1 (CIFAR-100, 20k images, 8000 steps / 1.024M presentations), measured 2026-09-03
on val_public balanced accuracy with a 133M-param ResNet-34 (width 160, res 48, SGD-nesterov
lr 0.2, wd 2e-3, ls 0.1, cosine):

- **Cutout(12px, applied at 32px before upsampling) is the single biggest lever found:
  raw 0.7058 -> 0.7251, raw+EMA ensemble 0.7137 -> 0.7343.** Far above the ~0.5% run-to-run
  sigma. Nine prior attempts had never tried it; they plateaued near 0.72-0.73.
- **Weight EMA is worth ~+0.9% but only as an ensemble member.** EMA alone is *worse* than
  the final weights (0.7173 vs 0.7251); the average of raw + EMA log-probs beats both
  (0.7343). Costs zero extra presentations, so it is free score.
- **Mixup(alpha 0.2, p 0.5) on top of cutout is a wash** (0.7329 vs 0.7343, same seed).
- **flip5 TTA (flip + 2px shifts) is worse than plain flip TTA** on the good models
  (0.7282 vs 0.7343). Shift TTA hurts; horizontal flip alone is the right choice.

**Why:** at 51 epochs over 20k images a 133M-param net memorises (train loss ~0.9 with
ls 0.1); masking augmentation is the binding constraint, not capacity or schedule.

**How to apply:** start from cutout + raw/EMA ensembling before touching architecture,
width, depth, or LR schedule. See [[cifar100-budget-eval-noise]] and
[[cifar100-marginal-balancing]].
