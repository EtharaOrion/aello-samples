---
name: cifar10-h100-94pct-recipe
description: "Working airbench-style CIFAR-10 recipe reaching 94.9% in ~14s on one H100, plus the two hyperparameters that were mis-set and cost ~1pp each"
metadata: 
  node_type: memory
  type: project
  originSessionId: dae12940-c072-47c7-aeee-8fb4801424ac
  modified: 2026-08-18T07:01:04.454Z
---

Verified 2026-08-18. 6.08M-param convnet: frozen 2×2 ZCA whiten conv (3→24) → GELU →
three `conv/maxpool/BN/GELU + conv/BN/GELU` groups (128/384/512) → MaxPool(3) → bias-free
head. SGD+Nesterov m=0.85, summed CE + label smoothing 0.2, triangular LR, 64× LR on
BN/whiten biases, bs=1024. **32 epochs = 12.3s, 94.88% test** with 6-way TTA in `forward`.

Two settings carried from the airbench reference were wrong for a longer schedule, each
worth ~1pp — larger than any architecture change tried:
- **weight decay is applied per step**, so 3× the epochs applies 3× the total decay.
  0.0306→0.0153→0.0051 gave 92.9→93.9→94.2%. Use ~0.005 at 32 epochs.
- **logit scale**: 1/9 → 1/32 was +1.5pp at 12 epochs. Plateau is broad (1/128…1/32).

Cutout 12–16px is the only augmentation lever that beat the noise floor (+0.5pp). Weight
EMA *hurt* (−1.3pp at decay 0.95). Dirac conv init, `lr_end`, pad=3, bn_momentum,
whiten_bias_epochs, label smoothing 0.1: all within noise. Epochs plateau hard past 32.

Noise floor matters more than any single result: 5 seeds on a 5k val give sd=0.35pp, which
is ≈ the binomial sd of a 5k set — so 2-seed screens only resolve effects ≳0.5pp. Fit ZCA
over *all* training images in chunks; the PNG cache is class-sorted so a prefix sample is
all one class. See [[cifar10-verifier-ladder-gotchas]].
