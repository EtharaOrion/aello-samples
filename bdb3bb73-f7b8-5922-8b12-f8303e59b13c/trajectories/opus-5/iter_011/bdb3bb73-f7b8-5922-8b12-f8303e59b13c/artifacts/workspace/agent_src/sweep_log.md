# Experiment log

Every number below is top-1 accuracy on a **fixed 5 000-image slice held out of
the CIFAR-10 train split** (seeded permutation, identical across all runs; the
held-out images are excluded from training in every sweep run). The test split
was not read during any of this. `n` = number of repeats; where `n>1` the
figure is the mean and the individual runs are listed.

All sweep runs used `--holdout 5000 --no-ckpt`, so they trained on 45 000
images — about 0.5 pp below what the same config reaches with the full 50 000.

## 0. Noise floor

Re-running an **identical config and seed** (`lr 2.4, 40 ep, seed 0`) gave
**94.60%** and **94.28%** in two separate invocations. Nothing differs but
cudnn's autotuning, which picks different convolution algorithms run to run, so
the arithmetic differs slightly and the trajectories diverge.

Seed-to-seed spread within one config was similar (lr 0.6: 94.84 / 94.66 /
94.60; lr 2.4: 94.28 / 94.32 / 94.38).

**Working noise floor: ~0.3 pp.** Single runs were used only to *reject*
clearly bad settings; every accepted decision was confirmed with 2–3 repeats.
A 0.2 pp single-run "improvement" was treated as no evidence at all.

## 1. First pass — and why it was thrown away

| sweep | result |
| --- | --- |
| lr ∈ {0.3, 0.6, 1.2, 2.4} @ 26 ep | 92.98 / 93.22 / 93.04 / 92.98 — flat |
| wd ∈ {2e-4, 5e-4, 1e-3, 1.5e-3, 5e-3, 1.5e-2, 4e-2} @ 26 ep | 93.04 / 93.22 / 93.12 / 92.88 / 91.92 / 88.18 / 81.06 — monotone decline |
| epochs ∈ {26, 40, 60} | 93.22 / 93.56 / 93.42 — saturated by 40 |
| dirac init / BN changes @ 40 ep | 93.28 (neither) / 93.68 (dirac) / 93.38 (BN) / 93.64 (both) |

LR had almost no effect, weight decay only ever hurt, and the model plateaued
at ~93.5% however long it trained. A model whose loss is insensitive to its
step size is usually badly conditioned rather than well-tuned, which pointed at
the input stage.

**The bug.** `zca_patch_filters` built the filter matrix as
`evecs.T / sqrt(evals + eps)`. Row *i* of `evecs.T` is eigenvector *i* and
needs scaling by `1/sqrt(λ_i)`, but that expression broadcasts the divisor
across the trailing axis, dividing *column j* by `sqrt(λ_j)` — a different
matrix. The whitening conv was therefore not whitening. Fix:
`evecs.T / sqrt(evals + eps)[:, None]`.

| | 26 ep | 40 ep |
| --- | --- | --- |
| broken | 93.22 | 93.56 |
| fixed | **94.54** | **94.86** |

Every sweep above was re-run afterwards, since a first layer that changes scale
invalidates the LR and weight-decay optima.

## 2. Learning rate, after the fix

Single-run screen @ 40 ep: 0.15 → 93.68, 0.3 → 94.28, 0.6 → 94.86,
1.2 → 94.30, 2.4 → 94.60. Spread is at the noise floor, so the top two were
repeated 3×:

| lr | runs | mean |
| --- | --- | --- |
| **0.6** | 94.84 / 94.66 / 94.60 | **94.70** |
| 2.4 | 94.28 / 94.32 / 94.38 | 94.33 |

All three lr=0.6 runs beat all three lr=2.4 runs, so the 0.37 pp gap is real
despite each individual comparison being marginal. **lr = 0.6.**

## 3. Cutout — the one large win

@ 40 ep, lr 0.6, wd 5e-4, n=2 unless noted:

| cutout | runs | mean |
| --- | --- | --- |
| 0 | 94.84 / 94.66 / 94.60 (n=3) | 94.70 |
| 8 | 94.76 / 94.70 | 94.73 |
| 12 | 95.02 / 94.82 | 94.92 |
| **16** | **95.46 / 95.26** | **95.36** |
| 20 | 95.04 / 95.16 | 95.10 |
| 24 | 94.84 (n=1) | 94.84 |

+0.66 pp over no cutout, twice the noise floor, with a clean unimodal peak at
16 px either side. **cutout = 16.**

## 4. Schedule length

| config | runs | mean |
| --- | --- | --- |
| 40 ep, cutout 16 | 95.46 / 95.26 | **95.36** |
| 56 ep, cutout 16 | 95.42 / 95.36 | 95.39 |
| 56 ep, cutout 12 | 94.94 / 95.06 | 95.00 |
| 56 ep, no cutout | 94.68 / 94.66 | 94.67 |

56 epochs is indistinguishable from 40 (0.03 pp, an order of magnitude inside
the noise floor) while costing 40% more wall-clock. **40 epochs.** Cutout
does not need a longer schedule to pay off here.

## 5. Promoted configuration

```
epochs 40 · bs 500 · lr 0.6 (linear warm-up 20%, linear decay to 2%)
SGD nesterov, momentum 0.85 · wd 5e-4 on weights only · label smoothing 0.2
augment: ±3 px translate, horizontal mirror, 16 px cutout
widths 128/384/512 · dirac init · BN momentum 0.4, scale frozen · 6-way TTA
```

Chosen entirely from the holdout numbers above, then trained once on the full
50 000-image train split with seed 0 and scored through
`evaluate_ladder.py`, which replays the grading path exactly (test PNGs →
`ToTensor` → `Normalize` → `torch.jit.load(...).eval()` → batches of 512 →
`argmax(1)`):

| | v1 (pre-cutout, promoted early as a safe artefact) | **v2 (promoted)** |
| --- | --- | --- |
| final test acc | 95.02% | **95.50%** |
| first epoch ≥ 94% | 33 (94.14%) | **35 (94.50%)** |
| reconstructed t\* | 15.85 s | **17.03 s** |
| ladder span / total wall-clock | 18.73 s / 19.23 s | 18.98 s / 19.48 s |

v2 wins on accuracy and on crossing margin (94.50% vs 94.14% at the crossing
epoch). Both are far inside the 28 s reference anchor, so the extra 1.2 s of
t\* costs nothing. v1 stayed on disk as the graded ladder until v2 had been
built and verified, and v2 replaced it with a directory rename so each
checkpoint kept the modification time its own `save()` wrote.

## 6. Robustness of the promoted run

The promoted config was re-trained end to end on the full 50 000-image train
split with two additional seeds, and the final checkpoint of each was scored
through the same grading replica:

| seed | final test acc |
| --- | --- |
| **0 (promoted)** | **95.50%** |
| 1 | 95.47% |
| 2 | 95.16% |

Worst case is 1.16 pp clear of the 94% threshold, so the crossing does not
depend on a lucky run. These two runs were scored and then deleted — seed 0
was fixed before they were launched and remains the promoted ladder, so the
test split was not used to select among them.

## 7. Things tried that did not survive

* **Higher weight decay** (up to 4e-2): monotonically worse, both before and
  after the whitening fix. This architecture regularises through augmentation,
  not decay.
* **Longer schedules** (56, 60 epochs): flat past 40 epochs with or without
  cutout.
* **dirac init and the BN changes in isolation**: +0.4 pp and +0.1 pp
  respectively, both inside the noise floor on their own. Kept anyway — they
  are free, and the direction was consistent across the pre- and post-fix
  ablations — but they are not claimed as wins.
* **Very large cutout** (24 px): starts to destroy the signal, −0.5 pp vs 16.
