# What was tried, and what was measured

Metric below is **balanced accuracy on `val_public.npz` (2000 images)** — the local split.
It is never trained on. The graded split is untouched apart from the forward pass that
produces the submitted predictions.

## Budget accounting (per submitted run)

- `steps` = optimizer updates, hard-capped at 8000 (`BUDGET_STEPS`).
- `presentations` = augmented training views fed to the network, hard-capped at 1,024,000
  (`BUDGET_PRESENTATIONS`). One augmented view per sample per step; batch 128 x 8000 steps
  hits the cap exactly. The optional BN-recalibration path reserves its forward passes out
  of the same cap before training starts, so the cap can never be exceeded.
- Both counters are printed in the run JSON. Test-time augmentation is inference only.

## Runs (all 8000 steps / 1,024,000 presentations, single training run each)

| tag | change vs baseline | val single view | val all-views | val best combo | wall clock |
|---|---|---|---|---|---|
| `A_base` | baseline (res 48, full TrivialAugment) | 0.7294 | 0.7328 | 0.7372 | (shared GPU) |
| `B_se_ema` | + squeeze-excite, EMA weights + BN recal | 0.7164 | 0.7296 | 0.7318 | (shared GPU) |
| `E_taprob50` | TrivialAugment applied to 50% of samples | 0.7331 | 0.7349 | 0.7354 | 350 s |
| `F_res56` | + progressive resolution 32 -> 56 | 0.7320 | 0.7415 | 0.7468 | 369 s |
| `G_final` | F + a 4th shift-TTA view, writes submission | 0.7408 | 0.7406 | 0.7443 | 382 s |

`G_final` is the submitted run: 8000 steps, 1,024,000 presentations, 382.3 s wall clock
(340.1 s training + 42.1 s inference), predictions from the val-selected combination
`raw+rawS` with tau=1.0. `F_res56` (same config, one fewer TTA view) measured 0.7468 with
its own val-selected combination; the bootstrap sd on this 2000-image split is 0.0095, so
the two are indistinguishable and the run with the cleaner provenance was kept.

A and B shared one GPU, so their wall clock is not a standalone measurement and their
absolute numbers are slightly depressed by different cuDNN kernel selection; they are
comparable to each other, not to E/F/G. E, F and G ran alone.

Findings:

- **Squeeze-excite + EMA hurt** (-0.5%). Dropped.
- **Softer augmentation helped slightly** (+0.4% single view). TrivialAugment-Wide
  magnitudes are tuned for 200+ epoch schedules; this budget is ~51 epochs.
- **Progressive resolution was the real win**: 35% of steps at 32x32, then 56x56, at the
  same wall clock as training everything at 48x48 (+0.66% all-views). Shift-TTA gains grow
  with resolution (shift 2: 0.7413, shift 3: 0.7468), which is why a 4th shift view was
  added for the final run.
- Earlier iterations had already ruled out CutMix (-1%) and wd 2e-3 (neutral).

Run-to-run / TTA numerical noise on 2000 val images is about +-0.5%, so only the
resolution change is larger than the noise floor; the others are directional.

## Inference

The (weight-set x TTA-view) combination and the prior-correction exponent tau are chosen
on `val_public` only, then applied unchanged to the graded images. Training-set class
counts are not uniform (166-231 per class) while the metric is *balanced* accuracy, so
dividing the averaged probabilities by `prior**tau` is a defensible correction; tau is
selected on val alongside the view combination.
