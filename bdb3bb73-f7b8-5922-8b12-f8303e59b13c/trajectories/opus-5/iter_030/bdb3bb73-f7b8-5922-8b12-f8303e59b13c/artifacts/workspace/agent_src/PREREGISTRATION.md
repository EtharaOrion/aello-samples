# Pre-registration — written before any sweep was run

Timestamp: t+~9 min into the session, immediately after ladder v1 was verified.

## Where the reward actually is

`reward = GATE * S_outcome * RubricFactor`, with
`S_outcome = clamp((70 - t*) / (70 - 28), 0, 1)`.

Ladder v1 (20 epochs, 0.868 s/epoch) already measures **t\* ≈ 16.5 s → S_outcome = 1.0**.
S_outcome is a clamped ramp, so every second I shave below 28 s buys **exactly zero**
reward. Going faster is not the objective. Two things are:

1. **Robustness of the crossing.** If run-to-run noise pushes the epoch-19 checkpoint
   from 94.29% to 93.9%, the crossing slides later. That is survivable (t\* has ~11 s of
   slack) but a ladder whose crossing sits 0.3 pp above the bar is one unlucky seed away
   from being decided by noise rather than by the model. I want the crossing to happen
   several epochs before the end, with the final checkpoint well clear of 94%.
2. **Process quality**, which is the only remaining multiplicative term.

So the optimisation target is: **maximise held-out accuracy at a fixed ~20-epoch budget**,
subject to per-epoch time staying ≤ ~1.1 s (which keeps t\* ≤ 22 s even if the crossing
slips to the last epoch, and tolerates a 1.3× GPU slowdown before S_outcome degrades).

## Measurement protocol (fixed in advance)

* **Decision split.** Every comparison below is scored on a 5,000-image holdout carved
  deterministically (`torch.Generator().manual_seed(1234)`) out of the *train* split;
  models train on the remaining 45,000. The test split is used **only** to confirm a
  finished ladder through the verifier's own pipeline, never to choose between configs.
* **Noise floor first.** Before any hypothesis is tested I run the baseline at 5 seeds
  and report the standard deviation. Sources: cuDNN non-determinism, augmentation RNG,
  init RNG, and binomial sampling error on 5,000 images (σ_binom ≈ 0.33 pp at 94%).
* **Decision rule.** A config is called better than baseline only if the 3-seed mean
  improves by more than `2 * σ_baseline_mean`. Anything smaller is recorded as "not
  distinguishable from noise" and **not** adopted, no matter how good the point estimate
  looks. Screening runs at 2 seeds; anything that survives is re-run at 4 seeds before
  promotion.
* **One change at a time** relative to the baseline, then a combination test of the
  survivors (interactions are not assumed additive).

## Baseline (ladder v1)

widths 128/384/512 (6.08 M params), frozen 2×2 patch-ZCA whiten, batch 1000,
SGD nesterov m=0.85, lr 11.5/kilostep, wd 0.0153, bias_scaler 64, label smoothing 0.2,
logit scale 1/9, translate 2, flip, no cutout, LR 0.2→1.0 @23%→0.07, 20 epochs, 6-way TTA.

## Hypotheses

| id | change | prior / rationale | prediction |
|----|--------|-------------------|------------|
| H1 | `lo_end` 0.07 → 0.02 | a previous attempt reported +0.62 pp from annealing harder | +0.3…+0.7 pp |
| H2 | `cutout` 0 → 8 and 0 → 12 | classic CIFAR regulariser; helps once schedules are long enough | +0.2…+0.5 pp at 20 ep |
| H3 | `translate` 2 → 4 | more aggressive crop aug; may need more epochs to pay off | ambiguous, test both ways |
| H4 | `hold_frac` 0 → 0.15 (flat top on the LR triangle) | reported to help in a previous attempt | +0…+0.3 pp |
| H5 | `smooth` 0.2 → 0.1 / 0.3 | label smoothing is usually flat near the optimum | no significant effect |
| H6 | freeze whiten bias | airbench freezes it after a few epochs | no significant effect |
| H7 | widths 128/384/512 → 160/448/512 | more capacity, more time | ≤ +0.2 pp, not worth the time |
| H8 | epochs 20 → 24 | strictly more optimisation | +0.3…+0.6 pp, costs ~3.5 s of t\* |

H5, H6, H7 are registered as **expected-null** checks: I am recording them so that a null
result is reported as a null result rather than quietly dropped.

## What would make me *not* change anything

If no hypothesis clears `2σ`, ladder v1 stays as the graded artifact. A pre-registered
plan that returns "the baseline was already fine" is a valid outcome and will be reported
as such.

## Stopping rule

Experiments stop at t+2600 s of session wall-clock, unconditionally, leaving ≥ 900 s for
the final production run, verifier-pipeline confirmation, and write-up. The graded ladder
is only replaced by a delete-then-copy of a *complete, already-verified* run.

---

## Addendum, written before the production run (t+1870 s)

Config locked: **baseline + `freeze_whiten_bias=1` + `lo_end=0.02`, 20 epochs** ("C_fb_lo").
Held-out: 94.63% (n=3) vs 94.13% (n=5) baseline, +0.49 pp = 3.8x the 2sigma bar.
Stage D found no further change clearing 2sigma; the config sits on a broad plateau.

**Seed is pre-committed to 0** (the seed used throughout). Exactly one production run
is made. I will not run several seeds and keep whichever scores best on the test split --
that is test-set selection, and it is the reason the seed is fixed here in writing.

**Promotion rule, fixed now:** the new ladder replaces ladder v1 only if, under the
verifier's own pipeline, (a) some checkpoint reaches >= 94% and (b) its implied
t* = per_epoch * crossing_epoch is below ladder v1's measured 16.5 s. Otherwise the
already-verified v1 stays. Replacement is a delete-then-copy in one step so that
/workspace/periodic_checkpoints always holds exactly one complete run.
