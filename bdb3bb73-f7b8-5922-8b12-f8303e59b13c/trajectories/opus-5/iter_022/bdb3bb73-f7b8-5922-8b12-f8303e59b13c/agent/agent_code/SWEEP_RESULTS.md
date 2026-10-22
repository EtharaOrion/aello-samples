# Sweep results

Protocol, candidates and promotion rule were fixed in `PREREGISTRATION.md` before any
run. 42 runs total, all on the 5,000-image train-split holdout. The test split was
never fed to a model during training or selection.

Metric `acc3` = mean holdout top-1 over a run's final 3 epochs (6-way TTA).

## Noise floor — the headline caveat

The baseline's 4 seeds gave σ_run = **0.079 pp**, which turned out to be a badly
optimistic estimate off 3 degrees of freedom. Pooling the within-group variance of all
nine ≥3-seed groups (19 df) gave σ_pooled = **0.167 pp**, over twice as large. Every
promotion decision below uses the pooled figure, which raises the bar from 0.121 pp to
0.254 pp and disqualifies three candidates that the round-1 σ would have promoted.

## Round 1 — one factor at a time

Screening at 1 seed, then the top 4 at 3 seeds. The screen is worth reading only as a
demonstration of the winner's curse:

| candidate | screen (1 seed) | confirmed (3 seeds) |
|---|---|---|
| `lr=1.8` | 0.9433 (best but one) | **0.9408** — below baseline |
| `mom=0.80` | 0.9449 (best) | 0.9430 |
| `epochs=24` | 0.9436 | 0.9432 |
| `wd=1.4e-3` | 0.9427 | 0.9429 |

`lr=1.8` screened +0.23 pp above baseline and confirmed at −0.03 pp. Ranking on a single
seed would have shipped it.

Against baseline mean 0.9410 (n=4) with the 0.254 pp bar, **no round-1 candidate was
individually significant**. `epochs=24` (+0.214), `mom=0.80` (+0.194) and `wd=1.4e-3`
(+0.188) all sat just under it. Three near-orthogonal knobs each buying ~+0.2 pp is
what motivated round 2.

Clear losers, for the record: `bs=500` (0.9333, never crossed 94%), `mom=0.90` (0.9361),
`whiten_bias_epochs=0` (0.9381), `label_smoothing=0.1` (0.9390), `wd=2.8e-3` (0.9390).
Freezing the whitening bias from step 0 costs 0.3 pp, so the three warm-up epochs it
gets are doing real work.

## Round 2 — walking past the edges, and stacking

Round 1's two best moves sat at the edge of their swept ranges, so round 2 stepped past
them and tested whether they combine. 3 seeds each, no 1-seed screen.

| candidate | mean `acc3` (n=3) | Δ vs baseline | bar | crossing epochs | verdict |
|---|---|---|---|---|---|
| `mom0.80+lr1.8` | 0.9445 | +0.343 pp | 0.254 | 19, 18, 18 | **promote** |
| `mom0.80+wd1.4e-3` | 0.9439 | +0.283 pp | 0.254 | 18, 17, 18 | **promote** |
| `mom=0.75` | 0.9419 | +0.086 pp | 0.254 | 19, 19, 18 | not significant |
| `lr=2.1` | 0.9409 | −0.010 pp | 0.254 | 19, 19, 18 | not significant |

The interesting result: **neither `lr=1.8` nor `wd=1.4e-3` was significant on its own,
but either one stacked with `momentum=0.80` clears the bar.** Pushing momentum further
down (0.75) or lr further up (2.1) both give the gain back, so 0.80 and 1.8 are near a
genuine optimum rather than on a slope.

## Promotion

The two winners are separated by 0.060 pp against a 0.273 pp two-group bar — they are
statistically indistinguishable, so the pre-registered tie-break applies: prefer the
earlier crossing. `mom0.80+wd1.4e-3` crosses at a mean epoch 17.7 versus 18.3, so it is
the promoted configuration.

*(`round2.py`'s automatic printout picks the higher mean and names `mom0.80+lr1.8`; it
implements only the significance test, not the tie-break clause. The tie-break is
applied here by hand, as registered.)*

**Shipped:** baseline with `momentum=0.80, wd=1.4e-3`, 20 epochs, seed 0.

## Graded run

Single run, config fixed in advance, trained on the full 50,000 train images.

| | |
|---|---|
| first checkpoint ≥ 94% | **epoch 18, 94.49%** |
| final checkpoint (epoch 20) | **94.79%** |
| `t_star` (verifier formula) | **16.43 s** → S_outcome **1.0000** |
| checkpoint span, epochs 1→20 | 17.64 s |
| total run wall clock (reported) | 21.99 s |

The holdout predicted crossing at epochs 18/17/18 while training on 45k; the graded run
on 50k crossed at 18. Holdout accuracy ran ~0.4 pp below test accuracy throughout,
consistent with the 10% smaller training set, and the ranking transferred.

## What I would test next

`momentum=0.80 + lr=1.8 + wd=1.4e-3` — the three-way stack was never run, and the
two two-way stacks both landed above the bar. Left untested rather than shipped
unvalidated.
