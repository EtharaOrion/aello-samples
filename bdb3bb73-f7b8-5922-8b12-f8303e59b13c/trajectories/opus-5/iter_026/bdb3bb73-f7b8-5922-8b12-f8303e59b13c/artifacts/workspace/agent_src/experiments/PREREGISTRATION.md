# Pre-registered tuning protocol (written before any sweep run)

## Non-negotiables
* The CIFAR-10 **test split is not read at all** during tuning. Every tuning
  number below is holdout accuracy computed from the *train* split.
* Test-split numbers are produced exactly once per promoted ladder, by
  `verify_ladder.py`, as the correctness check the task asks for.

## Split
A fixed, deterministic permutation of the 50,000 train images (torch CPU
generator, seed 1234567). The last 5,000 of that permutation are the holdout;
the model trains on the remaining 45,000. Identical for every run, so all
comparisons are paired on the same data.

## Metric
Holdout top-1 using the same 6-way TTA that is baked into the shipped
checkpoint, so the tuning metric measures the artefact that is actually graded.

## Screening budget
18 epochs for every screening run (baseline ladder is 22 epochs on 50k; 18 on
45k is the matched compute point).

## Stage 0 -- noise floor
Baseline config x 6 seeds. Report mean, sd across seeds (`sigma_1`) and the
standard error of a k-seed mean (`sigma_1/sqrt(k)`). cuDNN autotuning and the
augmentation RNG both contribute; this number sets the bar everything else
must clear.

## Stage 1 -- one-factor-at-a-time screen (1 seed = 0 per cell)
| factor | levels |
|---|---|
| lr | 8.0, **11.5**, 16.0, 22.0 |
| weight_decay | 0.0100, **0.0153**, 0.0230 |
| momentum | 0.80, **0.85**, 0.90 |
| label_smoothing | 0.10, **0.20**, 0.30 |
| bias_scaler | 32, **64**, 128 |
| bn_momentum | 0.4, **0.6**, 0.8 |
| peak_frac | 0.15, **0.23**, 0.35 |
| lr_end | 0.02, **0.07**, 0.15 |
| translate pad | **2**, 3 |
| widths | 96/320/448, **128/384/512**, 160/448/640 |
| epochs | 14, **18**, 22 |
(bold = baseline level)

**Decision rule, fixed in advance:** a level survives only if its single-seed
holdout accuracy beats the *stage-0 baseline mean* by more than `1.0 * sigma_1`.
A single seed cannot establish an effect; surviving only buys a ticket to
stage 2. `epochs` is scored separately because it trades directly against t*.

## Stage 2 -- multi-seed confirmation
Compose at most 4 candidate configs from the surviving levels. Run each, plus
the baseline, for 4 seeds (0-3). Promote the candidate with the highest 4-seed
mean, and only if `mean(cand) - mean(base) > 2 * sqrt(2) * sigma_1 / sqrt(4)`
(a ~2-sigma paired-difference bar). If nothing clears it, keep the baseline --
"no change" is an acceptable outcome.

## Stage 3 -- promotion
Retrain the winner on all 50,000 images. Pick the epoch count E so that the
expected first crossing of 94% has >= 3 epochs of slack before E and the
projected `t* = E * mean_epoch_sec` stays under 20 s (the outcome score
saturates at 28 s, so extra speed below that buys nothing and margin is worth
more than milliseconds). Verify the new ladder with `verify_ladder.py`, and
only then swap it in for the graded ladder.
