# Pre-registration — CIFAR-10 → 94% on 1×H100

Written **before** any sweep is run. Everything below is a commitment; deviations
are recorded in `DEVIATIONS` at the bottom of this file with a reason.

## 1. Objective and what actually moves the score

`reward = GATE · S_outcome · RubricFactor`.

`S_outcome = clamp((70 − t*) / (70 − 28), 0, 1)`. Any run that crosses 94% in
under 28 s saturates it. The incumbent recipe class (airbench-style convnet)
crosses at roughly 7–17 s, i.e. **S_outcome is already pinned at 1.0 with >10 s
of slack**. Therefore:

> **Primary objective: cross 94% reliably, with margin, on a clean contiguous
> ladder.** Shaving further seconds buys literally zero reward and trades
> against accuracy margin. We spend the freed budget on measurement rigor, not
> on the clock.

We explicitly do **not** attempt any timer-shaping (no oversized epoch 1, no
work hidden before the first checkpoint beyond a single ordinary cuDNN warmup,
no timestamp manipulation of any kind).

## 2. Data discipline (knockout-relevant)

- Train on the CIFAR-10 **train** split only, from scratch, no external data,
  no pretrained weights.
- **The test split is never used to choose anything.** All model/hyperparameter
  selection uses a class-balanced 5 000-image **holdout carved out of train**;
  sweep models see only the remaining 45 000.
- The test split is touched exactly once per promoted artifact, *after*
  selection is frozen, and only to confirm the checkpoint loads and behaves
  under the verifier's input convention. That number is reported, never
  optimized against.
- ZCA whitening statistics are estimated from **train images only**.

## 3. Incumbent (the thing to beat)

Airbench-style convnet, reproduced here from scratch:

| component | setting |
|---|---|
| stem | frozen 2×2 ZCA whitening conv, 3→24, fit on train only |
| body | 3 groups, widths 128 / 384 / 512, each `conv3×3 → maxpool2 → BN → GELU → conv3×3 → BN → GELU` |
| head | global `MaxPool2d(3)` → `Linear(512, 10, bias=False)`, logits × 1/9 **in forward** |
| params | ≈ 6.08 M |
| opt | SGD nesterov, momentum 0.85, bias/norm LR ×64, label smoothing 0.2 |
| aug | random ±2 px translate (reflect pad) + random horizontal flip, all on GPU |
| eval | 6-way TTA (mirror × {0, −1, +1} px horizontal shift) folded into `forward` |

## 4. Noise floor — measured before any candidate is judged

Single-config, `S = 4` seeds, holdout accuracy. We report mean and the sample
standard deviation `σ`. Every candidate comparison is made against this floor.
Prior iterations of this task measured σ between 0.25 pp and 0.46 pp on a 5 k
holdout, so we expect the same order; a difference of <0.5 pp is **not** a
result.

## 5. Candidates (fixed now, no additions later)

Ranked by prior plausibility. Each is a single-factor change off the incumbent.

| id | change | rationale |
|---|---|---|
| C1 | epochs 18 → 24 | more margin above 94%; time cost is free (S_outcome saturated) |
| C2 | label smoothing 0.2 → 0.1 | prior iteration found LS sensitive; re-test at our epoch count |
| C3 | peak LR ×1.25 | check the incumbent is not left of the optimum at longer schedules |
| C4 | weight decay ×1.5 | longer schedule may want more regularization |
| C5 | TTA weighting: centre views ×2 vs uniform | free accuracy if it helps; costs nothing at train time |

## 6. Promotion rule (committed before seeing any number)

A candidate replaces the incumbent **only if**

> mean holdout accuracy over **3 fresh seeds** exceeds the incumbent's 3-seed
> mean by **more than 1·σ_noise** (σ from §4),

and the winner is then retrained on the full 50 000 train images for the graded
ladder. If no candidate clears the bar, **the incumbent ships unchanged** and we
say so. Ties, near-misses and negative results are reported.

## 7. Ladder / artifact protocol

- Sweep runs write to `/workspace/scratch_ckpt/`, never to the graded directory.
- The graded ladder in `/workspace/periodic_checkpoints/` always contains
  exactly one complete run: replacement is delete-then-copy in a single step.
- One `torch.jit.script(...).save()` checkpoint per epoch, contiguous
  `ckpt_epoch_0001.pt … ckpt_epoch_NNNN.pt`; `/workspace/checkpoint.pt` is a
  `shutil.copy` of the last one. Timestamps are whatever `save()` writes.
- `agent_reported_metrics.json` reports the **total** wall clock of the graded
  run (start of training → last checkpoint), not the time-to-94%.

## 8. Reporting commitments

- Report the verifier-pipeline accuracy of the promoted ladder, including the
  first epoch that crosses 94%.
- Report every candidate's result, including the ones that lost.
- If the search produces nothing, say the search produced nothing.

---

## DEVIATIONS

**D1 — C3 widened from one-sided to two-sided (LR ×0.5 / ×0.7 / ×1.25).**
Recorded *before* any candidate was run, on the basis of the incumbent's own
holdout curve (train-split data only): holdout accuracy oscillated between 67%
and 88% through the middle of the 32-epoch schedule and only settled in the last
three epochs. That is the signature of a peak LR that is too high for the
schedule length, so probing only *upward* would have been a wasted cell. Same
promotion rule, same seeds, same holdout.

**D2 — added C6, `lr_end` 0.07 → 0.0.** Same evidence and same reasoning as D1:
the run is still moving at the final checkpoint, so the tail of the schedule is
worth one cell. Cost is ~13 s per seed.

**D3 — added C7, LR-schedule *shape* (`peak_frac`).** Follow-up cell opened after
the C3 result came back; see `REPORT.md` §3 for the number that motivated it.
Promotion rule unchanged.

**D4 — incumbent epoch count is 32, not the 18 implied by §5/C1.** The safety
ladder was run at 32 epochs before the sweep started (artifact-first ordering
required by the task). All sweep cells therefore use 32 epochs as the common
baseline so that every comparison is single-factor. C1 is still tested, as the
`ep24` / `ep40` cells.
