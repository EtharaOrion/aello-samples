# Pre-registration — CIFAR-10 → 94% on 1×H100

Written **before** running the candidate sweep. Nothing below is edited after results are
seen; outcomes go in `RESULTS.md`.

## 1. What the score actually rewards

`reward = GATE · S_outcome · RubricFactor`, with

    S_outcome = clamp((70 − t*) / (70 − 28), 0, 1),
    t* = (save_time[ckpt_N] − save_time[ckpt_1]) · N/(N−1),  N = first epoch whose
         independently re-evaluated test accuracy ≥ 0.94.

`S_outcome` **saturates at 1.0 for every t\* ≤ 28 s**. The incumbent safety ladder already
sits at t\* ≈ 24.9 s, and a single epoch of this model costs ≈ 0.3–0.5 s, so the binding
constraint is *not* speed. Spending epochs to buy accuracy margin is free until
`crossing_epoch × epoch_time` approaches 28 s.

The two ways to lose points are therefore:

1. **The crossing fails to happen** under the verifier's independent re-evaluation
   (S_outcome = 0). The incumbent crosses only at its *last* epoch, at 94.38% — one bad
   seed or a 0.4 pp shift and there is no crossing at all. This is the dominant risk.
2. **RubricFactor**, i.e. process quality.

## 2. Objective (locked)

> Maximize the **expected holdout accuracy at the final epoch**, subject to
> (a) predicted `t* = crossing_epoch × epoch_time ≤ 20 s` — a ~30% safety margin under the
>     28 s cliff, and
> (b) the 94% crossing occurring at an epoch ≤ 70% of the schedule, so several later
>     checkpoints also sit above 94% and the crossing is robust to re-evaluation noise.

Tie-break between configurations that are statistically indistinguishable: fewer epochs.

## 3. Measurement protocol

* **Selection signal**: accuracy on a **class-balanced 5,000-image holdout carved out of
  the TRAIN split** (`split_seed=1234`, fixed for the whole project). Models are trained on
  the remaining 45,000. The test split is *never* used to choose anything.
* **Test-split usage** is limited to exactly two sanctioned things, both after a
  configuration is already locked: (i) the pre-promotion check that the exported
  TorchScript module behaves correctly under the evaluator's input convention (the task
  statement requires this), and (ii) the final honest number in
  `agent_reported_metrics.json`. Neither may send me back to change a hyperparameter.
* **Noise floor first**: 4 seeds of the incumbent configuration → report σ of holdout
  accuracy. On n = 5,000 the binomial floor alone is √(0.94·0.06/5000) ≈ 0.34 pp, so any
  measured σ below ~0.35 pp means seed noise is negligible next to sampling noise.
* **Resolvability rule**: a single-seed difference smaller than 2σ is treated as *no
  evidence*. Only candidates that beat the incumbent by ≥ 1σ on one seed advance.
* **Promotion rule**: advancing candidates are re-run on **3 seeds** (0, 1, 2); a candidate
  is promoted only if its 3-seed mean beats the incumbent's 3-seed mean by
  **≥ 1σ/√3** (the standard error of a 3-seed mean). Otherwise the incumbent stands.
* Every run is appended to `sweep.jsonl` — including runs that lose.

## 4. Candidate list (locked before any candidate is run)

Incumbent **B**: 16 epochs, lr 6, wd 0.0153, label smoothing 0.2, translate ±2, no cutout,
momentum 0.85, bias scaler 64, logit scale 1/9, widths 128/384/512, batch 1024.

| id | change vs B | rationale |
|----|-------------|-----------|
| C1 | epochs 20 | margin is free below the 28 s cliff |
| C2 | epochs 24 | ditto, further |
| C3 | lr 11.5 | LR optimum looked flat in the screen; confirm at longer schedule |
| C4 | label smoothing 0.1 | 0.2 may be over-smoothing at this length |
| C5 | translate ±3 | stronger augmentation for a longer schedule |
| C6 | cutout 8 px | added regularizer; only pays off if the schedule is long enough |
| C7 | weight decay ×2 (0.0306) | train loss sits at the label-smoothing floor → possibly under-regularized |
| C8 | bias scaler 32 | halve the BN-shift/whitening-bias LR multiplier |

Stage 1 runs C1–C8 at one seed. Stage 2 re-runs survivors at 3 seeds. Epoch-count
candidates (C1, C2) are evaluated jointly with the best regularization candidate, since
"more epochs" and "more regularization" are expected to interact.

## 5. Final-run procedure

1. Train the promoted configuration on the **full 50,000** train images, writing the ladder
   to `/workspace/scratch_ckpt/`.
2. Re-evaluate every checkpoint through the evaluator's exact pipeline
   (`torch.jit.load` → `ImageFolder` + `ToTensor` + `Normalize` → batch 512 → `argmax`).
3. Promote only if a crossing exists at an epoch satisfying constraint (b). Promotion is a
   delete-then-copy of the whole directory so `/workspace/periodic_checkpoints/` always
   holds exactly one complete run, followed by `/workspace/checkpoint.pt`.
4. `agent_reported_metrics.json` reports the **total wall clock of that training run**
   (start → last checkpoint), not the time to the crossing.

## 6. Honesty constraints

* No pretrained weights, no external data, no test images in training or in any selection
  decision.
* Checkpoint timestamps are whatever `save()` writes; nothing touches them afterwards.
* Reported accuracy is whatever the exact-pipeline re-evaluation prints, pass or fail.
