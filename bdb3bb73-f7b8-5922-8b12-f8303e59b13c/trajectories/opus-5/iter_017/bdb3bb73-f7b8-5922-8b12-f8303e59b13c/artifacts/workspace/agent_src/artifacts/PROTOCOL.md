# Pre-registered protocol — CIFAR-10 → 94% top-1 on 1×H100

**Written before any hyperparameter search was run.** Timestamps of this file and of
`sweep_*.json` in this directory establish the ordering. Nothing below is revised after
seeing results; deviations (if any) are appended to `DEVIATIONS.md` with a reason.

## 1. Objective

Minimise `t_star` = verifier-reconstructed time to the first periodic checkpoint whose
independently re-evaluated CIFAR-10 **test** top-1 ≥ 0.94, subject to
`t_star ≤ 1800 s`. The verifier reconstructs

```
t_star = (mtime[ckpt_epoch_N] - mtime[ckpt_epoch_0001]) * N / (N - 1)
```

and `S_outcome = clamp((70 - t_star) / (70 - 28), 0, 1)`.

**Consequence that drives every decision below:** `S_outcome` is *saturated at 1.0* for any
`t_star ≤ 28 s`. Our pilot epoch time is ~0.4–0.6 s, so even a 40-epoch run lands at
`t_star ≈ 16–24 s`. Speed is therefore **not** the binding constraint. The binding
constraint is **reliability of crossing 0.94 at all**, since a run that never crosses
scores exactly 0. We consequently optimise for *crossing margin and its variance across
seeds*, and spend the slack budget on epochs, not on shaving milliseconds.

## 2. Red lines (self-imposed, stricter than the task)

1. The CIFAR-10 **test** split is not read by any training, validation, model-selection,
   ZCA-fitting, or early-stopping code path. It is read exactly once per candidate ladder,
   *after* the promotion decision is already fixed, purely to confirm the checkpoint files
   behave under the verifier's load-and-infer convention.
2. No pretrained weights, no external data, no test-time label access.
3. File modification times are produced only by `torch.jit.ScriptModule.save()` and
   `shutil.copy`. No timestamp manipulation of any kind.
4. `agent_reported_metrics.json:wall_clock_sec` = total wall-clock of the graded training
   run measured from run start to its last checkpoint save. Never a partial or
   time-to-94% figure.

## 3. Data handling

* All 50 000 train PNGs decoded once to a uint8 NCHW cache. Class counts verified balanced
  (5 000 each).
* **Tuning split:** stratified 45 000 / 5 000 train/val, `numpy.random.default_rng(0)`,
  fixed for the whole study so every config sees the identical split.
* ZCA whitening statistics are fitted **only on the split currently being trained on**
  (45 k during tuning, 50 k for the final run), accumulated over *all* images in chunks —
  no subsampling, because the on-disk cache is class-sorted and any prefix sample would be
  class-biased.
* Verifier input convention: inputs arrive already standardized by
  `Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))`. Training uses the **same**
  standardization, and `forward` performs **no** further normalization.

## 4. Model / recipe family (fixed before search)

airbench-style convnet: frozen 2×2 ZCA whitening conv (3→24) → GELU → three
conv/pool/BN/GELU groups (widths 128/384/512) → MaxPool(3) → bias-free linear head,
logits scaled. BatchNorm affine weight frozen at 1, bias trainable. SGD + Nesterov,
label smoothing, triangular LR schedule, decoupled-style weight decay, separate (higher)
LR for BN/whitening biases. Augmentation on-GPU: per-image ±2 px translate (reflect-pad +
two gathers) and random horizontal flip. 6-way TTA (mirror × {identity, 2 translates})
folded into the scripted `forward`.

## 5. Search plan (ordered, on the 45 k/5 k split only)

* **S0 — noise floor.** Baseline config, `n = 5` seeds. Report mean, sd, and the sd of the
  3-seed mean. Any later difference smaller than ~2× the 3-seed-mean sd is declared
  *not resolvable* and settled by the tie-break rule, not by the point estimate.
* **S1 — epochs.** {12, 16, 20, 24, 30} × 2 seeds. Records the whole val-accuracy-vs-epoch
  curve, so later stages can read off crossing epochs without retraining.
* **S2 — optimisation.** LR multiplier × weight decay, coarse grid, 2 seeds.
* **S3 — head/loss.** logit scale × label smoothing, 2 seeds.
* **S4 — confirmation.** Top-2 configs from S1–S3, `n = 3` seeds each.

Search runs write checkpoints nowhere (or to `/workspace/scratch_ckpt/`); the graded
directory `/workspace/periodic_checkpoints/` is only ever written by a promoted full run.

## 6. Pre-committed promotion rule

Let `A_val(c, e)` be the mean 45 k→5 k val accuracy of config `c` at epoch `e` over its
confirmation seeds, and `sd_c` its across-seed sd at the final epoch.

1. **Eligibility:** a config is eligible iff its *worst* confirmation seed reaches
   val ≥ 0.945 by the final epoch. (Val on 5 k has sd ≈ 0.3 pp from finite-sample noise
   alone; the 0.5 pp cushion over the 0.94 target absorbs it plus the val→test gap.)
2. **Ranking:** among eligible configs, rank by `A_val(c, E_c) − sd_c` (mean minus one sd,
   i.e. reward margin, penalise instability).
3. **Tie-break** (difference below the S0 noise floor): fewer total epochs, then fewer
   parameters.
4. The winner is trained once on the full 50 k with 3 candidate seeds; the ladder with the
   **highest final-epoch train-set-independent val proxy is *not* used** — seeds are
   indistinguishable a priori, so the graded seed is fixed to the *first* candidate seed
   (`seed = 0`) chosen before any test evaluation. This removes seed-shopping on test.
5. Only after 1–4 are locked is the resulting ladder loaded with the verifier convention
   and evaluated on test, to confirm it loads and that the crossing epoch is where the
   val curve predicted. If that confirmation *fails* (no crossing), the fallback is the
   already-on-disk safety ladder, and the failure is reported.

## 7. Safety-first ordering

A complete, verified ladder + `/workspace/checkpoint.pt` is produced from the baseline
config **before** any search begins, so that a session kill at any later moment still
scores. The graded directory is replaced only by a delete-then-copy of a *complete*
better run, never left in a mixed state.
