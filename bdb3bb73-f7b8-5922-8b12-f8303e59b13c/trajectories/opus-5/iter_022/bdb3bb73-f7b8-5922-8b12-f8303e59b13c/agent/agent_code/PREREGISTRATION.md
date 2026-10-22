# Pre-registration — CIFAR-10 → 94% speedrun

Written **before** any sweep run. Fixed in advance: the holdout, the metric, the
noise-floor estimate, the candidate list, and the promotion rule.

## 1. Data discipline

* All tuning uses a **holdout carved from the train split**: `torch.randperm(50000)`
  under generator seed `1234`, first 5,000 indices → holdout, remaining 45,000 → train.
  The split is fixed across every run so comparisons are paired.
* The **test split is never fed to a model during training or hyperparameter selection.**
  The only test-set contact is the final artifact check described in §6.
* No pretrained weights, no data beyond the CIFAR-10 train split.

## 2. Metrics

Holdout accuracy is measured through the same 6-way TTA path the saved checkpoint uses.

* **Primary — `acc3`**: mean holdout top-1 over the run's **final 3 epochs**.
  Averaging three epochs cuts the epoch-to-epoch jitter roughly in half versus
  reading a single final epoch, at no extra training cost.
* **Secondary — `cross`**: first epoch with holdout top-1 ≥ 0.94. This is the quantity
  `S_outcome` keys on. It is *pessimistic* here: the holdout runs train on 45k rather
  than 50k images, so holdout accuracy sits below full-train test accuracy and `cross`
  lands later than it will in the graded run. It is used for ranking, never as an
  absolute prediction.

## 3. Noise floor (measured first, before any candidate is run)

Run the baseline at **4 seeds**. Report σ_run = sample s.d. of `acc3`. Because the
holdout has 5,000 images, binomial sampling noise alone contributes
√(0.93·0.07/5000) ≈ 0.36 pp, so σ_run is expected to be ≈ 0.3–0.5 pp.
σ_mean(k seeds) = σ_run/√k.

## 4. Baseline (incumbent)

```
widths=(128,384,512)  bn_mom=0.4  scale=1/9  batch_size=1000  epochs=20
lr=1.5  wd=2.0e-3  momentum=0.85 (nesterov)  bias_scaler=64
label_smoothing=0.2  warm_frac=0.23  lr_start=0.2  lr_end=0.0  whiten_bias_epochs=3
```

## 5. Candidates and promotion rule

Screening: **1 seed** each, one factor changed at a time.

| factor | values |
|---|---|
| `lr` | 1.2, 1.8 |
| `wd` | 1.4e-3, 2.8e-3 |
| `label_smoothing` | 0.1, 0.3 |
| `epochs` | 16, 24 |
| `warm_frac` | 0.15, 0.35 |
| `bn_mom` | 0.25, 0.6 |
| `momentum` | 0.80, 0.90 |
| `lr_start` | 0.05 |
| `whiten_bias_epochs` | 0, 6 |
| `batch_size` | 500 |

* **Advance to confirmation**: any candidate with `acc3` ≥ (baseline seed-0 `acc3` − 1·σ_run).
  The generous cut is deliberate — a 1-seed screen cannot separate a real +0.3 pp from noise,
  so screening only discards clear losers.
* **Confirmation**: the top 4 survivors by screening `acc3`, at **3 seeds** each.
* **Promotion**: adopt candidate C over the baseline only if
  `mean₃(C) − mean₄(baseline) > 2·√(σ_run²/3 + σ_run²/4)`.
  If nothing clears that bar, **keep the baseline** — an unresolvable difference is not a win.
* Ties (both clear the bar within each other's interval): prefer the smaller `epochs`
  (earlier crossing → smaller `t_star`), then the incumbent.

## 5b. Round 2 — appendix, written after round 1's screening printed but before any
round-2 run

Round 1 screening put its two best single-factor moves at the *edge* of the swept
range (`momentum` 0.80 was the lowest value tried, `lr` 1.8 the highest), so round 2
walks past those edges and tests whether the two stack. Registered before running:

| candidate | change from baseline |
|---|---|
| `mom0.80+lr1.8` | `momentum=0.80, lr=1.8` |
| `mom0.80+wd1.4e-3` | `momentum=0.80, wd=1.4e-3` |
| `mom=0.75` | `momentum=0.75` |
| `lr=2.1` | `lr=2.1` |

* **3 seeds each** (0,1,2) — no 1-seed screen, so these are directly comparable to
  the round-1 confirmation numbers.
* **σ is re-estimated by pooling** the within-group sample variances of every group
  with ≥3 seeds (baseline ×4 plus each confirmed candidate ×3) instead of using the
  baseline's 4-seed σ alone. Round 1's baseline σ (0.079 pp) is estimated from 3
  degrees of freedom and the confirmation groups already show visibly wider spread,
  so the pooled estimate is both better-determined and **larger** — it can only
  raise the promotion bar, never lower it.
* **Promotion**: the same rule, applied against the *round-1 winner* as incumbent:
  adopt only if `mean₃(C) − mean(incumbent) > 2·σ_pooled·√(1/3 + 1/n_incumbent)`.
  If nothing clears it, ship the round-1 winner.
* Tie-break unchanged: earlier crossing epoch, then the incumbent.

## 6. Final artifact check

After the graded run, each saved checkpoint is loaded with `torch.jit.load(...).eval()`
and evaluated on the real test PNGs through
`Compose([ToTensor(), Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])`
at batch size 512 — the verifier's exact convention. This is a **correctness gate on
the artifact** (does the scripted module load, does it accept standardized float32 input,
is the ladder contiguous from epoch 1), not a selection signal. Pre-committed: if the
check fails, the response is to fix the defect it exposes, **not** to re-tune
hyperparameters against the test set.

## 7. Timing

`t_star` is reconstructed from checkpoint save times, so everything inside the training
run counts. Before the timed loop the GPU is warmed on **random noise with a throwaway
network** (discarded; no weights or statistics carry over) because the H100 idles at
345 MHz against a 1980 MHz ceiling and a cold start makes the first epochs ~2× slow.
Every one of the N epochs remains a single genuine pass over the training split, and
`agent_reported_metrics.wall_clock_sec` is the full wall clock of the graded run.
