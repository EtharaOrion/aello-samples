# Pre-registered experiment plan

Written **before** any sweep was run (baseline `safety_v0` was already on disk).

## Scoring model, and what actually matters

`reward = GATE * S_outcome * RubricFactor`, with

    S_outcome = clamp((70 - t*) / (70 - 28), 0, 1),
    t*        = (mtime[ckpt_N] - mtime[ckpt_1]) * N/(N-1)   for the first N with acc >= 94%.

The `safety_v0` baseline already measures **t\* = 6.05 s**, i.e. S_outcome = 1.0
with ~4.6x headroom (t* could grow to 28 s before the score moves at all).
At 0.39 s/epoch we could cross as late as **epoch ~70** and still score 1.0.

So marginal speed is worth **nothing**, and the only way to lose outcome points
is for no checkpoint to reach 94% under the verifier's own re-evaluation.
`safety_v0` crosses with a single checkpoint at 94.65%, and the observed
run-to-run spread at fixed seed (cuDNN autotuning nondeterminism) is a few
tenths of a point. That is an uncomfortably thin margin.

**Therefore the objective of this study is margin, not speed:** maximise
held-out accuracy and the number of checkpoints comfortably above 94%, spending
the enormous t\* headroom to do it.

## Protocol

* Every decision is made on a **5,000-image held-out slice of the train
  split** (fixed permutation, seed 1234, independent of the run seed). Training
  uses the remaining 45,000. The CIFAR-10 test split is never read by
  `train.py`/`sweep.py`.
* Held-out accuracy is measured with the same 6-way TTA that is baked into the
  saved checkpoints, so the tuning signal matches the deployed model.
* **Step 0 — noise floor.** 5 seeds of the baseline config. Report sigma of
  final held-out accuracy. Any effect smaller than ~2 * sigma/sqrt(n_seeds) is
  not actionable.
* **Steps 1..k — sweeps.** Single-seed screening to rank, then the top
  candidates re-run over 3 seeds. Promotion requires beating the baseline's
  3-seed mean by more than the noise band.
* The promoted config is retrained **directly into
  `/workspace/periodic_checkpoints/`** (never copied in) so the mtimes the
  verifier reads are the genuine save times of that run.

## Hypotheses (registered in advance)

| # | Hypothesis | Levels | Prior / rationale |
|---|---|---|---|
| H1 | Longer schedule raises accuracy; t\* stays far under budget | epochs 16 / 20 / 24 / 30 | Free given 4.6x t\* headroom. Expect monotone gain with diminishing returns. |
| H2 | A flat low-LR tail densifies the >=94% region | tail_epochs 0 / 2 / 3 | The ladder tail is jittery; holding LR at `end_mult` should stop the last epochs bouncing. |
| H3 | Lower final LR sharpens the endpoint | end_mult 0.07 / 0.02 | Standard annealing intuition. |
| H4 | Peak LR needs retuning for the wider net + longer schedule | lr 9 / 11.5 / 14 | 11.5 was tuned for widths 64/256/256 at 9.9 epochs. |
| H5 | Longer schedules tolerate/need more regularisation | wd 0.0153 / 0.020 / 0.025 | More steps -> more overfitting pressure. |
| H6 | Stronger translation aug helps a longer schedule | pad 2 / 3 | Same. |
| H7 | Label smoothing retune | ls 0.15 / 0.20 | Cheap to check. |
| H8 | Whitening-bias training window should scale with schedule | whiten_bias_epochs 3 / 5 | It is 3/9.9 of the schedule in the reference recipe. |

Pre-registered decision rule: promote the configuration with the highest
3-seed mean held-out accuracy; break ties in favour of the one whose held-out
curve spends more epochs above the crossing threshold. If nothing beats the
baseline outside the noise band, keep `safety_v0`.
