# Pre-registration — CIFAR-10 speedrun, attempt 24

Written **before** any sweep was run, and not edited afterwards. Results and the
promotion decision live in `RESULTS.md`.

## 1. What is actually being optimised

`reward = GATE * S_outcome * RubricFactor`, and
`S_outcome = clamp((70 - t*) / (70 - 28), 0, 1)`.

The insurance run already on disk when this document was written crosses 94% at
epoch 18 with **t\* = 23.4 s**, i.e. `S_outcome = 1.0` already. Every second
shaved below 28 s is worth **zero** additional reward. Therefore the sweep does
**not** optimise for speed. It optimises for:

1. **Robustness of the crossing** — the verifier re-evaluates independently, and
   run-to-run seed noise on this task is ~0.3–0.5 pp. A ladder whose first
   crossing checkpoint reads 94.02% is one unlucky draw away from *no crossing at
   all* and `S_outcome = 0`. Accuracy margin is the thing with real expected
   value.
2. **A comfortable t\* safety margin** — target `t* <= 15 s`, roughly half the
   reference anchor, so that timing jitter cannot push us over.

So the sweep objective is: **maximise held-out top-1 accuracy at a fixed
18-epoch budget**, subject to `t* <= 15 s` on the final full-train run.

## 2. Validation protocol (no test data is used for any decision)

* **Held-out split**: a fixed 5,000-image slice of the 50,000-image *train*
  split, selected by `torch.randperm(50000, generator=seed 1234)[:5000]`. The
  same slice is used for every candidate and every seed.
* Sweep runs train on the remaining **45,000** images only.
* The **test split is never used to choose anything**. It is used exactly once
  at the end, to confirm that the promoted ladder behaves under the verifier's
  own `ImageFolder -> ToTensor -> Normalize` pipeline the way it does under
  ours. Held-out accuracy on 45k-trained models is expected to read ~0.5 pp
  *below* full-train test accuracy; the sweep only uses it for **ranking**.

## 3. Noise floor

Before ranking anything, the baseline config is run with **4 seeds** and the
per-seed standard deviation `sigma` of held-out accuracy is recorded. Any
difference smaller than the noise floor is reported as "not distinguishable"
and must not drive a decision.

## 4. Candidates (frozen list, one-factor-at-a-time from baseline)

Baseline: `lr=11.5, momentum=0.85, weight_decay=0.0153, label_smoothing=0.2,
bias_scaler=64, warmup_frac=0.23, lr_init_frac=0.2, lr_final_frac=0.0,
widths=(128,384,512), bn_momentum=0.6, scaling_factor=1/9, whiten_bias_epochs=3,
batch=1000, epochs=18, fp16`.

| id | change |
|----|--------|
| A  | baseline |
| B  | `lr = 9.0` |
| C  | `lr = 14.0` |
| D  | `weight_decay = 0.0110` |
| E  | `weight_decay = 0.0200` |
| F  | `label_smoothing = 0.10` |
| G  | `label_smoothing = 0.30` |
| H  | `momentum = 0.80` |
| I  | `momentum = 0.90` |
| J  | `lr_final_frac = 0.07` |
| K  | `warmup_frac = 0.13` |
| L  | `bn_momentum = 0.8` |
| M  | `scaling_factor = 1/6` |
| N  | `widths = (160, 448, 512)` |
| O  | `whiten_bias_epochs = 6` |

If wall-clock forces a cut, candidates are dropped from the **bottom of the
table upwards**, and the cut is reported.

## 5. Decision rule (committed in advance)

1. **Round 1** — every candidate on seeds {0, 1}. Baseline additionally on
   seeds {2, 3} to measure `sigma`.
2. **Round 2** — every candidate whose round-1 mean is `>= baseline mean` is
   re-run on seeds {2, 3}, giving 4 seeds each.
3. A candidate is a **real winner** iff
   `mean(cand, 4 seeds) - mean(A, 4 seeds) > 2 * sigma * sqrt(2/4)`
   (≈ two standard errors of the difference). Anything else is declared
   indistinguishable from baseline.
4. **Promotion**: if one or more real winners exist, the single best is combined
   with the baseline and re-validated (4 seeds) before shipping. If **no**
   candidate is a real winner, the **baseline ships unchanged** — a null result
   is a result, and picking the top of a noisy leaderboard is exactly the error
   this protocol exists to prevent.
5. **Epoch count for the shipped run** is chosen *after* the config is fixed, as
   the smallest budget whose 3-seed mean held-out accuracy leaves
   `>= 4 * sigma` of headroom over 94%, so that the verifier's independent
   re-evaluation crosses with margin rather than by luck.
6. **Seed for the shipped run**: seed 0, fixed in advance. The shipped ladder is
   whatever seed 0 produces — no cherry-picking the best of several draws.

## 6. Red lines

* No pretrained weights of any kind. The whitening layer is estimated from the
  train split at the start of each run (~40 ms) and is not a stored weight.
* No test-split input ever reaches the model during training.
* No external data.
* File timestamps are left entirely to `save()` / `shutil.copy`.
* `agent_reported_metrics.json` reports the **total** wall-clock of the shipped
  run, from process start to the last checkpoint — never a time-to-94%.
