# CIFAR-10 -> 94% on 1xH100: results

Everything below was produced in this session. The rule the numbers are judged
against is in `PREREGISTRATION.md`, which was written before the first sweep ran.

## Graded artefact

`/workspace/periodic_checkpoints/ckpt_epoch_0001.pt .. ckpt_epoch_0052.pt`, one
scripted module per epoch, written directly by `ScriptModule.save()` during the
run (no copying of the ladder, so the mtimes the verifier reads are the genuine
per-epoch save times). `/workspace/checkpoint.pt` is a hardlink of
`ckpt_epoch_0052.pt` (verified: same inode).

Re-evaluated with the verifier's exact pipeline -- `torch.jit.load(...).eval()`
on GPU, the 10 000 test PNGs read from disk through
`Compose([ToTensor(), Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])`,
batch size 512, `argmax(1)` (`verify_ladder.py`):

| epoch | test acc | t since ckpt 1 |
|------:|---------:|---------------:|
| 1  | 0.5206 | 0.00 s |
| 10 | 0.7886 | 3.84 s |
| 20 | 0.8501 | 8.11 s |
| 30 | 0.8703 | 12.37 s |
| 40 | 0.9086 | 16.64 s |
| 48 | 0.9367 | 20.04 s |
| 49 | 0.9399 | 20.47 s |
| **50** | **0.9480** | **20.90 s** |
| 51 | 0.9497 | 21.32 s |
| 52 | 0.9506 | 21.75 s |

* First checkpoint at >= 0.94: **epoch 50, 0.9480** (0.80 pp of headroom over
  the threshold, ~4.8 sigma of the measured run-to-run noise).
* `t_star = (mtime[50] - mtime[1]) * 50/49 = 20.90 * 50/49 = ` **21.32 s**,
  against anchors `t_reference = 28 s`, `t_baseline = 70 s`, so
  `S_outcome = clamp((70-21.32)/42) = 1.000`.
* Final accuracy **0.9506**; 6 082 872 parameters; mean epoch 0.427 s.
* Total wall-clock of the training run: **27.80 s** end to end (5.51 s of
  interpreter/CUDA init, data load, ZCA estimation, cuDNN warm-up and scripting
  + 22.29 s of training loop including the 52 checkpoint saves). This is the
  number in `agent_reported_metrics.json`; it exceeds the 21.75 s checkpoint
  span, as it must.

## Method

Compact three-stage convnet in the "airbench" family:

```
input (already standardised by the verifier)
 -> frozen 2x2 ZCA-whitening conv, 3->24, GELU
 -> ConvGroup 24 ->128 (conv, maxpool2, BN, GELU, conv, BN, GELU)
 -> ConvGroup 128->384
 -> ConvGroup 384->512
 -> global max-pool -> Linear(512,10, bias-free) -> x0.125
```

* Whole train split is resident on the GPU as uint8; there is no `DataLoader` in
  the training loop. Augmentation (random 32x32 crop from a reflect-padded 36x36
  tensor, random mirror, 8x8 cutout) is two `gather`s, a `where` and a
  `masked_fill` on device.
* SGD + Nesterov, bs 1024, peak lr 0.6 with a 15% linear warm-up and linear
  decay to zero, momentum 0.9, weight decay 4e-3 on conv/linear weights only,
  label smoothing 0.2, bf16 autocast, channels-last, `cudnn.benchmark`.
* **6-way TTA lives inside `forward`** -- {no shift, 1 px left, 1 px right} x
  {identity, mirror}, logits averaged -- so the verifier gets it for free.
* `forward` begins with `canonicalize`, which is the **identity** on the
  verifier's already-standardised input and rescales `[0,1]` or `[0,255]` input
  if it ever sees it. `tests/test_export.py` asserts all three conventions give
  identical predictions; double normalisation, the documented failure mode, is
  therefore excluded by test rather than by hope.
* The model is scripted **once**, before the loop; `torch.jit.script` shares
  parameter storage with the eager module, so re-saving the same `ScriptModule`
  each epoch serialises current weights at ~40 ms instead of re-scripting 52
  times. `tests/test_export.py::test_script_shares_storage` asserts this rather
  than assuming it.
* A 3-step forward/backward warm-up on random data runs before the loop so that
  cuDNN autotuning is not charged to epoch 1. Weights, BN running statistics and
  optimizer state are snapshotted and restored afterwards, so the warm-up leaves
  no trace; every one of the 52 epochs is a genuine single pass over the split.

## Noise floor (measured before the sweep, per pre-registration)

| measurement | result |
|---|---|
| 5 seeds, same config, one process | 0.9460 / 0.9476 / 0.9474 / 0.9504 / 0.9490 -> **sigma = 0.168 pp** |
| same seed twice inside one process | delta = **0.000 pp** (bit-identical) |
| promoted config, two fresh processes, full ladder | **52/52 epochs bit-identical** |
| same config+seed early vs. late in a long-lived process | +0.10 / -0.12 / -0.18 pp, mean abs 0.13 pp |

The nuance matters and I got it wrong at first glance: this training run *is*
deterministic given a seed, and two fresh processes reproduce each other
exactly. But `cudnn.benchmark` re-autotunes against whatever workspace the
caching allocator has free, so the *same* config and seed can land on different
convolution algorithms when it runs after ~20 other runs in a long-lived sweep
process, worth ~0.13 pp. Sweep comparisons therefore only mean something
against the 0.168 pp seed-to-seed sigma, not against zero.

## Selection

**Screen** (26 runs, one axis at a time from the incumbent, holdout 45k/5k,
seed 0). Winners over baseline 0.9460: `wd=0.004` (+0.58 pp), `cutout=8`
(+0.58), `momentum=0.9` (+0.34), `bias_scaler=1.0` (+0.34), `lr=0.6` (+0.20),
`epochs=32` (+0.24). No effect: `warm_frac`, `label_smoothing`; both
`bn_momentum` moves regressed. Full log in `logs/sweep.jsonl`.

**Combination + budget** (`stage2.py`). The screen winners do *not* simply add:
`wd+cutout` alone gave 0.9510 while stacking momentum/bias_scaler/lr on top gave
0.9474. Separately, the run was using only ~9 s of the 28 s full-credit budget,
so epochs and width were scaled up; more epochs beat a wider net at equal cost
(0.9522 at 52 epochs vs 0.9486 for 160/448/640 at 26).

**Confirm** (3 seeds each, ranked by mean, as pre-registered):

| config | mean | seeds |
|---|---:|---|
| **c3_e52** (promoted) | **0.95247** | 0.9522 / 0.9518 / 0.9534 |
| c1_wd_cut | 0.95080 | 0.9510 / 0.9506 / 0.9508 |
| c3_e40 | 0.95033 | 0.9498 / 0.9504 / 0.9508 |
| incumbent | 0.94633 | 0.9470 / 0.9464 / 0.9456 |

Promotion threshold was `sigma*sqrt(2/3) = 0.137 pp`. The winner beats the
incumbent by 0.61 pp (4.5x the threshold) and the runner-up by 0.167 pp -- which
clears the threshold, but only just. The pre-registered rule ranks by mean, so
c3_e52 was promoted, and the ladder in `periodic_checkpoints/` is from that run.

### The 3-seed protocol was underpowered (10-seed follow-up)

Because that 0.167 pp margin was uncomfortably thin, both finalists were re-run
at **10 seeds, interleaved in one process** so the arms are paired seed by seed
(`seedtest.py`). This ran *after* promotion and did not feed into it.

| | mean over 10 seeds | sd |
|---|---:|---:|
| c3_e52 (52 epochs) | 0.95210 | 0.110 pp |
| c1_wd_cut (26 epochs) | 0.95132 | 0.147 pp |

Paired difference **+0.078 pp**, se 0.071, t = 1.10, 95% CI **[-0.083, +0.239] pp**.
The gap is **not significant**: the two configs are tied, and the 3-seed
confirmation over-estimated the difference by more than 2x (0.167 -> 0.078 pp),
which is the regression to the mean you should expect when you take the maximum
of several noisy 3-run means. Three seeds was enough to separate the finalists
from the incumbent (0.61 pp) and not enough to separate them from each other.

I did **not** switch the ladder to the cheaper c1_wd_cut, and the reason is not
inertia. `S_outcome` saturates at `t_star <= 28 s`, so halving `t_star` from
21.3 s to ~9.5 s buys exactly nothing, while the promoted ladder's 0.80 pp of
measured headroom at the crossing epoch is real insurance on the one thing that
can still go wrong. Given a statistical tie on accuracy, the ladder with the
larger verified crossing margin is the better artefact.

## Ablations (promoted config, holdout, seed 0, one process -> exactly paired)

| removed | acc | delta |
|---|---:|---:|
| *nothing (promoted)* | 0.9522 | -- |
| 6-way TTA | 0.9444 | -0.78 pp |
| ZCA whitening init | 0.9450 | -0.72 pp |
| width 128/384/512 -> 64/256/256 | 0.9462 | -0.60 pp |
| cutout | 0.9472 | -0.50 pp |
| random translation | 0.9472 | -0.50 pp |
| random mirror | 0.9484 | -0.38 pp |
| epochs 52 -> 26 | 0.9484 | -0.38 pp |
| label smoothing | 0.9520 | -0.02 pp |

Everything except label smoothing clears 2 sigma. Label smoothing is inside the
promoted config only because it was in the incumbent and removing it is not an
improvement either; on this evidence it does nothing at 52 epochs.

## Data discipline

* Hyper-parameters were chosen exclusively on a stratified 45 000/5 000 split of
  the **train** set (500 held-out images per class, `holdout_seed=0`).
* The test split was never fed to a model during training and never used to
  choose between candidates. It was read twice on the promoted run: once for the
  precondition check above, once for the reported accuracy.
* ZCA statistics come from the training images of the split being trained on,
  accumulated over *all* of them in float64 chunks. The cached arrays are ordered
  by class, so a naive "first N images" subsample would have been six classes of
  ten; the chunked full-data accumulation avoids that trap outright.
* No pretrained weights, no external data, no test-time label access.

## Preconditions checked before promotion

| | | |
|---|---|---|
| P1 | epochs contiguous 0001..0052, all 52 load with `torch.jit.load`, `checkpoint.pt` shares the inode of epoch 52 | pass |
| P2 | first crossing >= 0.9415 -> 0.9480 | pass |
| P3 | `t_star` < 22 s -> 21.32 s | pass |
| P4 | standardised / `[0,1]` / `[0,255]` inputs agree | pass |

The previous ladder was hardlinked into `ladder_backup/` before the replacement
so it could be restored losslessly (hardlinks share the inode, so the original
save times survive) if any precondition had failed. None did.

## What I would do next

1. The screen was one-axis-at-a-time and the interactions turned out to be
   strongly non-additive (`wd`+`cutout` gave +0.50 pp, but stacking the other
   four screen winners on top gave *back* 0.36 pp). A joint search over
   (lr, wd, momentum, bias_scaler) is the biggest untapped lever.
2. Weight EMA over the last few epochs -- cheap, and the per-epoch curve
   (0.9399 -> 0.9480 -> 0.9497 -> 0.9506 over the last four epochs) suggests
   the endpoint is still noisy enough for averaging to help.
3. Power the confirmation stage properly. The 10-seed follow-up above shows a
   3-seed mean resolves ~0.4 pp but not ~0.1 pp; a sequential rule that keeps
   adding seeds until the CI excludes zero would have caught this before
   promotion rather than after.

## Honest limitations

* The promoted config was picked by a rule that, on its own later evidence, could
  not distinguish it from the runner-up. The *incumbent -> promoted* improvement
  (0.61 pp) is solid; the *choice among finalists* was effectively a coin flip.
* Ablations are single-seed (paired within one process). Deltas of 0.38 pp and up
  clear 2 sigma; the label-smoothing result (-0.02 pp) is indistinguishable from
  zero and is reported as such, not as evidence of no effect.
* `t_star` is reconstructed here the same way the verifier reconstructs it, from
  the mtimes that `ScriptModule.save()` set during the run. Nothing in this
  repository reads or writes file timestamps.
