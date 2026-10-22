# CIFAR-10 to 94% top-1 from scratch on one H100

A whitening-convnet trained from random init on the 50,000-image CIFAR-10
train split. No pretrained weights, no external data, no test-split images in
the training loop.

## Deliverables

| path | what |
|---|---|
| `periodic_checkpoints/ckpt_epoch_NNNN.pt` | one `torch.jit.script(...).save()` per epoch, contiguous from 0001 |
| `checkpoint.pt` | hardlink to the last periodic checkpoint |
| `agent_reported_metrics.json` | final accuracy + **total** wall clock of the graded run |
| `experiments/` | pre-registration, raw sweep logs, verification output |

## Files

| file | role |
|---|---|
| `prep_data.py` | one-time decode of the PNG folders into uint8 `.npy` caches |
| `common.py` | model, ZCA whitening init, GPU augmentation, fast eval |
| `train.py` | the training run; writes the checkpoint ladder |
| `sweep.py` | the pre-registered holdout study |
| `verify_ladder.py` | re-scores a ladder through the grader's exact pipeline |
| `promote.py` | swaps a verified scratch ladder into the graded directory |

## Method

**Architecture** (6.1M / 10.4M params depending on width) follows Keller
Jordan's CIFAR-10 airbench:

```
Conv2d(3, 24, k=2)          <- FROZEN ZCA whitening, learned bias only
GELU
ConvGroup(24  -> w1)        conv3x3 - maxpool2 - BN - GELU - conv3x3 - BN - GELU
ConvGroup(w1  -> w2)
ConvGroup(w2  -> w3)
MaxPool2d(3) -> flatten
Linear(w3, 10, bias=False)
x * (1/9)                   <- constant logit scale, applied in forward
```

Details that matter:
* The whitening filters are the eigenvectors of the 2x2 patch covariance of the
  **train** split, in +/- pairs and scaled by `1/sqrt(lambda + 5e-4)`. The
  weights are frozen; only the bias trains, and only for the first 3 epochs.
* BatchNorm gammas are frozen at 1.0. Only the betas train -- the following
  conv can absorb any per-channel rescale, so the gamma is a redundant
  parameter that costs optimisation steps.
* Convs use a partial-Dirac init on the first `in_channels` filters, which
  gives each group an identity path at step 0.
* fp16 weights/activations, fp32 BatchNorm statistics, `channels_last`.
  The loss uses `reduction='sum'`, which scales gradients by the batch size and
  keeps fp16 gradients clear of underflow without a `GradScaler`.

**Data path.** The whole train split is standardized once and kept resident on
the GPU as a single reflect-padded fp16 tensor. Augmentation (random 32x32 crop
out of 36x36, random mirror) is two `gather`s and a `where` -- no DataLoader,
no host traffic, nothing to starve the GPU.

**Test-time augmentation is inside the checkpoint.** `forward` averages 6
views: {identity, -1px, +1px diagonal translate} x {as-is, mirrored}, weighted
0.25/0.25 for the untranslated pair and 0.125 each for the four translated
ones. The scripted artefact the grader loads therefore does TTA on its own.

**Input convention.** The grader feeds float32 NCHW batches that are *already*
per-channel standardized with mean (0.4914, 0.4822, 0.4465) and std (0.2470,
0.2435, 0.2616). Training tensors are standardized with those exact constants
and `forward` performs **no** normalization of its own, so there is no double
normalization. `verify_ladder.py` re-decodes the test PNGs with torchvision's
`ImageFolder` + `ToTensor` + `Normalize` and re-scores every checkpoint through
`torch.jit.load(...).eval()`, i.e. the grader's pipeline end to end, and
deliberately shares no code with the training data path so a mismatch would
show up as a low number rather than cancelling out.

## Timing discipline

The grader rebuilds the clock from checkpoint save times, so the training loop
does exactly one thing between two saves: one genuine pass over all 50,000
training images. No evaluation, no synchronising logs. Accuracy is measured
afterwards, out of band. cuDNN autotuning is warmed up before training on a
throwaway model fed random noise, so no epoch is charged for the algorithm
search and every epoch costs the same.

Promotion uses `os.link`, not `shutil.copy`: a copy would stamp every
checkpoint with the moment of the copy and make a ~15 s run look like a
fraction of a second. Hardlinks keep the real save times. No file timestamp is
ever modified.

## Results

### The graded ladder

Config: widths 128/384/512 (6.08 M params), batch 1000, SGD-nesterov momentum
0.85, lr 11.5 / weight decay 0.0153 in airbench's per-1024-example units,
bias & BN-beta LR scaled 64x, label smoothing 0.2, whitening bias frozen after
epoch 3, 24 epochs, triangular LR peaking at 23% of the decay phase and
decaying to 0.02 of peak, then **held** at 0.02 of peak for the final 3 epochs.

Independently re-scored by `verify_ladder.py` through the grader's exact
pipeline (`ImageFolder` -> `ToTensor` -> `Normalize` -> `torch.jit.load().eval()`,
batch 512, `argmax(1)`):

| epoch | test top-1 | | epoch | test top-1 |
|---|---|---|---|---|
| 1 | 0.6810 | | 13 | 0.8734 |
| 2 | 0.7727 | | 14 | 0.8860 |
| 3 | 0.7873 | | 15 | 0.8829 |
| 4 | 0.7821 | | 16 | 0.9003 |
| 5 | 0.7854 | | 17 | 0.9234 |
| 6 | 0.8537 | | 18 | 0.9303 |
| 7 | 0.8346 | | 19 | 0.9396 |
| 8 | 0.8212 | | 20 | **0.9472** |
| 9 | 0.8656 | | 21 | **0.9511** |
| 10 | 0.8461 | | 22 | **0.9518** |
| 11 | 0.8742 | | 23 | **0.9526** |
| 12 | 0.8883 | | 24 | **0.9514** |

* first checkpoint at/above 94%: **epoch 20** (0.9472)
* best: **0.9526** (epoch 23)
* checkpoints at/above 94%: **5** of 24
* ladder span 26.61 s over 24 epochs -> **t\* = 23.17 s**, S_outcome = 1.0
* total wall clock of that run, data prep and cuDNN warmup included: **32.14 s**


### How that config was chosen

Every tuning number below is **holdout** accuracy: a fixed 5,000-image slice
of the *train* split, with the model trained on the other 45,000. The test
split was not opened during tuning. The protocol was written down first, in
`experiments/PREREGISTRATION.md`.

**Stage 0 - noise floor.** Baseline config, 6 seeds, 18 epochs:

```
  0.9404  0.9420  0.9352  0.9386  0.9384  0.9406
  mean 0.9392   sigma_1 0.238 pp   sem of a 4-seed mean 0.119 pp
```
This is the bar: any single-run 'improvement' under ~0.24 pp is noise.

**Stage 1 - one-factor-at-a-time screen** (22 cells, seed 0 each),
sorted by holdout accuracy:

```
  widths=(160, 448, 640)     0.9430  +0.38 pp  (+1.60 sigma)  <- survives
  weight_decay=0.01          0.9412  +0.20 pp  (+0.84 sigma)
  bn_momentum=0.4            0.9404  +0.12 pp  (+0.50 sigma)
  bn_momentum=0.8            0.9402  +0.10 pp  (+0.42 sigma)
  label_smoothing=0.1        0.9398  +0.06 pp  (+0.25 sigma)
  lr_end=0.02                0.9398  +0.06 pp  (+0.25 sigma)
  peak_frac=0.15             0.9394  +0.02 pp  (+0.08 sigma)
  lr=8.0                     0.9392  +0.00 pp  (+0.00 sigma)
  peak_frac=0.35             0.9392  +0.00 pp  (+0.00 sigma)
  label_smoothing=0.3        0.9382  -0.10 pp  (-0.42 sigma)
  epochs=22                  0.9380  -0.12 pp  (-0.50 sigma)
  bias_scaler=32.0           0.9378  -0.14 pp  (-0.59 sigma)
  momentum=0.8               0.9368  -0.24 pp  (-1.01 sigma)
  lr=16.0                    0.9366  -0.26 pp  (-1.09 sigma)
  widths=(96, 320, 448)      0.9366  -0.26 pp  (-1.09 sigma)
  epochs=14                  0.9366  -0.26 pp  (-1.09 sigma)
  bias_scaler=128.0          0.9362  -0.30 pp  (-1.26 sigma)
  weight_decay=0.023         0.9360  -0.32 pp  (-1.35 sigma)
  lr=22.0                    0.9354  -0.38 pp  (-1.60 sigma)
  pad=3                      0.9354  -0.38 pp  (-1.60 sigma)
  lr_end=0.15                0.9332  -0.60 pp  (-2.52 sigma)
  momentum=0.9               0.9328  -0.64 pp  (-2.69 sigma)
```
The airbench hyperparameters are already at a local optimum for this net: lr,
weight decay, momentum, label smoothing, bias scaler, BN momentum, LR peak
position and crop padding all fail to clear baseline + 1 sigma, and several
are clearly worse. Only the width increase survived. Note also that
`epochs=22` scored *below* `epochs=18` -- on 45k images this net is already
converged at 18 epochs, so buying accuracy margin with a longer schedule does
not work at that operating point.

**Stage 2 - 4-seed confirmation** of the surviving direction (baseline reuses
the 6 stage-0 seeds):

| config | n | mean | sd | delta vs base | Welch t | s/epoch |
|---|---|---|---|---|---|---|
| `base` | 6 | 0.9392 | 0.24 pp | +0.00 pp | +0.00 | 0.623 |
| `W160` | 4 | 0.9426 | 0.14 pp | +0.34 pp | +2.86 | 1.016 |
| `W160+wd010` | 4 | 0.9406 | 0.26 pp | +0.14 pp | +0.86 | 0.903 |
| `W160+bn040` | 4 | 0.9427 | 0.15 pp | +0.35 pp | +2.85 | 1.015 |
| `W192(posthoc)` | 4 | 0.9430 | 0.13 pp | +0.38 pp | +3.31 | 1.136 |

The pre-registered promotion bar was delta > 0.34 pp. All three wide variants
clear it and are statistically indistinguishable from one another, so the
effect is 'more width', not any particular width. `W192` was a post-hoc probe
added after width survived, and is labelled as such rather than presented as
part of the registered plan.


### The two findings that actually changed the artefact

**1. The ladder tail, not the peak, was the real risk.** Every schedule tried
up to this point produced a ladder with *exactly one* checkpoint above 94%.
Per-epoch holdout curves (`experiments/curve.log`) show why -- with a
triangular LR that is still decaying at the last step, the second-to-last
epoch sits 0.4-1.0 pp below the last:

```
base_E22 lr_end=0.07 seed0   last 5 epochs: .9118 .9260 .9270 .9338 .9380
W160_E18 lr_end=0.02 seed0   last 5 epochs: .9140 .9206 .9364 .9368 .9436
```

A ladder like that is one unlucky draw away from having *no* crossing
checkpoint, which scores zero -- and the run-to-run spread is ~0.24 pp, so that
is not a remote possibility. The fix is to stop the decay early and hold the LR
at 2% of peak for the last 3 epochs. It costs nothing and flattens the tail
(`experiments/tail_hold_check.log`):

```
base_E24 tail3 seed0   last 6 epochs: .9318 .9384 .9406 .9412 .9416 .9418
base_E24 tail3 seed1   last 6 epochs: .9334 .9416 .9434 .9436 .9430 .9438
W160_E21 tail3 seed0   last 6 epochs: .9310 .9368 .9428 .9428 .9418 .9430
```

The shipped ladder has 5 checkpoints above 94% and 4 above 95%, so the crossing
survives a bad draw.

**2. The width win did not survive the change of operating point.** Stage 2
promoted `W160` on a +0.34 pp margin (t = +2.86) measured at 18 epochs with no
LR tail hold. Re-measured at the schedule actually being shipped, `W160_E21_tail3`
and `base_E24_tail3` both average 0.9428 on holdout -- a delta of 0.00 pp -- while
`W160` costs ~1.6x per epoch. So the baseline width was kept.

This is worth stating plainly: a +2.9-sigma, 4-seed, pre-registered result
evaporated when one other knob moved. Two seeds per arm is thin evidence for the
re-measurement, but it is a point estimate of exactly zero against a 60% cost
increase, and the decision only needs the sign. The general lesson is that a
hyperparameter effect measured at one schedule is not a property of the model,
and the last check before promotion should re-measure at the final operating
point rather than trusting the sweep.

### Honest limitations

* `sigma_1 = 0.24 pp` was measured across seeds *within one process*. Comparing
  the same seed across processes showed shifts of up to ~0.5 pp, because
  `cudnn.benchmark` selects different convolution algorithms depending on what
  ran before it, and fp16 training amplifies the resulting rounding differences.
  So 0.24 pp understates the true run-to-run spread, and the stage-1 1-sigma
  screen was correspondingly lenient. The multi-seed stage-2 gate is what
  protects the conclusions; the single-seed screen only nominates candidates.
* Stage 1 varied one factor at a time, so interactions between hyperparameters
  are invisible to it.
* The GPU was shared with another tenant for most of this session (100%
  utilisation and 74 GB in use by a process outside this container). Epoch time
  for the identical configuration ranged from 0.49 s on an idle device to
  1.16 s under contention, which is why t* for the shipped ladder is 23.2 s
  rather than the ~12 s the same run costs on an idle H100. Accuracy is
  unaffected; only the clock is. S_outcome saturates at t* <= 28 s, so the
  contention did not cost outcome score, but it would have on a busier device.
* The 24-epoch full-train recipe was replicated end to end at a second seed
  (`experiments/verify_replication_seed1.json`). Both runs behave the same way:

  | run | first crossing | peak test top-1 | ckpts >= 94% | t* |
  |---|---|---|---|---|
  | seed 0 (**shipped**) | epoch 20 | 0.9526 | 5 / 24 | 23.17 s |
  | seed 1 (replication) | epoch 19 | 0.9518 | 6 / 24 | 21.93 s |

  last 6 checkpoints, seed 0: 0.9396 0.9472 0.9511 0.9518 0.9526 0.9514
  last 6 checkpoints, seed 1: 0.9410 0.9467 0.9508 0.9513 0.9506 0.9518

  Which ladder to ship was fixed **before** the replication was scored -- seed 0
  was already promoted and the decision was pre-committed -- so the test split
  was not used to choose between them.

### Reproducing

```bash
python prep_data.py                     # decode PNGs -> .npy caches (once)
python train.py --ckpt-dir /workspace/scratch_ckpt/cand   # SHIPPED config, ~15-32 s
python verify_ladder.py --dir /workspace/scratch_ckpt/cand --out v.json
python promote.py --src /workspace/scratch_ckpt/cand --verify-json v.json \
                  --wall-clock <seconds reported by train.py>
```
`sweep.py --stage 0|1|cand:{...}` reruns the holdout study; `selfcheck.py`
re-checks the deliverables. `train.py`'s CLI defaults are the shipped
config (`SHIPPED`); `DEFAULTS` is deliberately left at the *screening*
baseline so re-running `sweep.py` reproduces the logged holdout numbers.

