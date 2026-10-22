# CIFAR-10 to 94% top-1 on one H100

Trains a compact convolutional network from scratch on the CIFAR-10 train split
and writes one TorchScript checkpoint per epoch to
`/workspace/periodic_checkpoints/`.

## Reproducing

```bash
python prepare_data.py train test        # decode PNGs -> uint8 .npy caches (~1s)
bash promote.sh                          # repro run + graded run + verify + validate
```

`promote.sh` runs the whole delivery path: an independent-seed reproducibility run
into a scratch directory, then the graded run trained **directly into**
`/workspace/periodic_checkpoints/`, then the verifier-replica evaluation, then the
machine-checked precondition suite.

## Files

| file | role |
| --- | --- |
| `prepare_data.py` | one-time PNG -> uint8 `.npy` decode for train and test |
| `core.py` | model, GPU augmentation, whitening init, optimiser, schedule |
| `train_run.py` | one training run; held-out mode for tuning, ladder mode for delivery |
| `sweep.py` / `analyze_sweep.py` | held-out hyperparameter search and noise-aware summary |
| `ablations.py` | isolates the contribution of TTA, whitening, and augmentation |
| `finalize.py` | writes the graded ladder, `checkpoint.pt`, and the metrics file |
| `verify_ladder.py` | re-implements the verifier's evaluation and clock |
| `validate_checkpoints.py` | 12 machine-checked preconditions on the delivered ladder |

## Method

**Architecture** (airbench lineage, 6.08M parameters). A frozen 2x2 patch-whitening
convolution (3 -> 24 channels) feeds three `conv3x3 -> maxpool -> BN -> GELU ->
conv3x3 -> BN -> GELU` groups of width 128 / 384 / 512, then a global max-pool and a
bias-free linear head scaled by `logit_scale`. The whitening filter is the
eigen-decomposition of the *training-split* 2x2 patch covariance, emitted with both
signs so the following GELU sees each direction in both polarities. That is a
data-dependent initialisation recomputed from scratch on every run — no pretrained
weights are loaded anywhere in this repository.

**Data path.** All 50,000 training images are standardised once and held on the GPU
as a single fp16 tensor. Each epoch materialises one augmented copy of the whole
set (per-sample random translate within a 3px reflect pad, random horizontal flip,
6px cutout) with a single gather, so there is no dataloader, no host-to-device copy,
and no CPU work inside the training loop.

**Optimisation.** Nesterov SGD, momentum 0.85, batch 1024, bf16 autocast,
channels-last. Weights get weight decay 0.01; BN parameters and biases get no decay
and 4x the learning rate. Linear warmup over the first 12% of steps to lr 0.2, then
linear decay to 2% of peak. Label smoothing 0.1. 24 epochs.

**Test-time augmentation is part of `forward`.** The model averages logits over the
image, its mirror, and a 1px horizontal translate of both (weighted 1, 1, 0.5, 0.5).
Folding this into the module means the verifier's single `model(x)` call gets the
full model, and TTA costs nothing in the measured training window.

**Input-convention handling.** The verifier feeds batches that are already
`ToTensor() + Normalize(...)` standardised. `forward` expects exactly that, but
sniffs `min`/`max` and rescales `[0,1]` or `[0,255]` batches if it sees them —
standardised CIFAR batches always contain values near -2, so the three conventions
are unambiguous. Check `L4` in the validation suite confirms all three produce
identical predictions, which makes double-normalisation impossible rather than
merely unlikely.

## Keeping the test split out of the loop

Every design decision — architecture width, learning rate, weight decay, label
smoothing, logit scale, bias LR multiplier, cutout, pad, warmup, batch size, epoch
count, EMA — was selected on a fixed 5,000-image held-out slice of the **train**
split (split seed 1234, identical across all runs). `train_run.py` and `sweep.py`
never open the test directory; `validate_checkpoints.py` check `L8b` asserts this
statically.

The test split is read in exactly one place, `verify_ladder.py`, and only *after* a
candidate is fully trained, to confirm that the saved TorchScript artefacts behave
under the verifier's exact pipeline. It is a correctness check on the delivery path,
not a selection signal: the config in `final_cfg.json` was frozen from held-out
results before the graded run was trained.

## Measurement honesty

**The clock is real.** The verifier rebuilds `t*` from checkpoint mtimes. The graded
ladder is therefore trained *directly into* `/workspace/periodic_checkpoints/` —
never trained elsewhere and copied in, because `shutil.copy` does not preserve
mtimes and a copied ladder would report copy-times (a sub-second span) instead of
real per-epoch training times. No timestamp is ever modified by any code here.

**Nothing is deferred out of the measured window.** Augmentation, optimisation and
TorchScript serialisation all happen between the epoch-1 and epoch-N saves. The only
work before the window is data loading, whitening init, and one warmup step for
cuDNN autotuning; that is reported as `warmup_sec` and is a fraction of a second.

**`torch.compile` is deliberately not used.** It roughly halves per-step time but
costs ~15s of compilation. For a 24-epoch run that is a net *loss* in true
end-to-end wall clock, and its only effect on the score would be to move real
compute outside the mtime window the verifier measures. Both reasons point the same
way, so the run is eager-mode throughout.

**`S_outcome` is saturated, so the remaining budget was spent on margin, not
speed.** With `t_reference = 28s` and a measured `t* ~ 10s`, going faster earns
nothing, while failing to cross 94% earns zero. Choices were made accordingly: 24
epochs rather than the 20 that also clears the bar, and the accuracy-vs-time
trade-offs below resolved toward accuracy whenever they were free.

## Results

Graded ladder, re-evaluated by `verify_ladder.py` (which reproduces the verifier's
pipeline: `ImageFolder` over the raw test PNGs, `ToTensor()` + `Normalize(...)`,
batch 512, `torch.jit.load(path).eval()`, `argmax(1)`):

| quantity | value |
| --- | --- |
| first checkpoint >= 94% | `ckpt_epoch_0023.pt`, 94.17% |
| `t*` = (mtime[23] - mtime[1]) x 23/22 | **9.33 s** |
| `S_outcome` at t_ref 28s / t_base 70s | **1.00** (clamped) |
| final checkpoint (epoch 24) | **94.83%** |
| mean epoch time | 0.405 s (includes the per-epoch TorchScript save) |
| total run wall clock, process start to last save | 12.44 s |
| validation suite | 12 / 12 checks pass |

Three independent seeds of the same config, each trained end to end:

| seed | first >= 94% | `t*` | final acc |
| --- | --- | --- | --- |
| 0 (graded) | epoch 23, 94.17% | 9.33 s | 94.83% |
| 1 (scratch) | epoch 23, 94.17% | 9.31 s | 95.00% |
| 2 (scratch) | epoch 23, 94.48% | 9.30 s | 94.91% |

All three cross at the same epoch, within 0.03 s of the same `t*`, and finish
within 0.17% of each other (mean final accuracy 94.91%). Each was trained end to
end with `finalize.py` and scored by the verifier-replica pipeline; none is a
re-evaluation of another's weights.

### Hyperparameter search

Three stages on the 5,000-image held-out train slice: 49 configurations over 74
training runs, plus 16 more in the ablations below.

*Stage 1* — one factor at a time, 31 configs, 1 seed. Almost every delta landed
inside the noise floor; what it reliably established was the *losing* side of each
axis: `weight_decay=0.04` (-0.82%), `widths=(64,256,256)` (-0.70%),
`batch_size=512` (-0.56%), `final_frac=0.10` (-0.42%), `pad=1` (-0.36%).

*Stage 2* — the promising directions combined, 3 seeds each. Combining is what
lifted the effect above the noise floor:

| config | val acc (3 seeds) | vs baseline |
| --- | --- | --- |
| C3 = cutout 6 + pad 3 + ls 0.1 + logit_scale 0.25 + bias_lr 4x | 0.9515 | +0.39% |
| C3 + widths (160,448,640) | 0.9525 | +0.48% |
| C3 + EMA | 0.9525 | +0.49% |
| augmentation part alone (cutout 6, pad 3) | 0.9494 | +0.17% |
| optimiser part alone (ls, logit_scale, bias_lr) | 0.9467 | -0.09% |
| baseline | 0.9477 | — |

Neither half of C3 does much alone; together they gain 0.39%. The optimiser
changes only pay off once the stronger augmentation is there to be exploited.

*Stage 3* — refinement around C3, 2 seeds each: `epochs=20` 0.9472, `epochs=24`
0.9511, `epochs=28` 0.9513; `lr=0.28` 0.9512, `weight_decay=0.006` 0.9516,
`label_smoothing=0.05` 0.9505. The schedule saturates at 24 epochs and no
remaining single axis moves the result.

### Two negative results, and what was dropped

**EMA does nothing here.** Every EMA setting returned the same number to four
decimals — `ema_start_frac` 0.4 / 0.6 / 0.8 and `ema_decay` 0.90 / 0.95 / 0.98 all
gave 0.9510-0.9512. The cause is visible in the schedule: the LR is already at 2% of
peak when the average starts, so the weights barely move and the EMA converges to
the final weights. It was removed rather than kept as inert complexity
(`ema_start_frac > 1` disables it, and the code is retained only so the ablation is
reproducible).

**The wider model was not adopted.** `widths=(160,448,640)` was ahead by +0.10%
(3 seeds, stage 2) and +0.13% (2 seeds, stage 3) — consistent in sign, but pooled
over 5 runs that is still under one standard error, and it costs 43% more wall
clock. Not adopted: an unestablished tenth of a percent is not worth a 43% slowdown
even when the slowdown is affordable.

### Noise floor

This deserves stating plainly because it governs how the numbers above should be
read. A 5,000-image held-out slice has a binomial standard error of about **0.32%**
at p = 0.945. Runs are also not bitwise reproducible even at a fixed seed —
`cudnn.benchmark` and non-deterministic reduction kernels are left on for speed, and
the identical `baseline, seed 0` configuration returned 0.9462 in stage 1 and 0.9490
in stage 2. So single-seed differences below roughly 0.6% carry no information.
`analyze_sweep.py` prints the standard error next to every mean for this reason, and
stage 2 and 3 comparisons were run at 3 and 2 seeds respectively.

### What tuning did and did not buy

Tuning moved final accuracy from 94.73% (untuned baseline ladder) to 94.83-95.00%,
but `t*` barely moved: 9.28 s untuned vs 9.33 s tuned, both crossing at epoch 23.
That is expected — `t*` is set by the length of the schedule and the per-epoch cost,
not by where the accuracy curve sits, and the crossing lands near the end of the
schedule either way. The accuracy gain is worth having as threshold margin; it was
never going to show up in the score, which was already clamped.

## Honest limitations

- The 94% crossing is at epoch 23 of 24, so the ladder clears the bar with two
  checkpoints to spare, not ten. The safety margin lives in the *final* accuracy
  (94.83%, roughly 0.8% above threshold) rather than in the number of crossing
  checkpoints. A run whose epoch-23 checkpoint landed 0.2% low would cross at epoch
  24 instead and report `t* = 9.7 s` — same `S_outcome`.
- Held-out selection used 45,000 training images while the delivered model trains on
  all 50,000. The configuration ranking is assumed to transfer across that 10% data
  change; it was not re-validated at 50,000, since doing so would have required the
  test split.
- Test accuracy is reported for the three runs that were fully evaluated. Three
  seeds is enough to show the result is not a lucky draw, but not enough to be read
  as a tight confidence interval.

## Ablations

`ablations.py` (2 seeds) plus a 6-seed paired re-run of the TTA arm
(`tta_more.py`), all on the same held-out train slice.

| ablation | val acc | delta |
| --- | --- | --- |
| **A. test-time augmentation** (6 seeds, paired: one set of weights per seed, three forward variants) | | |
| no TTA | 0.9456 | — |
| mirror only | 0.9501 | **+0.45%** |
| mirror + 1px shift (shipped) | 0.9503 | +0.47% |
| **B. patch whitening** (2 seeds) | | |
| frozen, covariance-initialised | 0.9514 | — |
| same layer, random init, trainable | 0.9514 | **+0.00%** |
| **C. training augmentation** (2 seeds) | | |
| translate + flip + cutout (shipped) | 0.9514 | — |
| no random translate | 0.9414 | **-1.00%** |
| no cutout | 0.9464 | **-0.50%** |

**Augmentation is the load-bearing component.** Removing the random translate
costs a full point — more than every hyperparameter in the search combined.

**The mirror is the entire TTA effect.** The extra 1px-shift pass adds +0.02% and
wins only 4 of 6 paired comparisons, which is indistinguishable from zero. It is shipped
because it is marginally ahead and was already validated end to end, but a
mirror-only `forward` would be the simpler equivalent and would halve the
verifier's evaluation cost. Calling this a tie rather than a win is the honest
reading: it is the one shipped component whose benefit is not established, only
un-contradicted.

**Frozen patch whitening does not help at this schedule length.** Frozen
covariance-derived filters and a randomly-initialised trainable layer of the same
shape tie to four decimals. The whitening trick earns its keep in the ~10-epoch
airbench regime where the network has no time to learn first-layer filters; at 24
epochs it has the time, so the initialisation stops mattering. It is kept because
it costs nothing (the layer exists either way, and freezing it slightly reduces the
backward pass), but it should not be credited for this model's accuracy.
