# CIFAR-10 → 94% top-1 from scratch, single H100

**Result: first checkpoint ≥ 94% is `ckpt_epoch_0019.pt` at 94.33%; the run finishes
at 94.76% (epoch 20). Verifier-style t\* = 6.84 s. Total training wall clock 9.10 s.**

Reference anchors for this task are t_reference ≈ 28 s and t_baseline ≈ 70 s, so t\*
lands ~4× under the reference and S_outcome is saturated at 1.0 with wide margin.

## Files

| file | role |
| --- | --- |
| `prepare_data.py` | one-off decode of the PNG folders into uint8 `.npy` caches (train and test cached by separate invocations) |
| `model.py` | architecture, frozen-whitening init, TTA-folded `forward` |
| `train.py` | training loop; reads **only** the train cache; writes the checkpoint ladder |
| `evaluate.py` | scores saved checkpoints under the verifier's exact input convention; **the only file that reads the test split** |
| `verify_ladder.py` | structural self-check of the delivered ladder |
| `sweep.sh` | 3-seed holdout sweep driver used for every configuration decision |

Reproduce: `python prepare_data.py --split train && python train.py --epochs 20 --widths 128 384 512 --seed 0`

## Method

An airbench-lineage convnet, trained from scratch — no pretrained weights, no
external data, no test data in the training path.

**Architecture** (6.08 M params at the chosen width): a frozen 2×2 conv
initialized to the ZCA-whitening basis of training-set patches (12 eigenvectors
stored as ± pairs → 24 channels), GELU, then three `conv3×3 → maxpool2 → BN → GELU →
conv3×3 → BN → GELU` groups at widths 128/384/512, global max-pool, and a bias-free
linear head scaled by 1/9. BN scales are frozen at 1 (redundant with the following
conv) and convs get a partial-identity (Dirac) init, which is what lets a net with no
residual connections converge in 20 epochs.

**Throughput.** The whole train split lives on the GPU as one padded fp16 tensor, so
there is no `DataLoader` and no host round-trip per step. Augmentation (±2 px random
translate + horizontal mirror) is two `gather`s plus a `where`, giving every image an
independent offset without looping over the 25 shift combinations. Net is fp16 with
fp32 BN, channels-last. Result: **0.36 s/epoch** for 50k images.

**Optimization.** SGD + Nesterov (momentum 0.85), sum-reduced cross-entropy with 0.2
label smoothing, LR specified per-1024-examples (11.5) and divided by the factor by
which momentum inflates the step, decoupled weight decay 0.0153, biases on a 64× LR.
Triangular schedule: 0.2× → 1.0× peak at 23% of steps → 0.07× at the end. The
whitening layer's bias trains for 3 epochs then freezes.

**TTA is part of the model.** `forward` is the deployed inference function and folds
6-way TTA (mirror, plus two 1-px diagonal translates each mirrored) into itself, so
the verifier gets it for free. The training loop calls `core()`, the single
un-augmented pass.

## Input-convention handling (the documented failure mode)

The verifier feeds tensors that are **already** standardized by
`ToTensor() + Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))`. Training
feeds the network exactly the same quantity, so `forward` does no normalization of
its own. As a belt-and-braces guard, `_standardize` detects `[0,255]` and `[0,1]`
inputs by magnitude and rescales those, but for the verifier's convention that branch
is dead code — standardized CIFAR-10 is bounded by [-1.99, 2.13], so the >3.0 and
"max ≤ 1 and min ≥ 0" tests cannot fire.

This was confirmed, not assumed. `evaluate.py --check-equivalence` builds the test
set through torchvision's `ImageFolder` + `DataLoader` with the verifier's exact
transform and compares it against the cached fast path: **max |Δx| = 4.8e-7, labels
identical, N = 10000**. Both paths score the ladder identically (epoch 19 = 94.33%,
epoch 20 = 94.76%), and the observed input range printed by the evaluator is
`[-1.989, 2.126]` — i.e. the real verifier convention.

## How configurations were chosen

Every decision was made on a **5,000-image slice held out of the train split**, at 3
seeds per config; the split is fixed (RNG 1234) and independent of the training seed
so all configs see the same images. The test split was not read until the ladder was
already on disk. Noise floor: a 5k holdout has a ±0.31 pp binomial SE, so single-run
gaps below ~0.7 pp are not signal.

| config (holdout, 3 seeds) | final acc | s/epoch |
| --- | --- | --- |
| 16 ep, 64/256/256 | 94.15 ±0.12 | 0.230 |
| 24 ep, 64/256/256 | 94.36 ±0.14 | 0.228 |
| 16 ep, 96/320/384 | 94.62 ±0.20 | 0.339 |
| 20 ep, 96/320/384 | 94.65 ±0.05 | 0.338 |
| 12 ep, 128/384/512 | 94.41 ±0.15 | 0.409 |
| 16 ep, 128/384/512 | 94.81 ±0.23 | 0.406 |
| **20 ep, 128/384/512** | **95.11 ±0.16** | **0.404** |
| 24 ep, 128/384/512 | 94.98 ±0.10 | 0.403 |
| 16 ep, 192/512/768 | 94.99 ±0.18 | 0.664 |

Width buys much more than depth of schedule: +0.66 pp going 64→128 at fixed epochs,
while 16→24 epochs at fixed width buys +0.21 pp. Past 128/384/512 it flattens (192
wide is no better and 64% slower per epoch), and 24 epochs is no better than 20.

Ablations at the chosen config (3 seeds, holdout):

| variant | final acc | Δ |
| --- | --- | --- |
| chosen | 95.11 ±0.16 | — |
| no TTA | 94.25 ±0.18 | **−0.86** |
| random (trainable) 2×2 conv instead of frozen whitening | 94.45 ±0.19 | **−0.66** |
| peak LR 9 instead of 11.5 | 94.98 ±0.04 | −0.13 (noise) |
| peak LR 14 instead of 11.5 | 94.95 ±0.08 | −0.16 (noise) |

TTA and the whitening init are both worth keeping — each clears the noise floor by
2–3×. LR 11.5 sits on a flat optimum, so it was left alone rather than tuned into
the noise.

## Timing honesty

The scored clock is rebuilt from checkpoint mtimes, so everything between the epoch-1
and epoch-N saves is real work. Two choices follow from that:

- **No `torch.compile`.** Its ~15 s warmup would sit *before* the epoch-1 save and
  would shrink measured t\* without making training one millisecond faster. cuDNN
  autotuning is likewise left to happen inside epoch 1 rather than pre-warmed.
- **The graded ladder was trained in place.** Copying a ladder in from a scratch
  directory would stamp all 20 files with near-identical mtimes and destroy the
  timing signal, so the final run wrote straight into `periodic_checkpoints/` after
  the previous ladder was deleted. The measured per-epoch spacing is 0.356–0.380 s
  (mean 0.360), i.e. genuine and even.

`agent_reported_metrics.json` reports `wall_clock_sec = 9.101`, the **total** wall
clock of the graded run from process start to the last checkpoint. The epoch-1 →
epoch-20 span is 6.833 s, so the reported figure is the larger, inclusive one.
`accuracy = 0.9476` is the final checkpoint's independently measured test accuracy.

The model is scripted once before training; the `ScriptModule` shares parameter
storage with the eager module, so each per-epoch `.save()` serializes that epoch's
live weights. This is asserted, not assumed — `verify_ladder.py` check [4] confirms
no two consecutive checkpoints hold identical head weights, and the monotone
accuracy curve (76.0% → 94.8%) is itself the evidence.

## Self-check output

```
[1] 20 checkpoints, epochs 1..20  contiguous-from-1: True
[2] span 6.83s  per-epoch min/mean/max = 0.356/0.360/0.380s  monotone: True
[3] all 20 load, return float32 (512,10), finite
[4] consecutive-identical weight pairs: 0 (must be 0)
[5] checkpoint.pt == ckpt_epoch_0020.pt: True
LADDER OK
```

## Robustness

The graded ladder is **seed 0, pre-committed** before any test-split number for this
architecture was looked at. Seeds 1 and 2 were run afterwards on the full 50k purely
to measure variance — they were *not* used to pick a winner, which would be test-set
selection:

| seed | epoch 18 | epoch 19 | epoch 20 |
| --- | --- | --- | --- |
| **0 (graded)** | 93.90 | **94.33** | **94.76** |
| 1 | 93.52 | 94.39 | 94.67 |
| 2 | 93.86 | 94.35 | 94.95 |

All three cross at epoch 19 within a 0.06 pp band and finish at 94.79 ±0.12. The
crossing is a property of the recipe, not of a lucky seed. Note also that the
delivered run has **two** checkpoints above the line: even if epoch 19 were re-scored
just under 94%, epoch 20 (94.76%, a 0.76 pp margin) crosses at t\* ≈ 7.2 s, which is
still S_outcome = 1.0. There is no realistic path to a miss.

One honest caveat: held-out *train* images read ~0.35 pp optimistic relative to the
real test split (95.11% holdout vs 94.76% test). That gap is consistent across
configs, so it does not affect the ranking the sweep was used for, but the holdout
numbers above should not be read as test-accuracy predictions.

## Constraints

No pretrained weights and no weight-zoo initialization: every parameter is either
randomly initialized or, for the frozen 2×2 conv, computed from the covariance of
training-split patches. No external data. `train.py` never opens the test directory
or the test cache; the test split is read only by `evaluate.py`, which scores
checkpoints that are already on disk and therefore cannot influence any weight or any
hyperparameter choice. No file timestamps were manipulated — `save()` set every mtime
in the ladder, and `/workspace/checkpoint.pt` is a plain `shutil.copy` of epoch 20.
