# CIFAR-10 → 94 % top-1 on 1×H100 — method and decision log

## Result

| | |
|---|---|
| Graded ladder | 32 contiguous scripted checkpoints, `periodic_checkpoints/ckpt_epoch_0001..0032.pt` |
| First checkpoint ≥ 94 % (verifier pipeline) | **epoch 29 → 94.27 %** |
| `t*` = (mtime[29] − mtime[1]) · 29/28 | **12.27 s** (budget: 28 s ⇒ `S_outcome` saturates at 1.0) |
| Final checkpoint | **95.06 %** |
| Full training-run wall clock | 17.13 s (13.12 s of it is the ladder span) |
| Model | 6.08 M params, trained from scratch, CIFAR-10 train split only |

## How the objective was read

`S_outcome = clamp((70 − t*)/(70 − 28), 0, 1)`. The *first* full run of the
session already produced `t* = 13.3 s`, i.e. S_outcome was pinned at 1.0 with
more than 2× headroom. Shaving further seconds buys literally nothing. The only
way left to lose reward is a ladder that fails to cross 94 % when the verifier
re-evaluates it. So the session objective was set as:

> maximise accuracy **margin** at the crossing epoch and at the final epoch,
> subject to keeping `t*` comfortably below 28 s

and every subsequent decision followed from that, rather than from chasing speed.

## Setup

- **Data** — `cache_data.py` decodes the 60 k PNGs once into uint8 NCHW `.npy`.
  Class order = sorted class-directory names = `ImageFolder`'s order, so label
  ids match the verifier's.
- **No test leakage** — every hyperparameter decision is made on a *fixed* 5 000-image
  holdout carved from the 50 k train split (`HOLDOUT_SEED = 1234`, never changed).
  `verify.py` is the only code that reads the test split; it runs only to confirm
  that a *finished* ladder loads and scores as expected. No test-derived number
  ever fed back into a configuration choice.
- **Model** (`model.py`) — frozen 2×2 eigen-whitening conv (3→24, kept in ± pairs
  so the following GELU sees both signs) → three `conv–pool–BN–GELU–conv–BN–GELU`
  groups at widths 128/384/512 → global max-pool → bias-free linear → `× logit_scale`.
  The whitening init is checked numerically: per-channel output std ≈ 1.00,
  |mean| < 0.02.
- **Augmentation** is entirely on-GPU — images are reflect-padded once to 38×38,
  then each batch costs two `gather`s (random ±3 px translate) and one `where`
  (mirror). No DataLoader, no host↔device traffic in the training loop.
- **TTA** — 6-way ({no shift, ±1 px} × {identity, mirror}) folded into `forward`,
  so the verifier gets it just by calling `model(x)`.
- **Recipe** — SGD + Nesterov (m = 0.85), bs 1024, label smoothing 0.2, linear
  warmup 15 % then linear decay to 2 % of peak, bf16 autocast, channels-last,
  no weight decay on 1-D params.

## Two implementation details that matter for the verifier's clock

1. **Checkpoint cost lands inside the measured interval.** `torch.jit.script()`
   costs ~0.4 s; doing it per epoch would have roughly doubled the effective
   epoch time *as measured by the verifier*. `CheckpointWriter` scripts one
   eval-mode fp32 copy **before** the timed loop, then each epoch only restamps
   its `state_dict` and re-serializes → 0.42 s/epoch with checkpointing vs
   0.36 s/epoch without.
2. **Warmup happens before epoch 1.** cudnn autotuning and allocator warmup run
   ahead of the loop, so no epoch is anomalously slow or fast. Measured
   inter-checkpoint gaps: min 0.420 s, max 0.424 s, mean 0.423 s over 31 gaps.
   Each epoch is an *exact* single pass over all 50 000 images — the ragged final
   batch is kept, not dropped (an earlier version dropped an 848-image tail,
   which made epochs 1.7 % cheaper than a true pass; that was removed).

## Noise floor — established before drawing conclusions

- The 5 000-image holdout has binomial s.e. `sqrt(0.94·0.06/5000) = 0.34 pp`.
- Re-running one *identical* config (same seed) gave 93.92 % then 94.18 %: cudnn
  autotuning and non-deterministic reduction order alone move the answer 0.26 pp.
- Three seeds of the base config: 94.38 / 94.08 / 94.06 → **mean 94.17, sd 0.18 pp**.

⇒ Single-seed differences below ~0.5 pp are not evidence. Coarse sweeps were
therefore read as "which *region* is good", and finalists ranked by a 3-seed mean.

## Decision log

All val numbers are the held-out 5 k, models trained on the remaining 45 k.

| # | Question | Evidence | Decision |
|---|---|---|---|
| 0 | Baseline sanity | lr 0.4 / 16 ep → 88.3 %, val oscillating wildly | LR far too high |
| 1 | LR (16 ep) | 0.02→92.4 · 0.05→93.2 · 0.1→93.1 · 0.2→91.3 · 0.3→90.2 | clean unimodal peak at 0.05–0.1 |
| 2 | Epoch scaling | lr 0.05: 16→93.2, 24→93.6, 32→93.9; lr 0.1 flat | still improving at 32 ep |
| 3 | **Safety ladder** | lr 0.05 / 32 ep / full split → test **94.04 %**, `t*` 13.3 s | promoted at t+9 min so a valid ladder existed early |
| 4 | LR × WD (12 cfg) | good region is `lr·wd ≈ 0.001–0.0025`; wd 0.12 collapses (91.8–92.6) | lr 0.05, wd 0.02 |
| 5 | One-factor screen (13 cfg) | `logit_scale` 0.05→93.6, 0.11→94.2, 0.25→94.6, **0.5→94.8**; `pad` 3→94.7; momentum / warmup / final-LR / label-smoothing all within noise | push `logit_scale`, try pad 3, leave the rest |
| 6 | logit_scale × LR at pad 3 | lr 0.05: 0.5→**95.18**, 1.0→95.16, 2.0→94.68, 4.0→93.34; lr 0.02 collapses to ~93 | logit_scale 0.5; LR optimum shifted up with pad 3 |
| 7 | LR × WD re-check | lr 0.05/0.08/0.12 at wd 0.02 → 95.18/95.10/95.00 (flat plateau); wd 0.045 worse | keep lr 0.05, wd 0.02 — broad optimum, not a knife edge |
| 8 | cutout & epochs | cutout 8 helps only at short schedules (20 ep: +0.4 pp) and *hurts* at 32 ep | cutout 0 |
| 9 | **Finalists, 3 seeds** | 32 ep → 95.18/95.00/95.20 = **95.13 ± 0.11**; 40 ep → 94.84/94.96/95.08 = 94.96 ± 0.12 | 32 epochs — no worse and 25 % cheaper |

**Promoted configuration**: `epochs 32, bs 1024, lr 0.05, wd 0.02, momentum 0.85,
label_smoothing 0.2, warmup 0.15, final_lr_frac 0.02, logit_scale 0.5, pad 3,
cutout 0, widths (128, 384, 512)`.

### Confirmatory 2×2 ablation — one decision replicated, one refuted

Decisions 5–6 (`logit_scale`, `pad`) came from *single-seed* screens, so they
were re-run afterwards as a full 2×2 factorial, 3 seeds per cell, all with the
identical (final) code path — `stageG.log`, `stageH.log`:

| `logit_scale` | `pad` | seeds 0/1/2 (val %) | mean | sd |
|---|---|---|---|---|
| 0.11 | 2 | 94.14 / 94.46 / 94.42 | 94.34 | 0.17 |
| 0.11 | 3 | 94.70 / 94.64 / 94.22 | 94.52 | 0.26 |
| 0.5 | 2 | 94.98 / 95.30 / 94.56 | 94.95 | 0.37 |
| 0.5 | 3 | 94.86 / 95.04 / 95.04 | **94.98** | 0.10 |

- **Main effect of `logit_scale` 0.11 → 0.5: +0.54 pp.** With pooled sd ≈ 0.25 pp
  the standard error of that contrast is ≈ 0.14 pp, so this is ~3.7σ — **real**,
  and it is the single largest win of the session.
- **Main effect of `pad` 2 → 3: +0.11 pp — ~0.7σ, i.e. not significant.** The
  +0.55 pp that the one-seed screen reported for `pad 3` **did not replicate**;
  it was mostly noise. `pad 3` is retained in the promoted config only because it
  is nominally ≥ `pad 2` in both cells and the promoted ladder is already
  verified at 95.06 %; it is not claimed as an improvement.

This is exactly the failure mode the noise-floor measurement was meant to catch,
and it is why finalists were ranked on 3-seed means rather than single runs.

## Verification actually performed

`verify.py` reproduces the verifier exactly: `ImageFolder` over the test PNGs,
`ToTensor() + Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))`,
float32 NCHW batches of 512, `torch.jit.load(path).eval()`, `argmax(1)`. Every
one of the 32 graded checkpoints is scored this way (`verify_final.json`).

Additional checks on the promoted `checkpoint.pt`:

| Check | Result |
|---|---|
| Standardized input (the verifier's path) | 95.02 % |
| `[0,1]` input | 95.02 % |
| `[0,255]` input | 95.02 % |
| Predictions at bs 512 vs bs 128 | 100 % agreement |
| Output dtype / device | float32 / cuda |
| Ladder epoch numbers contiguous 1..32 | yes |

The `_to_standard` guard branches on the batch min/max. A 512-image standardized
batch always contains pixels below −0.5 (black pixels map to −1.99), so the
standardized path can never be misrouted into a double normalization — the
failure mode that silently collapses accuracy under the verifier.

**Reproducibility of the promoted config on the test split**: three independent
full-split runs of the frozen configuration scored 95.02 / 95.02 / 95.06 %.
The configuration was fixed *before* any of these runs from holdout evidence
alone; these numbers are reporting, not selection.

## Reported metrics

`agent_reported_metrics.json` = `{"accuracy": 0.9506, "wall_clock_sec": 17.13}`.
`wall_clock_sec` is the **whole** graded training run — process launch through
the last checkpoint write — which is deliberately larger than the 13.12 s
checkpoint span (setup, data load, whitening init, warmup, epoch 1).

## Files

| | |
|---|---|
| `cache_data.py` | PNG → uint8 `.npy` cache |
| `model.py` | network, whitening init, TTA, input-convention guard |
| `train.py` | training loop, GPU augmentation, `CheckpointWriter` |
| `sweep.py` | in-process sweep driver → `sweeps.jsonl` (every run ever made) |
| `verify.py` | verifier-replica evaluation of a whole ladder |
| `stage{B..H}.log`, `sweeps.jsonl` | raw sweep output |
| `verify_final.json` | per-epoch test accuracy + `t*` for the graded ladder |
