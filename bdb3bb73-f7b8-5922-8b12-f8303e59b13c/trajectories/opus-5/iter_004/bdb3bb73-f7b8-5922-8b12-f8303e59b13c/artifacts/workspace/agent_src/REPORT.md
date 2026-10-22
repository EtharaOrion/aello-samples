# CIFAR-10 → 94% top-1 on 1×H100

**Delivered:** 24 checkpoints in `/workspace/periodic_checkpoints/` (contiguous `ckpt_epoch_0001..0024`)
plus `/workspace/checkpoint.pt` (byte-identical copy of epoch 24).

| quantity | value |
|---|---|
| first checkpoint ≥ 94% (verifier-faithful re-eval) | **epoch 23, 94.15%** |
| **t\*** = (mtime[23] − mtime[1]) × 23/22 | **16.58 s** |
| S_outcome = (70 − t\*)/(70 − 28), clamped | **1.0000** |
| final checkpoint (epoch 24) accuracy | **94.86%** |
| total wall clock of the graded run | 19.03 s |
| mean epoch (incl. per-epoch checkpoint save) | 0.721 s |
| contract checks passed (`validate.py`) | **20 / 20** |

Reproduced on three seeds — all cross at epoch 23 with t\* = 16.58–16.60 s
(94.15% / 94.39% / 94.71%). Seed 0 was **pre-committed before any test number was
looked at**, and it happened to be the *weakest* of the three; it is nevertheless
the graded ladder, because picking a seed by test accuracy would be test-set
selection.

## Method

Architecture (airbench lineage, ~7 M params): a **frozen 2×2 ZCA whitening conv**
initialised from the train-set patch covariance → three `conv → maxpool → BN →
GELU → conv → BN → GELU` groups of width 128/384/512 → 3×3 maxpool → bias-free
linear head scaled by 1/9. BN scales are frozen at 1 (the following conv absorbs
them); only BN shifts train.

Training: SGD + Nesterov (momentum 0.85), batch 1024, 24 epochs, peak LR 14.0 with
a 0.2 → 1.0 → 0.07 triangular schedule peaking at 23% of steps, weight decay
0.0153, label smoothing 0.2; biases/BN-shifts get 64× LR at matched effective
decay. LR and WD are normalised by `batch_size·(1+1/(1−momentum))` so the tuned
constants survive batch-size/momentum changes. The net trains in fp16.

Data path: all 50 000 train images sit on the GPU as uint8, standardised once into
a reflect-padded fp16 tensor. Random ±2 px translation (two `gather`s), horizontal
flip, and shuffling all run on device — the training loop does zero host↔device
traffic and no `DataLoader`.

Inference: checkpoints are exported as a scripted `DeployNet` that (a) harmonises
the input range and (b) averages logits over mirror + two 1-px diagonal
translations (6 forward passes). TTA lives *inside* `forward`, so the verifier
gets it for free.

## Protocol: how the test split was and wasn't used

Every model-selection decision was made on a **held-out 5 000-image slice of the
train split** (500/class, deterministic stride-10 pick); the other 45 000 were the
sweep's training set, identical across configurations so comparisons are paired.
The training loop performs **no evaluation of any kind** — accuracy is only ever
computed offline from saved weights.

The binomial SE of a 5 000-image estimate at 94% is 0.34%, so the harness prints
"treat gaps < 0.67% (2 SE) as noise" above every table, and I applied that rule
even when it cost me the nominally-best config (the last row of the table below). The test
split was read only by `verify_ladder.py`, to measure the finished artefact the
same way the verifier does.

## Results

Held-out accuracy, mean over seeds. **Bold** = adopted.

| experiment | result | verdict |
|---|---|---|
| width 64/256/256 → 96/320/384 → **128/384/512** | 93.58 → 94.51 → **94.72** | real (+1.14%, ≫ noise) |
| whitening: **ZCA frozen** / random frozen / random learned | **95.07** / 94.02 / 94.02 | real (+1.05%, ≫ noise) |
| TTA level 0 / 1 / **2** (paired, same weights) | 94.16 / 94.79 / **94.93** | real (+0.77%, same sign on 3/3 seeds) |
| peak LR 9.0 / **11.5→14.0** | 95.09 / 95.19 | within noise; took 14.0 (nominally best in 2 sweeps) |
| epochs 18 / **24** / 30 (3 seeds) | 94.75 / **94.97** / 95.01 | within noise → took the *cheapest* |
| cutout 8 | 94.69 vs 95.01 | no gain; dropped |
| pad 4 vs **pad 2** | 94.91 vs 94.80 | within noise; kept simpler |
| label smoothing 0.1 vs **0.2** | 94.79 vs 94.80 | within noise |
| weight decay 0.022 vs **0.0153** | 94.82 vs 94.80 | within noise |
| BN momentum 0.6 / 0.3 / 0.1 | 95.19 / 95.22 / 95.24 | within noise; kept 0.6 |
| **wider 192/512/640 @ 20 ep** vs 128/384/512 @ 24 ep | 95.26 vs 95.00 | **+0.26% < 0.67% → not adopted** |

Only three factors — width, ZCA whitening, and TTA — cleared the noise floor.
Everything else is a plateau, and reporting it as such is the honest read: the
2-seed sweep had `lr14 @ 24 ep` at 95.19%, but re-running the same config at 3
seeds gave 94.97%, a 0.22% swing from seed count alone.

### One real bug, found and fixed

The first implementation multiplied the base LR by 0.2 *and* had the LambdaLR
schedule ramp from 0.2, applying the warmup floor twice — peak LR was ~5× too
low. Symptom: 34 epochs (93.58%) scored *worse* than 24 (93.94%), which is
backwards and is what prompted the audit. After the fix, 16 epochs beat the old
24. `speedrun.py:294` now carries a comment pinning the ownership of that factor.

### Deliberately not done

* **`torch.compile`** — roughly halves per-epoch time, but its ~15 s warm-up sits
  *outside* the timed window (t\* is rebuilt from checkpoint mtimes). Using it
  would have cut reported t\* without making the run genuinely faster. Since
  S_outcome is already clamped at 1.0 with 1.7× headroom (16.6 s vs the 28 s
  reference), it buys nothing real and I left it off.
* **Chasing accuracy past ~95%** — S_outcome saturates at t\* ≤ 28 s, so accuracy
  above the crossing threshold only buys robustness, not score. The structure of
  the triangular schedule means accuracy jumps only in the final ~3 epochs, so
  the crossing epoch is inherently near the end of *any* schedule length; margin
  therefore comes from final accuracy (94.86%, i.e. +0.86%), not from adding
  epochs.

## Verification

`validate.py` is a hard pass/fail suite over the graded ladder; all 20 checks pass.

* ladder non-empty, epochs contiguous 1..24, `ckpt_epoch_0001.pt` present
* mtimes strictly increasing; per-epoch gaps uniform (0.720–0.724 s)
* all 24 files have distinct MD5s (no stale/duplicated checkpoint)
* `checkpoint.pt` MD5 == epoch-24 MD5
* `.npy` cache is **bit-identical** to `ImageFolder + ToTensor + Normalize`
  (max |Δ| = 0.00e+00), so `verify_ladder.py` really is the verifier's pipeline
* checkpoints `torch.jit.load` on **GPU and CPU**; output is (512,10) float32 and finite
* predictions are **batch-size independent** (512 / 256 / 1000 agree exactly)
* predictions are **input-convention invariant**: standardized, [0,1] and [0,255]
  batches all give the same accuracy. `DeployNet._to_standard` passes standardized
  input through untouched — the double-normalisation trap in the brief — and only
  rescales when the observed range says otherwise
* epoch 1 and epoch 24 are genuinely different models (max |Δlogit| = 7.7)

Constraints: trained from scratch (no pretrained weights are loaded anywhere — the
only tensors read from disk are the two `.npy` caches of the CIFAR-10 *train*
PNGs), no external data, no test-split input during training, and no file
timestamp is ever modified — `save()` sets every mtime and `/workspace/checkpoint.pt`
is written with `shutil.copy`.

## Files

| file | role |
|---|---|
| `prepare_data.py` | one-off PNG → uint8 `.npy` decode (1.3 s) |
| `speedrun.py` | model, GPU data pipeline, `DeployNet`, training loop |
| `train.py` | entry point; clears any old ladder, trains, writes checkpoints |
| `sweep.py` | held-out model selection with the noise floor printed |
| `ablate_tta.py` | paired TTA ablation |
| `verify_ladder.py` | verifier-faithful re-eval + t\* reconstruction |
| `validate.py` | 20 hard contract checks |
| `sweep_*.json`, `verify_final.json`, `last_run_stats.json` | raw results |

Reproduce the graded ladder:

```bash
python prepare_data.py
python train.py --out /workspace/periodic_checkpoints --final /workspace/checkpoint.pt \
  --config "{'widths':(128,384,512),'lr':14.0,'epochs':24,'whiten_init':'zca','cutout':0,'pad':2,'tta':2,'seed':0}"
python verify_ladder.py && python validate.py
```
