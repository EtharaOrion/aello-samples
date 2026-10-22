# CIFAR-10 → 94% top-1 from scratch on 1×H100

**Result (independently re-checked with the verifier's own pipeline):**

| quantity | value |
|---|---|
| first checkpoint ≥ 94% | `ckpt_epoch_0021.pt` — **94.05%** |
| `t_star` (verifier formula, from checkpoint mtimes) | **8.92 s** |
| final checkpoint (`ckpt_epoch_0024.pt`) | **94.93%** |
| ladder | 24 scripted checkpoints, epochs 1…24, one genuine pass each |
| total training wall clock (run start → last checkpoint) | **11.81 s** |
| reference / baseline anchors | 28 s / 70 s → `S_outcome = 1.0` |

Reproducibility: three full runs (seeds 0/1/2) all crossed 94% at epoch 21–22,
`t_star` = 8.92 / 9.33 / 9.34 s, final accuracy 94.93 / 94.73 / 94.68 %.
The graded ladder is seed 0, which was trained and promoted *before* the other
two were run — the reproducibility runs went to a scratch directory and were
never eligible for promotion, so nothing was cherry-picked.

## Method

An airbench-lineage convnet (Keller Jordan's CIFAR-10 speedrun family), tuned here
on a held-out slice of the training split.

* **Frozen whitening front-end.** A 2×2 convolution whose weights are the
  eigenvectors of the training set's 2×2 patch covariance, scaled by
  `1/sqrt(λ+ε)` and stored as `±` pairs (24 output channels). Weights are frozen;
  only its bias trains, and only for the first 3 epochs. This conditions the
  first layer for free and is what makes a ~20-epoch schedule viable.
* **Three conv groups** (128 → 384 → 512), each `conv3×3 → maxpool2 → BN → GELU →
  conv3×3 → BN → GELU`, then a 3×3 max-pool to 1×1 and a bias-free linear head
  (zero-initialised, output scaled by 1/9).
* **BatchNorm with a frozen unit scale** and a learned shift only — this is what
  lets the run tolerate the very large learning rate.
* **Optimiser:** Nesterov SGD, momentum 0.85, sum-reduced cross-entropy with
  label smoothing 0.2. `--lr`/`--wd` are quoted per 1024 examples and divided by
  the Nesterov step-amplification factor `1 + 1/(1-m)`, so their meaning does not
  drift when batch size or momentum change. Schedule: linear warmup over the
  first 23% of steps, then linear decay to 7% of peak. BN shifts and the whitening
  bias get 64× the weight learning rate.
* **Data path:** all 50 000 training images live on the GPU as one standardized
  tensor. Each epoch's augmentation (random ±2 px translation via two `gather`s
  off a reflect-padded copy, random horizontal flip) is generated for the whole
  epoch in a handful of kernels, so the input pipeline never stalls the GPU.
  Training runs in bf16 autocast on channels-last fp32 master weights.
* **Test-time augmentation is folded into the model.** `forward` averages
  mirror-TTA over the image and two 1-pixel diagonal translations (6 passes).
  The checkpoint is therefore self-contained: the verifier gets the TTA without
  doing anything special.

Batch size 1000 divides 50 000 exactly: every epoch is exactly 50 steps over the
full training split, no dropped tail, no oversized first epoch.

### Input-scale robustness

The verifier feeds tensors that are **already standardized** (`ToTensor` →
`Normalize`, range ≈ [-2, 2.1]). `forward` canonicalises the input instead of
assuming: if `max > 3` it treats the batch as [0,255], else if `min ≥ -0.05` as
[0,1], otherwise as already standardized. `check_input_conventions.py` confirms
the promoted checkpoint scores **94.93% under all three conventions**, and that
the decision is batch-size independent. This is the failure mode that silently
collapses accuracy to 10–30% under the verifier, so it is checked explicitly
rather than assumed.

## How the tuning was done (no test-split leakage)

`train.py` and `sweep.py` never open the test split. Every hyper-parameter
decision was scored on a fixed, deterministic 5 000-image holdout carved out of
the 50 000 training images (`--holdout 5000`, seeded independently of the run
seed, so all configurations see the same validation images). The test split is
read only by `evaluate_ladder.py` / `check_input_conventions.py`, after
training, to reproduce the verifier's measurement — which the task explicitly
asks for before promoting a ladder.

Sweep results (all runs appended to `sweep_results.jsonl`):

| stage | finding |
|---|---|
| 1 (16 one-factor probes) | width is the dominant lever (+1.3%); `logit_scale` 1.0 is catastrophic (81%), 0.25 helps; lower `wd`, higher `lr`, `head-init=default` all mildly positive; cutout and `warmup 0.1` neutral-to-negative |
| 2 (combinations on the wide net) | `wd=0.008` best single change; the stage-1 gains did **not** stack — combinations landed inside the noise band |
| 3 (repeats) | seed-to-seed σ ≈ 0.1%; **32 epochs is not better than 24** (94.64 vs 94.75) — the schedule saturates, so the extra 8 epochs were dropped |
| 4 (TTA) | none 93.87% → mirror 94.36% → mirror+translate **94.59%**, consistent across seeds |

A methodological caveat worth recording: the *same* config and seed measured
94.76% in one sweep process and 94.06% in another, while repeats *within* one
process spread by only ~0.1%. Cross-process comparisons are therefore not
trustworthy here (nondeterministic kernels at a very large learning rate), and
every decision above was made from comparisons made inside a single sweep
process.

## Deliberate non-optimisations

* **`torch.compile` is off for the graded run.** It works and it is faster per
  epoch (0.184 s vs 0.424 s), but compilation costs ~15 s and happens before
  epoch 1 — i.e. *outside* the window the verifier measures (`t_star` is rebuilt
  from checkpoint 1 → checkpoint N). Turning it on would cut the reported
  `t_star` to ~4.6 s while making the actual job take ~20 s instead of 11.8 s.
  Since `S_outcome` is already saturated (8.92 s against a 28 s reference),
  that trade buys a smaller number by moving work out of view rather than by
  doing less work, so the graded run is eager. `--compile 1` remains available.
* **Accuracy over further speed.** `t_star = 8.92 s` is already ~3× under the
  reference anchor and `S_outcome` is clamped at 1.0, so additional speed is
  worth nothing while margin above 94% is worth something (it is what makes the
  crossing epoch reproducible). The 24-epoch schedule was chosen because it is
  the point where more epochs stopped helping — not because time ran out.

## Files

| file | role |
|---|---|
| `prepare_data.py` | decode the CIFAR-10 PNG folders once into uint8 `.npy` caches (order/labels taken from `ImageFolder`, so they match the verifier) |
| `data.py` | GPU-resident split + whole-epoch on-device augmentation |
| `model.py` | `CifarNet`, whitening initialisation, TorchScript-safe `forward` (input canonicalisation + TTA) |
| `train.py` | training loop + per-epoch scripted checkpoint ladder |
| `sweep.py` | held-out hyper-parameter sweeps (training split only) |
| `evaluate_ladder.py` | replica of the verifier: `torch.jit.load` → 512-batch standardized test tensors → `argmax`, plus the `t_star` formula from mtimes |
| `check_input_conventions.py` | standardized / [0,1] / [0,255] and batch-size checks |
| `final_check.py` | asserts every deliverable invariant (contiguity, TorchScript loadability, increasing mtimes, hardlink, metrics consistency) |
| `ladder_eval.json` | per-epoch verified accuracy of the promoted ladder |
| `scratch_eval_seed{1,2}.json` | the two reproducibility runs (their checkpoints were scratch-only and have been deleted, so exactly one ladder exists on disk) |

## Reproduce

```bash
python prepare_data.py
python train.py --epochs 24 --widths 128 384 512 --wd 0.008 --tta 2 \
    --out-dir /workspace/periodic_checkpoints --final
python evaluate_ladder.py --dir /workspace/periodic_checkpoints
python check_input_conventions.py --ckpt /workspace/checkpoint.pt
python final_check.py
```

Timeline of this session: a valid 32-epoch ladder (t\* 12.70 s, 94.72%) was on
disk and verified within the first ~12 minutes as a floor; the sweeps ran after
that, and the ladder was only replaced once the 24-epoch recipe had been
measured end-to-end and shown to be better on both axes.

`--final` hardlinks the last periodic checkpoint to `/workspace/checkpoint.pt`
(`os.link`, so no timestamp is written by hand anywhere in this repo; every
checkpoint mtime is whatever `ScriptModule.save()` set during the run).
