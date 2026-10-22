# CIFAR-10 → 94% top-1 from scratch, 1×H100

**Delivered ladder** (`/workspace/periodic_checkpoints/`, 28 scripted checkpoints, one per epoch):

| quantity | value |
|---|---|
| first checkpoint ≥ 94% (independent re-eval, verifier pipeline) | `ckpt_epoch_0027.pt`, **94.64%** |
| grader clock at that epoch, `(save[27]−save[1])·27/26` | **13.16 s** (reference 28 s, baseline 70 s) |
| last checkpoint (`ckpt_epoch_0028.pt` = `/workspace/checkpoint.pt`) | **94.68%** |
| total training wall clock (process start → last checkpoint) | 17.40 s (13.15 s of it inside the ladder) |
| parameters / epochs / mean epoch | 6.08 M / 28 / 0.487 s |

Trained from scratch on the 50 000-image CIFAR-10 train split only. No pretrained weights, no
external data, no test images anywhere in training or in hyperparameter selection.

## 1. Method

**Data.** The PNG folders are decoded once into a uint8 GPU tensor (`prep_data.py`); the training
loop never reads the filesystem. Images are standardised with the same constants the grader uses
(mean 0.4914/0.4822/0.4465, std 0.2470/0.2435/0.2616) and kept reflect-padded in fp16.

**Model** (`train.py:Net`, 6.08 M params).

* frozen 2×2 ZCA-whitening conv, 24 filters = ±(eigenvectors of the train-patch covariance scaled
  by 1/√(λ+5e-4)); the ± pair keeps both signs alive through the following GELU;
* three `conv3×3 → maxpool2 → BN → GELU → conv3×3 → BN → GELU` groups, widths 128/384/512;
* global `maxpool3` → bias-free linear head, logits scaled by 1/9;
* BatchNorm scales are frozen at 1 (only the shift is learned), head is zero-initialised.

**Augmentation**, entirely on-device: random ±2 px translate (two `gather`s on the pre-padded
tensor), random horizontal flip (one `where`), and a 12×12 cutout (`masked_fill`).

**Optimisation.** SGD + Nesterov (momentum 0.85), batch 500 → exactly 100 steps/epoch, i.e. one
genuine pass over the 50 000 training images per epoch. LR 0.30 (×8 for BN shifts and the whitening
bias, which are excluded from weight decay), triangular schedule with 15% warmup decaying linearly
to 0, weight decay 5e-3, label smoothing 0.2, bf16 autocast, channels_last, TF32, cudnn benchmark.

**Test-time augmentation is inside `forward`** so the grader gets it for free: mirror plus a 1-px
translate pair, 6 views combined as `0.6·(x + mirror) + 0.4·mean(shifted pairs)`.

**Checkpointing.** The ScriptModule is built once and shares parameter storage with the eager
model, so each epoch costs one `scripted.save(...)` (~20 ms) rather than a re-script. Nothing else
happens between epochs — no evaluation inside the graded run — so the ladder's save times are the
training time and nothing else.

## 2. Verifier contract

`forward` takes float32 NCHW input that is **already standardised** (`ToTensor` + `Normalize`) and
does not normalise again. `verify.py` re-checks each delivered checkpoint through exactly the
grader's path — `torch.jit.load(p).eval()`, batch 512 of the real 32×32 PNGs under
`Compose([ToTensor(), Normalize(...)])`, `argmax(1)` — which is where every accuracy number in this
file for the delivered ladder comes from (observed input range [-1.99, 2.13]). `check_artifacts.py`
additionally confirms: epochs contiguous 1..28, every checkpoint loads and returns finite (512, 10)
logits, save times strictly increasing, `checkpoint.pt` byte-identical (SHA-256) to
`ckpt_epoch_0028.pt` (hardlinked), and the reported wall clock ≥ the ladder span. All pass.

## 3. How the configuration was chosen

Every decision below was made on a **fixed 5 000-image holdout carved out of the train split**
(`torch.randperm(50000, seed=1234)[:5000]`, identical in `train.py` and `verify.py`); models used for
tuning saw only the remaining 45 000. The test split was read exactly once, *after* the ladder had
already been trained and promoted, purely to report the numbers in the table above.

**Noise floor first.** Four seeds of one config gave 94.12 / 94.26 / 94.30 / 94.42 → **σ ≈ 0.12 pp**,
and re-running an identical seed moved the result by 0.06 pp (cudnn non-determinism). So single-run
gaps below ~0.25 pp were treated as noise, and finalists were re-run on **fresh seeds (11, 12)** that
had not been used for screening, to avoid selecting on a lucky screening draw.

**Screening → finalists.** Roughly 40 holdout runs (`runs.jsonl`, `sweep*.log`):

| round | what moved | what did not |
|---|---|---|
| A: LR 0.125→2.0, wd, momentum, warmup, bias-LR | LR ≥ 1.0 clearly bad (92.5 / 91.2); wd 0.01 bad (−0.5 pp) | LR 0.15–0.4 flat within noise; momentum 0.9, warmup 0.3, bias-scaler all ≤ 1σ |
| B: capacity + schedule | widths 128/384/512 > 64/256/256 (+0.4 pp); **cutout 8–12 + 32 ep** (+0.35 pp over 24 ep, ~3σ) | 40 epochs (−0.3 pp), translate 3–4, label smoothing 0.1, widths 192/512/640 |
| C: finalists, seeds 11+12 | 32ep/cut12/lr.25 = 94.76, 28ep/cut12/lr.30 = 94.71, 32ep/cut8/lr.40 = 94.52, 24ep/cut8/lr.40 = 94.12, narrow 32ep/cut12 = 94.02 | — |
| D: speed/margin frontier | nothing cleared the bar: narrow 24ep 93.48, narrow 28ep 93.84, mid-width 96/320/384 94.13, wide 20ep 94.15 | — |

Note that longer schedules only help *with* cutout: 32 epochs without cutout scored 93.90, below the
24-epoch run (94.28) — the extra epochs are useful only once the extra regularisation is there.

**Pre-committed promotion rule.** Maximise the 2-seed holdout mean subject to estimated t\* ≤ 15 s,
tie-broken toward the faster config when within 1 SEM (0.09 pp) of the best. Round C's top two are
0.05 pp apart, so the rule selects the cheaper **28 ep / cutout 12 / LR 0.30** — 4 fewer epochs for a
difference well inside the noise floor.

**Why the ladder was not made faster.** `S_outcome` saturates at t\* ≤ 28 s and the delivered ladder
sits at 13.2 s, so speed below that buys nothing while margin above 94% is the binding risk. Round D
tested the frontier anyway under a pre-set bar (≥ 94.35 pp holdout mean, i.e. ≈ 0.6 pp expected test
margin): the cheapest credible candidates (mid-width, or wide/20 epochs, t\* ≈ 8–10 s) all landed at
94.1 pp, ~0.6 pp below the promoted config, so the existing ladder was kept rather than traded for
speed that is not scored. The promoted run crosses at 94.64% — a 0.64 pp cushion.

## 4. What each component is actually worth

`ablate.py`, 2 seeds per row, same holdout, promoted recipe = 94.65% (`ablation.json`):

| variant | holdout | Δ |
|---|---|---|
| promoted recipe | 94.65 | — |
| no random translate | 94.18 | **−0.47** |
| learned first layer instead of frozen ZCA | 94.25 | **−0.40** |
| no cutout | 94.41 | **−0.24** |
| learnable BN scales | 94.49 | −0.16 |
| no label smoothing | 94.66 | +0.01 |

Re-scoring one fixed set of weights at each TTA level: single view 94.05, +mirror 94.64,
+mirror&translate 94.65. So **mirror TTA is worth +0.59 pp and the four extra translated views add
+0.01 pp** — three quarters of the inference cost buys nothing measurable. Likewise label smoothing
0.2 is inert here. Both are still present in the delivered ladder (it was trained and verified
before these rows existed, and neither hurts accuracy); they are the first two things to drop if
this recipe is taken further, along with a re-test of BN-scale freezing, whose +0.16 pp is only
~1.3σ.

The two components that clearly earn their place are the augmentations and the frozen ZCA layer —
the whitening filter bank is not decoration, it is worth 0.4 pp on its own.

**Reproducibility of the delivered result.** The promoted recipe was re-run end to end on an
unseen seed (7) into a scratch directory and scored the same way (`repro_eval.json`): it crosses at
the *same* epoch 27 (94.26%) and ends at 94.87%, versus epoch 27 / 94.64% and 94.68% for the
delivered seed-0 ladder. The crossing epoch, and therefore t\*, is stable to ~0.5 s across seeds;
the crossing accuracy varies by ~0.4 pp. This run was reporting only — the delivered ladder was
already fixed and was not swapped for it. Its checkpoints were deleted so that exactly one ladder
exists in `/workspace/periodic_checkpoints/`.

Note also that the delivered ladder is safe twice over: if epoch 27 (94.64%) were re-scored below
the bar, epoch 28 (94.68%) crosses at t\* = 13.64 s, still far inside the 28 s reference.

## 5. Reproduce

```bash
python prep_data.py                      # PNGs -> uint8 tensor cache (once)
python train.py --epochs 28 --cutout 12 --lr 0.3 --widths 128 384 512 \
                --holdout 0 --seed 0 --out /workspace/periodic_checkpoints
python verify.py --dir /workspace/periodic_checkpoints --split test --every 1
python check_artifacts.py
```

`python sweep.py '[{"epochs":28,"cutout":12,"lr":0.3,"widths":[128,384,512],"seed":11}]' tag` runs
holdout experiments (they default to `--holdout 5000 --save 0`, never writing the graded ladder).

## 6. Files

| file | role |
|---|---|
| `prep_data.py` | PNG → uint8 tensor cache |
| `train.py` | model + training + per-epoch scripted checkpointing |
| `sweep.py` | many holdout configs in one process |
| `verify.py` | re-evaluation through the grader's exact input pipeline; rebuilds t\* from save times |
| `ablate.py` | component ablation of the promoted recipe (§4) |
| `check_artifacts.py` | ladder/contract self-check |
| `runs.jsonl`, `sweep*.log` | full log of every run behind §3 |
| `final_eval.json` | per-epoch test accuracy of the delivered ladder |
| `ablation.json`, `ablation.log` | raw ablation results |
| `agent_reported_metrics.json` | 94.68% final checkpoint, 17.40 s total training wall clock |
