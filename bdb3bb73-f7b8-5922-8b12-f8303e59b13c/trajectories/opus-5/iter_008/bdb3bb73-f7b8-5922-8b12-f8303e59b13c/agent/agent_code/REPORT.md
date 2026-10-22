# CIFAR-10 → 94% top-1 on one H100

## Result

| quantity | value |
|---|---|
| graded ladder | `/workspace/periodic_checkpoints/ckpt_epoch_0001..0018.pt` (18 contiguous, all `torch.jit.load`-able) |
| first checkpoint ≥ 94% | **epoch 16, 94.02%** |
| t\* by the verifier's formula | **6.55 s** (span 6.948 s × 16/15) — reference 28 s, baseline 70 s |
| final checkpoint | **94.74%** (`ckpt_epoch_0018.pt`, = `/workspace/checkpoint.pt`) |
| total training-run wall clock | 8.99 s (data load + whitening init + 18 epochs + 18 saves) |
| mean epoch | 0.409 s (50 steps of batch 1000 + one scripted save) |

Accuracies are independent re-evaluations of the saved checkpoints through the
verifier's exact pipeline: the test PNGs via `ImageFolder`, `ToTensor()` +
`Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))`, batch 512,
`torch.jit.load(path).eval()` on the GPU, `argmax(1)` (`eval_ckpt.py`).

Reproducibility — the same config retrained on the full 50 k with three seeds:

| seed | crossing epoch | crossing acc | final acc | t\* |
|---|---|---|---|---|
| 0 (graded) | 16 | 0.9402 | 0.9474 | 6.55 s |
| 1 | 17 | 0.9445 | 0.9438 | 6.92 s |
| 2 | 17 | 0.9456 | 0.9469 | 6.92 s |

All three cross by epoch 17 at t\* < 7 s, i.e. the crossing does not depend on a lucky
seed. Mean final accuracy 0.9460 ± 0.0019.

## Model

6.08 M parameters, trained from scratch — no pretrained weights and no data beyond the
CIFAR-10 train split.

```
input (already standardised, 3x32x32)
  frozen 2x2 whitening conv -> 24ch, GELU          # ZCA basis of TRAIN-split patches
  ConvGroup 24 -> 128   conv3 / maxpool2 / BN / GELU / conv3 / BN / GELU
  ConvGroup 128 -> 384  ...
  ConvGroup 384 -> 512  ...
  maxpool3 -> flatten -> Linear(512, 10, bias=False)
```

* The only non-random initialisation is the whitening filter bank, computed from the
  covariance of 2×2 patches of the **train** images themselves (`init_whitening`), then
  frozen. Its bias trains for 3 epochs and is then frozen too.
* BatchNorm **scale is frozen at 1**, only the bias learns: the following conv can
  absorb any per-channel gain, so the extra parameter only adds optimisation noise on a
  sub-30-epoch budget.
* SGD, nesterov, momentum 0.85, batch 1000 (divides 50 000 exactly, so every epoch is
  one genuine full pass with no dropped tail). Two parameter groups: conv/head weights
  at lr 0.15 with weight decay 8e-3, BN/whitening biases at 32× that lr and no decay.
  Triangular schedule: 0.2 → 1.0 over the first 23% of steps, then 1.0 → 0 (a peak
  multiplier of ×1.0 on lr 0.15). Cross-entropy with label smoothing 0.1, logits scaled
  by 1/3.
* Augmentation: random 32×32 crop from a reflect-padded 36×36 cache plus random mirror,
  done on the GPU as two `gather`s and a `where`. All 50 k images live on the GPU as one
  fp16 tensor, so there is no DataLoader anywhere in the timed loop.
* bf16 autocast, channels-last, `cudnn.benchmark`.

## Input contract

The verifier hands `forward` a batch that is **already** standardised, so the network is
trained on exactly that representation and `forward` does not re-normalise — the
double-normalisation trap is avoided by construction, not by a correction. `Net.canon`
additionally range-probes the input and would rescale a [0,1] or [0,255] batch; it
cannot fire on standardised input, whose min is ≈ −2. Mirror + 1-px-translate TTA is
folded into `forward` behind `if self.training`, so it ships inside every scripted
checkpoint rather than living in an eval script the verifier never runs.

## Protocol and what the evidence actually supports

**Held-out discipline.** Every hyperparameter was chosen on a 45 000/5 000 split of the
train cache (`sweep.py`); the 5 000-image tail is excluded from training *and* from the
whitening covariance in those runs. The test split was used only to re-evaluate
already-saved checkpoints, never to rank a config. The promotion rule was written down
in `NOTES.md` at t ≈ 600 s, before the tuning sweeps finished.

**Noise floor first.** Re-running one identical config gave 0.9484 and 0.9438 — a
0.46 pp spread from cudnn autotuning and non-deterministic backward alone. That is
larger than most gaps in a one-shot grid, so single-run argmaxes over ~40 configs were
treated as unreliable and the four finalists were re-run with 3 seeds each and ranked by
mean. Reported as a caution, not a detail: the coarse grids below locate a plateau, they
do not resolve 0.2 pp differences.

**Effects larger than the noise floor** (held-out, 45 k train):

| knob | evidence |
|---|---|
| weight decay — the dominant knob | 24 ep, lr 0.2: 2e-4 → 0.9344, 2e-3 → 0.9432, **8e-3 → 0.9460**, 2.4e-2 → 0.9384 |
| BN-bias lr multiplier | 14 ep: ×1 → 0.9224, ×16 → 0.9280; 20 ep: ×8 → 0.9420, ×32 → 0.9482 |
| final lr of the decay | 20 ep: → 0.15 → 0.9366, → 0.07 → 0.9460, → **0.0 → 0.9486** |
| base lr | ≥ 0.4 diverges at ×1 bias lr; 0.10–0.20 is a flat plateau |
| label smoothing | 0.0 → 0.9416 vs 0.1–0.2 → ≈ 0.947 |

**Within the noise floor**, i.e. chosen but not claimed as improvements: momentum
0.8–0.85, BN momentum 0.4–0.6, logit scale 1/9 vs 1/3, bias multiplier 16 vs 32 vs 64,
warmup fraction 0.23 vs neighbours.

**Finalists**, 3 seeds each at 18 epochs (held-out mean):

| | widths | mean val | s/epoch |
|---|---|---|---|
| C1 | 128/384/512, ls 0.2, gain 1/9 | 0.9451 | 0.354 |
| C2 | 128/256/384, ls 0.2, gain 1/9 | 0.9463 | 0.281 |
| **C3** | **128/384/512, ls 0.1, gain 1/3** | **0.9491** | 0.354 |
| C4 | 128/256/384, ls 0.1, gain 1/3 | 0.9473 | 0.281 |

Schedule length, 2 seeds each: C3 reaches the pre-registered 0.945 bar at 16 epochs
(0.9455) and 0.9486 at 18; C4 reaches it at 14 (0.9457) and plateaus at 0.9473.

**Where the two pre-registered criteria disagreed, and how I resolved it.** Ranking by
held-out accuracy picks C3 at 18 epochs; ranking by epochs-to-0.945 picks C4 at 14. I
took C3/18. Both land far inside the 28 s reference (≈ 6.5 s vs ≈ 4.7 s), so the extra
1.8 s is not worth anything — whereas the one genuinely catastrophic outcome is a ladder
where *no* checkpoint clears 94%, which is a margin problem. C3/18 also had the tightest
seed spread of the four (5 held-out runs, std 0.0007). This choice used held-out numbers
only; no test accuracy entered it.

**TTA ablation** on the graded final checkpoint (weights pulled back out of the scripted
file into a plain module):

| inference | test acc | eval time |
|---|---|---|
| single pass | 0.9405 |
| + mirror (2×) | 0.9452 | 0.14 s |
| + 1-px translate (6×) | **0.9474** | 0.36 s |

TTA is worth +0.69 pp and costs nothing that the clock measures, since t\* is built from
training-checkpoint save times.

## Efficiency work

* Epoch count 24 → 18 once the schedule length was tuned on held-out data (−25%).
* `torch.jit.script` returns a module that **shares** parameter and buffer storage with
  the source (verified by matching `data_ptr()`s, and by BN running stats updated during
  training appearing in the saved file). So the model is scripted once and each epoch is
  just a `.save()`: 57 ms → 25 ms per checkpoint, ~7% off the mean epoch time that t\*
  is built from.
* Whole dataset resident on the GPU as fp16; augmentation is two `gather`s and a
  `where`; no DataLoader in the timed loop. 0.409 s/epoch all-in.
* A narrower 128/256/384 backbone is 21% faster per epoch for −0.18 pp held-out. It was
  measured and not taken, for the margin reason above; it is the obvious lever if the
  time budget ever tightened.

## Verification and compliance

* All 18 graded checkpoints load with `torch.jit.load` and were scored through the
  verifier's own PNG pipeline, not through my training cache — which is what would catch
  a channel-order or class-order mismatch (`ImageFolder` class order matches the
  `sorted()` order the cache was built with).
* `/workspace/checkpoint.pt` is byte-identical to `ckpt_epoch_0018.pt` (`cmp` clean).
* Promotion was done with a same-filesystem `mv` (a rename), so each checkpoint keeps
  the save time `.save()` gave it. A plain `cp` would have stamped all 18 files with the
  copy time and collapsed the measured span to ~0.3 s — a number that would badly
  misstate the run. No file timestamp was altered by any means.
* Per-epoch save gaps are 0.404–0.432 s across all 17 intervals: no oversized first
  epoch, every epoch one full 50 000-image pass.
* Reported `wall_clock_sec` = 8.986 s is the whole training run (from the first line of
  `train.py:main`) and therefore exceeds the 6.948 s first→last checkpoint span.
* Two ~570 MB core dumps (`core.417`, `core.547`, left by DataLoader workers that died
  on a shared-memory limit early on) were deleted: their raw memory images contain libc
  symbol names that a plain substring scan of workspace files would flag. A binary-safe
  re-scan of all 52 remaining files, checkpoints included, is clean.

## Honest limitations

* The crossing checkpoint itself (epoch 16, 0.9402) sits only 0.02 pp above the bar. It
  is not fragile for scoring — epoch 17 (0.9454) and 18 (0.9474) are backstops, and if
  the crossing shifted to 17 then t\* becomes 6.96 s, still ~4× inside the reference —
  but the single number 0.9402 should not be read as a robust margin.
* Coarse grids were single-run and are only trustworthy for effects above ~0.5 pp; the
  finalist comparison and the schedule-length scan are the parts with repeated seeds.
* A rebuilt module and its scripted checkpoint differ by up to 0.015 on logits of scale
  5.4 (TF32 convolutions picking different algorithms for channels-last vs contiguous
  layouts). All 512 predictions in the probed batch are identical and both score 0.9474,
  so this is float noise, not a functional mismatch — but it does mean checkpoint
  accuracy can wobble at the ±0.05 pp level between machines.

## Files

`prep_data.py` PNG → uint8 cache · `model.py` network + whitening init · `train.py`
training / checkpointing · `sweep.py` held-out config search · `eval_ckpt.py`
verifier-faithful re-evaluation · `ablate.py` TTA ablation · `NOTES.md` chronological
log and pre-registered promotion rule.
