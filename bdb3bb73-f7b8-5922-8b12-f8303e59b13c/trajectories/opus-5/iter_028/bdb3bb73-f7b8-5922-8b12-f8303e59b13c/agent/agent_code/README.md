# CIFAR-10 → 94% top-1 from scratch on one H100

## Result

Graded ladder in `/workspace/periodic_checkpoints/`: 24 checkpoints, epochs 1..24, one
genuine pass over the 50k train split each.

| quantity | value |
| --- | --- |
| first checkpoint ≥ 94% | `ckpt_epoch_0020.pt`, **94.17%** |
| t\* (grader's formula, from checkpoint save times) | **16.17 s** |
| implied S_outcome | **1.0000** (saturates at t\* ≤ 28 s) |
| final `ckpt_epoch_0024.pt` | **94.34%** |
| checkpoints ≥ 94% | 5 (epochs 20–24) |
| total run wall clock | 25.16 s (checkpoint span 18.43 s) |

Accuracies are measured with `verify.py`, which reproduces the grader's pipeline exactly:
`torch.jit.load(path).eval()` on GPU, batch size 512, test PNGs through
`ToTensor()` + `Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))`, `argmax(1)`.

## Model

airbench-lineage convnet, 6.08M params (`train.py`):

- frozen 2×2 ZCA whitening conv (3→24, ±eigenvector pairs), bias trained for 3 epochs then frozen
- three conv/pool/BN/GELU groups at widths 128 / 384 / 512; BN scale frozen, shift learned
- `MaxPool2d(3)` → bias-free `Linear(512,10)` → ×1/9 logit scale
- fp16 trunk with fp32 BN, channels_last; SGD + Nesterov (m=0.85), label smoothing 0.2,
  linear warmup to peak at 23% of training then linear decay to 2% of peak
- all 50k images resident on GPU as uint8; augmentation is on-device (two gathers for the
  per-sample random translate, a `where` for the mirror, a masked fill for cutout)

**Input convention.** The exported model consumes *already-standardized* float32 NCHW input
and does **not** normalize again. ZCA statistics are estimated on standardized training
images so train and grading agree. Verified end-to-end: 94.34% under the grader convention
vs 41.93% if fed raw [0,1] — a double-normalization bug could not have gone unnoticed.

**TTA** (6-way: {identity, mirror} × {centre, −1px, +1px}, weighted 0.6/0.4) is folded into
the scripted `forward` as a single batched trunk pass, so it costs nothing on the timed path.

**Per-epoch export cost.** The `Exported` wrapper is `torch.jit.script`-ed **once**; it shares
parameter storage with the live net, so each epoch pays only serialization (~30 ms), not
recompilation. Confirmed by checkpoints having distinct, monotonically improving accuracies.

## Protocol

All hypothesis testing ran on a **held-out 5k slice of the train split** (train on 45k).
The test split is never loaded by `train.py` or `sweep.py`; test accuracy is measured only
afterwards by `verify.py`, on already-saved checkpoints. Plans were written to
`plan1.py`/`plan2.py`/`plan3.py` *before* their runs, each with a fixed decision rule:
promote only on a 2-seed mean beating the incumbent by more than the measured noise floor.

**Noise floor:** 3 baseline seeds gave sd = **0.06 pp**.

**Diagnosis that drove everything.** Full-train accuracy plateaued at ~93.9% for *both* 18
and 24 epochs, so the schedule was not the binding constraint — the model was
regularization-limited. Round 1 confirmed this.

### Round 1 — augmentation (baseline 94.01)
| arm | val | Δ |
| --- | --- | --- |
| cutout12 | 94.58 | **+0.57** ✅ |
| cutout8 + translate4 | 94.46 | +0.45 |
| cutout8 | 94.31 | +0.30 |
| whiten-bias 6 epochs | 93.97 | −0.04 ❌ null |
| translate4 | 93.87 | −0.14 ❌ null |

### Round 2 — where cutout stops helping
cutout14 94.45, cutout16 94.46 (sd 0.45), cutout12+translate4 94.47 — all **below** cutout12,
so 12 is the optimum. cutout12 @ 32 epochs scored 94.66, only +0.08 over its own sd of 0.23:
**inside the noise, so rejected** by the pre-registered rule — and it would have cost 33% more
wall clock for no scoring benefit.

### Round 3 — optimizer constants (never re-tuned for this wider trunk)
| arm | val |
| --- | --- |
| ls 0.10 | 94.68 ± 0.25 |
| incumbent | 94.58 ± 0.11 |
| wd 0.0230 | 94.56 ± 0.23 |
| lr 14 / lr 9 | 94.42 / 94.35 |

**Null result — nothing promoted.** The leader gains +0.10 pp against its own sd of 0.25.
The constants inherited from airbench's narrower 64/256/256 net are already near-optimal here.

### Reliability of the promoted recipe (full train split, fresh seeds)
| seed | first ≥94% | t\* | final | #ckpts ≥94% |
| --- | --- | --- | --- | --- |
| 0 (graded) | ep 20 | 16.17 s | 94.34 | 5 |
| 1 | ep 21 | 16.26 s | 94.43 | 4 |
| 2 | ep 20 | 14.64 s | 94.24 | 4 |

3/3 seeds cross, all with ~12 s of headroom under the 28 s saturation point. The graded ladder
is the seed-0 run; picking the best-scoring seed here would be selecting on the test split, so
it was deliberately not done.

## Caveats

- **Run-to-run variation.** Two identical-command full-train runs (same seed, same config)
  gave 94.21%/crossing ep19 and 94.34%/crossing ep20 — ~0.13 pp and ±1 epoch apart. The
  held-out sweep, by contrast, reproduced its per-seed values bit-exactly across two separate
  processes. This GPU is shared (an external process held it at 100% utilization for the whole
  session), so timing-based `cudnn.benchmark` autotuning is the most plausible cause; I did not
  have budget to isolate it further. Mitigation: the recipe is chosen to leave 5 crossing
  checkpoints and ~12 s of t\* headroom, so ±1 crossing epoch does not change S_outcome.
- Because of that variation, held-out deltas below ~0.2 pp are not trustworthy on 2 seeds.
  This is exactly why round 3 was not promoted.
- The GPU contention also means absolute epoch time (~0.80 s) is likely worse than an
  uncontended H100 would give; t\* would improve, not degrade, on a quiet machine.

## Files

`prep_data.py` PNG→uint8 cache · `train.py` model + training + export · `verify.py` grader-replica
scoring · `sweep.py` held-out harness · `plan{1,2,3}.py` pre-registered plans ·
`sweep{1,2,3}.json` raw results · `reliability.py` multi-seed robustness check

Reproduce the graded ladder:

```bash
python prep_data.py
python train.py --epochs 24 --seed 0 --ckpt /workspace/periodic_checkpoints \
                --set lr_end=0.02 --set cutout=12
python verify.py --dir /workspace/periodic_checkpoints
```

## Compliance

Trained from scratch on the CIFAR-10 train split only — no pretrained weights, no external
data, no extra downloads. The test split is read exclusively by `verify.py`, after training,
from saved checkpoints. Checkpoint mtimes are whatever `save()` wrote during the run and were
never altered; `/workspace/checkpoint.pt` is an `os.link` hardlink of `ckpt_epoch_0024.pt`
(same inode, hence same mtime). `agent_reported_metrics.json` reports the full 25.16 s run
wall clock, which exceeds the 18.43 s first→last checkpoint span.
