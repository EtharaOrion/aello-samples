# CIFAR-10 → 94% top-1 from scratch on one H100

A compact convnet (6.08M params) trained from scratch on the CIFAR-10 train
split, writing one TorchScript checkpoint per epoch. No pretrained weights, no
external data, no test-split input during training.

## Layout

| file | role |
|------|------|
| `prepare_data.py`   | decode the PNG folders once into uint8 tensors (`/workspace/cache`) |
| `speedrun/net.py`   | model, frozen patch-whitening init, TTA inference wrapper |
| `speedrun/data.py`  | GPU-resident train tensors + on-device augmentation |
| `speedrun/train.py` | training loop, optimiser/schedule, per-epoch checkpointing |
| `run_train.py`      | run one job, write a ladder |
| `sweep.py`          | held-out hyper-parameter sweep (train split only) |
| `evaluate_ladder.py`| re-evaluate a ladder through the **verifier's exact** input pipeline |
| `finalize.py`       | ladder sanity checks, `checkpoint.pt`, reported metrics |
| `PREREGISTRATION.md`| protocol + candidate list + decision rule, frozen before sweeping |
| `RESULTS.md`        | what the sweep actually said and what was shipped |

Reproduce the shipped artefact with:

```bash
python prepare_data.py
python run_train.py --outdir /workspace/periodic_checkpoints --clean \
                    --epochs <E> --seed 0
python evaluate_ladder.py /workspace/periodic_checkpoints
```

## Method

**Architecture.** A frozen 2×2 patch-whitening convolution (24 filters, ZCA
basis of 2×2 patches estimated from the train split) feeds three
conv/pool/BN/GELU groups of width 128/384/512 and a bias-free linear head with a
1/9 logit scale. The whitening layer is *data-derived, recomputed from scratch
each run in ~40 ms* — it is not a stored or pretrained weight. BatchNorm keeps a
trainable bias but a frozen unit weight: the scale is redundant with the
following convolution and only adds optimiser noise.

**Data path.** All 50,000 training images live on the GPU as a single
reflect-padded fp16 tensor, so there is no DataLoader, no host→device copy and
no CPU work inside the loop. Random horizontal flip and random ±2 px
translation are folded into *one* index tensor and applied with a single
`gather` — 13 ms per epoch, ~1% of step time.

**Optimisation.** Nesterov SGD, momentum 0.85, batch 1000 (exactly 50 steps ×
50,000 images = one genuine pass per epoch), label smoothing 0.2, and a
triangular LR schedule (23% warmup). The learning rate is divided by
`1024·(1 + 1/(1−momentum))` so that `lr` and `momentum` are roughly independent
knobs. BatchNorm biases get a 64× larger LR than weights; the whitening bias is
trained for the first 3 epochs and then frozen.

**Inference.** 6-way test-time augmentation (mirror × {identity, −1 px, +1 px}
translate) is folded into the checkpoint's `forward`, so the verifier gets it
for free from a plain `torch.jit.load(...)(x)`.

## Input convention

`forward` expects exactly what the verifier feeds: float32 NCHW already
standardised by `Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616))`. It
does **not** re-normalise such input. As a guard it also accepts raw `[0,1]` or
`[0,255]` tensors; the three ranges are disjoint (standardised CIFAR-10 has
min ≈ −1.99 and max ≈ 2.13, so it can match neither the `[0,1]` test, which
requires min ≥ −0.01, nor the `[0,255]` test, which requires max > 8), so the
guard cannot misfire on correctly-preprocessed input. Every accuracy number in
`RESULTS.md` marked *verifier pipeline* was produced by
`ImageFolder → ToTensor → Normalize → torch.jit.load(...).eval()`, batch 512,
`argmax(1)` — i.e. the grader's own path, not our training tensors.

## Timing

The verifier reconstructs `t_star` from the checkpoints' save times. The final
run therefore writes **directly into** `/workspace/periodic_checkpoints/`
(after clearing it) rather than being copied in from scratch space: copying a
finished ladder would stamp every file within a fraction of a second of every
other and report a training speed that never happened. Only
`/workspace/checkpoint.pt` is a copy of the last periodic checkpoint, as the
task specifies. Sweeps and experiments never write to the graded directory.
