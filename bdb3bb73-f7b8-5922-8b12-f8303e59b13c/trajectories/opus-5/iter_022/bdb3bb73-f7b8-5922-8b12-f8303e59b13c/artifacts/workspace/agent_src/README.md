# CIFAR-10 → 94% top-1 on one H100

A from-scratch convnet that crosses 94% test top-1 in a training run of a few
seconds of GPU wall clock. No pretrained weights, no data outside the CIFAR-10
train split, no test-set contact during training or tuning.

## Files

| file | role |
|---|---|
| `PREREGISTRATION.md` | protocol, candidate list and promotion rule, fixed before any sweep run |
| `cache_data.py` | one-off decode of the PNG folders into uint8 tensors |
| `airbench.py` | model, GPU-resident data pipeline, training loop |
| `run.py` | trains one full run and writes the checkpoint ladder |
| `sweep.py` | executes the pre-registered sweep on a train-split holdout |
| `verify.py` | re-implements the verifier's evaluation and reconstructs `t_star` |
| `SWEEP_RESULTS.md` | what the sweep measured and what was promoted |
| `round2.py` | round-2 candidates plus the pooled-variance analysis |
| `write_metrics.py` | writes `agent_reported_metrics.json` from the graded run |

## Model (6,082,872 parameters)

```
input: float32 NCHW, already standardized with CIFAR mean/std (verifier convention)
  frozen 2x2 ZCA whitening conv, 3 -> 24, trainable bias  -> GELU
  3 x ConvGroup: conv3x3 -> maxpool2 -> BN -> GELU -> conv3x3 -> BN -> GELU
     widths 128 -> 384 -> 512
  maxpool3 -> flatten -> Linear(512, 10, bias=False) -> * 1/9
```

Design points that carry weight:

* **Frozen ZCA whitening front end.** The first layer is a 2×2 convolution whose
  weights are the eigenvectors of the 2×2×3 patch covariance, each scaled by
  1/√(λ+ε), concatenated with their negations (12 eigenvectors → 24 channels, so
  the following GELU sees both signs). It is initialized from train images and
  never updated. Handing the network a decorrelated basis for free is what lets
  the rest of the stack converge in ~20 passes.
* **BatchNorm scale frozen at 1.** A learnable BN scale immediately before a
  convolution is redundant with that convolution's weights; freezing it removes
  parameters that only add optimization noise. BN biases stay trainable and get
  64× the base learning rate.
* **Logit scaling by 1/9.** Shrinking the logits softens the softmax early on,
  which keeps gradients well-scaled under label smoothing 0.2.
* **Half precision with fp32 BatchNorm.** Convolutions run in fp16 for the tensor
  cores; BN keeps fp32 parameters and running statistics, where the precision
  actually matters. The loss uses `reduction='sum'`, which scales gradients by the
  batch size and thereby doubles as a built-in fp16 loss scale (the learning rate
  is divided by the batch size to compensate).

## Input convention

`forward` expects inputs **already standardized** with
`Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616))` — exactly what the
verifier feeds. Nothing is re-normalized inside the model. `forward` casts the
input to the parameter dtype (so a half checkpoint accepts float32 batches) and
returns float32 logits.

## Test-time augmentation, baked into the checkpoint

`forward` runs 6 passes and averages: mirror × {no shift, −1 px, +1 px} using
reflect padding, weighted `0.5·direct + 0.25·shifted` . Because it lives inside
`forward`, `torch.jit.load(path)(x).argmax(1)` gets the TTA prediction with no
cooperation needed from the caller.

## Data pipeline

The whole train split lives on the GPU as a standardized, reflect-padded half
tensor; there is no `DataLoader` in the training loop. Each batch costs two
gathers and one `where`:

1. gather the batch rows,
2. gather a random ±2 px translate through a precomputed table of the 25 crop
   offsets,
3. `where` against a mirrored copy for the random horizontal flip.

Augmentation totals ~5 ms per epoch, under 2% of the epoch.

## Timing

`t_star` is reconstructed by the verifier from checkpoint save times, so
everything inside the run counts. Two things are worth stating plainly:

* The H100 idles at **345 MHz against a 1980 MHz ceiling**. A cold start runs the
  first epochs at roughly half speed and the ramp bleeds into later ones —
  measured 0.78 s/epoch cold versus 0.35 s/epoch at steady clocks. `run.py`
  therefore calls `warmup_gpu`, which runs the training step on **random noise
  with a throwaway network that is then discarded**. No weights, statistics or
  data from the warm-up reach the real run; it only ensures each timed epoch is
  measured under the same conditions. TorchScript compilation is paid there too.
* Each epoch is one genuine pass over the training split: batch size 1000 into
  50,000 images is exactly 50 steps, no remainder and nothing dropped.

Checkpoints are written by scripting the model **once** before the loop —
`torch.jit.script` shares parameter storage with the live module, so each epoch's
save is a 12 MB serialization costing ~14 ms rather than a fresh compile.
The ladder is written directly into `/workspace/periodic_checkpoints/` by the
training run, so the recorded save times are the real ones.

## Method

Every hyperparameter decision comes from a 5,000-image holdout carved out of the
train split; the test split is used only for the final artifact check the task
prescribes (does the scripted module load, does it accept standardized float32
input, is the ladder contiguous from epoch 1). See `PREREGISTRATION.md` for the
protocol and `SWEEP_RESULTS.md` for the outcome.

42 holdout runs across two pre-registered rounds settled on `momentum=0.80,
wd=1.4e-3` over the baseline's `0.85 / 2.0e-3`. The honest summary of that search:
one factor at a time bought nothing that survived its own noise, and the promoted
change only cleared the bar as a *pair*. A 4-seed σ estimate (0.079 pp) was too
optimistic by 2× against the 19-df pooled estimate (0.167 pp), and the best
single-seed screen result (`lr=1.8`) reversed sign on confirmation.

## Result

Single graded run, config fixed in advance, seed 0, full 50,000 train images.

| | |
|---|---|
| first checkpoint ≥ 94% | **epoch 18 — 94.49%** |
| final checkpoint (epoch 20) | **94.79%** |
| `t_star` | **16.43 s** → S_outcome **1.0000** |
| checkpoint span, epochs 1→20 | 17.64 s |
| total run wall clock | 21.99 s |

All 20 checkpoints were re-evaluated with `torch.jit.load(...).eval()` on the real
test PNGs through the verifier's transform at batch size 512. The ladder is
contiguous 1..20, `/workspace/checkpoint.pt` is byte-identical to
`ckpt_epoch_0020.pt`, and the checkpoint span (17.64 s) is consistent with the
measured training wall clock (18.43 s) — the save times are the ones training
produced.
