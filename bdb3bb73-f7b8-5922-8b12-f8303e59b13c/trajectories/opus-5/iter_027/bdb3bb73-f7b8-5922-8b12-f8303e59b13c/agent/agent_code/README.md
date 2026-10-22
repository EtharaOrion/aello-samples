# CIFAR-10 to 94% top-1 on one H100

Train from scratch, emit one scripted checkpoint per epoch, cross 94% top-1 on
the test split as early as possible inside a 1800 s budget.

```
python prep_data.py                                       # decode PNGs -> cached uint8 (2.7 s)
python train.py --ckpt-dir /workspace/periodic_checkpoints ...   # the graded run
python evaluate.py --dir /workspace/periodic_checkpoints  # verifier-identical re-eval
python finalize.py --wall-clock-sec ... --accuracy ...    # link checkpoint.pt + metrics
```

| file | role |
|---|---|
| `prep_data.py` | decodes the 60k PNGs once into GPU-ready uint8 tensors |
| `model.py` | network, ZCA whitening init, `Deploy` (input-robust + TTA) wrapper |
| `train.py` | the training run; writes `ckpt_epoch_NNNN.pt` per epoch |
| `evaluate.py` | re-evaluates a ladder through the verifier's exact input pipeline |
| `sweep.py` | held-out hyperparameter study (never reads the test split) |
| `finalize.py` | publishes `checkpoint.pt` + `agent_reported_metrics.json` |
| `experiments/plan.md` | the hypotheses, registered before the sweep was run |
| `experiments/results.md` | what the sweep measured and what was promoted |

## The model

A widened re-implementation of the "airbench" CIFAR speedrun architecture,
6.08 M parameters, trained in pure fp16 (fp32 BatchNorm) with `channels_last`:

```
standardized NCHW input
  -> 2x2 conv 3->24, weights = frozen ZCA eigenbasis of train patches   (32 -> 31)
  -> GELU
  -> ConvGroup 24 -> 128    conv3 / maxpool2 / BN / GELU / conv3 / BN / GELU
  -> ConvGroup 128 -> 384
  -> ConvGroup 384 -> 512
  -> MaxPool2d(3)  ->  Linear(512, 10, bias=False)  ->  x (1/9)
```

Details that are easy to get wrong, and cost real accuracy when you do:

* **The whitening conv is a data-dependent init, not a pretrained weight.** Its
  filters are the ZCA eigenbasis (scaled by `1/sqrt(lambda + 5e-4)`, both signs)
  of every 2x2x3 patch of the *training* images. Weights frozen; the bias is
  trained for the first few epochs only.
* **BatchNorm scale is frozen at 1**; only the bias is learned.
* **The `1/9` logit scale is a layer in `forward`**, not something folded into
  the head's initialization. Folding it into init instead multiplies the
  effective learning rate on the head by 9.
* `Deploy.forward` re-normalizes *only if needed* — see below.

## Training loop

The whole train split lives on the GPU as fp16, pre-padded by `pad` px
(reflect). There is no DataLoader and no host<->device traffic inside an epoch.
Augmentation is two `gather`s (random crop), one `where` (horizontal flip) and
one `masked_fill` (cutout). Optimizer is nesterov SGD on a summed loss with
label smoothing, so the learning rate is quoted in per-example units and
rescaled by the momentum `kilostep` term; biases get a 64x larger LR; the LR
follows a linear ramp to a peak at 23% of training and then a linear decay,
optionally held flat over the last few epochs.

cuDNN autotuning is warmed up on a throwaway network **before** the timed loop,
so epoch 1 is not inflated by algorithm selection.

## Input convention — the trap

The verifier evaluates with

```python
model = torch.jit.load(path).eval()
x = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                             (0.2470, 0.2435, 0.2616))])(png)
pred = model(x_batch_of_512).argmax(1)
```

i.e. inputs arrive **already standardized**, range ~[-1.99, 2.13]. A model that
standardizes again inside `forward` looks fine on a [0,1] harness and collapses
under the verifier. `Deploy.forward` therefore detects the convention from the
batch statistics, branchlessly (no host sync, TorchScript-safe):

| observed | assumed convention | action |
|---|---|---|
| `max > 20` | `[0, 255]` | `/255`, then standardize |
| `min >= -0.02` | `[0, 1]` | standardize |
| otherwise | already standardized | pass through |

`evaluate.py --mode both` checks the cached-tensor path against a real
`ImageFolder` + `transforms` pass: **max abs difference 0.000e+00**, labels
identical. So the accuracies reported here are computed on byte-identical
inputs to the verifier's.

## Test-time augmentation

6 views are folded into the scripted `forward`, so the verifier gets them just
by calling `model(x)`: `{identity, +1 px diagonal, -1 px diagonal}` x
`{no flip, horizontal flip}`, combined as
`0.5 * mirror_only + 0.5 * mean(translated)`.

## No test-set leakage

`train.py` and `sweep.py` read `cache/train_u8.pt` only. Every hyperparameter
decision was made on a **5,000-image held-out slice of the train split** (fixed
permutation, seed 1234, independent of the run seed), training on the remaining
45,000. `evaluate.py` is the only thing that reads the test split, and it runs
strictly *after* a configuration is frozen, to report the number the verifier
will independently recompute.

## Timing and honesty

The verifier owns the clock: it reconstructs
`t* = (mtime[ckpt_N] - mtime[ckpt_1]) * N/(N-1)` for the first checkpoint N at
>= 94%. Consequences that shaped the code:

* `ScriptModule.save(...)` happens **inside** the timed loop, so the measured
  per-epoch cost includes producing the deliverable.
* The graded run writes **directly** into `/workspace/periodic_checkpoints/`.
  A ladder is never copied in from scratch space — copying would stamp every
  file with the copy time and fabricate a faster run.
* `agent_reported_metrics.json.wall_clock_sec` is the **total** wall clock of
  the training process that produced the ladder (process start to last
  checkpoint), which `finalize.py` asserts is >= the first-to-last checkpoint
  span before writing the file.
* Every epoch is one genuine pass over the training split.

See `experiments/results.md` for the measured numbers.
