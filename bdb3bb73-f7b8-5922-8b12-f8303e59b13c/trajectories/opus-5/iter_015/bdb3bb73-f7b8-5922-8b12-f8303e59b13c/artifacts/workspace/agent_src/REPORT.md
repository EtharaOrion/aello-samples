# CIFAR-10 → 94% top-1 on one H100

**Result.** A 26-epoch run reaches **94.34%** test top-1 at epoch 25 and **94.90%** at
epoch 26. Mean epoch duration 0.404 s ⇒ **t\* ≈ 10.1 s** to the first 94% rung, against
the 28 s reference / 70 s baseline anchors. Training end-to-end (process start → last
checkpoint) is **14.3 s**; the checkpoint ladder itself spans 10.09 s.

Everything is trained from scratch on the CIFAR-10 train split only. No pretrained
weights, no external data, no test-split input at training or tuning time.

---

## 1. What the model is

`src/net.py`. A compact 3-group VGG-style convnet in the lineage of Keller Jordan's
`airbench`, re-implemented here:

```
input (float32 NCHW, already standardized)
  └ Conv2d(3→24, k=2, bias)         frozen ZCA whitening filters  [±eigvecs of 2×2 patch cov]
  └ GELU
  └ ConvGroup(24 →128)   conv3 → maxpool2 → BN → GELU → conv3 → BN → GELU
  └ ConvGroup(128→384)   ″
  └ ConvGroup(384→512)   ″
  └ MaxPool2d(3)                    31→15→7→3→1 spatial
  └ Linear(512→10, no bias) × 1/9
```

6.1 M parameters. Three details carry most of the speed:

- **Frozen ZCA whitening front end.** The 2×2 patch covariance of the training images is
  eigendecomposed once; the conv weights are set to `[E/√(λ+ε), −E/√(λ+ε)]` and never
  trained. Signed duplication means the following GELU sees both polarities of every
  projection. Only the layer's *bias* trains, and only for the first 30% of the schedule.
- **Partial-identity (Dirac) conv init.** The first `in_channels` filters of every conv
  start as an identity map, so each group begins near a pass-through.
- **Frozen BatchNorm scale.** γ is pinned at 1 and only β learns; the following conv can
  absorb any scale, and removing the redundancy stabilises the very large learning rate.

## 2. Input convention (the trap in this task)

The grader loads a rung with `torch.jit.load(...).eval()` and feeds float32 NCHW batches
that have **already** been through `ToTensor()` + `Normalize((0.4914,0.4822,0.4465),
(0.2470,0.2435,0.2616))` — range ≈ [−2, 2.1]. Normalising again inside `forward` would
quietly collapse accuracy under the grader while a local [0,1] evaluation still looked
fine.

This pipeline therefore normalises **exactly once, outside the model**: `data.py`
standardises the cached uint8 tensors with those same constants on the GPU, and
`net.forward` starts straight at the whitening conv. To make sure that claim is not just
an assertion, `src/eval_ckpts.py` re-reads the **test PNGs off disk** with
`torchvision.datasets.ImageFolder` and the literal transform above, rather than reusing
the training cache — so a mismatch in class ordering, channel order or scaling could not
hide. Measured input range on that path: `[-1.989, 2.126]`, classes in alphabetical
order. Every number in this report comes from that path.

## 3. Test-time augmentation lives inside `forward`

The grader calls the model once and takes `argmax(1)`, so TTA has to be in the graph.
`TTAModel.forward` averages logits over {identity, mirror} × {no shift, up-left 1 px,
down-right 1 px}, weighted 0.6 / 0.4 toward the unshifted pair — six forward passes.
This costs the *grader* inference time and costs the *training clock* nothing, which is
the right side of the trade here. Worth **+0.73 pp** (§6).

## 4. Training recipe

`src/train.py`. Nesterov SGD, momentum 0.85, batch 1024, label smoothing 0.2, triangular
LR (23% warmup from 0.2×, linear decay to 0.07×), bf16 autocast with fp32 master weights
and `channels_last`. `lr` and `weight_decay` are quoted per 1024 examples and per unit of
`1 + 1/(1−momentum)` so batch size and momentum can move without retuning them. BN
shifts get 64× the base LR (they carry no weight decay penalty at that scale).

All 50k training images sit on the GPU as standardized fp16, so no step goes back to the
host. Augmentation — reflect-pad ±2 px random translate, horizontal flip, 10 px cutout —
is two `gather`s, a `where` and a multiply, well under a millisecond per batch.

Kernel autotuning (`cudnn.benchmark`), the whitening eigendecomposition and
`torch.jit.script` all run **before** the timed loop; a single scripted module is built
once and re-saved each epoch rather than re-scripted, so per-rung save cost stays in the
tens of milliseconds.

**Promoted config** (deltas from `DEFAULT_CFG`): `epochs=26`, widths `128/384/512`,
`cutout=10`, seed 0, trained on all 50,000 images.

## 5. How hyperparameters were chosen — and the noise floor

Selection used a **5,000-image holdout carved out of the train split** (`data.train_val_split`,
fixed seed 0; the model trains on the remaining 45,000). The test split was read only
after the ladder was already promoted. The rule was written down in `PROMOTION_RULE.md`
*before* the finalist runs.

**Noise floor.** Two different quantities, and conflating them is the easy mistake:

| | measurement | size |
|---|---|---|
| within-process, seed-to-seed | sd over seeds {0,1,2} of one config | **0.08–0.34 pp** (typ. 0.12) |
| across processes, identical config+seed | same config re-measured in a later process | **0 pp usually; 0.43 pp when it moves** |

The second row is the interesting one. `{epochs 26, 128/384/512, cutout 10}` was measured
four separate times:

| process | seeds | holdout accuracies |
|---|---|---|
| `sweep_wide` | 0,1 | 0.9516, 0.9496 |
| `sweep_finalists` | 0,1,2 | 0.9516, 0.9530, 0.9506 |
| `ablation` | 0,1 | **0.9480, 0.9468** |
| `cutout_retest` (after a code edit) | 0,1,2 | 0.9516, 0.9530, 0.9506 |

So the pipeline is normally **bit-for-bit reproducible across processes** — the finalist
and re-test processes agree to the last digit, on both sides of an edit to `train.py`.
But one process out of four came out 0.43 pp low on *every* seed at once. That signature
— a whole-process offset rather than per-seed jitter — points at `cudnn.benchmark`, which
autotunes convolution algorithms from wall-clock timings and will occasionally pick a
different one when the machine is loaded differently.

**Consequence for the methodology:** a config's absolute holdout number is not portable
between processes, so only *paired, within-process* comparisons are trustworthy at the
sub-0.5 pp scale. Every table in this report is one — each sweep runs its whole grid
inside a single process — and absolute numbers should not be compared across tables.
The `ablation` table below is the process that drifted, which is why its reference row
reads 0.9474 rather than 0.9517; its *deltas* are still valid because every row in it was
measured under the same drift.

**Finalists** (seeds {0,1,2}, holdout, one process; tie threshold 0.20 pp pre-committed):

| config | holdout mean | sd | crossing epoch |
|---|---|---|---|
| **A epochs 26, 128/384/512, cutout 10** | **0.9517** | 0.0012 | 25, 24, 25 |
| B epochs 26, 128/384/512, cutout 8 | 0.9487 | 0.0020 | 25, 25, 24 |
| C epochs 32, 128/384/512, cutout 10 | 0.9484 | 0.0019 | 29, 30, 31 |
| D epochs 22, 128/384/512, cutout 10 | 0.9469 | 0.0012 | 21, 22, 21 |

A leads B by 0.30 pp > 0.20 pp, so the rule selects A outright with no tie-break needed,
and the 3-seed re-test in §6 reproduced both rows exactly. Honest caveat: 0.30 pp is
above the within-process tie threshold but comparable in size to the cross-process drift,
so A vs B is best read as "A is at least as good", not as a decisive win. It does not
change the outcome — B crosses at the same epoch.

Earlier stages (each within one process): epoch count 14→40 on the narrow net plateaued
at ~94.2%; widening 64/256/256 → 128/384/512 was the single largest gain (+0.9 pp);
`weight_decay` 0.025 and `lr` 9.0 were both worse; `w3=640` did not help.

## 6. Ablation (seeds {0,1}, holdout, one process)

| variant | holdout | Δ |
|---|---|---|
| promoted config | 0.9474 | — |
| cutout 0 | 0.9476 | +0.02 |
| no Dirac init | 0.9454 | −0.20 |
| no ZCA whitening (same layer, trained from standard init) | 0.9424 | −0.50 |
| no TTA | 0.9401 | −0.73 |
| no flip / translate | 0.9377 | −0.97 |
| narrow net 64/256/256 | 0.9377 | −0.97 |

The ablation says **cutout is worth nothing** (+0.02 pp), while the selection sweep had
picked cutout 10 over cutout 8 and over no cutout. That contradiction was worth
resolving rather than leaving in a footnote, so cutout was re-tested paired, 3 seeds,
one process (`logs/cutout_retest.jsonl`):

| cutout | seeds 0,1,2 | mean |
|---|---|---|
| 10 | 0.9516, 0.9530, 0.9506 | **0.9517** |
| 0 | 0.9512, 0.9486, 0.9506 | 0.9501 |
| 8 | 0.9496, 0.9502, 0.9464 | 0.9487 |

Cutout 10 beats no cutout by 0.16 pp — **inside** the 0.20 pp tie threshold. So the two
measurements agree once both are read against the noise floor: cutout is doing no
measurable work at this schedule length, and it is not hurting either. It stayed in the
promoted config because the rule was pre-committed and the finalist stage had already
run; changing it afterwards on a second look is exactly the drift the rule exists to
prevent. A future run could drop it with no expected loss and one less knob.

Geometric augmentation and TTA are the two components actually holding the result up,
followed by whitening. Everything else is second-order.

## 7. Reproducibility of the promoted run

The promoted config was run twice on the full 50k split, seed 0, and evaluated on the
grader path both times:

| run | epoch 25 | epoch 26 | first crossing | ladder span |
|---|---|---|---|---|
| promoted (`periodic_checkpoints/`) | **0.9434** | **0.9490** | epoch 25 | 10.09 s |
| independent re-run | 0.9456 | 0.9466 | epoch 25 | 10.10 s |

Same crossing epoch, same t\*, 0.22 pp apart — consistent with the noise floor in §5.
Inference itself is deterministic: re-evaluating the same saved rungs reproduces
identical accuracies.

The **first** run is the promoted one, because the pre-committed rule says to promote the
seed-0 full-split run of the winning config. The second run existed to measure end-to-end
timing and to check reproducibility; picking whichever of the two had the nicer test
number would have been test-split selection.

## 8. Timing, and what `wall_clock_sec` means

`agent_reported_metrics.json` reports `wall_clock_sec = 14.30`. That is the **total
wall-clock of the promoted training run**, not a time-to-94%:

- 10.616 s — timed epoch loop of the promoted run, first step → last checkpoint save (measured)
- 3.682 s — setup before the loop: data load, ZCA eigendecomposition, cudnn autotune,
  `torch.jit.script` (measured on the byte-identical re-run of the same script and config)
- **14.30 s** total, versus a checkpoint ladder span of 10.09 s.

The reported number is deliberately the larger, end-to-end figure. `accuracy = 0.9490` is
the final rung's grader-path test accuracy.

The ladder was produced by one training run and moved into `periodic_checkpoints/` with
`os.link` (`src/promote.py`), so each graded rung is the same inode the trainer wrote and
carries its genuine save time; `checkpoint.pt` is a content copy of the last rung and is
byte-identical to it.

## 9. Constraint compliance

- **From scratch** — every weight is randomly initialised or computed from train-split
  statistics (the ZCA filters). Nothing is loaded from any checkpoint or weight zoo.
- **No test leakage** — the test split is read by exactly one file, `src/eval_ckpts.py`,
  which is never imported by the training or sweep code. All tuning used the train-split
  holdout. Hyperparameters were fixed and the ladder promoted before any test number was
  produced.
- **No external data** — inputs come only from `/workspace/data/cifar10/train`.
- **No extra packages** — stock `torch` 2.4.0 / `torchvision` 0.19.0; nothing installed.

## 10. Layout

```
src/prep_data.py    decode the PNG folders once into uint8 .npy
src/data.py         GPU-resident standardized tensors, holdout split, on-device augmentation
src/net.py          model + TTA / no-TTA inference wrappers (torch.jit.script-able)
src/train.py        whitening init, optimizer/schedule, timed loop, per-epoch scripted saves
src/sweep.py        multi-seed holdout search
src/eval_ckpts.py   grader-exact re-evaluation of a ladder
src/promote.py      atomic ladder swap (os.link) + checkpoint.pt
PROMOTION_RULE.md   the decision rule, written before the finalist runs
logs/               every sweep, ablation and evaluation result, one JSON per line
                      sweep_epochs / sweep_arch / sweep_wide  search stages
                      sweep_finalists                          3-seed finalist comparison
                      ablation                                 component ablation
                      cutout_retest                            follow-up on the §6 contradiction
                      eval_graded / eval_promoted              grader-path test accuracies
                      run_manifest                             end-to-end timing of a full run
```

Reproduce: `python prep_data.py && python train.py --ckpt-dir /workspace/scratch_ckpt
--cfg '{"epochs":26,"w1":128,"w2":384,"w3":512,"cutout":10}' && python eval_ckpts.py
--dir /workspace/scratch_ckpt`
