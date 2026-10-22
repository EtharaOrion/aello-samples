# Results — CIFAR-10 → 94 % top-1 from scratch on 1×H100

## Graded artifact

| | |
|---|---|
| Ladder | `/workspace/periodic_checkpoints/ckpt_epoch_0001.pt … 0032.pt` (32 files, contiguous) |
| Final | `/workspace/checkpoint.pt` (hardlink of `ckpt_epoch_0032.pt`) |
| **First checkpoint ≥ 0.94 (test)** | **`ckpt_epoch_0030.pt`, acc 0.9418** |
| **t\*** = span × N/(N−1) | **11.86 s** → `S_outcome = clamp((70−11.86)/(70−28)) = 1.0` |
| Final-epoch test acc | **0.9488** |
| Training run wall-clock | 13.89 s (ladder span 12.25 s, mean epoch 0.395 s) |
| Params | 6 082 872 |

Three checkpoints (30, 31, 32) independently clear 0.94, so the crossing does not hinge on
a single marginal file: even if epoch 30 re-evaluated a hair below threshold, epochs 31/32
(0.9439 / 0.9488) put t\* at 12.65 s and `S_outcome` still at 1.0.

All accuracies above are from `verify_ladder.py`, which reproduces the verifier convention
exactly: `torch.jit.load(path).eval()` on GPU, test PNGs through
`Compose([ToTensor(), Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])`,
batch size 512, `argmax(1)`.

## What the model is

Frozen 2×2 ZCA-whitening conv (3→24, fitted on the training split only) → GELU → three
`conv/maxpool/BN/GELU + conv/BN/GELU` groups (128/384/512) → MaxPool(3) → bias-free linear
head, logits × 1/32. BatchNorm affine weight frozen at 1, bias trained. SGD+Nesterov
(momentum 0.85), summed cross-entropy with label smoothing 0.2, triangular LR (peak at 23 %
of steps), decoupled-style weight decay, 64× LR on BN/whitening biases. All 50 k images
live on the GPU as uint8; per-epoch augmentation (±2 px translate via two gathers, random
flip, 16 px cutout) is on-device, so there is no dataloader in the loop. 6-way TTA
(mirror × {identity, ±1 px}) is folded into the scripted `forward`, so the verifier's single
`model(x)` call gets the ensemble.

## How the recipe was found

Everything below was measured on a **stratified 45 k/5 k split of the train set**. The test
split was read only by `verify_ladder.py`, after the promotion decision was already fixed.

**Noise floor first (S0).** 5 seeds of the baseline: sd = **0.35 pp** single-seed,
**0.20 pp** for a 3-seed mean. A 5 k val set has a binomial sd of 0.34 pp at this accuracy,
so essentially *all* the observed spread is finite-val-set noise — seed-to-seed training
variance is small. This number is what made the rest of the study interpretable: it set the
bar below which a difference is not a result. Most of the 15-factor S3 screen fell below it.

**The plateau and what broke it.** S1 (epoch budget) showed a hard ceiling — 16 ep 93.3 %,
24 ep 93.8 %, 32 ep 94.1 %, 40 ep 94.0 %. Rather than accept 32–40 epochs as "the answer",
the flatness was treated as evidence that something else was capping accuracy. S2 found it:
weight decay is applied **per step**, so running 3× airbench's epoch count applies 3× its
total decay. 0.0306 → 0.0153 → 0.0051 gave 92.9 % → 93.9 % → 94.2 %, monotone in both LR
columns and ~4× the noise floor. Learning rate itself was insensitive (5.75/11.5/17 all
≈ 94.1 %). Earlier, the same reasoning applied to a slow-converging first baseline had
already identified the logit scale (1/9 → 1/32, +1.5 pp at 12 epochs).

**S3/S3b factor screen (15 + 8 configs, 2 seeds).** Only cutout beat the noise floor, and
it did so as a *trend* rather than a point estimate: 6→94.13 %, 8→94.32 %, 10→94.49 %,
12→94.63 %, 16→94.70 %. Weight EMA clearly hurt (−0.1 pp at 0.85, −1.3 pp at 0.95); Dirac
conv init, `lr_end`, `pad=3`, `bn_momentum`, `whiten_bias_epochs` and lower label smoothing
were all within noise. 48 epochs bought nothing over 32.

**S4 confirmation (3 fresh seeds) and the pre-committed rule.** The rule (PROTOCOL §6) was
written before the search: eligible iff the *worst* seed reaches val ≥ 0.945; rank by
mean − sd; graded seed fixed at 0 a priori. It resolved cleanly — three of four finalists
failed eligibility on their worst seed, and the survivor also won the rank metric outright:

| config | mean | sd | worst | eligible |
|---|---|---|---|---|
| **cutout16 @32ep** | 0.9467 | 0.0009 | **0.9458** | ✅ |
| combo+dirac @32ep | 0.9465 | 0.0039 | 0.9420 | ✗ |
| cutout12+dirac @32ep | 0.9447 | 0.0008 | 0.9440 | ✗ |
| cutout12 @24ep | 0.9427 | 0.0029 | 0.9404 | ✗ |

The top three are separated by ~0.2 pp, i.e. *below* the noise floor — they are not
meaningfully different, and the eligibility criterion (a worst-case bar, not a mean) is what
actually did the work. That is the intended behaviour: with ties this tight, picking by mean
would have been seed-shopping.

## Leave-one-out ablation of the promoted recipe

Run *after* promotion, so it cannot have influenced selection. 2 seeds, 45 k→5 k val.

| variant | val acc | Δ |
|---|---|---|
| promoted (all on) | 0.9444 | — |
| − cutout | 0.9397 | −0.47 pp |
| − TTA (single view) | 0.9381 | −0.63 pp |
| − ZCA whitening (random init) | 0.9358 | −0.86 pp |
| − tuned weight decay (→0.0153) | 0.9344 | −1.00 pp |
| − tuned logit scale (→1/9) | 0.9342 | −1.02 pp |

Every component clears the noise floor. Notably the two largest contributors are the two
hyperparameters that were *mis-set* in the starting recipe — the search paid for itself
twice over relative to any architectural change.

## Correctness checks

* **Input convention.** The graded checkpoint scores 0.9995 on a train subset under the
  verifier's standardized convention, versus 0.5449 for `[0,1]` inputs and 0.7681 if
  standardization were applied twice. `forward` does no normalization; the model is
  unambiguously matched to the verifier's convention, not accidentally compensating.
* **Ladder integrity.** 32 files, epoch numbers contiguous 1..32, mtimes strictly
  increasing, span 12.25 s consistent with the 13.89 s measured run wall-clock.
* **Promotion without restamping.** The winning ladder was moved into the graded directory
  with `os.link` (hardlink → same inode → the mtimes written by `ScriptModule.save()`
  during the run are preserved verbatim) and swapped in by `rename`, so a complete ladder
  was in place at every instant. Copying would have restamped all 32 files to within a
  second of each other and destroyed the genuine timing signal.
* **ZCA fit is unbiased.** The on-disk cache is class-sorted, so the whitening statistics
  are accumulated over *every* image of the training split in chunks rather than from a
  prefix sample, which would have been all-airplane.
* **Reported metrics.** `agent_reported_metrics.json` reports the full training-run
  wall-clock (13.89 s ≥ the 12.25 s checkpoint span), not the time to 94 %.

## Honest limitations

* The 0.9418 at the crossing epoch carries only 0.18 pp of margin over the threshold. This
  is not load-bearing (epochs 31/32 also cross, and `S_outcome` saturates well before t\*
  matters), but the *specific* crossing epoch would move under a different seed.
* S3/S3b screened at 2 seeds against a 0.25 pp noise floor, so it can only detect effects
  ≳0.5 pp. Several factors reported as "within noise" may be real but small; the study was
  deliberately powered to find the large mis-settings, not to rank the small ones.
* `S_outcome` saturates at t\* ≤ 28 s and this run sits at 11.86 s, so roughly 2.4× of the
  speed headroom is unused. Given that a run scoring 0 if it fails to cross is the only real
  risk, that headroom was deliberately spent on crossing margin rather than on shaving
  epochs — see PROTOCOL §1.
* Val (45 k-trained) systematically understates the 50 k-trained test number by ~0.2–0.4 pp;
  the promotion rule's 0.945 bar was set to absorb this, but it was not independently
  calibrated.

## Files

`train.py` model/data/augmentation/training loop · `run_ladder.py` full-50 k run + ladder ·
`verify_ladder.py` verifier-convention re-evaluation (only test reader) · `sweep.py` +
`sweep_main.py`/`sweep_s3.py`/`sweep_s3b.py`/`sweep_s4.py` held-out study ·
`ablate.py` leave-one-out · `cache_data.py` PNG decode · `reproduce.sh` end-to-end repro ·
`artifacts/PROTOCOL.md` pre-registration · `artifacts/DEVIATIONS.md` · `artifacts/*.json`
raw sweep and verification records.
