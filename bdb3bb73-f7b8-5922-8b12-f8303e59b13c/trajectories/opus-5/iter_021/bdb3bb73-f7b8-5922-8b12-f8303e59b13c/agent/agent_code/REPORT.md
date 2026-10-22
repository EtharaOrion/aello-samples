# CIFAR-10 → 94% on 1×H100 — result and method

**Shipped:** 32-epoch ladder, **94.83%** test top-1, first checkpoint ≥94% at
**epoch 30**, reconstructed **t\* = 19.6 s**, total training wall clock 23.4 s.

Reproduce with `python promote.py --epochs 32 --lr-end 0.0 --seed 0`.

---

## 1. What the score actually rewards, and how that shaped the work

`S_outcome` is `clamp((70 − t*) / (70 − 28), 0, 1)`. The very first ladder I put
on disk crossed at t\* = 12.4 s, i.e. **S_outcome was already pinned at 1.0
before any tuning happened**, with >15 s of slack. Every second saved after that
is worth exactly zero, and every accuracy point above 94.0% is worth exactly
zero too.

So I did not optimize the clock. I spent the budget on measuring properly, and
used the slack to buy *margin* — distance above the 94% line, and distance below
the 28 s line — because margin is the only thing that protects the score against
variance. `PREREGISTRATION.md` states this before any experiment was run.

## 2. Setup

Airbench-style convnet reproduced from scratch, 6 082 848 parameters: frozen 2×2
ZCA whitening stem (3→24, eigen-filters emitted as ±pairs so the following GELU
keeps both signs), three `conv → maxpool → BN → GELU → conv → BN → GELU` groups
at widths 128/384/512, global `MaxPool2d(3)`, bias-free `Linear(512,10)`, logits
scaled by 1/9 in `forward`. SGD nesterov, momentum 0.85, bias/norm LR ×64, label
smoothing 0.2, batch 1000 (divides both 50 000 and 45 000, so every epoch is
exactly one full pass with no ragged tail). Augmentation is entirely on-GPU: two
`gather`s for a ±2 px translate plus a `where` for the mirror. 6-way TTA
(mirror × {0, −1, +1} px shift) is folded into the scripted `forward`.

**Data discipline.** All selection used a fixed class-balanced 5 000-image
holdout carved from *train* (`HOLDOUT_SEED = 1234`); sweep models saw only the
other 45 000. ZCA statistics come from train images only. The test split is read
in exactly one place, `verify_ladder.py`, and only to check a frozen artifact.

## 3. Noise floor first

Four seeds of the incumbent, holdout accuracy: 94.26 / 94.42 / 93.64 / 94.28.

> **mean 94.15%, σ = 0.35 pp.** Promotion bar: beat the incumbent mean by >1σ.

This is the whole reason the results below are readable. A 0.3 pp "improvement"
here is nothing, and several candidates landed exactly there.

## 4. Every cell I ran

All at 32 epochs unless noted, holdout accuracy, Δ vs the incumbent's 4-seed mean.

| candidate | n | mean | Δ | verdict |
|---|---|---|---|---|
| incumbent (`lr_end` 0.07) | 4 | 94.15 | — | baseline |
| **C6 `lr_end` → 0.0** | **3** | **94.86** | **+0.71 (2.0σ)** | **PROMOTED** |
| C3 peak LR ×0.5 | 2 | 94.42 | +0.27 | inside noise |
| C3 peak LR ×0.7 | 2 | 94.03 | −0.12 | inside noise |
| C3 peak LR ×1.25 | 2 | 94.22 | +0.07 | inside noise |

Then, re-based on the promoted `lr_end = 0` baseline (94.86):

| candidate | n | mean | Δ vs 94.86 | verdict |
|---|---|---|---|---|
| C1 epochs → 48 | 3 | 94.91 | +0.05 | inside noise, **not promoted** |
| C1 epochs → 64 | 2 | 94.85 | −0.01 | inside noise |
| C7 `peak_frac` → 0.10 | 2 | 94.96 | +0.10 | inside noise |
| C3 peak LR ×0.5 (re-probe) | 2 | 94.78 | −0.08 | inside noise |
| C2 label smoothing → 0.1 | 3 | 94.93 | +0.07 | inside noise, **not promoted** |
| C4 weight decay ×1.5 | 2 | 94.80 | −0.06 | inside noise |
| translate → 3 px | 2 | 94.99 | +0.13 | inside noise |
| epochs → 20 | 2 | 94.46 | −0.40 | worse |
| epochs → 16 | 2 | 94.26 | −0.60 | worse |

**Both near-misses died on their third seed, and that is the headline finding.**

| cell | seed 0 | seed 1 | 2-seed Δ | seed 2 | 3-seed Δ |
|---|---|---|---|---|---|
| C2 label smoothing 0.1 | 95.28 | 95.02 | **+0.29** | 94.50 | **+0.07** |
| C1 epochs 48 | 95.10 | 94.90 | **+0.14** | 94.74 | **+0.05** |

At two seeds, label smoothing 0.1 looked like a second real win (+0.29 pp,
consistent across both seeds, both above every baseline seed). The third seed
came back 94.50 and the effect evaporated. The 48-epoch cell did the same thing.
Two seeds agreeing is not evidence — with σ = 0.35 pp, a 2-seed mean still has a
0.25 pp standard error, and *both* of these cells would have been promoted by a
2-seed protocol. Neither is real.

**One winner out of twelve candidates.** Decaying the LR the whole way to zero
instead of stopping at 7% of peak is worth +0.71 pp — 2.0σ, and it reproduced on
all three seeds. Everything else — peak LR over a 2.5× range, schedule shape,
schedule length from 32 to 64 epochs — was flat inside the noise floor. The peak
LR is *not* the lever here, which is the opposite of what the incumbent's wildly
oscillating mid-training holdout curve suggested (that oscillation is BN
dynamics, not divergence; it resolves entirely in the last few epochs).

These last three cells were nearly lost: a `pgrep -f "sweep.py r3"` wait-loop I
wrote matched its own command line and spun instead of running the batch, which
cost ~7 minutes before I noticed. I shipped the graded artifact first and came
back for them, so the pre-registered candidate list is complete.

## 5. The GPU is shared, and that changed the shipping decision

Identical 32-epoch runs came back at 11.4 s, 15.1 s, 24.4 s and 25.0 s. That
looked like my measurement being broken, but `nvidia-smi` showed the card at
**100% utilization and 543 W drawn by a process outside my container**, at full
1980 MHz clocks with no throttle flags. Throughput here swings ~3× for reasons I
do not control.

That matters because t\* is reconstructed from checkpoint save times, so
contention inflates it. The first two attempts to ship the 32-epoch config were
**refused by the pre-registered validity gate** at t\* = 36.1 s and 41.1 s —
those runs reached 95.04% and 95.16% test accuracy and I threw them away anyway,
because 95.16% at t\* = 41 s scores `S_outcome = 0.69` while 94.18% at
t\* = 12.4 s scores 1.0. Accuracy above 94% is worthless; the clock is not.

I fell back to a 20-epoch ladder (94.72%, t\* = 16.4 s, promoted), then retried
32 epochs when contention eased and it passed at t\* = 19.6 s. The gate never
lets a worse ladder overwrite a better one: it refuses the swap unless the new
ladder is contiguous, contains a ≥94% checkpoint, and reconstructs to t\* ≤ 26 s.
A failed attempt costs ~90 s and leaves the incumbent untouched.

## 6. Shipped ladder (verifier-replica re-evaluation)

`verify_ladder.py` loads each checkpoint with `torch.jit.load(path).eval()` and
feeds float32 NCHW batches of 512 test PNGs through exactly
`Compose([ToTensor(), Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616))])`.

| epoch | 29 | 30 | 31 | 32 |
|---|---|---|---|---|
| test acc | 0.9380 | **0.9437** | 0.9457 | **0.9483** |

- first crossing: **epoch 30**, t\* = (save₃₀ − save₁) × 30/29 = **19.59 s**
- 32 contiguous checkpoints, `checkpoint.pt` is a byte-identical copy of
  `ckpt_epoch_0032.pt` (md5-checked)
- ladder span 19.74 s ≤ reported wall clock 23.39 s ✓

`forward` also auto-detects `[0,1]` and `[0,255]` inputs and normalizes only if
the batch minimum is non-negative — a standardized CIFAR batch always contains
values near −2, so the verifier's already-normalized input passes through
untouched. This removes the double-normalization failure mode entirely rather
than relying on my getting the convention right.

## 7. Honest limitations

- The +0.71 pp `lr_end` win is measured on a 5 000-image holdout; its standard
  error is ~0.33 pp, so the effect is real but the *size* is loosely pinned.
- Final test accuracy across three full-train runs of the same 32-epoch config
  was 95.04 / 95.16 / 94.83 — a ~0.3 pp spread. The shipped 94.83% is the
  *lowest* of the three; I shipped it because it is the run whose ladder passed
  the timing gate, not because it was the best draw.
- The two most promising candidates (label smoothing 0.1, 48 epochs) were each
  taken to 3 seeds and both landed at Δ ≈ +0.05 pp. They are not promoted and I
  do not believe they are real. With more budget I would put seeds 4–6 on them
  rather than open a new sweep axis, but I would expect nothing.
- `σ = 0.35 pp` is itself estimated from only 4 seeds, so the bar has real
  uncertainty around it. Every "inside noise" verdict above inherits that.
- Only `lr_end` was ever varied *and* confirmed at 3 seeds against a 4-seed
  baseline. One confirmed effect out of twelve candidates is the honest yield.
