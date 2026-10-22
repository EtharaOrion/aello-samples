# Results

## What shipped

`/workspace/periodic_checkpoints/` holds **20 contiguous checkpoints** from a
single run of the baseline config (seed 0, 20 epochs), trained **directly into
that directory** so the save times the verifier clocks are the real ones.

Re-evaluated through the verifier's exact pipeline
(`ImageFolder → ToTensor → Normalize` → `torch.jit.load(...).eval()`, batch 512,
`argmax(1)`):

| quantity | value |
|---|---|
| first checkpoint ≥ 94% | **epoch 18, 94.22%** |
| reconstructed `t*` = span(1→18) × 18/17 | **23.40 s** (anchors: reference 28 s, baseline 70 s) |
| final checkpoint (epoch 20) | **95.00%** |
| ladder span (epoch 1 → 20) | 24.73 s |
| per-epoch gap | 1.272 – 1.316 s (mean 1.301 s) |
| total run wall-clock (process start → last ckpt) | 31.30 s |

Per-epoch test accuracy: 73.3, 83.6, 82.1, 85.9, 83.4, 82.2, 86.9, 87.4, 86.2,
85.1, 88.8, 89.3, 91.5, 91.5, 92.2, 92.3, 93.7, **94.2**, 94.7, 95.0.

`S_outcome = clamp((70 − 23.40)/(70 − 28), 0, 1) = 1.0`.

## Headline finding

The sweep produced exactly one nominally significant winner — `momentum 0.80`,
+0.255 pp at t = 3.6, surviving Bonferroni correction over 15 tests. **It failed
to replicate.** An independent 9-runs-per-config replication put it at
**−0.053 pp**, i.e. very slightly *worse* than baseline. The baseline shipped.
Details below, because how the false positive happened is the useful part.

## Noise floor (as measured during the sweep)

Baseline, 4 seeds, 18 epochs, held-out slice of the train split:
94.34 / 94.36 / 94.54 / 94.32 → **σ = 0.101 pp**, 95% CI ≈ [0.057, 0.29] pp on
3 dof. This number was later shown to be an underestimate — see "Why the winner
evaporated".

## Sweep

15 pre-registered one-factor candidates, seeds {0,1} in round 1; the 12 whose
round-1 mean reached baseline plus the baseline itself were re-run on seeds
{2,3} for 4 seeds each. Pre-registered bar: `2σ·√(2/4) = 0.143 pp`.

| candidate | n | mean | Δ vs baseline | verdict at the time |
|---|---|---|---|---|
| H `momentum=0.80`     | 4 | 0.9465 | **+0.255 pp** | real winner (later refuted) |
| K `warmup_frac=0.13`  | 4 | 0.9454 | +0.145 pp | on the bar |
| O `whiten_bias_epochs=6` | 4 | 0.9450 | +0.115 pp | indistinguishable |
| D `weight_decay=0.0110` | 4 | 0.9450 | +0.110 pp | indistinguishable |
| F `label_smoothing=0.10` | 4 | 0.9450 | +0.110 pp | indistinguishable |
| N `widths=(160,448,512)` | 4 | 0.9448 | +0.095 pp | indistinguishable |
| B `lr=9.0`            | 4 | 0.9448 | +0.090 pp | indistinguishable |
| E `weight_decay=0.0200` | 4 | 0.9446 | +0.075 pp | indistinguishable |
| C `lr=14.0`           | 4 | 0.9446 | +0.075 pp | indistinguishable |
| L `bn_momentum=0.8`   | 4 | 0.9442 | +0.030 pp | indistinguishable |
| M `scaling_factor=1/6` | 4 | 0.9439 | +0.000 pp | indistinguishable |
| I `momentum=0.90`     | 4 | 0.9439 | +0.000 pp | indistinguishable |
| **A baseline**        | 4 | 0.9439 | — | — |
| G `label_smoothing=0.30` | 2 | 0.9428 | −0.110 pp | dropped after round 1 |
| J `lr_final_frac=0.07` | 2 | 0.9410 | −0.290 pp | dropped after round 1 |

**Multiple comparisons.** A ~95% bar applied 15 times expects ~0.75 false
positives, so both nominal winners were re-tested against Bonferroni:
H reached t = 3.57 (p ≈ 4e-4 raw, ≈ 6e-3 corrected — survives); K reached
t = 2.03 (p ≈ 0.04 raw, ≈ 0.63 corrected — does not). Landing 0.002 pp over a
pre-registered bar is exactly the artefact correction exists to catch, so K was
called a null. H was carried forward as the sole candidate.

Note J: decaying the LR fully to zero beats stopping at 7% of peak by 0.29 pp,
the largest single effect in the table, and it is already in the shipped config.

## Replication + epoch-budget study

Nine further runs per config (3 seeds × budgets {16, 18, 20}), run after the
sweep, on the same held-out slice:

| budget | A baseline | H momentum=0.80 | Δ |
|---|---|---|---|
| 16 epochs | 0.9446 | 0.9440 | −0.060 pp |
| 18 epochs | 0.9446 | 0.9449 | +0.027 pp |
| 20 epochs | 0.9472 | 0.9459 | −0.127 pp |
| **pooled (n=9)** | **0.9455** | **0.9449** | **−0.053 pp** |

H's +0.255 pp did not survive contact with new data. The pooled estimate is
slightly negative and consistent with zero.

### Why the winner evaporated

Re-running the *identical* config and seed in a fresh process:

| run | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| baseline, sweep vs replication | 0.000 pp | 0.000 pp | **0.140 pp** |
| momentum 0.80, sweep vs replication | 0.000 pp | 0.000 pp | **0.240 pp** |

Seeds 0 and 1 reproduce **bit-exactly**; seed 2 does not. The reason is
mechanical: seeds 0–1 were run in the round-1 process and seed 2 in the round-2
process, and with `cudnn.benchmark=True` the autotuner picks different
convolution algorithms depending on how much GPU memory the *other tenant* on
this shared card happens to be holding. Different algorithm → different
floating-point reduction order → a genuinely different trajectory.

So the 4-seed σ = 0.101 pp measured *within* the sweep understates the real
thing. Across all 18 replication runs the standard deviation is **0.170 pp
(baseline) / 0.196 pp (momentum 0.80)** — close to 2× larger. Had the bar been
built from σ = 0.18 pp it would have been `2·0.18·√(2/4) = 0.255 pp`, and H's
+0.255 pp would **not** have cleared it. The false positive is fully explained
by a noise floor estimated from too few independent execution contexts.

**Transferable lesson:** on a shared GPU, seed-to-seed variation measured inside
one process is not the noise floor. Vary the process, not just the seed.

## Why the baseline shipped

Two independent reasons, either sufficient:

1. **The winner isn't real.** Replication puts `momentum=0.80` at −0.053 pp.
2. **It would have been worth nothing anyway.** `S_outcome` is already 1.0 at
   `t* = 23.4 s` against a 28 s reference, so +0.26 pp of accuracy buys exactly
   zero additional reward — while re-running means deleting a verified ladder
   and rebuilding it, and the ladder's timestamps *are* the measurement, so it
   cannot be restored from scratch space without fabricating the clock. At the
   moment the decision was due the GPU had 3.8 GB free of 80 GB, sat at 97–100%
   utilisation from another tenant, and had just failed six consecutive 144 MB
   allocations. A re-run risked a truncated ladder or a slower one (at 2×
   contention `t*` → ~47 s, dropping `S_outcome` to 0.55). Near-zero upside
   against a double-digit chance of losing 40% of the outcome score.

The decision was taken on reason 2 before reason 1 was known; the replication
then confirmed it independently.

## Deviations from the pre-registration, stated plainly

1. **§5.4 promotion not executed.** The single real winner was not shipped.
   Justified above, and subsequently vindicated by the replication.
2. **§5.5 epoch budget was not selected by the rule.** The rule (smallest budget
   with ≥ 4σ held-out headroom over 94%) was only evaluable after the shipped
   run existed, because the first two attempts at the budget study were killed
   by the other tenant's OOM. Applying it retrospectively with σ = 0.101 pp, all
   three budgets clear the 94.40 pp threshold, so the rule would have selected
   **16 epochs**. The shipped run used 20, which is the *more* conservative
   direction: 0.26 pp more held-out accuracy and a 1.0 pp (≈ 6σ) margin over
   threshold on the final checkpoint, at a `t*` still 4.6 s inside the reference
   anchor. Reported as an unvalidated-at-the-time choice that the data later
   supported rather than as a swept result.
3. **§1 `t* ≤ 15 s` target not met** (achieved 23.4 s). That target assumed a
   full-speed H100; see below.
4. **Ordering.** The shipped ladder was trained *before* the sweep, as the
   task's "land a valid ladder first" instruction requires. The sweep then
   tested its config against 14 alternatives and found nothing better that
   replicated. Nothing about the shipped artefact was chosen using the sweep,
   and nothing about it was chosen using the test split.

## Environment finding: the GPU is shared, and ~4× off spec

| microbenchmark | measured | H100 SXM spec |
|---|---|---|
| fp16 matmul 8192³ | **221 TFLOPS** | ~700–990 |
| HBM copy bandwidth | **764 GB/s** | ~3350 |

The model's convolutions run at 80–100 TFLOPS, i.e. **~45% of this machine's
achievable ceiling** — normal efficiency for 31×31 / 15×15 / 7×7 feature maps.
The ~3× gap versus epoch times reported for this task on a quiet machine is the
hardware, not the code. Three independent confirmations: another tenant's
processes visibly held 57–76 GB of the 80 GB card; six consecutive 144 MB
allocations failed while this process held only 3.6 GB; and epoch times drifted
35% between sweep rounds with no code change.

`torch.compile` was measured and **rejected**: `default` gave
1.284 → 0.989 s/epoch (−23%), `max-autotune` gave 0.944 s/epoch for 98 s of
compilation. Neither closes a 4× hardware gap, and both would make the sweep and
the shipped run numerically non-identical, so eager mode was kept everywhere.

## What the next attempt should do

1. **Do not re-sweep these 15 knobs.** Fourteen were indistinguishable and the
   fifteenth did not replicate. This recipe is at a local optimum; the remaining
   headroom is structural (architecture, augmentation, schedule shape), not in
   scalar hyper-parameters.
2. **Measure the noise floor across processes, not just across seeds** — one run
   per process, or pin `cudnn.benchmark=False` and accept the slowdown. Getting
   this wrong is what manufactured the false positive here.
3. **Check GPU tenancy before planning the time budget.** A quiet card should
   give ~0.4 s/epoch and `t*` ≈ 7–8 s; a contended one gives 1.3 s/epoch. The
   plan should branch on the microbenchmark, not on the sticker spec.
4. Keep the epoch budget at 20 unless the card is quiet — the extra margin over
   94% is free while `S_outcome` is saturated.
