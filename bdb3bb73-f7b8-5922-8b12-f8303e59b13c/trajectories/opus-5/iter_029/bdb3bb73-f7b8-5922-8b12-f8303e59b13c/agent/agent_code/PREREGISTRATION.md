# Pre-registration — written BEFORE the sweep results were observed

## Objective
Maximise held-out accuracy at a fixed epoch budget, because `S_outcome` is already
saturated (`t* <= 28 s` gives 1.0 and our epochs cost ~0.8 s, so anything up to ~30
epochs scores full marks). The binding risk is therefore **not** speed, it is that the
verifier's independent re-evaluation lands below 0.94 on the checkpoint we believed
crosses. Extra accuracy buys crossing-checkpoint margin and a longer tail of
above-threshold checkpoints.

## Measurement protocol
- **Selection metric**: top-1 with the 6-way in-forward TTA on a **held-out slice of
  the TRAIN split** — 5,000 images, drawn with a fixed generator (seed 1234) that is
  independent of the training seed, so every config sees the identical split.
- The **test split is never used to choose anything**. It is read only by
  `eval_ladder.py`, after training, to verify a saved ladder against the verifier's
  exact pipeline (the task statement explicitly directs us to do this).
- **Noise floor**: at p≈0.94 on n=5000 the binomial standard error is
  sqrt(.94*.06/5000) = 0.335 pp. Seed-to-seed training variance adds to this. So a
  single-run difference below ~0.7 pp carries no information and must not be acted on.
- Each config is run with seeds {0, 1}; configs are compared on the **mean over seeds**.

## Decision rule (fixed in advance)
1. Incumbent = `lr_final 0.07, cutout 0, dirac 1, bn_weight_grad 0` (24 epochs).
2. A challenger is promoted only if its 2-seed mean beats the incumbent's 2-seed mean
   by **more than 0.35 pp** (≈1 SE of the mean of two runs). Otherwise keep the
   incumbent — ties go to the incumbent, not to the larger number.
3. The winner is then re-run at 3 seeds to confirm before it goes into the graded
   ladder. If confirmation fails to hold the margin, fall back to the incumbent.
4. Epoch count for the graded run is chosen so that the expected first crossing leaves
   **>= 4 checkpoints above 0.94** and the projected `t*` stays **<= 24 s**, keeping
   headroom against the 28 s S_outcome saturation point in case of GPU contention.

## Sweep grid (2 seeds each)
- `lr_final` in {0.07, 0.02} — iteration-27 notes report a real gain from annealing
  harder at the end.
- `cutout` in {0, 8} — iteration-28 notes report regularisation was the bottleneck.
Full 2x2 factorial so the interaction is visible rather than assumed.

## Safety invariant
A complete, verified ladder is already on disk (32 epochs, first crossing epoch 31 at
0.9444, t* 23.8 s) plus `/workspace/checkpoint.pt`. It is only replaced by a ladder
that has been re-evaluated end-to-end with `eval_ladder.py` and shown to be better.
