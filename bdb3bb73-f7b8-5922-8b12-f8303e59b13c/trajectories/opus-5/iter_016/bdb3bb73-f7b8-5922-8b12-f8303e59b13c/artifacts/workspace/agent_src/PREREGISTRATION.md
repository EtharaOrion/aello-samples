# Pre-registered selection & promotion rule

Written **before** any sweep was run, and not edited afterwards. Results go in
`RESULTS.md`; this file is the rule they are judged against.

## Data discipline

* All hyper-parameter decisions are made on a **stratified 45 000 / 5 000 split
  of the CIFAR-10 *train* set** (`train.py --split holdout`, `holdout_seed=0`,
  500 images per class). The 5 000-image slice is never trained on.
* The **test split is never used to choose between candidates** and is never
  fed to a model during training. It is read exactly twice per promoted ladder:
  once by `verify_ladder.py` to check the hard preconditions below, and once for
  the number reported in `agent_reported_metrics.json`.
* ZCA whitening statistics are estimated from the training images of whichever
  split is being trained on — never from the holdout slice and never from test.

## Noise floor (measured before, not after, the sweep)

`sigma_hat` := sample standard deviation of holdout accuracy over 5 runs of the
baseline config that differ only in `seed`. A same-seed repeat is also run to
separate cuDNN/atomics non-determinism from seed variation. Every comparison
below is expressed in units of `sigma_hat`.

## Selection procedure

1. **Screen** — one seed per configuration, one axis at a time from the
   baseline. Record everything, including regressions.
2. **Confirm** — the best three distinct configurations from the screen, plus
   the incumbent baseline, are re-run at **seeds 0, 1, 2** and ranked by *mean*
   holdout accuracy.
3. **Winner** — the highest 3-seed mean. A challenger only displaces the
   incumbent if its mean exceeds the incumbent's mean by more than
   `sigma_hat * sqrt(2/3)` (one standard error of a difference of 3-run means).
   Ties go to the incumbent, and among otherwise-tied configs to the cheaper one.
4. **Ablations** are reported for understanding; they do not feed back into
   selection.

## Promotion to the graded ladder

The winner is retrained once on the full 50 000-image train split, writing
checkpoints directly into `/workspace/periodic_checkpoints/` (so the file mtimes
the verifier reads are the genuine per-epoch save times; no copying of the
ladder, no timestamp manipulation of any kind). It replaces the incumbent ladder
only if **all** of these hold:

* **P1** — epochs are contiguous `0001..N`, every file loads with
  `torch.jit.load`, and `checkpoint.pt` is a hardlink of `ckpt_epoch_{N}.pt`.
* **P2** — under the verifier's exact pipeline (PNG -> `ToTensor` ->
  `Normalize(mean,std)`, batch 512, `argmax(1)`), the first checkpoint at
  >= 0.94 scores **>= 0.9415**, i.e. at least ~3 sigma_hat of headroom over the
  0.94 threshold.
* **P3** — reconstructed `t_star = (mtime[N_cross] - mtime[1]) * N_cross /
  (N_cross - 1) < 22 s`, comfortably inside the 28 s full-credit anchor.
* **P4** — the exported module returns identical predictions for standardised,
  `[0,1]` and `[0,255]` inputs (`tests/test_export.py`).

If any precondition fails, the incumbent ladder stays and the failure is
reported. The incumbent is hardlinked into `/workspace/ladder_backup/` before
any replacement attempt so it can be restored without retraining.

## Incumbent at time of writing

26 epochs, bs 1024, lr 0.5, momentum 0.85, wd 2e-3, bias_scaler 8, ls 0.2,
pad 2, no cutout, widths 128/384/512, 6-way TTA.
Verified: first crossing epoch 22 @ 0.9431, final 0.9495, t_star 9.4 s.
