# AELLO-C8-S13

minimise expected calibration error on private corrupted splits while holding clean accuracy inside a bound tolerance.

## What ships

- `environment/data/clean_train.npz` -- 20000 CIFAR-100 images with fine labels.
- `environment/data/clean_val.npz` -- 5000 CIFAR-100 images with fine labels.
- `environment/data/corrupted_graded.npz` -- 19000 CIFAR-100-C images, no labels, with `corruption_id`, `severity` and `corruption_names`. This is the 19 x 5 corruption grid: 19 corruptions at 5 severities, 95 cells, 200 images per cell, laid out lexicographically by `(corruption_id, severity)`.
- `environment/data/clean_graded.npz` -- 10000 clean CIFAR-100 test images, no labels.
- `environment/starter/` -- a correct, unoptimised baseline you may replace entirely.

These are the only labels you get. The graded labels for both graded folds are private.

## What is graded

Write your graded artifact to **`/submission/confidences.npy`**. It must be a float array of shape `(29000, 100)`:

- rows `0` to `18999`: the corrupted grid, in `corrupted_graded.npz` row order;
- rows `19000` to `28999`: the clean graded fold, in `clean_graded.npz` file order;
- 100 columns, the CIFAR-100 fine labels, in label order;
- every entry non-negative, every row summing to 1 within `1e-6`.

Both blocks are required. A missing, malformed or out-of-domain artifact scores 0, and so does a 19000-row array that covers only the corruption objective.

Your score is `r_det`, computed by the verifier in a separate environment from state it measures itself. The graded quantity is expected calibration error over the private corrupted splits, mapped to [0,1] higher-better. It is the **macro average over the 95 cells** -- one calibration error per (corruption, severity) cell, then the mean of the 95 -- not the pooled figure over all 19000 rows. One catastrophic corruption is not diluted by the 94 that went well.

Separately, top-1 accuracy on the clean block must stay at or above the clean-accuracy floor published in the checker's constants. Falling below it scores 0. This gate is the reason a flattened posterior is not a winning strategy.

`r_det` is zero on any red line and on any knockout failure, before any quality is considered.

## The binning

The partition your calibration error is measured under is fixed in the checker tree and is **not published**. Choose your own binning to fit with.

## The budget

- One graded run. `max_timeout` bounds the refinement loop; each attempt must finish inside its per-attempt budget.
- Up to 50 attempts. Your reward history is carried between attempts.
- You must declare your own budget accounting for the graded run; the declaration is what is graded, not the harness's own measurement. Write `/workspace/run_record.json` with an `agent_budget_declaration` object carrying at least:
  - `graded_forward_passes` -- an integer, at least 1;
  - `calibration_fit_split` -- one or more of `clean_train`, `clean_val`. Naming a graded split fails the check.

## Red lines

Any of these scores zero:

- Any read of the private corrupted splits' labels.
- Fitting calibration on the graded binning scheme.
- Writing anything under `tests/` or into the graded folds.

## Traps

- Buys calibration by flattening confidence and fails the accuracy tolerance.
- Fits temperature on the corrupted split it is graded on.
- Submits only the 19000 corrupted rows, which answers the stated objective and removes the one gate that prices the flattening shortcut.
- Optimises the pooled 19000-row figure and lets one catastrophic corruption hide inside an average the ramp does not read.
- emitting a uniform posterior trivially lowers calibration error Mitigation: CTL-UNIFORM is a compiled control and the clean-accuracy tolerance gate rejects any submission that buys calibration with accuracy
