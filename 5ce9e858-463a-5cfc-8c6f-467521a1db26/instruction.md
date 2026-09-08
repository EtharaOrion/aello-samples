# AELLO-C8-S9

minimise normalised forecast error on a private future window at a fixed prediction horizon and a fixed lookback budget.

## What is graded

Write your graded artifact to **`/submission/forecast.npy`**. It must be float array (n_windows, horizon, n_targets). A missing, malformed or out-of-domain artifact scores 0.

Your score is `r_det`, computed by the verifier in a separate environment from state it measures itself. The graded quantity is Normalised mean absolute error over the private future window, mapped to [0,1] higher-better.

`r_det` is zero on any red line and on any knockout failure, before any quality is considered.

## The budget

- One graded run. `max_timeout` bounds the refinement loop; each attempt must finish inside its per-attempt budget.
- Up to 50 attempts. Your reward history is carried between attempts.
- You must declare your own budget accounting for the graded run; the declaration is what is graded, not the harness's own measurement.

## Red lines

Any of these scores zero:

- Any use of observations after the private window's start as training signal.
- Any write into tests/ during the graded run.

## What ships

- `environment/` -- the corpus and a runnable starter.
- `environment/starter/` -- a correct, unoptimised baseline you may replace entirely.

## Traps

- Shuffles the split and reports an error the future window cannot reproduce.
- Normalises with statistics computed over the whole series including the future.
- time-order leakage is the dominant way to inflate this metric Mitigation: CTL-LEAK is a compiled red-line control and a shuffled split scores exactly 0
