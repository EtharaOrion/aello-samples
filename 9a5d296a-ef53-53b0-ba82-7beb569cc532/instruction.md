# AELLO-C8-S10

maximise accuracy on the private test nodes at a fixed total neighbourhood-sampling budget per epoch.

## What is graded

Write your graded artifact to **`/submission/preds_c8s10.npy`**. It must be integer label array, one row per graded instance. A missing, malformed or out-of-domain artifact scores 0.

Your score is `r_det`, computed by the verifier in a separate environment from state it measures itself. The graded quantity is Accuracy over the private test nodes.

`r_det` is zero on any red line and on any knockout failure, before any quality is considered.

## The budget

- One graded run. `max_timeout` bounds the refinement loop; each attempt must finish inside its per-attempt budget.
- Up to 50 attempts. Your reward history is carried between attempts.
- You must declare your own budget accounting for the graded run; the declaration is what is graded, not the harness's own measurement.

## Red lines

Any of these scores zero:

- Any read of the private test nodes' labels.
- Any write into tests/ during the graded run.

## What ships

- `environment/` -- the corpus and a runnable starter.
- `environment/starter/` -- a correct, unoptimised baseline you may replace entirely.

## Traps

- Stacks depth until representations over-smooth and accuracy falls below the MLP control.
- Spends the whole fan-out budget on the first hop.
- the OGB leaderboard publishes strong recipes for this exact split Mitigation: the sampling budget is authored and is not the leaderboard setting, so a transferred recipe exceeds it and is rejected by the budget check
