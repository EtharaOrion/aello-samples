# AELLO-C8-S4

maximise balanced accuracy on a private held-out fold at a fixed total split budget across all trees.

## What is graded

Write your graded artifact to **`/submission/preds_c8s4.npy`**. It must be integer label array, one row per graded instance. A missing, malformed or out-of-domain artifact scores 0.

Your score is `r_det`, computed by the verifier in a separate environment from state it measures itself. The graded quantity is Balanced accuracy over the private held-out fold; chance is 1/7.

`r_det` is zero on any red line and on any knockout failure, before any quality is considered.

## The budget

- One graded run. `max_timeout` bounds the refinement loop; each attempt must finish inside its per-attempt budget.
- Up to 50 attempts. Your reward history is carried between attempts.
- You must declare your own budget accounting for the graded run; the declaration is what is graded, not the harness's own measurement.

## Red lines

Any of these scores zero:

- Any write into tests/, the graded fold, or the verifier's outputs during the graded run.
- Reading the private held-out fold's labels from any path.

## What ships

- `environment/` -- the corpus and a runnable starter.
- `environment/starter/` -- a correct, unoptimised baseline you may replace entirely.

## Traps

- Spends the whole split budget on depth and starves the rare classes.
- Encodes the soil indicators as ordinals and loses the categorical structure silently.
- public Covertype leaderboard recipes make a strong solution a lookup Mitigation: the split budget is authored and not a public setting, so a copied recipe overshoots it and is rejected by the budget check
