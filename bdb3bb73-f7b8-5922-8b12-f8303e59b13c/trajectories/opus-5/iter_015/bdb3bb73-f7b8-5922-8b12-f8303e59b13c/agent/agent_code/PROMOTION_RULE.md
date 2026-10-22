# Pre-committed promotion rule

Written **before** the finalist runs, so the choice cannot be rationalised after seeing
the numbers.

## Measurement protocol

- All hyperparameter selection uses a **5,000-image holdout carved out of the CIFAR-10
  *train* split** (`data.train_val_split`, fixed seed 0). The 10,000-image test split is
  never read during training or tuning.
- Measured noise floor (identical config, different seeds, `sweep_epochs.jsonl`):
  per-seed sd on holdout accuracy is **0.08–0.34 pp**, typically ~0.12 pp. With 3 seeds
  the standard error of the mean is therefore ~0.07–0.20 pp.
- **Any two configs whose 3-seed holdout means differ by < 0.20 pp are treated as tied.**

## Finalists

| id | config (deltas from `DEFAULT_CFG`) |
|----|------------------------------------|
| A  | epochs 26, widths 128/384/512, cutout 10 |
| B  | epochs 26, widths 128/384/512, cutout 8  |
| C  | epochs 32, widths 128/384/512, cutout 10 |
| D  | epochs 22, widths 128/384/512, cutout 10 |

## Decision rule

1. Run all finalists on seeds {0, 1, 2} with the 5,000-image holdout.
2. Rank by mean holdout accuracy.
3. Among configs tied with the leader (within 0.20 pp), pick the one with the **fewest
   epochs** (cheapest run, earliest crossing, smallest `t*`).
4. Retrain the winner on the **full 50,000-image train split** with seed 0 and promote
   that ladder. No test-split number is consulted before this point.
5. The test split is read exactly once afterwards, by `eval_ckpts.py`, purely to *report*
   the ladder's accuracy and to confirm the input convention matches the grader. If that
   run showed the ladder failing to cross 94% it would be a bug report, not a signal to
   re-pick hyperparameters.

## Reporting rule

`agent_reported_metrics.json` reports the **total wall-clock of the promoted training
run** (start of the timed loop to the last checkpoint save), not a time-to-94%.
