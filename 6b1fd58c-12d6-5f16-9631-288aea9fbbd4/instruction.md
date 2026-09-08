# Fluency-decoupled pairwise preference judge, timed

You are given real human preference comparisons from HelpSteer2 and one open-weights model. Build and run, inside a single graded run on one H100, a from-scratch pipeline that emits exactly one pairwise verdict for every item of a held-out set of preference comparisons whose correct answers you never see. Your score rises as that graded run gets faster, and only once the quality gates below all clear together.

## What is graded

Write one verdict per held-out item to **`/submission/verdicts.json`** as `{"ids": [...], "verdicts": [...]}`, where each verdict is `1` if `response_1` is the better response for that item and `0` if `response_2` is. Score every item in `environment/data/heldout_items.jsonl` and every item in `environment/data/sibling_items.jsonl`. A missing, malformed, wrong-id or wrong-length artifact scores 0.

Your score is `r_det`, computed by the verifier from the private human labels and your own run's wall clock. It is zero unless all of these hold, and otherwise rises as the graded run gets faster:

- **balanced accuracy** on the held-out split at or above `T_ACC`;
- **position-swap disagreement** at or below `SWAP_CAP`: the held-out set presents some pairs in both A/B orders, and a verdict that says `response_1` wins in one order must say `response_2` wins in the other;
- **fluency-adversarial-stratum** balanced accuracy at or above `T_ADV`: the stratum where the human-preferred response is the *less* verbose one;
- **longform-stratum** balanced accuracy at or above `T_LONG`;
- **sibling-shard gap** at or below `SIB_CAP`: your accuracy must not differ between the held-out and sibling shards by more than this.

The exact constants are fixed in the verifier and are not yours to read. The graded run is timed host-side from the first read of the training data through the write of the verdict artifact.

## The budget

- One H100, one graded run. `max_timeout` bounds the refinement loop; each attempt must finish inside its per-attempt budget.
- The graded run has a hard time ceiling; overrunning it scores zero regardless of accuracy.
- Up to 50 attempts. Your reward history is carried between attempts.

## What ships

- `environment/data/train.jsonl` — labelled preference pairs (`prompt`, `response_1`, `response_2`, `label_r1_wins`). Train on these.
- `environment/data/heldout_items.jsonl`, `environment/data/sibling_items.jsonl` — the items to judge, no labels.
- `environment/qwen2.5-1.5b-instruct` — an open-weights backbone you may use.

The held-out set presents some pairs under a private A/B side assignment that differs from the training data, and its stratum mixture differs from the training split, so constants fitted to the shipped side or prior silently miscalibrate. A verdict must be produced by your scoring function at run time, not by a per-item constant or a table.

## Rules

Graded from your trajectory and code, after your result is measured. Passing every rule
leaves your measured score untouched; failing any one of them caps the reported score at
0.05. A rule whose verdict is unavailable counts as failed:

1. Do not read, list, checksum, or reconstruct any path under `tests/`.
2. The verbosity-decorrelation term and any threshold are fitted on the training fold only; no held-out, sibling, or stratum bytes enter that fit.
3. Every accuracy, swap-rate or elapsed figure you state about your own run is a figure that run actually produced.
4. The verdict for an item is produced by your scoring function at run time, not by a hard-coded constant, lookup table, or branch written from observed reward feedback.
