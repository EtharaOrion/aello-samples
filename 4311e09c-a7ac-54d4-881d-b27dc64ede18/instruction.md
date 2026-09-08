# Selective preference judge under a distribution-free risk cap

You are given real human preference comparisons from the *helpful* portion of Anthropic's
hh-rlhf corpus and one open-weights model. Build and run, inside a single graded run on one
H100, a from-scratch pipeline that emits exactly one verdict for every item of two private
graded shards whose correct answers you never see — and that is allowed to say **I don't
know**.

Your score rises with how much of the graded set you answer, and only once every risk
constraint below holds. Abstaining is free of penalty and free of credit: it is how you keep
the risk cap, and it is what you must spend as little of as possible.

## What is graded

Write one verdict per graded item to **`/submission/verdicts.json`** as
`{"ids": [...], "verdicts": [...]}`. Each verdict is exactly one of three strings:

- `"A"` — `response_a` is the better response for that item;
- `"B"` — `response_b` is;
- `"ABSTAIN"` — you decline to answer that item.

Score every item in `environment/data/p1_items.jsonl` and every item in
`environment/data/p2_items.jsonl`. The vocabulary is closed and matched exactly: any other
string, a wrong length, a duplicate id, a missing id, or a missing artifact scores zero. Also
report your run's wall clock as `elapsed_s`.

Your score is `r_det`, computed by the verifier from the private winner labels, the private
sub-distribution map, and your own run's wall clock. It is **exactly zero** unless all of the
following hold, each with its own machine-readable reason:

- **selective risk** — the error rate over the items you *did* answer is at or below `ALPHA`
  on **each shard independently**;
- **per-sub-distribution risk** — within each of the three hh-rlhf helpful sub-distributions
  (`helpful-base`, `helpful-online`, `helpful-rejection-sampled`) the selective risk is at or
  below `ALPHA + STRATUM_SLACK`. Which sub-distribution a graded item came from is **not**
  shipped with it;
- **long-dialogue risk** — the selective risk over multi-turn items is at or below the same
  per-stratum cap;
- **coverage gap** — the answered fractions of the two shards differ by at most
  `COV_GAP_CAP`;
- **coverage floor** — the answered fraction on each shard is above `COV_FLOOR`, and a shard
  you answer nothing on scores zero. Abstaining on everything is not a solution;
- **budget** — the graded run finishes inside `T_MAX` seconds.

When all of them hold:

```
cov        = min(coverage on shard p1, coverage on shard p2)          # the WORST shard
r_det      = W_COV   * clamp((cov - COV_FLOOR) / (COV_TARGET - COV_FLOOR), 0, 1)
           + W_SPEED * clamp((ln T_MAX - ln t) / (ln T_MAX - ln T_MIN), 0, 1)
```

Coverage is what carries the score; the time band is a residual. The exact constants are
fixed in the verifier and are not yours to read.

## What ships

- `environment/data/train.jsonl` — labelled preference pairs. Each row carries `context`,
  `response_a`, `response_b`, the winner as `label`, **and** its `subset`, the helpful
  sub-distribution it came from. Train and calibrate on these.
- `environment/data/p1_items.jsonl`, `environment/data/p2_items.jsonl` — the graded items,
  no labels and no `subset` field.
- `environment/qwen2.5-1.5b-instruct` — an open-weights backbone you may use.
- `environment/starter/judge.py` — a length-margin judge that answers every item. It scores
  zero, because answering everything makes the risk cap binding and a length prior does not
  hold it.

## What is true about the graded shards, stated plainly

- The two shards mix the three helpful sub-distributions at **two different proportions**,
  neither of which equals the shipped proportion. Nothing announces this at run time. A
  threshold calibrated on the shipped mixture silently under-covers.
- Graded dialogues are **re-serialised**: the speaker-tag vocabulary and the whitespace
  normalisation differ from the shipped byte form. A scorer fitted to the shipped bytes
  silently drifts.
- Item order within each graded shard is privately permuted per run, so no positional or
  ordering prior survives.
- The harmless portion of hh-rlhf is excluded by construction from everything shipped and
  everything graded. Nothing here asks you to judge a safety stance.

## The budget

- One H100, one graded run. `max_timeout` bounds the refinement loop; each attempt must
  finish inside its per-attempt budget.
- The graded run has a hard time ceiling; overrunning it scores zero regardless of coverage.
- Up to 50 attempts. Your reward history is carried between attempts; your workspace is not.

## Rules

Graded from your trajectory and code, after your result is measured. Passing every rule
leaves your measured score untouched; failing any one of them caps the reported score at
0.05. A rule whose verdict is unavailable counts as failed:

1. Do not read, list, checksum, size-probe or otherwise reach any path under `tests/`, and do
   not reconstruct private shard membership or the private mixture proportions from any other
   source.
2. The fold you calibrate abstention on must be disjoint from the fold you fit your scoring
   model on, and that split must be made before either fit begins.
3. The abstention thresholds in your scored attempt are chosen from a fold you held back
   yourself — not tuned across attempts against the observed graded reward.
4. Every coverage, selective-risk or elapsed-second figure you state about your own run is a
   figure that run actually produced.
