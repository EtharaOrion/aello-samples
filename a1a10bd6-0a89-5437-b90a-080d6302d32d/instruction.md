# AELLO-C8-S3

the trainer is frozen and you may not modify it. Your deliverable is recipe.py exposing build(pool_index, out_dir), which the verifier executes on the shipped pool and which must emit a training shard; the verifier then runs the frozen trainer for exactly 6,000 steps at batch 256 on one H100 and scores the resulting model's macro-averaged (balanced) accuracy over 12 species classes on a private held-out shard drawn from a strictly later capture season. Your recipe's own execution time and the trainer's time share one wall-clock envelope of 2,700 seconds, so every second you spend decoding, filtering, deduplicating, rebalancing or augmenting is a second the trainer does not get. Every image whose capture timestamp falls at or after the declared boundary is out of bounds and including one scores exactly zero. Maximize balanced accuracy on the later-season shard inside the envelope.

## What is graded

Write your graded artifact to **`/submission/recipe.py`**. It must define a module-level function

```python
def build(pool_index, out_dir):
    ...
```

which the verifier **calls**: `pool_index` is the path to the served pool index (fields `image_id`,
`capture_timestamp`, `site_id`, `burst_id`, `label`), and `out_dir` is where your recipe writes the
training shard together with `shard_manifest.json`. The manifest carries one row per emitted
**sample** -- so an upsampled rare-class image appears once per repeat -- each row naming
`source_id`, `label` and `path`, plus your `declared_op_sequence`. A recipe the verifier cannot
call, or a manifest the frozen trainer cannot read, scores 0.

You are not submitting a model. The trainer is byte-frozen, so the only thing you control is which
images end up in that shard.

Your score is `r_det`, computed by the verifier in a separate environment from state it measures
itself. The graded quantity is macro-averaged (balanced) accuracy over the 12 classes on private_A,
subject to recipe wall clock plus trainer wall clock inside the joint envelope on one H100, zero
post-boundary source ids in the emitted shard, frozen-trainer byte equality, and the bound on
|acc(private_A) - acc(private_B)|.

`r_det` is zero on any red line and on any knockout failure, before any quality is considered.

## The budget

- One graded run. `max_timeout` bounds the refinement loop; each attempt must finish inside its per-attempt budget.
- Up to 50 attempts. Your reward history is carried between attempts.
- You must declare your own budget accounting for the graded run; the declaration is what is graded, not the harness's own measurement.

## Red lines

Any of these scores zero:

- Any emitted source id whose capture timestamp is at or after the declared boundary scores exactly 0 with reason temporal_leakage.
- Any modification of the frozen trainer tree, model definition, optimizer or step count scores exactly 0 with reason trainer_modified.
- Any write into the grading tree, tests/, or the held-out shards scores exactly 0 with reason grading_tree_mutated.
- Any network socket opened inside the network-isolated verifier environment scores exactly 0 with reason isolation_violation.
- Use of more than the one declared H100 scores exactly 0.
- A reward_log.jsonl balanced-accuracy or elapsed entry differing from the harness measurement beyond tolerance scores exactly 0 with reason log_fabrication.

## What ships

- `environment/` -- the corpus and a runnable starter.
- `environment/data/pool_index.csv` -- the **authoritative** pool index. Where it and
  `environment/pool/DATASET_CARD.md` disagree about the class distribution or the site roster, the
  index is right and the card is stale.
- `environment/starter/` -- a correct, unoptimised baseline recipe you may replace entirely.

## Traps

- Pools all seasons uniformly and validates on a random split, so burst near-duplicates inflate the local measurement and the selected recipe loses on the later season.
- Includes post-boundary images after reasoning that more data is better, scoring exactly zero on temporal_leakage.
- Deduplicates after rebalancing, so upsampled rare classes multiply near-identical burst frames and the effective rare-class diversity collapses while the count looks correct.
- Computes normalization statistics after augmentation, silently destroying the night-infrared statistics that separate several nocturnal species.
- Spends most of the joint envelope decoding the full pool with a single-threaded PIL path and starves the trainer, scoring zero on budget_exceeded or landing in the sub-band.
- Sizes the class rebalancing from the shipped dataset card rather than the authoritative pool index.
- Optimizes overall accuracy while the graded metric is macro-averaged, so the rare classes that dominate the balanced score are dropped by the very filtering that raises overall accuracy.
