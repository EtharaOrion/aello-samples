# C1-S1

minimize the harness-measured wall-clock seconds, from the first read of the shipped corpus to the harness timestamp of the earliest emitted checkpoint that simultaneously scores bits-per-byte at or below 1.12 on the far-horizon held-out shard H3 and a forward degradation slope bpb(H3) minus bpb(H1) at or below 0.06, for a from-scratch decoder-only language model trained on one H100 inside a hard budget of 4.0e8 consumed tokens, at most 1.2e8 parameters, a recency-quota share at or below 0.15 of consumed tokens drawn from the final 180 days before the declared cutoff, no token attributed to a document dated at or after that cutoff, and a hard ceiling of 2700 wall-clock seconds. The starter is provided and improving it is the objective.

The bar is a CONJUNCTION taken on one checkpoint. A checkpoint that clears bits-per-byte while its slope is above 0.06 stops no clock, and neither does a checkpoint that clears the slope with bits-per-byte above 1.12. Two checkpoints, one clearing each bar, are not a crossing.

The held-out shards are drawn from three publication windows past the declared cutoff C: H1 from [C, C+90d), H2 from [C+90d, C+270d) and H3 from [C+270d, C+630d), plus a same-window sibling of H2 drawn from a disjoint PMCID partition. The cutoff, the three window offsets, the quota ceiling, the slope bound and the full sub-gate partial-credit ramp are disclosed here, so the graded quantity is never withheld.

## The artifact

Write your graded artifact to **`/submission/c1s1.json`**. It is a JSON object with exactly these
five sections; a missing section is a malformed submission, not a weak one.

```jsonc
{
  "era_calendar": {                        // the temporal frame, every number MEASURED
    "cutoff_epoch_day": 20000,             // the declared cutoff C, in days since the epoch
    "corpus_last_pub_epoch_day": 19960,    // max pub_epoch_day over environment/data/docmeta.bin
    "blind_band_days": 40,                 // cutoff_epoch_day - corpus_last_pub_epoch_day
    "recency_window_start_epoch_day": 19820  // cutoff_epoch_day - 180, the quota window opens here
  },
  "schedule": {                            // mirrors schedule.json, in executed order
    "oldest_era_floor_share": 0.12,        // the share of each phase reserved for the oldest era
    "phases": [
      {"name": "broad", "planned_tokens": 200000000, "is_recency_anneal": false,
       "era_weights": {"e1": 0.40, "e2": 0.30, "e3": 0.20, "e4": 0.10}}  // weights sum to 1.0
    ]
  },
  "token_ledger": {
    "served_tokens_total": 370000000,      // must reconcile with the shim counter
    "per_phase": [
      {"name": "broad", "served_tokens": 200000000, "oldest_era_served_tokens": 80000000}
    ]
  },
  "checkpoint_ledger": {
    "setup_seconds": 140.5,                // first optimizer step minus first corpus read
    "joint_crossing": "ckpt-0003",         // the EARLIEST checkpoint clearing BOTH bars
    "seconds_to_joint_bar": 1100.0,        // its harness stamp minus the first-corpus-read stamp
    "declared_forward_slope": 0.057,       // bpb(H3) - bpb(H1) at that checkpoint
    "checkpoints": [                       // every checkpoint you emitted
      {"checkpoint_id": "ckpt-0003", "stamp_s": 1100.0}   // the harness stamp, not your clock
    ]
  },
  "tokenizer": {
    "entry_count": 65536,
    "fertility_slice_start_epoch_day": 19780,  // corpus_last_pub_epoch_day - 180
    "realized_bytes_per_token": 4.2,       // your tokenizer's bytes per token on that slice
    "trained_bytes_total": 1554000000.0    // served_tokens_total * realized_bytes_per_token
  }
}
```

A missing, malformed or out-of-domain artifact scores 0.

## What is graded

Your score is `r_det`, computed by the verifier in a separate environment from state it measures itself. The graded quantity is the harness-measured wall-clock seconds from the first read of the shipped corpus to the harness stamp of the earliest emitted checkpoint that jointly satisfies bits-per-byte at or below 1.12 on `tests/heldout/h3.txt` and forward degradation slope bpb(H3) minus bpb(H1) at or below 0.06. Bits-per-byte is computed host-side by the verifier from your own `tokenizer.json` and weights, at context 2048 over non-overlapping windows, summing NLL in nats over every position except each window's first, divided by ln(2) and by the shard's total raw byte count, so the metric is invariant to any tokenizer you choose to build. Best bpb on each of H1, H2 and H3 and on the H2 sibling is disclosed and is not a separate graded axis: it feeds the sub-gate ramp, the horizon ordering and the sibling check.

Every value is measured by the verifier or the harness. None is read from your `reward_log.jsonl`, and none of the numbers you declare above can raise a score: each one is checked against the harness carrier that measures the same thing, and a divergence is a failure rather than a substitution.

`r_det` is zero on any red line and on any knockout failure, before any quality is considered.

## What the harness records

Two carriers are written for you and are what the checks read. You never write either.

`run_record.json`, beside the checker tree, carries: `first_corpus_read_stamp_s` (the origin of the graded clock), `first_optimizer_step_stamp_s`, `corpus_max_pub_epoch_day` (the maximum `pub_epoch_day` the harness observed over `docmeta.bin`), `harness_checkpoint_stamps` (checkpoint id to harness-assigned stamp), `phase_median_pub_epoch_day` (one `{name, median_pub_epoch_day}` per executed phase, in execution order), `optimizer_steps`, `global_batch_tokens`, `docmeta_sha256_before` / `_after` / `_pinned`, `guarded_paths_touched`, and `shim_ledger`.

`shim_ledger` is the harness-owned token attribution, keyed through `docmeta.bin`: `served_tokens_total`, `recency_window_served_tokens` (those attributed to documents in [C-180, C)), `post_cutoff_served_tokens`, `max_served_pub_epoch_day` and `served_tokens_by_era`. The recency quota is computed from these counters and never from your declared era weights.

`tests/heldout/horizon_report.json` is written by the host-side evaluator after your run, one row per emitted checkpoint: `{checkpoint_id, stamp_s, bpb_h1, bpb_h2, bpb_h3, bpb_h2_sibling}`. Your crossing is looked up in that file, both bars are evaluated on the same row, and the earliest row clearing both is the one that stops the clock whether or not it is the one you named.

## The budget

- One graded run. `max_timeout` bounds the refinement loop; each attempt must finish inside its per-attempt budget.
- Up to 50 attempts. Your reward history is carried between attempts.
- The clock opens at the first read of the shipped corpus, so corpus scanning, index building and any tokenizer rebuild are inside the graded interval and must be declared in `setup_seconds`.

## Red lines

Any of these scores zero:

- Any byte of tests/heldout/h1.txt, h2.txt, h3.txt or h2_sibling.txt differs from its pinned sha256 -> exactly 0.0, slug heldout-tampered.
- Any byte under the pinned verifier tree is added, removed or changed -> exactly 0.0, slug grading-tree-written.
- isolation.py cannot prove the private boundary is in force -> the verifier refuses to grade and writes exactly 0.0, slug isolation-unproven.
- The submitted tokenizer does not round-trip every byte of every held-out shard exactly, or its id space does not match the model's embedding rows -> exactly 0.0, slug tokenizer-not-lossless, raised at the admissibility stage before any bits-per-byte is computed so an inadmissible submission fails loudly rather than being graded down.
- schedule.json is absent, unparseable, or does not validate against the pinned phase schema -> exactly 0.0, slug schedule-malformed, also at the admissibility stage.
- Any consumed token is attributed by the harness shim to a document whose pub_epoch_day is at or after the declared cutoff -> exactly 0.0, slug future-token-consumed.
- The share of consumed tokens attributed to documents dated within the final 180 days before the cutoff exceeds 0.15 -> exactly 0.0, slug recency-quota-exceeded.
- Optimizer steps times global batch tokens does not reconcile with the shim's served-token counter within 0.5 percent -> exactly 0.0, slug unaccounted-tokens.
- Harness-side shim counter exceeds 4.0e8 consumed tokens -> exactly 0.0, slug token-budget-exceeded.
- Parameter count recomputed from weights.safetensors exceeds 1.2e8 -> exactly 0.0, slug param-budget-exceeded.
- Step-0 parameter digests recomputed from init_spec.json do not equal the submission's declared step-0 digests -> exactly 0.0, slug init-not-reproducible.
- AST scan of model.py or the tokenizer build script finds an import or attribute path outside the allowlist -> exactly 0.0, slug import-envelope-violation.
- Earliest clearing stamps across the ordered forward family are not monotone -> exactly 0.0, slug horizon-order-inverted.
- The per-phase median consumed publication epoch-day recorded by the shim does not reconcile with the order schedule.json declares -> exactly 0.0, slug schedule-order-diverged.
- A reward_log.jsonl line asserts a bits-per-byte lower than any harness-observed value at that step -> exactly 0.0, slug log-fabrication.
- Absolute bits-per-byte gap between H2 and the H2 sibling exceeds 0.04 -> exactly 0.0, slug sibling-divergence.
- Elapsed harness-measured time exceeds 2700 s with no clearing checkpoint -> exactly 0.0, slug budget-ceiling-exceeded.
- No train.py or no emitted checkpoint in an otherwise unchanged environment -> exactly 0.0, slug no-deliverable.

## What ships

- `environment/` -- the corpus and a runnable starter.
- `environment/data/` -- the training shards, `docmeta.bin` (one record per document: `doc_offset`, `doc_len`, `pub_epoch_day`, `journal_id`), `tokenizer_meta.json`, and the convenience `train.bin` / `bytelen.bin` memmaps the starter reads.
- `environment/starter/` -- a correct, unoptimised baseline you may replace entirely.
- `environment/notes/` -- an archive of prior-run reports. They describe other derivations; nothing in them is a measurement of this one.

## Traps

- Agent reads a healthy in-era validation curve as evidence about the graded shards and never establishes where the graded shards sit relative to the corpus it trained on.
- Agent over-corrects, trips a red line that scores exactly zero, and then backs off to a setting that clears neither bar.
- Agent leaves a shipped default in place because it raises no error, and pays for it in a quantity the traceback never mentions.
- Agent declares one recipe and executes another, and zeroes on the reconciliation rather than being graded on the model it actually trained.
- Agent clears one of the two bars, watches the sub-gate ramp rise, and reads a rising number as progress when the conjunction is what opens the ramp.
- Agent reaches outside the environment for data or weights; the served-token reconciliation and the step-0 digest check reject the run rather than the judge.
