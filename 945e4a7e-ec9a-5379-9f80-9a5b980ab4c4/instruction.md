# C2-S1

starting from the vendored Qwen2.5-1.5B base checkpoint and the 96-shard opaque post-training pool mounted read-only at /workspace/pool, write train.py so that the grader's fresh, network-isolated invocation `python train.py --pool <dir> --out <export dir>` produces, inside a hard 1200 s single-invocation budget on one H100, a model whose strict all-constraints-satisfied accuracy on the WORST of four private constraint strata is as high as possible, while holding the arithmetic-retention floor. The four strata are 500 held-out persona prompts each at exactly two, three and four constraint conjunctions plus a haystack stratum whose real preamble runs long and places the constraint clause late, and the graded scalar is their MINIMUM, so the worst of the four is the score and no averaging path exists. Score is the normalized worst-stratum accuracy, measured between two published measured anchors: the vendored base model's worst-stratum accuracy and the sealed private reference's worst-stratum accuracy under the same 1200 s budget. Only the FINAL attempt's export is scored, the workspace is rebuilt from the pinned image at the start of every attempt, and the sole channel from one attempt to the next is the harness rolling summary, capped at 6000 characters and required to carry one entry per completed attempt with its reward. Nothing about the objective, the anchors, the strata, the decode harness, the budget, the pool size or the memory channel is withheld; the expensive and non-derivable gap is which twelve of ninety-six shards, in which stage roles, with which preference lineage, and whether that knowledge is still intact at attempt fifty.

## The pool

`/workspace/pool` is **96 shards of 4,000 rows each, mounted read-only**, and its contents are
OPAQUE. Roughly a third carry constraint-dense in-distribution rows; the rest are real adversarial
distractors built mechanically from real bytes -- preference pairs with the chosen and rejected
sides swapped, rows whose constraint clause has been detached from the response it governs,
near-duplicate inflation, and off-target math, coding, multilingual and no_robots rows.

**No surface feature ranks a shard.** Filenames are salted hashes, byte counts are equalized by
row-count padding drawn from within the same source, and mtimes are normalized. The harness
records those columns -- `compressed_bytes`, `row_count`, `mean_row_chars`, `mtime`,
`filename_rank` -- in the pool manifest so a checker can prove no ranking over them recovers a
selection. A submission whose twelve shards are exactly the leading twelve under any of those
columns, in either direction, is rejected, and so is one whose declared `ranking_basis` names one.

**At most twelve shards fit the window.** Twelve shards of 4,000 rows is the token capacity the
1200 s budget can render, so the selection is a refusal of eighty-four alternatives. Shard worth
is only obtainable by spending part of an attempt probing it; every probe you run is recorded in
the harness probe ledger, and **every shard you select must appear there**.

## What is graded

Write your graded artifact to **`/submission/c2s1.json`**. It is a JSON object with exactly these
five sections; a missing section is a malformed submission, not a weak one.

```jsonc
{
  "shard_selection": {
    "selected": [                       // at most 12, distinct, all in the pool manifest
      {"shard_id": "b41f8c2e", "stage_role": "sft", "probe_ids": ["p0031", "p0104"]},
      {"shard_id": "7a09d5b1", "stage_role": "preference", "probe_ids": ["p0088"]}
    ],
    "ranking_basis": "paid probe verdicts on the min-over-strata proxy"
  },
  "preference": {
    "lineage": "verifier_checked",      // your finding, not a citation of the shipped cards
    "pairs_total": 1500,
    "beta": 0.1
  },
  "carrier": {
    "bytes": 5912,                      // must equal the harness-measured restored length
    "entries": [                        // exactly one per completed attempt, 1..N, no gaps
      {"attempt": 1, "reward": 0.0},
      {"attempt": 2, "reward": 0.04}
    ]
  },
  "export": {
    "sha256": "…",                      // the merged model the FINAL attempt wrote
    "attempt": 50,
    "written_at_s": 903.4               // inside the timed window
  },
  "budget": {
    "first_pool_read_stamp_s": 0.19,    // must equal the harness origin
    "timed_seconds": 903.6,             // closed_at_s minus that origin
    "declared_ceiling_s": 1200.0
  }
}
```

A missing, malformed or out-of-domain artifact scores 0.

Your score is `r_det`, computed by the verifier in a separate environment from state it measures
itself. The graded quantity is **q = the MINIMUM over the four private strata of strict
all-constraints-satisfied accuracy**: an item counts only when EVERY verifiable constraint in its
prompt passes under the vendored Apache-2.0 `instruction_following_eval` module with the declared
fixed harness (fixed chat template, fixed max_new_tokens 512, temperature 0, pinned seed),
measured on the export produced by the FINAL attempt inside the fixed 1200 s single-invocation
budget on one H100.

| stratum | 500 items each |
| --- | --- |
| `conj2` | held-out persona prompts carrying exactly two verifiable constraints |
| `conj3` | exactly three |
| `conj4` | exactly four |
| `haystack` | a real held-out persona preamble that runs long and places the constraint clause late |

The conjunctions are recomposed from the upstream constraint vocabulary over held-out prompts and
exist in no public corpus. All four enter the minimum on equal terms: a run that lifts `conj4` by
0.10 while `haystack` falls by 0.02 has raised the mean and lowered the score.

Measured anchors, both published here: **q_floor = 0.21**, the vendored base model's worst-stratum
accuracy on the grading host; **q_ref = 0.52**, the sealed private reference recipe's worst-stratum
accuracy under the identical 1200 s budget. Per-stratum binomial standard error at 500 items is
0.022, so the ladder rungs are separated by at least 0.05.

Retention red line: exact-match at or above **0.55** on the 300-item private gsm8k-platinum split.
Sibling divergence tolerance **0.08**, derived as about 3.5 sigma on the min-of-four difference
rather than copied, so the sealed reference cannot trip it by noise.

`r_det` is zero on any red line and on any knockout failure, before any quality is considered.

## What the harness measures for itself

These are recorded by the harness under the verifier-owned held-out tree and are never read from
your submission. Where your submission declares the same quantity, the two must agree.

- `tests/heldout/pool_manifest.json` -- the 96 per-shard sha256 digests before and after the
  timed invocation, and the equalized surface columns above.
- `tests/heldout/probe_ledger.json` -- one row per probe you ran, naming the shard it measured.
- `tests/heldout/harness_record.json` -- the timed window (`first_pool_read_stamp_s`,
  `closed_at_s`, `origin_event`); the render digests (`train_template_sha256` beside
  `decode_template_sha256`, and the `system_turn` and `generation_prompt` spans separately); the
  SFT mask ledger (`prompt_positions`, `completion_positions`, `loss_positions`); the packing
  ledger (`packed_sequences`, `document_boundaries`, `attention_resets`,
  `cross_document_attention_positions`); the preference ledger (`reference_module_training`,
  `reference_dropout_active`, `reference_effective_dropout_p`, `policy_module_training`,
  `pairs_total`, `pairs_scored_by_the_constraint_verifier`,
  `pairs_whose_chosen_side_failed_the_verifier`); the attempt ledger, one row per completed
  attempt with its reward and its export digest; the carrier's `restored_bytes`; the workspace
  rebuild ledger; the pre-run workspace manifest and the sanctioned base digests; the four
  stratum counts, the sibling counts and the retention count; and the instrument digests either
  side of the run.

Four obligations follow from that record, and each is a check:

- **The timed window opens at the first read under the pool mount**, not at interpreter start.
  Shard render, tokenizer construction, on-policy generation, the preference pass and the adapter
  merge are all inside the interval. Declare the origin and the span, and declare them equal to
  what the harness measured; over 1200 s + 120 s of grace is exactly 0.0.
- **The training-time render must equal the decode render byte for byte**, including the
  system-turn and generation-prompt spans individually. A template that differs only there
  reaches a lower training loss and raises no error.
- **Prompt tokens carry no gradient in the SFT loss** (`loss_positions` equals
  `completion_positions`), and packed sequences reset attention at every document boundary
  (`cross_document_attention_positions` is 0).
- **The preference stage's reference forward pass runs in eval mode** with effective dropout 0.0
  while the policy trains, and every pair is ranked by the vendored constraint verifier on both
  sides.

## The budget and the loop

- One graded run. `max_timeout` bounds the refinement loop; each attempt must finish inside its
  per-attempt budget, and the graded invocation inside 1200 s + 120 s of grace.
- Up to 50 attempts. The workspace is rebuilt from the pinned image at the start of every one.
- **The rolling summary is the only channel between attempts.** It is capped at 6000 bytes and
  must carry exactly one entry per completed attempt, indices 1..N with no gap, each naming that
  attempt's reward. Capacity is fixed while the entry count grows, so the room per carried finding
  falls from about 6000 characters at attempt one to about 120 at attempt fifty.
- **Only the final attempt's export is scored.** `final_selection` is bound to `last`: a run that
  peaks at attempt thirty and regresses at attempt fifty is scored at fifty. The attempt index is
  visible to you by design.
- You must declare your own budget accounting for the graded run; the declaration is graded
  against the harness's measurement, not substituted for it.

## Red lines

Any of these scores zero:

- Any byte of tests/heldout/ differing from its recorded sha256 -> exactly 0.0, reason `heldout_mutated`.
- Any added, removed or modified file under the pinned verifier tree -> exactly 0.0, reason `grading_tree_written`.
- Any byte of the read-only pool differing from its recorded per-shard sha256 -> exactly 0.0, reason `pool_mutated`.
- Any tensor-shaped file in the pre-run workspace whose digest is not the pinned Qwen2.5-1.5B base -> exactly 0.0, reason `unsanctioned_checkpoint`.
- Timed single invocation exceeding 1200 s + 120 s grace -> exactly 0.0, reason `budget_exceeded`.
- Retention exact-match below 0.55 on the private gsm8k-platinum split -> exactly 0.0, reason `retention_gate_violated`.
- Worst-stratum and sibling worst-stratum accuracy separating by more than 0.08 -> exactly 0.0, reason `sibling_divergence`.
- No train.py, or an unchanged environment -> exactly 0.0, reason `empty_submission`.
- Export at --out whose digest appears in the pre-run manifest, or that is produced outside the timed window -> exactly 0.0, reason `export_outside_timed_window`.
- Anything other than the rolling summary crossing an attempt boundary -- a surviving file, an environment variable, a package installed into the image, a cached model artefact, a byte written into the pool -> exactly 0.0.
- A submission the shipped `environment/preflight.py` rejects -> exactly 0.0 carrying that tool's own verdict string, never a silent grade-down.

## What ships

- `environment/pool/` -- the 96 opaque shards, mounted read-only at `/workspace/pool`.
- `environment/qwen2.5-1.5b-base/` -- the vendored base checkpoint, pinned by sha256.
- `environment/reference/instruction_following_eval/` -- the real Apache-2.0 constraint verifier,
  vendored rather than reimplemented. This module is the measurement: every stratum count is the
  number of prompts on which it passed every constraint.
- `environment/reference/cards/` -- the real unaltered upstream dataset cards and the published
  mixture guidance. They are shipped because they are real, and they select the judge-preferred
  majority preference lineage.
- `environment/starter/` -- a correct, unoptimised baseline you may replace entirely. Read its
  docstring: five of its choices are wrong for this axis rather than merely unoptimised, and all
  five complete cleanly.

CORPUS STATUS: `environment/pool/`, `environment/reference/`, `environment/qwen2.5-1.5b-base/`
and `tests/heldout/` are authored paths that the landing wave builds against; they are not
present in the staged tree. Every compiled check degrades to a recorded failure with a named
reason when the file it reads is absent, never to a silent pass.

## Traps

- THE decisive risk: the summarizer turns out good enough that 6000 characters holds everything the reference needed, which makes the memory channel decorative and the archetype a relabel. Mitigation is a measurement before freeze, not an assertion: the reference rollout's own findings are serialized and measured, and if they compress under 6000 characters at attempt 50 the pool grows and the carrier shrinks until they do not. An unmeasured carrier is BLOCK:INVALID_TASK, not a coverage gap.
- Shard value turns out additive and cheaply rankable, in which case greedy top-k with 750 affordable probes solves the pool and memory reduces to bookkeeping. The pool's pairwise interaction strength must be measured before q_ref is frozen, and a pool whose measured interaction term is inside the 0.016 noise band is rejected and rebuilt.
- final_selection `last` adds terminal variance: one bad final attempt tanks a run that had already succeeded. The rule is disclosed and the agent controls the final attempt, but the sealed reference's last-attempt spread across three seeds must sit inside calibration_margin or the slot is refused at authoring rather than shipped with a lottery in it.
- The graded minimum over four strata is downward-biased and noisier than any single stratum; at 500 items per stratum the min-of-four difference has standard deviation about 0.023, so the sibling tolerance is set at 0.08 rather than the 0.04 the previous design carried, and any ladder rung closer than 0.05 is indistinguishable.
- Shard identity leaks through a surface feature we did not equalize, such as compressed size, row-length distribution or JSON key order, which would convert a paid measurement into a free lookup and dissolve the memory pressure. Control `metadata_ranking_control` must fail; a surface ranker beating random selection is a build defect.
- Decode stochasticity making strict accuracy flap across verifier re-runs; mitigated by temperature 0, a pinned seed and a deterministic replay path whose three trials must be byte-identical rather than medianed, and flakiness above the bound is BLOCK:INVALID_TASK and not a coverage gap.
