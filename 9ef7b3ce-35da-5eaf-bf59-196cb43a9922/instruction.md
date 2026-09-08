# C3-S3

from four real source corpora provided in the environment, build a corpus of at most 1.0B tokens under the frozen tokenizer and emit it together with a mixture manifest, such that the frozen, hash-pinned trainer minimises the WORST of four per-domain held-out bits-per-byte values (web, encyclopedic, scientific, mathematical). Everything is disclosed: the four domains, the four held-out sources, the frozen trainer, the frozen tokenizer, the step budget, the objective and all three reward constants. The agent controls per-domain sampling weights, per-domain document selection and filtering, deduplication granularity, per-domain repetition policy, recombination of provided documents, and emission order. It controls nothing else.

## The four domains

| domain | source corpus | held-out shard |
| --- | --- | --- |
| `web` | `allenai/c4` config en | `tests/heldout/web.bin` |
| `encyclopedic` | `wikimedia/wikipedia` config 20231101.en | `tests/heldout/encyclopedic.bin` |
| `scientific` | `allenai/peS2o` | `tests/heldout/scientific.bin` |
| `mathematical` | `open-web-math/open-web-math` | `tests/heldout/mathematical.bin` |

The four shards are private and rotate over disjoint partitions of the same pinned snapshots on a schedule that is never announced. Which domain is currently the worst moves with them.

`environment/data/sources.json` maps every `source_id` to `{domain, sha256, bytes}`. **A span's domain is read from that file**, never from a label you attach to it. `environment/data/raw/` holds the four corpora; `environment/frozen/tokenizer.json` and `environment/frozen/trainer.py` are frozen and hash-checked; `environment/decontam/heldout_fingerprints.bin` is a one-way normalised 13-gram hash set over a rotating private superset of the four shards; `environment/reference/` ships the real upstream dataset cards and a real public per-document quality-classifier score file.

## What is graded

Write your graded artifact to **`/submission/c3s3.json`**. It is an object carrying:

| key | content |
| --- | --- |
| `ids` | the emitted record ids, in emission order |
| `outputs` | the per-record byte digests, aligned position for position with `ids` |
| `spans` | aligned with `ids`; each entry is a non-empty list of `{source_id, offset, length}` triples, the exact source spans that record was assembled from |
| `mixture_manifest` | see below |
| `corpus_sha256` | the digest of the emitted corpus |
| `consumed_tokens` | your count of the emitted corpus under the frozen tokenizer |

`mixture_manifest` carries:

| key | content |
| --- | --- |
| `weights` | one finite weight in [0,1] for each of the four domains and no other label; the four sum to 1 |
| `weight_basis` | must read `post_dedup_tokens` |
| `repetition_epochs` | per domain; 1 means emitted once |
| `dedup` | `{within_domain_granularity, across_domain_granularity}`; the across-domain granularity must be document level |
| `selection` | your per-domain document selection and filtering policy |
| `emission` | your emission curriculum |

A missing, malformed or out-of-domain artifact scores 0.

Your score is `r_det`, computed by the verifier in a separate environment from state it measures itself. The graded quantity is b_worst = max over the four domains of the per-domain held-out bits per byte produced by the frozen evaluator, reduced by max in the fixed order web, encyclopedic, scientific, mathematical. A mixture that lifts the mean while sinking one domain scores strictly worse, so the max-min structure is the objective rather than a side constraint. **The four per-domain values and the identity of the worst domain are never returned**: you receive one float and at most one reason code.

`r_det` is zero on any red line and on any knockout failure, before any quality is considered.

## The rules the checkers enforce

- **The cap is hard and it is counted for you.** The verifier encodes the emitted corpus under the frozen tokenizer and requires at most **1,000,000,000** tokens. Your `consumed_tokens` must equal that count rather than stand in for it, and the tokenizer's digest must be identical before and after the run.
- **Repetition is charged every time.** A span emitted k times costs k times against the cap and counts k times toward its domain's realised weight. Any domain declaring `repetition_epochs` above 1 must show a realised token count above its distinct-span token count.
- **Weights are post-dedup token shares.** Each declared weight must sit within **0.005** absolute of that domain's share of the realised token total, recomputed by the verifier by tracing every emitted span to its source document digest. A weight computed over documents, or over pre-deduplication counts, will miss this and the failure names which basis you actually used.
- **Per-domain realisation is exact integer token accounting**, not a floating-point proportion target: the verifier re-executes your manifest from a fresh checkout and requires the same `corpus_sha256` and the same four per-domain counts as equal integers.
- **Deduplicate at both scopes.** No surviving document-level duplicate may be attributed to two different domains.
- **Interleave.** The trainer consumes your corpus once, in the order you emit it. Each of the **eight** equal consecutive blocks of your emitted record order must contain at least one record from each of the four domains.
- **Every emitted byte is an exact span of a provided document.** Recombining fragments is allowed; rewriting and importing are not.
- **Every domain must receive tokens.** A domain with a zero share is still evaluated.

## The run record

Write `run_record.json` at the bundle root. You author:

- `agent_budget_declaration`: an object naming `build_seconds`, `train_seconds`, `per_domain_weights` (equal to `mixture_manifest.weights` key for key), `repetition_epochs` and `emission_curriculum`. The declaration is what is graded, not the harness's own measurement.
- `selection_evidence`: an object with `own_measurements` -- for each of the four domains at least one entry naming `estimator`, a positive `tokens_scored`, and `recorded_at`, the path of the artifact that measurement wrote -- and `cited_sources`, what your weight choice rests on. The shipped quality-classifier scores and the shipped dataset cards may be cited. A `cited_sources` list containing nothing else, or an `own_measurements` list that leaves a domain unmeasured, fails with reason `prior-asserted-over-measurement`.

The harness and the verifier write the rest of the record, including every measurement above; nothing you report about your own bits per byte reaches the graded path.

## The budget

- One graded run. `max_timeout` bounds the refinement loop; each attempt must finish inside its per-attempt budget.
- Build wall-clock at most **1200 s** and train wall-clock at most **900 s**, each timed by the host from first touch of the data. Both are exact-zero cliffs, not slopes.
- Up to 50 attempts. Your reward history is carried between attempts.
- You must declare your own budget accounting for the graded run; the declaration is what is graded, not the harness's own measurement.

## Red lines

Any of these scores zero:

- Any normalised 13-gram of the emitted corpus present in the held-out fingerprint set: exact 0.0 with reason eval-leak. The count is taken per domain and every one of the four must be zero.
- Any emitted byte not traceable to an exact span of a provided source document: exact 0.0 with reason foreign-content.
- Declared per-domain weights differing from realised per-domain token counts beyond the disclosed tolerance: exact 0.0 with reason declared-mixture-divergence. A span whose asserted domain disagrees with the source manifest fails the same way.
- Mixture manifest that does not re-execute to the emitted corpus digest: exact 0.0 with reason recipe-corpus-divergence.
- Consumed tokens above 1.0B, build above 20 minutes, or train above 15 minutes: exact 0.0 with the corresponding reason.
- Any write into the grading tree, the four held-out shards, the fingerprint file or the frozen trainer or tokenizer: exact 0.0 with reason grading-tree-write, heldout-touched or frozen-trainer-modified.
- Empty submission or unchanged environment: exactly 0.0, proven by the empty-submission screening atom.

## What ships

- `environment/` -- the four corpora, the source manifest, the frozen tokenizer and trainer, the fingerprint set, the reference artifacts, and a runnable starter.
- `environment/starter/` -- a correct, unoptimised baseline you may replace entirely.

## Traps

- Asserting a memorised published mixture (Pile, RedPajama, Dolma proportions or upsample-Wikipedia folklore) without measuring, and landing at or below the starter.
- Optimising the mean instead of the max-min, which the withheld per-domain breakdown makes easy to do by accident and expensive to detect.
- Following the shipped quality-classifier endorsement, which improves web and encyclopedic while sinking mathematics, the domain that then defines b_worst.
- Declared weights diverging from realised token counts because weights were computed over documents or pre-deduplication, scoring exactly 0.0 with reason declared-mixture-divergence.
- Spending the attempt budget on a four-dimensional weight sweep that random search would already have covered, leaving selection, dedup granularity, repetition policy and emission order untouched.
- Downloading external data despite the corpus being environment-provided, scoring exactly 0.0 with reason foreign-content.
