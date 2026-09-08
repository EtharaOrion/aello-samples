# C3-S2

from a fixed 12 GB raw web-text slice, build a curation pipeline from scratch that emits a training corpus and a machine-readable recipe manifest, such that the frozen, hash-pinned trainer consuming the first N tokens of your corpus in the order you emit them minimises held-out bits per byte on two private evaluation shards drawn from sources other than the training slice. The tokenizer, trainer, optimizer, schedule, seed, architecture and 6,000-step budget are all frozen and hash-checked. Selection, deduplication, filtering, decontamination and emission order are the entire design space; N is drawn from a private schedule inside the disclosed band [560M, 600M] tokens, so the recipe must be robust to where the single pass stops.

## What is graded

Write your graded artifact to **`/submission/c3s2.json`**. The trainer is frozen, so the graded
object is the **corpus** and the **recipe that emitted it**, never a model. The artifact is a JSON
object carrying, at minimum:

| key | meaning |
| --- | --- |
| `ids` | the emitted record ids, **in emission order** |
| `outputs` | the sha256 of each emitted record's bytes, aligned to `ids` |
| `recipe_manifest` | an ordered array of `{"op": <name>, "params": {...}}` entries; order is array position |
| `corpus_sha256` | the sha256 of the emitted corpus, which the manifest must re-execute to |
| `spans` | one `{"source_id", "offset", "length"}` triple per emitted record, aligned to `ids` |
| `consumed_tokens` | the token count of the consumed prefix under the frozen tokenizer |

A missing, malformed or out-of-domain artifact scores 0. So does a manifest expressed as prose, or
as a mapping keyed by operator name: order is what is graded and a mapping cannot carry it.

Your run record must carry an agent-authored `curation_declaration` naming `operator_order`,
`emission_curriculum`, `build_seconds` and `train_seconds`. `operator_order` must equal the `op`
sequence of the manifest you submitted; a declaration that disagrees with the manifest is recipe
drift and is caught before the verifier spends twenty minutes re-executing it.

Two orderings inside the manifest are checked directly, because neither is visible in any
intermediate state:

- every **decontamination** operator must come **after** byte **normalisation**;
- every **deduplication** operator must come **before** the **truncation** to the token budget.

Your score is `r_det`, computed by the verifier in a separate environment from state it measures
itself. The graded quantity is held-out bits per byte from the frozen evaluator, taken as the
**worse** of the two private shards: `b = max(b_P1, b_P2)`, P1 encyclopedic and P2 scientific.
Lower is better. Worse-of-two makes distribution-matching to one held-out source a mutually
exclusive wrong deliverable rather than a partial win.

`r_det` is zero on any red line and on any knockout failure, before any quality is considered.

## The budget

- One graded run. `max_timeout` bounds the refinement loop; each attempt must finish inside its per-attempt budget.
- Up to 50 attempts. Your reward history is carried between attempts.
- You must declare your own budget accounting for the graded run; the declaration is what is graded, not the harness's own measurement.

## Red lines

Any of these scores zero:

- Any normalised 13-gram of the emitted corpus present in the held-out fingerprint set: exact 0.0 with reason eval-leak.
- Any emitted byte that does not trace to an exact span of a provided raw document: exact 0.0 with reason span-containment-violation.
- Recipe manifest that does not re-execute to the emitted corpus digest: exact 0.0 with reason recipe-corpus-divergence.
- Consumed token prefix above the drawn budget N: exact 0.0 with reason token-budget-exceeded.
- Build above 20 minutes or train above 15 minutes host-measured: exact 0.0 with the corresponding wallclock reason.
- Any write into the grading tree, the held-out tree, the fingerprint file or the frozen trainer: exact 0.0 with reason grading-tree-write or frozen-trainer-modified.
- Empty submission or unchanged environment: exactly 0.0, proven by the empty-submission screening atom.

## What ships

- `environment/` -- the corpus and a runnable starter.
- `environment/starter/` -- a correct, unoptimised baseline you may replace entirely.

## Traps

- Order collapse: deduplication applied after truncation to the token budget, so the surviving duplicates were already fixed by the truncation point and dedup does nothing measurable.
- Decontamination applied before normalisation, so normalised held-out 13-grams survive in the emitted corpus and CHK-B5 scores the attempt exactly 0.0 while the agent's own pre-normalisation check reported zero hits.
- Recipe manifest drift: the emitted corpus reflects an interactive pipeline the manifest no longer describes, so verifier re-execution produces a different digest and scores 0.0 with reason recipe-corpus-divergence.
- Over-filtering: a high-quality threshold that leaves fewer than the required token budget, so the single pass repeats data or underfills, and bits per byte degrades below the starter.
- Emission order ignored: documents emitted in raw shard order, leaving the entire curriculum dimension of the design space unexplored while the agent spends the attempt budget on thresholds.
- Reproducing the published C4 recipe and stalling at exactly 0.10, because the published recipe is the starter.
