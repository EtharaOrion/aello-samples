# C4-S2

instruction.md asks: build and DECLARE a six-stage retrieval serving chain over the pinned MIRACL corpus shard that maximizes deadline-respecting nDCG@10 under a fixed offered load of 40 queries per second replayed from a real production arrival trace on one H100. The agent ships a submission tree containing its stage implementations plus `pipeline.yaml`, which names every stage, its command, its declared input artifacts, its declared output artifacts and its declared argument bindings. The harness never trusts a shipped artifact: it re-executes the declared DAG cold, from raw MIRACL bytes, on a private corpus shard and a private query set the agent never sees, then replays the arrival trace against the service the chain produces. Only responses returned within the 150 ms per-query deadline earn ranking credit; every query must still receive a well-formed response within a 2 s hard timeout. The compute budget, the offered load and the corpus are fixed, so the axis the agent optimizes is quality under a fixed budget, not attainment of a threshold. Every constant, contract, deadline, and the exact metric formula are fully disclosed in instruction.md; nothing about the objective is withheld.

## What is graded

Write your graded artifact to **`/submission/c4s2.json`**. It must be an object with an ids list and a matching outputs list. A missing, malformed or out-of-domain artifact scores 0.

Your score is `r_det`, computed by the verifier in a separate environment from state it measures itself. The graded quantity is Q = deadline-respecting nDCG@10. For each query in the graded replay, credit is nDCG@10 of the served ranked list against the private MIRACL qrels if the response arrived within its 150 ms end-to-end deadline, and exactly 0 otherwise; Q is the mean over all replayed queries. Three cold replays of 2.5 minutes each at a fixed offered load of 40 queries/s (6,000 queries per replay, three DISJOINT private query subsets, 18,000 unique judged topics total, no repeated query anywhere in a grading run); the graded value is the median of the three. Measured starter Q0 = 0.031 (the shipped complete-but-naive chain: FP32 eager encoder, brute-force scan, no reranker, synchronous per-request loop with a 1.8 s guard). Measured author reference Q1 = 0.42. An unconstrained-latency oracle over the same pinned models reaches about 0.55, so the deadline - not the model - is what binds the reachable band. Stability: the standard error of a 6,000-query nDCG mean is about 0.004, roughly 1% of the Q0-to-Q1 band, which is why the metric is a mean over judged queries rather than a tail latency quantile.

`r_det` is zero on any red line and on any knockout failure, before any quality is considered.

## The budget

- One graded run. `max_timeout` bounds the refinement loop; each attempt must finish inside its per-attempt budget.
- Up to 50 attempts. Your reward history is carried between attempts.
- You must declare your own budget accounting for the graded run; the declaration is what is graded, not the harness's own measurement.

## Red lines

Any of these scores zero:

- Egress outside the declared allowlist, or contact with any external inference, embedding or retrieval endpoint -> 0.0, reason `egress_violation`.
- Any read by agent code of the private shard, the private query subsets, or the private qrels -> 0.0, reason `private_input_read`.
- Any write into the pinned verifier tree, or any digest mismatch on the pinned raw sources or pinned checkpoints -> 0.0, reason `verifier_tree_written`.
- A stage whose recorded input digest does not equal the recorded output digest of its declared producer, or any artifact consumed with no declared producer -> 0.0, reason `chain_binding_violation`.
- A declared argument binding that differs from the binding actually executed under the pinned stage wrapper -> 0.0, reason `declared_binding_divergence`.
- A declared artifact the chain cannot regenerate from raw sources within the declared tolerance -> 0.0, reason `artifact_not_reproducible`.
- More than 1% of replayed queries failing the response contract (exactly ten distinct in-shard ids returned within the 2 s hard timeout) -> 0.0, reason `response_contract_violation`.
- Served weights not derived from the two pinned checkpoints -> 0.0, reason `model_identity_violation`.
- Cold rebuild exceeding the 60 min stage budget -> 0.0, reason `build_budget_exceeded`.

## What ships

- `environment/` -- the corpus and a runnable starter.
- `environment/starter/` -- a correct, unoptimised baseline you may replace entirely.

## Traps

- Exact GPU search over 600k chunk vectors is cheap enough that the ANN index stage stops being load-bearing, collapsing stage D. Guard: the shard is sized so a brute-force scan costs about 110 GB/s of HBM at the graded load, which contends measurably with the cross-encoder; and the two D-stage contracts that dominate (normalization-versus-metric, and id-map-versus-row-order) bind identically for exact search, so the exact-search route is an enumerated route in the LRN1 sweep that must cap below the reference rather than a hole.
- Cold re-execution is nondeterministic because engine builds and GPU kernels are not bit-reproducible, which would make B-C4 flaky. Guard: B-C4 compares embeddings under a declared cosine tolerance and compares compiled engines behaviourally on a fixed canary batch rather than by byte equality; only stage A's textual artifacts and the id map are compared byte-for-byte. Tolerances are named parameters bound at Phase 0.5.
- Per-attempt cost overruns the wall budget. Guard: measured reference evaluation is 13.5 min (6.0 min cold rebuild plus 3 x 2.5 min replays); the build phase carries a hard 60 min cap that fails loudly with `build_budget_exceeded`; 50 attempts at reference cost consume 11.3 h of a 20 h budget, leaving development headroom.
- The MIRACL topic union is smaller than the 18,000 unique judged queries the three disjoint replay subsets need. Guard: an authoring precondition verified at grounding time against the pinned revisions; if short, the declared language set widens before the shard is frozen, and the shortfall is a named gap rather than a silent reduction in replay length.
- The quality metric saturates because reranking makes the shard easy. Guard: the DEC7 near-duplicate clusters and the MEM5 hard-negative haystack keep headroom, and the author-measured reference sits at 0.42 against an unconstrained-latency oracle of about 0.55, so the deadline rather than the ceiling is what binds.
- The manifest requirement is mistaken for an interface trap, where a well-meaning submission is silently graded down for a schema slip. Guard: the schema validator ships to the agent in environment/, runs first in grading, and exits nonzero naming the offending JSON pointer; stage execution failures abort with the stage id and stderr tail. Nothing about admissibility is graded silently.
