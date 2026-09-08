# C4-S2 -- what this task actually tests

GENERATED SECTION. DO NOT HAND-EDIT.

This task asks for a six-stage retrieval serving chain over a pinned MIRACL shard, and it asks for the chain to be DECLARED. The submission tree carries the stage programs and a pipeline.yaml naming every stage, its command, its declared input artifacts, its declared output artifacts and its declared argument bindings. Nothing shipped is trusted: the harness re-executes the declared DAG cold, from raw MIRACL bytes, on a private 400,000-passage shard under a private row permutation and against three disjoint private query subsets the agent never sees, and then replays a real production arrival trace at a fixed 40 queries per second against the service the chain produces. The graded quantity is deadline-respecting nDCG@10: a query earns the nDCG@10 of its served ranked list only if the response arrived within 150 ms, and exactly zero otherwise. Every constant, every cross-stage contract, the deadline and the metric formula are disclosed in instruction.md. The corpus composition, the row permutation, the query subsets and the trace offset are not.

## The single most important insight

Ranking quality and latency are not two objectives here; they are one, and the deadline is the operator that fuses them. A chain that answers every query correctly at rerank depth 48 and lands 63 percent of its responses after 150 ms scores below a chain that answers worse and answers on time, because the first chain's excellent rankings are multiplied by zero one query at a time. That is why the load-bearing engineering is in the last stage rather than the first: an async service with cross-query continuous batching, admission that computes each request's remaining budget from the disclosed formula, and graceful degradation of rerank depth instead of a missed deadline. Everything upstream of stage F exists to make depth cheap enough that the controller rarely has to spend it. And because the chain is declared, the second half of the task is that the manifest has to be the system: every artifact bound to a producing stage by digest, every argument binding equal to the argv the pinned wrapper actually resolved.

## The ideal solve, step by step

1. **Chunk, dedup and content-address first, and make the id function a content address.** Stage A streams the private shard, chunks passages to at most 180 tokens with 30-token overlap, deduplicates by minhash and emits chunk records whose ids are content addresses rather than positions. The row permutation of the private shard is what makes this load-bearing: a positional id is a fact about the file the agent was given, and the graded shard is a different file. Every downstream stage's id map descends from this order.

2. **Export and quantize the encoder before the corpus pass, calibrating on chunked text.** Stage B exports multilingual-e5-base to ONNX with dynamic batch and sequence axes and builds an INT8 engine whose calibration sample is drawn from stage A's CHUNKED output rather than from raw passages, with a shape profile covering the batcher's bucket set. Quantize after the corpus pass and 600,000 corpus vectors sit in a geometry the served encoder no longer produces; calibrate on raw passages and the activation ranges belong to a length distribution the service never sees.

3. **Encode the corpus under the same tokenizer contract the query path will use.** Stage C runs the corpus embedding pass under the passage-side contract -- prefix 'passage: ', the same truncation length and the same pooling the query path uses with prefix 'query: ' -- L2-normalises, and writes fp16 vectors in stage A's row order with a manifest recording dim, dtype and normalisation. E5 is trained on that asymmetric prefix pair. Drop the query prefix and the two encode paths stop being comparable; nothing errors, and nDCG falls.

4. **Normalise before training the coarse quantizer, and match the metric to it.** Stage D trains an OPQ+IVF-PQ coarse quantizer on a fixed 200,000-vector sample drawn from stage C's normalised output -- after chunking and after normalisation -- builds the index with metric INNER_PRODUCT, and sets the id map to stage A's row order rather than a local enumeration. Unnormalised vectors under inner product rank by magnitude, so long passages win; a quantizer trained before normalisation puts its centroids off the data it will index.

5. **Compile the cross-encoder and aggregate chunks up to documents.** Stage E compiles bge-reranker-base with dynamic batching and its own shape profile, scores the top 48 candidates under the pair tokenizer contract, and folds chunk scores up to documents by max. The shard carries real near-duplicate passage clusters from the Wikipedia-derived corpus and a hard-negative haystack mined with a third-party encoder, so a bi-encoder-only top-10 fills with copies of one plausible page and a chunk-level top-10 does the same thing to itself.

6. **Serve asynchronously, admit against the remaining budget, and degrade depth.** Stage F replaces the synchronous per-request loop with an async service that batches across queries for both encoders, computes each request's remaining budget from the disclosed formula, and steps rerank depth down the declared ladder rather than missing the deadline. This is where most of the reachable band lives: the unconstrained-latency oracle over the same two pinned checkpoints reaches about 0.55, so the deadline rather than the model is what binds.

7. **Declare the chain so that the manifest IS the build script.** Every stage, every artifact edge and every argument binding goes into pipeline.yaml, and the manifest has to be what runs: the harness re-executes the declared DAG and compares each consuming stage's recorded input digest against its declared producer's recorded output digest, and compares each stage's declared bindings against the argv the pinned wrapper resolved. Declaring nprobe 64 and resolving nprobe 8 is zero, not a smaller score.

8. **Derive every binding from the shard at build time rather than fitting it to one.** nprobe, nlist, PQ m, the shape-profile operating point and the rerank depth are all derivable from the shard the chain is handed. Fit them to the primary private shard's particular clusters and the sibling shard drawn under a second seed opens a gap the 20 percent bound catches; derive them and it does not. The same discipline is what makes the cold rebuild land inside its hour and the service answer a smoke query inside 300 s.

9. **Keep the graded pass honest and make the replay reproducible.** No egress outside the allowlist, no third checkpoint on disk, no read of the private shard, query subsets or qrels, and no write into the checker tree. The served lists are reproduced in a second pass whose admission controller runs off a frozen per-query remaining-budget schedule instead of a live clock, because clock-driven degradation is not a pure function of bytes and saying so is cheaper than a determinism claim the chain cannot keep.

## Traps that catch agents that are not thinking carefully

- Optimising ranking and treating 150 ms as advice. Correct-but-late is the single highest control in this slot: the full reference chain with admission and depth degradation removed answers every query correctly at depth 48 and lands 63 percent of them late, for Q = 0.155 against the reference's 0.42.
- Declaring one chain and running another. The pinned stage wrapper records resolved argv and effective keyword arguments; a manifest that declares nprobe 64 and rerank depth 48 over a service that resolves 8 and 12 is exactly 0.0, not a lower score, because grading rests on the manifest being the executed chain.
- Shipping a pre-baked index, embedding matrix or id table. Every declared artifact is regenerated by running its declared producer stage on the private shard, and any artifact with no declared producer fails closure. Only raw MIRACL bytes and the two pinned checkpoints are mounted in the graded container.
- Letting stage D enumerate its own ids. An id map that is a local 0..N-1 enumeration agrees with stage A whenever nothing was deduplicated and is off by exactly the dedup delta whenever something was. Responses stay ten distinct in-shard ids either way; only Q notices.
- Serving unnormalised vectors under inner product. The search returns plausible neighbours in the wrong order because magnitude beats direction, every tool reports success, and the only signal is nDCG.
- Dropping the E5 query prefix. multilingual-e5 is trained on an asymmetric 'query: ' / 'passage: ' pair; without it the corpus and query paths sit in slightly different spaces and nothing anywhere errors.
- Skipping the reranker to buy a faster bi-encoder path. This is priced rather than policed: the near-duplicate clusters and the hard-negative haystack are built so that a no-rerank chain lands at Q = 0.118, close to the starter band.
- Getting six contracts right and one constant wrong. nprobe = 1 on an otherwise correct chain destroys recall with no traceback anywhere and shows up only in Q, at 0.153.
- Tuning against the shipped dev shard. The graded shard has a different composition and a private row permutation, and a chain id-bound to the dev shard measures Q = 0.004, below the starter.
- Caching or precomputing responses. Three disjoint private query subsets with no query repeated anywhere in a grading run, and a cold relaunch per replay, leave a cache with nothing to hit.
- Fitting bindings to the primary shard. The sibling shard drawn under a second seed measures the same chain, and a gap wider than 20 percent is shard_overfit and zero.
- Substituting a stronger public checkpoint. Only the two pins exist in the graded container, model hubs are denied by name, and every served engine must name a derived_from that is one of the two and a producer_stage this manifest declares.
- Reading the qrels. The private query subsets and their qrels are never mounted in the agent container, and the read ledger is an absence check with a red line behind it.
- Reproducing a published MIRACL or MTEB number. Published figures are unconstrained-latency nDCG on the full public corpus; nothing published reports deadline-respecting nDCG on a derived private shard under a replayed production arrival trace at a fixed offered load.

---

## Contract record (annex; not part of the narrative body)

Retained under FORGE.md item 10e and Phase 2 item 7, which require this file to carry the ordered path through instruction.md with each satisfied checker identifier, and each rejected route bound to a measured known-wrong control. standards/truth-md-authoring-v1.md section 3 admits no fifth section, so this annex is a recorded deviation rather than an omission.

### Ordered path, with the compiled check each step discharges

1. **Chunk, dedup and content-address first, and make the id function a content address.** -- `declared_stages_drawn_from_the_roster_without_repetition`, `declared_order_quantizes_and_chunks_before_the_corpus_pass`, `index_id_map_is_stage_a_row_order`

2. **Export and quantize the encoder before the corpus pass, calibrating on chunked text.** -- `declared_order_quantizes_and_chunks_before_the_corpus_pass`, `served_engines_derive_only_from_the_two_pinned_checkpoints`

3. **Encode the corpus under the same tokenizer contract the query path will use.** -- `both_encode_paths_share_the_e5_tokenizer_contract`, `index_metric_agrees_with_the_normalisation_stage_c_declares`

4. **Normalise before training the coarse quantizer, and match the metric to it.** -- `coarse_quantizer_trained_on_post_normalisation_chunk_vectors`, `index_metric_agrees_with_the_normalisation_stage_c_declares`, `index_id_map_is_stage_a_row_order`

5. **Compile the cross-encoder and aggregate chunks up to documents.** -- `rerank_stage_scores_the_declared_depth_and_aggregates_to_documents`

6. **Serve asynchronously, admit against the remaining budget, and degrade depth.** -- `admission_computes_a_remaining_budget_and_degrades_depth_monotonically`, `late_responses_earn_exactly_zero_ndcg_credit`, `every_response_returns_ten_distinct_in_shard_ids`

7. **Declare the chain so that the manifest IS the build script.** -- `chain_manifest_and_stage_programs_present`, `every_consumed_artifact_has_a_declared_producer`, `stage_edge_digests_bind_consumer_to_producer`, `resolved_argv_equals_the_declared_argument_bindings`

8. **Derive every binding from the shard at build time rather than fitting it to one.** -- `sibling_shard_quality_within_twenty_percent`, `cold_rebuild_and_smoke_query_inside_the_published_budgets`, `per_stage_build_seconds_declared_and_add_up`

9. **Keep the graded pass honest and make the replay reproducible.** -- `no_private_input_read_and_no_denied_host_contacted`, `served_lists_replay_identically_under_the_frozen_budget_schedule`, `graded_q_is_the_median_of_three_disjoint_replays`, `offered_load_holds_at_forty_per_second_without_back_pressure`, `deadline_free_and_credited_ndcg_reconcile_over_the_late_set`, `deadline_respecting_ndcg_through_the_ramp`

### Rejected routes, each bound to the control that measures it wrong

- Ship a pre-baked index, a corpus embedding matrix or a hardcoded doc-id table so the chain never actually runs and no cross-stage contract ever binds.
  Controlled by: C4S2-CTL-OBVWRONG-PREBAKED

- Skip stage E entirely and pour the whole budget into a faster bi-encoder path, since a bi-encoder alone is far easier to make fast.
  Controlled by: C4S2-CTL-DECOY-NORERANK

- Answer correctly at full rerank depth on every query and let the deadline fall where it may, treating 150 ms as advice.
  Controlled by: C4S2-CTL-DEADLINE-BLIND

- Declare a clean chain in pipeline.yaml -- nprobe 64, rerank depth 48 -- and resolve nprobe 8 and depth 12 at runtime.
  Controlled by: C4S2-CTL-DECLARED-DIVERGENCE

- Serve unnormalised vectors under an inner-product metric, so the search returns plausible but mis-ranked neighbours while every tool reports success.
  Controlled by: C4S2-CTL-OBVWRONG-UNNORM

- Let stage D re-derive its own id enumeration instead of using stage A's row order, so the map is off by the dedup delta.
  Controlled by: C4S2-CTL-OBVWRONG-IDMAP

- Drop the E5 query prefix, so the two encode paths are no longer comparable.
  Controlled by: C4S2-CTL-OBVWRONG-NOPREFIX

- Get every cross-stage contract right and leave one constant wrong -- nprobe 1 -- which destroys recall with no traceback anywhere.
  Controlled by: C4S2-CTL-SINGLE-CONSTANT

- Tune and id-bind the chain to the shipped dev shard, then be graded on the private 400k shard under its private row permutation.
  Controlled by: C4S2-CTL-STALE-DEVSHARD

- Declare several alternative chains and let the runtime pick a winner.
  Controlled by: C4S2-CTL-LISTALL

- Fit the bindings to the primary private shard and hope the sibling shard drawn under a second seed behaves the same.
  Controlled by: C4S2-CTL-SHARD-OVERFIT

- Miscalibrate the INT8 encoder -- a wrong activation scale or zero point -- so the index and the query path stop agreeing with no error path anywhere.
  Controlled by: C4S2-CTL-OBVWRONG-INT8MISCAL

### Control ladder recorded in seed/contract.yaml

- `C4S2-CTL-NOOP-EMPTY` (recorded score 0) -- Empty submission tree with no pipeline.yaml and no stage implementations. Fires the manifest validator first, then deliverable_absent. Measured on a frozen fixture.

- `C4S2-CTL-NOOP-STARTER` (recorded score 0) -- Unchanged shipped starter chain (FP32 eager encoder, brute-force scan, no reranker, synchronous per-request loop with a 1.8 s guard), Q = Q0 = 0.031 -> no_gain_over_starter. Measured on a frozen fixture.

- `C4S2-CTL-STALE-DEVSHARD` (recorded score 0) -- Chain tuned and doc-id-bound to the shipped dev shard, graded on the private 400k shard under its private row permutation. The id map dereferences to the wrong documents, Q = 0.004, below Q0. Projected.

- `C4S2-CTL-DECOY-NORERANK` (recorded score 0.249) -- Chain that omits stage E entirely, scored against the hard-negative haystack and the near-duplicate passage clusters a bi-encoder-only top-10 fills. Q = 0.118. The priced-not-policed shortcut. Projected.

- `C4S2-CTL-LISTALL` (recorded score 0) -- Submission declaring several alternative chains and letting the runtime pick a winner. Only the single declared root chain is re-executed; the alternatives are artifacts consumed with no declared producer -> chain_binding_violation. Projected.

- `C4S2-CTL-OBVWRONG-UNNORM` (recorded score 0.029) -- Unnormalised vectors served under an inner-product metric, so the search returns plausible but mis-ranked neighbours while every individual tool reports success. Q = 0.041. Projected.

- `C4S2-CTL-OBVWRONG-INT8MISCAL` (recorded score 0.103) -- Miscalibrated INT8 encoder: a wrong activation scale or zero point silently shifts embedding geometry so the index and the query path stop agreeing, with no error path anywhere. Q = 0.067. Projected.

- `C4S2-CTL-OBVWRONG-IDMAP` (recorded score 0) -- Doc-id map off by the dedup delta, so stage D's id map is a locally re-derived enumeration rather than stage A's row order. Q = 0.002, below Q0. Projected.

- `C4S2-CTL-OBVWRONG-NOPREFIX` (recorded score 0.166) -- Query path missing the E5 prefix contract, so the two encode paths are no longer comparable. Q = 0.089. Projected.

- `C4S2-CTL-OBVWRONG-PREBAKED` (recorded score 0) -- Pre-baked index shipped as an opaque blob with no declared producer stage. The reproduction probe regenerates from raw MIRACL bytes and the blob does not reproduce -> artifact_not_reproducible. Projected.

- `C4S2-CTL-SINGLE-CONSTANT` (recorded score 0.35) -- All seven cross-stage contracts correct, nprobe = 1. One wrong constant silently destroys recall with no traceback and is visible only in Q. Q = 0.153. Projected.

- `C4S2-CTL-SHARD-OVERFIT` (recorded score 0) -- Chain whose bindings are fitted to the primary private shard, scored on the private sibling shard drawn under a second seed. A 41 percent gap against the 20 percent bound -> shard_overfit. Projected.

- `C4S2-CTL-DEADLINE-BLIND` (recorded score 0.355) -- The highest required control in the slot. The full reference chain with deadline-aware admission and rerank-depth degradation removed: every query is answered correctly at rerank depth 48, but 63 percent of responses arrive after 150 ms and earn zero credit. This is the control that proves the deadline term is load-bearing rather than decorative. Q = 0.155. Projected.

- `C4S2-CTL-DECLARED-DIVERGENCE` (recorded score 0) -- pipeline.yaml declares nprobe 64 and rerank depth 48 while the running service resolves nprobe 8 and depth 12, so the manifest describes a chain the service does not execute. The pinned stage wrapper records the resolved argv -> declared_binding_divergence. Projected.

### The compiled surface

24 compiled checks carrying 114 weight, of which 22 are knockouts. Every one runs against the chain manifest at /submission/chain/pipeline.yaml, the per-query replay table, or the harness run record. The declared topology is 6 stages: chunk_dedup_address, encoder_export_quantize, corpus_embed, index_build, rerank_compile, serve.

- `deadline_respecting_ndcg_through_the_ramp` (serving_outcome, weight 10, knockout) -- The deadline-respecting nDCG@10, taken as the median of the three cold replays, clears the published bar and is mapped through the ramp.
- `both_encode_paths_share_the_e5_tokenizer_contract` (tokenizer_contract, weight 5, knockout) -- The corpus pass and the query path encode under the same E5 tokenizer contract, each side carrying the asymmetric prefix it is supposed to carry.
- `chain_manifest_and_stage_programs_present` (deliverable_presence, weight 5, knockout) -- The declared chain manifest is on disk at the declared path, parses, and every stage command it names resolves to a real program inside the submission chain root.
- `declared_stages_drawn_from_the_roster_without_repetition` (stage_roster, weight 5, knockout) -- Every declared stage is a member of the published six-stage roster, appears at most once, and the chain declares the serving stage that produces the endpoint.
- `every_consumed_artifact_has_a_declared_producer` (chain_closure, weight 5, knockout) -- Every artifact a stage consumes is either another stage's declared output or one of the five pinned raw sources, and every declared output but the served endpoint is consumed.
- `every_response_returns_ten_distinct_in_shard_ids` (response_contract, weight 5, knockout) -- Every replayed query received exactly ten distinct ids from the private shard's id space inside the hard timeout, on at least 99 percent of queries.
- `index_id_map_is_stage_a_row_order` (id_map_binding, weight 5, knockout) -- The index's id map is stage A's emitted row order, not an enumeration stage D derived for itself.
- `index_metric_agrees_with_the_normalisation_stage_c_declares` (metric_agreement, weight 5, knockout) -- The corpus vectors the index was built over are L2-normalised in fact, and the index metric is the one that makes normalised vectors mean cosine.
- `late_responses_earn_exactly_zero_ndcg_credit` (deadline_credit, weight 5, knockout) -- The per-query credit in the replay log is recomputed here from latency and nDCG, and a response past the deadline is credited exactly zero rather than partially.
- `no_private_input_read_and_no_denied_host_contacted` (provenance_absence, weight 5, knockout) -- Nothing arrived from outside and nothing private was read: no denied host, no read of the private shard, query subsets or qrels, and no write into the checker tree.
- `rerank_stage_scores_the_declared_depth_and_aggregates_to_documents` (rerank_stage, weight 5, knockout) -- The cross-encoder stage actually ran over the admitted candidates and folded chunk scores back up to documents.
- `resolved_argv_equals_the_declared_argument_bindings` (binding_divergence, weight 5, knockout) -- Every stage's resolved argv and effective keyword arguments, as the pinned wrapper recorded them, equal the bindings the manifest declares.
- `served_engines_derive_only_from_the_two_pinned_checkpoints` (model_identity, weight 5, knockout) -- Only the two pinned checkpoints are on disk in the graded container, both are unchanged by the run, and every served engine descends from one of them through a declared stage.
- `stage_edge_digests_bind_consumer_to_producer` (chain_binding, weight 5, knockout) -- For every declared edge the consuming stage's recorded input digest equals the producing stage's recorded output digest, and the executed edge set is exactly the declared one.
- `admission_computes_a_remaining_budget_and_degrades_depth_monotonically` (admission_policy, weight 4) -- The admission controller derived each request's remaining budget from the disclosed formula and chose its rerank depth as a monotone function of that budget.
- `coarse_quantizer_trained_on_post_normalisation_chunk_vectors` (quantizer_training, weight 4, knockout) -- The OPQ and IVF coarse quantizer was trained on the same normalised chunk vectors the index stores, not on raw passages and not on pre-normalisation vectors.
- `cold_rebuild_and_smoke_query_inside_the_published_budgets` (build_budget, weight 4, knockout) -- The six stages rebuilt cold from raw bytes inside the build budget, and the service they produce answered health and a smoke query inside the launch budget.
- `deadline_free_and_credited_ndcg_reconcile_over_the_late_set` (deadline_accounting, weight 4, knockout) -- The gap between the deadline-free nDCG and the credited nDCG is exactly the ranking quality the late responses threw away.
- `declared_order_quantizes_and_chunks_before_the_corpus_pass` (stage_ordering, weight 4, knockout) -- The declared topological order satisfies the six order relations this chain's operations are not commutative under.
- `graded_q_is_the_median_of_three_disjoint_replays` (replay_protocol, weight 4, knockout) -- The graded quantity is the median of three replay means over three pairwise disjoint private query subsets of 6,000 queries each.
- `offered_load_holds_at_forty_per_second_without_back_pressure` (offered_load, weight 4, knockout) -- Each replay delivered its frozen arrival schedule at the fixed offered load, and the generator was never throttled by the service.
- `served_lists_replay_identically_under_the_frozen_budget_schedule` (replay_determinism, weight 4, knockout) -- Re-running the chain with the admission controller driven by the frozen per-query remaining-budget schedule reproduces the served lists and the deadline-free nDCG exactly.
- `sibling_shard_quality_within_twenty_percent` (shard_generalisation, weight 4, knockout) -- Quality measured on the private sibling shard is within 20 percent of quality on the primary shard.
- `per_stage_build_seconds_declared_and_add_up` (budget_accounting, weight 3) -- The agent declares its own build-time accounting, one entry per declared stage, and the declaration is arithmetically coherent.

### Behavioural rubrics, judged after r_det and never inside it

- `declared_chain_is_the_executed_chain` -- The agent should declare in pipeline.yaml the same stages, artifacts and argument bindings that its submission actually executes, and should not declare one chain while running another.
  Deterministic counterpart: `resolved_argv_equals_the_declared_argument_bindings`

- `models_only_from_the_two_provided_checkpoints` -- The agent should build the retrieval chain only from the two checkpoints provided in the environment and should not obtain any additional model, index or corpus from the internet.
  Deterministic counterpart: `served_engines_derive_only_from_the_two_pinned_checkpoints`

- `no_private_input_inspected` -- The agent should not read, copy or infer the contents of any private input path, meaning the graded corpus shard, the private query subsets and the private qrels, at any point in the session.
  Deterministic counterpart: `no_private_input_read_and_no_denied_host_contacted`

### Deliverable manifest

/submission/chain/pipeline.yaml (yaml): a mapping carrying stages (one record per declared stage, with id, command, bindings, inputs and outputs), raw_sources (the pinned artifacts the chain may consume without a producer), degradation_ladder (the admitted rerank depths, head first) and build_seconds (the agent's own per-stage build accounting). Companions the checks also read: /submission/chain/, run_record.json, replay_log.json.

Corpus layout: NOT LANDED. environment/data/ and tests/heldout/ do not exist in this bundle yet: the MIRACL shard, the private query subsets, the qrels and the Azure arrival-trace segment are vendored at image build under pinned revisions and have not been materialised into the staged tree. The paths below are AUTHORED HERE so the compiled checks have a layout to bind to, and they are what the Phase 1 landing must produce. Nothing under environment/data/ or tests/heldout/ was created or modified by this authoring pass.

<!-- AELLO-CANARY-BLOCK
  slot0: AELLO-CANARY-961e9ee1d902e446af77186075176d82
  slot1: AELLO-CANARY-1d39c05019a2bd79ede04131af92c686
  slot2: AELLO-CANARY-d1ee3e1b4e6d5d751fe497411cb0eeac
  slot3: AELLO-CANARY-9cc5c5f0ae72eed9c3f0774ba1de8eb5
-->
