# C1-S2 -- what this task actually tests

GENERATED SECTION. DO NOT HAND-EDIT.

minimize the harness-measured wall-clock seconds, from the first read of the shipped corpus to the harness timestamp of the earliest emitted checkpoint whose bits-per-byte on the private held-out shard is at or below 1.02, for a from-scratch decoder-only language model trained on one H100 under a hard budget of 6.0e8 consumed tokens, at most 1.8e8 parameters, and a hard 3600 s ceiling. The submission must also declare recipe.json naming its ordered stage sequence, per-stage domain mixture weights, and stage token boundaries, and the harness verifies the executed data order against that declaration. The held-out mixture is disclosed as 55 percent Congressional Record, 30 percent peS2o, 15 percent Gutenberg, deliberately different from the shipped training mixture, so no graded obligation is withheld. The graded artifact is an object with an ids list and a matching outputs list, written to /submission/c1s2.json. The graded quantity is Harness-measured wall-clock seconds to the earliest emitted checkpoint scoring bits-per-byte <= 1.02 on tests/heldout/private.bin, evaluated at context 2048 over non-overlapping windows excluding each window's first position, in nats divided by ln(2) and by total shard bytes. Secondary, disclosed: best bits-per-byte achieved, used only for the sub-gate ramp. Additionally recorded for grading, not for the reward axis: the executed per-domain token trace from the harness memmap shim, compared against the declared recipe.json.

## The single most important insight

The reward is deterministic under frozen bytes and is zero on any red line before any quality is considered, so the first thing that matters is that the artifact exists, has the declared shape, and was produced without touching the checker tree or the graded fold. Only then does Harness-measured wall-clock seconds to the earliest emitted checkpoint scoring bits-per-byte <= 1.02 on tests/heldout/private.bin, evaluated at context 2048 over non-overlapping windows excluding each window's first position, in nats divided by ln(2) and by total shard bytes. Secondary, disclosed: best bits-per-byte achieved, used only for the sub-gate ramp. Additionally recorded for grading, not for the reward axis: the executed per-domain token trace from the harness memmap shim, compared against the declared recipe.json move the score.

## The ideal solve, step by step

1. **Open the clock before you can lose to it.** The graded interval begins at the harness's first read of the shipped corpus, not at the first optimizer step, so tokenizer construction, memmap open, dataloader warm-up and any corpus statistics pass are all charged. Read the three memmaps once, lazily, and do the mixture arithmetic on the bytelen tables rather than on the token streams. Measured host overhead on this grading machine is 0.2355 s against a reference crossing at 1450 s, so setup is not where the seconds are -- but a full statistics pass over 6.0e8 tokens before step one is, and it is inside the interval.

2. **Price the axis: it is time-to-bar, not quality.** The score is the stamp of the FIRST checkpoint at or below 1.02 bits per byte on the private held-out shard. A checkpoint that models beautifully after the crossing moves nothing and a checkpoint that never crosses is worth nothing, so the target is not the best model reachable in 3600 s but the earliest moment the bar is provably behind you. Every design decision is priced in seconds-to-1.02, which is a different objective from loss-at-budget and selects different recipes.

3. **Emit often, because the cadence is the resolution of the score.** The harness stamps checkpoints you emit; it cannot stamp a crossing that happened between two emissions. If you emit every 600 s you are charged for up to 600 s you had already earned, and if you emit every 30 s you are charged for at most 30. Emission is cheap relative to that overshoot, so a dense cadence near the expected crossing is a direct, unglamorous win on the graded axis and it is a lever no quality-graded slot has.

4. **Anchor on the shifted mixture, not the shipped one.** The held-out shards are built at 55 percent Congressional Record, 30 percent peS2o, 15 percent Gutenberg -- disclosed, and deliberately unlike the shipped training mixture. A validation split carrying the training mixture will improve while the graded number does not, which is what plateaus a run just above the bar for the whole budget. Build your own validation shard at the disclosed shares from held-back training documents and steer on that.

5. **Use the per-document domain labels as the lever they are.** Three labelled corpora with per-document labels mean the mixture is yours to choose at any ratio, per stage. The shifted target makes Congressional Record weight the dominant single term, but weighting it uniformly for the whole run is not the same as ordering the stages so that it arrives where it compounds. Order is the axis the reward actually rewards, and it is measured against a trace the agent cannot write.

6. **Declare the recipe you will actually execute.** recipe.json names the ordered stage sequence, the per-stage mixture weights and the stage boundaries in TOKENS. The harness memmap shim records the executed per-domain trace and compares it against that declaration at 0.5 percent of the 6.0e8 budget, or 3.0e6 tokens. A smooth anneal with no discrete boundaries diverges from any three-stage declaration and zeroes the run even when the model is good; declare what you execute, and if you want an anneal, declare its boundaries.

7. **Count in tokens; the starter counts in sequences.** The shipped starter accounts the budget in sequences. Copy that accounting and every boundary is displaced by the packing factor -- 1024, 2048 or 4096 -- while the run completes cleanly and the loss curve looks merely disappointing. The failure is attributed to mixture weights and the attempt budget is spent on the wrong axis. Reconcile your final boundary against the shim's summed per-domain counters before you trust any stage result.

8. **Keep the ordered family monotone and the sibling close.** Crossing stamps for 1.09, 1.02 and 0.96 must be non-decreasing, which one training trajectory gives for free and a spliced set of runs does not. The private and sibling shards are the same shifted distribution under one document-id hash partition, so a gap wider than 0.04 bits per byte at the crossing says you cleared one draw of documents rather than the distribution the bar is set on.

9. **Carry the state between attempts, in units you can replay.** Fifty attempts, reward history carried, and the only per-attempt signal arrives at the crossing or not at all. Record for each attempt the stage order, the boundary placement in tokens, the mixture per stage and the resulting crossing stamp, so a promising order can be reconstructed rather than remembered. An attempt whose configuration you cannot rebuild has contributed a number and no knowledge.

## Traps that catch agents that are not thinking carefully

- Agent measures a promising stage order early, does not record the boundary placement that produced it, and after twenty attempts cannot reconstruct which of its own configurations was responsible - the state it accumulated collapses into the summary.
- Agent optimizes bits-per-byte on the shipped validation split, which carries the training mixture, and never re-anchors to the disclosed shifted held-out mixture, plateauing just above the gate for the whole budget.
- Agent adopts the archive's 38-of-60 majority order because it is the modal recommendation, lands at 0.18, and reads the plateau as evidence the curriculum axis is exhausted rather than that the majority is wrong for this target.
- Agent declares a three-stage recipe.json and implements a smoothly annealed mixture with no discrete boundaries; the executed trace does not match the declaration within 0.5 percent and the run zeroes with recipe-divergence even though the model is good.
- Agent spends the attempt budget on optimizer and architecture rather than on the data recipe, because those axes give faster legible feedback, and never prices the ordering that the reward actually rewards.
- Agent computes stage boundaries in sequences rather than tokens after copying the starter's accounting, silently displacing every boundary, and attributes the resulting loss to mixture weights.
- The published curriculum heuristic - anneal toward the highest-quality domain in the final stage - is genuinely in the public literature and transfers partially to this task. Mitigation: Partially closed, and recorded honestly as partial. Control published-heuristic executes the anneal-to-quality recipe and is measured at 0.25, above the ramp but far below the gate-plus-speed band, because the shifted held-out mixture is 55 percent Congressional Record, the domain that heuristic anneals away from, and because the heuristic prices no stage boundary and no sequence-length schedule. This is the weakest-closed shortcut in the slot and it is named rather than claimed shut; if the measured control lands above 0.40 at feasibility, the slot is re-scoped rather than shipped.
- Declare a simple recipe.json to satisfy the ORDERING checker while executing an unrelated adaptive data policy, harvesting both the declaration reward and the adaptive gains. Mitigation: CK-RECIPE compares the declared stage boundaries against the harness memmap shim's own per-domain token trace, which the agent cannot write, at a 0.5 percent-of-budget tolerance; divergence is the recipe-divergence red line at exactly 0.0. CK-TREE additionally zeroes any attempt to edit the shim.
- Import a pretrained model or a public checkpoint under the open solver egress the Optimization lane mandates. Mitigation: CK-INIT recomputes step-0 tensor digests from the declared seed and bound init ops; CK-IMPORTS AST-scans the emitted model.py; the private 49,152-entry vocabulary permutation makes any external embedding table a random relabelling; CK-TOKENS caps consumed tokens harness-side.
- A public speedrun recipe for the boarded nanoGPT/FineWeb configuration is retrieved and transferred. Mitigation: No axis of the boarded configuration survives: three-domain ordered curriculum rather than a single web corpus, a shifted evaluation mixture, a private vocabulary and permutation, private held-out bytes, one H100, and time bounds anchored to our own measured reference. Control public-recipe-transfer executes the published recipe on this corpus and must land below the gate before Phase 0.5 binds.

---

## Contract record (annex; not part of the narrative body)

Retained under FORGE.md item 10e and Phase 2 item 7, which require this file to carry the ordered path through instruction.md with each satisfied checker identifier, and each rejected route bound to a measured known-wrong control. standards/truth-md-authoring-v1.md section 3 admits no fifth section, so this annex is a recorded deviation rather than an omission.

Ordered path through instruction.md, with the checker identifier each step satisfies:

1. `bits_per_byte_excludes_each_window_first_position` (compiled, weight 5, knockout) -- The bar is evaluated at context 2048 over non-overlapping windows, scoring every position but the first of each window, in nats over ln 2 over shard bytes.
2. `checkpoint_cadence_bounds_the_time_to_bar` (compiled, weight 4) -- The emission cadence the run declares is the cadence the harness stamped, so the resolution of the score is a declared quantity.
3. `clock_opens_at_first_corpus_read` (compiled, weight 5, knockout) -- The graded seconds run from the harness's first read of the shipped corpus, not from the first optimizer step.
4. `declared_crossing_is_the_earliest_checkpoint_to_clear_the_bar` (compiled, weight 5, knockout) -- The crossing the submission declares at the 1.02 bar is the earliest-stamped emitted checkpoint the harness measured at or below that bar.
5. `emitted_checkpoint_series_is_stamped` (compiled, weight 5, knockout) -- The graded run emitted at least one checkpoint and every emitted checkpoint carries the harness stamp the score is read off.
6. `executed_domain_segments_match_declared_stage_order` (compiled, weight 5, knockout) -- The domain-majority segments the harness observed are the stages the recipe declared, in the declared order and at the declared boundaries.
7. `harness_stamping_instruments_unwritten` (compiled, weight 5, knockout) -- The instruments that assign the graded stamp and record the domain trace are byte-identical either side of the graded run.
8. `heldout_shards_carry_the_disclosed_shifted_mixture` (compiled, weight 4) -- The graded shards are built at the disclosed held-out mixture, which is not the shipped training mixture.
9. `model_py_imports_confined_to_the_allowlist` (compiled, weight 3) -- model.py reaches only torch, math and safetensors, and reaches no module that could fetch weights.
10. `no_phase_escapes_the_graded_interval` (compiled, weight 4) -- Every phase the run accounts for lies inside the interval the clock measures, and setup is one of them.
11. `ordered_bar_family_crossed_in_non_decreasing_time` (compiled, weight 5, knockout) -- The easier bars of the ordered family are crossed no later than the harder ones.
12. `parameter_count_recomputed_from_the_shipped_weights` (compiled, weight 3) -- The parameter count that grades is the one recounted from weights.safetensors.
13. `per_domain_served_tokens_within_the_budget` (compiled, weight 5, knockout) -- The tokens that count against the 6.0e8 budget are the ones the harness shim counted, by domain.
14. `recipe_declares_order_mixture_and_token_boundaries` (compiled, weight 5, knockout) -- recipe.json declares an ordered stage sequence, a per-stage mixture over the three labelled corpora, and stage boundaries expressed as token offsets.
15. `reward_log_claims_no_better_than_the_harness_saw` (compiled, weight 3) -- No logged record claims a bits-per-byte the harness never observed.
16. `seconds_to_the_bar_ramp` (compiled, weight 10, knockout) -- Harness-measured seconds from the first corpus read to the earliest clearing checkpoint, mapped through the published descending ramp.
17. `sibling_shard_agrees_within_four_hundredths_bpb` (compiled, weight 4) -- The crossing is a property of the held-out distribution, not of one shard.
18. `stage_boundaries_counted_in_tokens_not_sequences` (compiled, weight 4) -- Stage boundaries are token offsets, not sequence counts scaled by the packing length.
19. `step_zero_digests_reproduce_under_the_declared_seed` (compiled, weight 3) -- Step zero was reached by initialising from the declared seed over the shipped vocabulary, not by loading weights.

Rejected routes, each bound to the control that measures it wrong:

- The published curriculum heuristic - anneal toward the highest-quality domain in the final stage - is genuinely in the public literature and transfers partially to this task. -- controlled by: Partially closed, and recorded honestly as partial. Control published-heuristic executes the anneal-to-quality recipe and is measured at 0.25, above the ramp but far below the gate-plus-speed band, because the shifted held-out mixture is 55 percent Congressional Record, the domain that heuristic anneals away from, and because the heuristic prices no stage boundary and no sequence-length schedule. This is the weakest-closed shortcut in the slot and it is named rather than claimed shut; if the measured control lands above 0.40 at feasibility, the slot is re-scoped rather than shipped.
- Declare a simple recipe.json to satisfy the ORDERING checker while executing an unrelated adaptive data policy, harvesting both the declaration reward and the adaptive gains. -- controlled by: CK-RECIPE compares the declared stage boundaries against the harness memmap shim's own per-domain token trace, which the agent cannot write, at a 0.5 percent-of-budget tolerance; divergence is the recipe-divergence red line at exactly 0.0. CK-TREE additionally zeroes any attempt to edit the shim.
- Import a pretrained model or a public checkpoint under the open solver egress the Optimization lane mandates. -- controlled by: CK-INIT recomputes step-0 tensor digests from the declared seed and bound init ops; CK-IMPORTS AST-scans the emitted model.py; the private 49,152-entry vocabulary permutation makes any external embedding table a random relabelling; CK-TOKENS caps consumed tokens harness-side.
- A public speedrun recipe for the boarded nanoGPT/FineWeb configuration is retrieved and transferred. -- controlled by: No axis of the boarded configuration survives: three-domain ordered curriculum rather than a single web corpus, a shifted evaluation mixture, a private vocabulary and permutation, private held-out bytes, one H100, and time bounds anchored to our own measured reference. Control public-recipe-transfer executes the published recipe on this corpus and must land below the gate before Phase 0.5 binds.

Deliverable manifest: /submission/c1s2.json (json), an object carrying checkpoint_series (every emitted checkpoint with its harness stamp and held-out bits-per-byte), bar_crossings (the earliest clearing checkpoint for each bar of the ordered family 1.09 / 1.02 / 0.96), recipe (the ordered stages with their per-domain mixtures and token boundaries, mirroring recipe.json), clock (the first-corpus-read origin, the graded seconds to the bar, the declared emission interval and the phase ledger) and init_spec (master seed, vocabulary entry count and step-0 tensor digests).

Control obligations recorded in seed/contract.yaml:

- `CTL-NOOP-STARTER` targets the unchanged starter's single-stage uniform mixture, the shortcut of doing nothing; best bpb at 6.0e8-token exhaustion 1.074, never reaching 1.02; recorded score 0.0526.
- `CTL-NOOP-EMPTY` targets no recipe.json and no emitted checkpoint, the degenerate empty submission; recorded score 0.
- `CTL-STALE-EMBEDDING` targets reuse a shipped or externally obtained embedding table instead of initialising from the declared seed; rejected by CK-INIT with slug init-not-reproducible; recorded score 0.
- `CTL-STALE-SHARD` targets grade against a prior attempt's held-out shard bytes; rejected by CK-DIGEST with slug heldout-tampered; recorded score 0.
- `CTL-DECOY-MAJORITY` targets the 38-of-60 archived stage order endorsed by the report majority, the shortcut of following the plurality of the haystack; best bpb 1.055 at exhaustion; recorded score 0.0552.
- `CTL-DISTRACTOR-HAYSTACK` targets the best recipe minable from the 60-report archive, the shortcut of mining instead of measuring; best bpb 1.062; recorded score 0.0543.
- `CTL-LISTALL-ORDERINGS` targets execute all six stage permutations inside one graded run and report the best, the list-all shortcut; trips CK-TOKENS with slug token-budget-exceeded; recorded score 0.
- `CTL-OW-SINGLE-DOMAIN` targets train exclusively on Congressional Record, the dominant held-out domain, the shortcut of matching the evaluation mixture directly; best bpb 1.128; recorded score 0.0453.
- `CTL-OW-BUDGET-MISCOUNT` targets reproduce the starter's sequence-unit token accounting so the declared budget is silently mis-scaled; best bpb 1.089; recorded score 0.0506.
- `CTL-ORDER-PERMUTED` targets stage contents and mixture weights held fixed, ORDER permuted only, the direct NOV2 order-sensitivity control and the strongest required control in the slot; best bpb 1.033, missing L* by 0.013 bpb; recorded score 0.0582.
- `CTL-PUBLISHED-HEURISTIC` targets the anneal-toward-the-highest-quality-domain-last curriculum from the public literature, the slot's named partially-closed shortcut; best bpb 1.041 at exhaustion, missing L* by 0.021 bpb; recorded score 0.0571.
- `CTL-PUBLIC-RECIPE-TRANSFER` targets the published nanoGPT/FineWeb-lineage speedrun recipe transferred verbatim; best bpb 1.101; recorded score 0.049.
- `CTL-SINGLE-CLUE-BEST` targets each archived report taken alone as the whole recipe, 60 runs, the shortcut that one report suffices; strongest single report best bpb 1.048; recorded score 0.0562.
- `CTL-ATK-HELDOUT-READ` targets read tests/heldout/ and fit to the graded bytes; recorded score 0.
- `CTL-ATK-SHIM-REWRITE` targets rewrite the harness memmap shim so the per-domain token trace matches a declaration it did not execute; recorded score 0.
- `CTL-ATK-RECIPE-POSTEDIT` targets post-edit recipe.json after the run so the declaration matches the trace rather than the plan; recorded score 0.
- `CTL-ATK-NONDET-KERNEL` targets disable the deterministic-kernel envelope (re-enable TF32 or a nondeterministic reduction) to buy throughput at the cost of replayability; recorded score 0.

<!-- AELLO-CANARY-BLOCK
  slot0: AELLO-CANARY-db8b1b3ed618b1b560f45ca25e91ec57
  slot1: AELLO-CANARY-45914acb9cc37312a164ee7b509ad070
  slot2: AELLO-CANARY-8ec4a1b6c989d9006c88fd0fa7ddf649
  slot3: AELLO-CANARY-32f7fc7c78381adbde8d73756d0bae61
-->
