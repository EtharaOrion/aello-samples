# AELLO-C8-S3 -- what this task actually tests

GENERATED SECTION. DO NOT HAND-EDIT.

the trainer is frozen and you may not modify it. Your deliverable is recipe.py exposing build(pool_index, out_dir), which the verifier EXECUTES on the shipped pool and which must emit a training shard; the verifier then runs the frozen trainer for exactly 6,000 steps at batch 256 on one H100 and scores the resulting model's macro-averaged (balanced) accuracy over 12 species classes on a private held-out shard drawn from a strictly later capture season. Your recipe's own execution time and the trainer's time share one wall-clock envelope, so every second you spend decoding, filtering, deduplicating, rebalancing or augmenting is a second the trainer does not get. Every image whose capture timestamp falls at or after the declared boundary is out of bounds and including one scores exactly zero. The graded artifact is a Python module at /submission/recipe.py, not a prediction array and not a checkpoint.

## The single most important insight

The graded object is a DATA SELECTION RECIPE, and the trainer that consumes it cannot be touched. Every lever you have acts through one channel only -- which images end up in the shard -- so the questions that decide the score are which frames, in what proportion, from which sites and seasons, and in what order the selection ops compose. Two structural facts about camera-trap data do most of the work: a trigger fires a burst of near-identical frames, so a random validation split leaks a frame's own twin into training and hands you a confident wrong measurement; and the metric is a macro-average over twelve classes, so it divides by twelve regardless of what the shard contains and the rare classes weigh as much as the common ones. On top of those sits a budget the recipe itself spends: decode time and training time come out of one envelope, so the cheapest correct filter is the one applied to the index before any pixel is read.

## The ideal solve, step by step

1. **Burst-aware dedup applied to the pool INDEX, before decode** One camera trigger writes three frames seconds apart at one site. Group the index by burst_id and keep one frame per burst BEFORE anything is decoded: the frames you drop are frames you never pay to decode, and decode time is charged to the same envelope as training. Doing this on the index rather than on decoded pixels is what takes the reference recipe stage from about 700 s to about 470 s. Its worth is measured on the Phase 2 ladder, never asserted here.

2. **Time-ordered internal validation on the latest pre-boundary weeks** Hold out the most recent pre-boundary weeks rather than a random sample, so your own signal is ordered the way the graded one is. A random split places the other two frames of a burst on both sides at once and returns a confident number the later season cannot reproduce; this is the measurement error that selects the losing recipe, and it is upstream of every other choice.

3. **Recency and rare-class reweighting with capped repeats** The metric is a macro-average over twelve classes, so it divides by twelve whatever the shard contains and the rare classes carry as much of it as wildebeest does. Reweight toward the latest seasons and toward the rare classes, and cap the repeats so upsampling does not eat the row cap. Size the reweighting from the SERVED pool index, never from the shipped dataset card.

4. **Daylight and night-infrared statistics computed before augmentation** About forty percent of the frames fired the infrared flash and are monochrome. Compute per-channel statistics separately for the two illumination regimes and compute them on un-augmented pixels. Pooling the regimes, or taking the statistics after photometric augmentation, erases the channel structure several nocturnal species are separated by, and nothing raises when you do it.

5. **Op order: dedup, decode, rebalance, statistics, geometry, photometric** Composition order is graded here, not recipe length. Rebalancing before dedup counts the three frames of one trigger as three samples and then multiplies them, so the row count looks right while rare-class diversity collapses. Declare the order in the manifest and execute the order you declared: the harness watches the sequence and compares.

6. **Spending the envelope so the trainer gets most of it** Recipe seconds and trainer seconds come out of one budget. Batched GPU decode at a resolution you choose, applied only to the frames that survived the index-level dedup, leaves the frozen trainer most of the envelope for its six thousand steps. A single-threaded decode of the whole pool spends the budget on images the shard was never going to keep.

7. **Selecting across the camera roster rather than down one clearing** Site is a field you can select on and the later-season shard is drawn across the roster. A correlated run of bursts from one clearing is not the same information as the same row count drawn across many sites, however good it looks on a split that shares those sites.

## Traps that catch agents that are not thinking carefully

- Pools all seasons uniformly and validates on a random split, so burst near-duplicates inflate the local measurement and the selected recipe loses on the later season.
- Includes post-boundary images after reasoning that more data is better, scoring exactly zero on temporal_leakage.
- Deduplicates after rebalancing, so upsampled rare classes multiply near-identical burst frames and the effective rare-class diversity collapses while the count looks correct.
- Computes normalization statistics after augmentation, silently destroying the night-infrared statistics that separate several nocturnal species.
- Spends most of the joint envelope decoding the full pool with a single-threaded PIL path and starves the trainer, scoring zero on budget_exceeded or landing in the sub-band.
- Sizes the class rebalancing from the shipped dataset card rather than the authoritative pool index.
- Optimizes overall accuracy while the graded metric is macro-averaged, so the rare classes that dominate the balanced score are dropped by the very filtering that raises overall accuracy.
- Lookup shortcut: Snapshot Serengeti has published deep-learning results (Norouzzadeh et al. 2018 and successors) and public WILDS-style domain-shift recipes, and the agent has open internet egress. Mitigation: Published work reports model accuracy from long unconstrained training with pretrained backbones on overall accuracy over the common species. This slot freezes the model, the optimizer and the step count, grades a macro-averaged metric over 12 classes including rare ones on a strictly later season, and puts the recipe's own execution cost inside the same envelope. The graded object is a budget-constrained data recipe, which no public repository publishes; a published model recipe cannot be pasted in because the trainer is byte-frozen and the deliverable is not a model.
- Data-recipe benchmarks (DataComp-LM style) and their published filtering heuristics could be transplanted. Mitigation: Those heuristics are text-corpus filters with no temporal boundary, no burst structure and no decode-cost accounting; transplanting them produces a shard that either exceeds the envelope or leaves the burst leakage untouched. The order-sensitivity binding (C5-ORDER) is the part no published heuristic list encodes, and NOV2's projected frontier target composes against order sensitivity rather than recipe length precisely for this reason.
- Fitting the recipe to one graded season across attempts by hardcoding camera sites or class priors observed to work. Mitigation: The graded later-season window rotates deterministically across a bound family of later windows keyed by the run index, C6-SIB zeroes a sibling gap above 0.02, and the STALE control requires a site-hardcoded recipe to fall below the dichotomizing threshold once the window rotates.
- Smuggling label information by encoding predictions or an ordering into the emitted shard rather than curating it. Mitigation: The trainer is frozen and consumes only images and labels under a published contract, C-STAGES rejects any shard outside that contract loudly, C8-COUNT caps the emitted shard size, and the labels the recipe may attach are restricted to the pool index's own labels, which C3-LEAK re-derives verifier-side.

---

## Contract record (annex; not part of the narrative body)

Retained under FORGE.md item 10e and Phase 2 item 7, which require this file to carry the ordered path through instruction.md with each satisfied checker identifier, and each rejected route bound to a measured known-wrong control. standards/truth-md-authoring-v1.md section 3 admits no fifth section, so this annex is a recorded deviation rather than an omission.

Ordered path through instruction.md, with the compiled check each step satisfies. The order is the order the recipe must do the work in, not alphabetical: every entry names a fact about THIS task's selection contract, and none of them would mean anything on a slot that grades a model instead of the data it was fed.

1. `recipe_exposes_build_pool_index_out_dir` (compiled, weight 5, knockout) -- The graded deliverable is a callable selection recipe, not a model and not a prediction array.

2. `selection_keyed_to_index_fields_not_row_offsets` (compiled, weight 3) -- The recipe addresses the pool by its named fields, not by row position.

3. `dedup_filters_the_index_before_any_decode` (compiled, weight 3) -- Burst deduplication is applied to the pool index, ahead of the decode it would otherwise pay for.

4. `at_most_one_frame_per_burst_id` (compiled, weight 4) -- No camera trigger contributes two different frames to the shard.

5. `internal_validation_holds_out_latest_preboundary_window` (compiled, weight 4) -- The recipe's own validation split is ordered in time, the way the graded one is.

6. `c3_leak` (compiled, weight 5, knockout) -- No frame captured at or after the declared boundary reached the shard.

7. `burst_dedup_precedes_rebalance_in_executed_ops` (compiled, weight 4) -- Deduplication runs before rebalancing, not after it.

8. `rebalancing_sized_from_pool_index_not_dataset_card` (compiled, weight 3) -- The authoritative class distribution is the served pool index, not the shipped dataset card.

9. `class_histogram_traced_to_the_served_index_not_literals` (compiled, weight 4) -- Class priors and site rosters are derived from the pool index that was served, not carried as constants in the source.

10. `c1_bal` (compiled, weight 4) -- The shard can support a macro-average over all twelve classes.

11. `selected_sites_span_the_served_camera_roster` (compiled, weight 3) -- The selection spans camera sites instead of concentrating on a handful of them.

12. `night_ir_statistics_computed_before_augmentation` (compiled, weight 4) -- Per-channel statistics are computed on un-augmented pixels and keep the two illumination regimes apart.

13. `declared_op_sequence_equals_the_observed_one` (compiled, weight 3) -- The composition order the manifest declares is the order the harness watched run.

14. `emitted_shard_manifest_admissible` (compiled, weight 5, knockout) -- build() left behind a shard the frozen trainer can actually consume.

15. `emitted_row_count_within_cap_counting_repeats` (compiled, weight 3) -- The shard's size is counted in rows, with every upsampled repeat counted separately.

16. `frozen_trainer_digests_unchanged_across_the_recipe` (compiled, weight 4) -- The emitted shard is the recipe's only channel to the frozen trainer.

17. `c2_budget` (compiled, weight 5, knockout) -- The recipe's own seconds and the trainer's seconds are charged against one envelope.

18. `replay_reproduces_shard_and_prediction_digests` (compiled, weight 3) -- A replayed graded run reproduces the shard and the model it trained, byte for byte.

19. `c10_log` (compiled, weight 3) -- The agent's own reward_log.jsonl agrees with what the harness measured.

20. `sibling_shard_gap_within_bound` (compiled, weight 3) -- The two matched later-season sibling shards agree on what the recipe taught.

21. `balanced_accuracy_over_twelve_classes_ramp` (compiled, weight 10, knockout) -- Macro-averaged balanced accuracy over the twelve classes on private_A, mapped through the published ramp.


Rejected routes, each bound to the control that measures it wrong:

- Lookup shortcut: Snapshot Serengeti has published deep-learning results (Norouzzadeh et al. 2018 and successors) and public WILDS-style domain-shift recipes, and the agent has open internet egress. -- controlled by: Published work reports model accuracy from long unconstrained training with pretrained backbones on overall accuracy over the common species. This slot freezes the model, the optimizer and the step count, grades a macro-averaged metric over 12 classes including rare ones on a strictly later season, and puts the recipe's own execution cost inside the same envelope. The graded object is a budget-constrained data recipe, which no public repository publishes; a published model recipe cannot be pasted in because the trainer is byte-frozen and the deliverable is not a model.
- Data-recipe benchmarks (DataComp-LM style) and their published filtering heuristics could be transplanted. -- controlled by: Those heuristics are text-corpus filters with no temporal boundary, no burst structure and no decode-cost accounting; transplanting them produces a shard that either exceeds the envelope or leaves the burst leakage untouched. The order-sensitivity binding (C5-ORDER) is the part no published heuristic list encodes, and NOV2's projected frontier target composes against order sensitivity rather than recipe length precisely for this reason.
- Fitting the recipe to one graded season across attempts by hardcoding camera sites or class priors observed to work. -- controlled by: The graded later-season window rotates deterministically across a bound family of later windows keyed by the run index, C6-SIB zeroes a sibling gap above 0.02, and the STALE control requires a site-hardcoded recipe to fall below the dichotomizing threshold once the window rotates.
- Smuggling label information by encoding predictions or an ordering into the emitted shard rather than curating it. -- controlled by: The trainer is frozen and consumes only images and labels under a published contract, C-STAGES rejects any shard outside that contract loudly, C8-COUNT caps the emitted shard size, and the labels the recipe may attach are restricted to the pool index's own labels, which C3-LEAK re-derives verifier-side.

Deliverable manifest: /submission/recipe.py (python_module) exposing build(pool_index, out_dir), which the verifier CALLS; it writes the training shard and shard_manifest.json under out_dir. The graded object is the selection, not a model.

Control obligations recorded in seed/contract.yaml:

- `CTRL-NOOP-EMPTY` targets no recipe.py present / unchanged environment; recorded score 0.
- `CTRL-NOOP-NAIVE` targets the shipped naive recipe unchanged (uniform sampling, resize plus horizontal flip only); it lands exactly at the a_lo anchor 0.5200, so the ramp term is exactly 0 and r_det is the sub-band ceiling; recorded score 0.05.
- `CTRL-STALE-POOL` targets REPLACES the inert window-rotation stale control. The pre-boundary pool handed to the graded run rotates across a bound family of three draws whose SEASON MIX and SITE ROSTER differ substantially (P1 weights S1-S3, P2 weights S4-S6, P3 mixes). This control hardcodes the camera-site roster and class priors observed under a previously served draw, so it sizes its rebalancing against a roster largely absent under another and the macro-average collapses on the rare classes; projected a = 0.598. The magnitude is NOT bounded by the C6-SIB red line, because C6-SIB constrains the two matched later-season HELD-OUT shards, not the pool composition.; recorded score 0.4859.
- `CTRL-STALE-OFFSET` targets a recipe keyed to exact pool-index row offsets rather than to fields; the run-indexed rotated subsample makes the offsets wrong and it fails loudly with shard_unreadable rather than being silently graded down; recorded score 0.
- `DECOY-CARD` targets decoy/distractor: sizing rebalancing from environment/pool/DATASET_CARD.md, whose class distribution and site roster contradict the pool index that instruction.md declares authoritative; it over-samples classes rare in the actual pool and under-samples the ones that dominate a macro-average; projected a = 0.571; recorded score 0.335.
- `CTRL-LIST-ALL` targets decoding and emitting the whole pool with every augmentation applied; trips C8-COUNT (400,000 sample cap) or C2-BUDGET, so breadth is not a route; recorded score 0.
- `CTRL-OBVIOUS-SINGLECLASS` targets a recipe emitting a single-class shard; macro-averaged accuracy about 1/12 = 0.083, deep in the sub-band 0.05*(0.083/0.520); recorded score 0.008.
- `CTRL-OBVIOUS-POSTBOUNDARY` targets a recipe that includes post-boundary images; C3-LEAK zeroes it on temporal_leakage; recorded score 0.
- `NC-NORM` targets the invariant-17 rejecting half for PRC1: the reference recipe with per-channel normalization statistics computed AFTER augmentation rather than before, which silently folds daylight statistics onto the roughly 40 percent night-infrared frames; projected a = 0.612; recorded score 0.5641.
- `ORDER-SENSITIVITY` targets NOV2's order proof doubling as a control: the reference with dedup and rebalance swapped, so rebalancing counts burst duplicates as distinct samples; projected a = 0.634, a reward drop of 0.313 from 1.0, which exceeds NOV2's required 0.15; recorded score 0.6871.
- `CTRL-EGRESS-PRIORS` targets NEW, closes the review-flagged shortcut: learning the graded window's class priors and site roster from publicly downloadable LILA BC later-season metadata in the egress-enabled sandbox and hardcoding them in recipe.py. The new C13-PRIORS checker allows no numeric literal array above 64 elements and no per-site or per-class constant table, and requires the emitted shard's class histogram to be derivable from the served pool index by the declared, executed op sequence C5-ORDER already records; a hardcoded prior table zeroes with reason priors_unprovenanced. The pool-draw rotation of CTRL-STALE-POOL is the second layer.; recorded score 0.
- `FEAS-JDG-FLIP` targets buying r_det by a route the rubrics forbid (external later-season metadata, trainer interference, fabricated logs): the reference at r_det = 1.0 with exactly one rubric flipped reports min(1.0, 0.40) = 0.400; recorded score 0.4.

<!-- AELLO-CANARY-BLOCK
  slot0: AELLO-CANARY-ad0326b653051bf3fff9b3d0f400c264
  slot1: AELLO-CANARY-16319a20be1c38a127ad5533a33025a5
  slot2: AELLO-CANARY-7b2bbeeef9a8c1bf504167a51fdb9fb3
  slot3: AELLO-CANARY-a13aa21e8dec1742842ba8b70712bb43
-->
