# C4-S3 -- what this task actually tests

GENERATED SECTION. DO NOT HAND-EDIT.

This slot grades a speed, not an answer. The agent writes a quantization recipe program that converts the pinned Qwen2.5-1.5B-Instruct checkpoint into an export for the provided mixed-precision decode runtime, and what is measured is how fast that export decodes: T is the geometric mean of tokens per second at batch 1 and at batch 16, over 512 decode steps after 64 warmup steps, median of 5 trials, measured host-side. The recipe has 8 minutes of calibration on one H100 to do it in, and the export it produces must average no more than 4.10 bits per weight over all 196 linear projections, recomputed from the packed bytes, while holding private PG-19 held-out perplexity within 1.03x of the bf16 reference. The graded deliverable is the recipe and its declared stage order, not only the artifact the recipe emitted: the harness re-executes the recipe and the regenerated export must match what was submitted.

## The single most important insight

The bit budget is a constraint and the perplexity gate is a constraint; neither is the objective, and a submission that satisfies both perfectly can still score zero on the graded axis. What earns reward is throughput, and throughput is bought by putting cheap formats where the network can afford them -- which is a question about per-layer sensitivity, the one quantity this bundle discloses nowhere. Everything else is disclosed on purpose: the format cost table, the 4.10 budget, the 1.030 gate, the 8-minute clock. Difficulty survives that disclosure because knowing that INT3 group-32 costs 3.5 bits and INT4 group-128 costs 4.125 is not knowing which of the 196 projections will tolerate INT3, and the 8-minute budget is too short to find out by measuring all of them. The second thing that matters is that the geometric mean across two batch sizes makes a one-sided win worthless: batch 1 rewards fewer kernel launches and batch 16 rewards fewer weight bytes, and the geomean of a doubled rate and a halved rate is the rate you started with.

## The ideal solve, step by step

1. **Price the table before you touch a weight** Every format in the runtime contract prices at payload_bits + (scale_bits + zero_point_bits) / group_size. Doing that arithmetic first settles the decoy in one line: uniform INT4 group-128 costs 4.125 bits and is over budget, so the recipe the vendor notes recommend by majority cannot be submitted at all, while INT4 group-256 at 4.0625 fits and INT3 group-32 at 3.5 leaves room to spend elsewhere.

2. **Establish the baseline in the same run you are graded in** Both graded quantities are ratios. s is T/T0 against the bf16 baseline throughput and p is perplexity over the bf16 reference perplexity, and both denominators have to be measured in the same run under the same protocol, on the same private slice, or the ratio's standard error is not the one the 3 percent gate margin was sized against.

3. **Spend the 8 minutes on sensitivity, not on completeness** A Fisher-style diagonal sensitivity proxy over a small C4 subset ranks projections far cheaper than measuring all 196, and the budget is a hard cliff rather than a soft cost: the exhaustive route measured at 11.4 minutes is a zero, not a slow pass. Accumulate the ranking across attempts, because the reward history is carried between attempts and the per-attempt budget is not.

4. **Allocate as a knapsack, not as a choice of one format** With a price per format and a sensitivity per layer, the allocation is a knapsack under 4.10 bits: FP8 per-channel on the few most sensitive projections, INT4 on the bulk, INT3 group-32 on the least sensitive MLP down-projections. Note that a good mixed allocation can spend MORE bits than the cheapest uniform one and still decode faster, which is the clearest sign that the budget is not the objective.

5. **Declare the order you actually execute** The stages compose and do not commute. Clipping-range search before rounding is a different recipe from rounding before clipping-range search, and group folding belongs last. Declare the order in the manifest, execute it, and expect the instrumented runtime to have traced it.

6. **Pack the bytes you declare and declare the bytes you pack** The average bits per weight is recomputed from the size of the shard files on disk, so padding that is packed but not declared inflates the recomputed average rather than hiding under a compliant manifest figure. Write the manifest from the packer's own output, not from the allocation you intended.

7. **Prove the packed buffer is what gets served** A layer declared INT3 that is served from an fp16 copy is fast to write and fails on effect: the runtime keeps a per-layer dequant-invocation counter over the graded decode, and a counter of zero on a layer the manifest does not call bf16_passthrough means the packed buffer was never touched.

8. **Check the cross-corpus gap before you submit** One wrong constant in the multi-stage numeric pipeline -- a clipping range, a smoothing alpha, a group boundary -- does not raise an error and does not stop the export from emitting fluent text. It shows up as a corpus-dependent quality gap, which is why retention is compared between the graded PG-19 slice and the private C4 sibling slice rather than looked at on one of them.

9. **Measure throughput the way the metric is defined** Fixed work, not fixed time: 64 warmup steps then 512 measured steps at each batch size, five trials, medians, geometric mean across the two batch sizes. Keep the trial spread inside the admissibility band, because a median drawn from a noisy five is a draw rather than a measurement.

10. **Make the recipe regenerable, and let it be regenerated** Seed from the frozen bundle bytes so the harness re-execution uses the same seed by construction. Expect the regeneration check to compare the per-layer format map exactly and the perplexity within half a percent, and never the packed bytes, because GPU reduction order makes bitwise equality of packed weights the wrong thing to demand of an honest run.

11. **Leave the upstream pin alone and stay off the hubs** The checkpoint is an input the recipe converts, not a slot to fill with a different model, and a public AWQ or GPTQ release of this exact model is the shortest wrong path in the whole slot. The allowlist excludes the hubs by name, the private slices are never opened, and the checker tree is never written.

12. **Understand which band you are in** Below the quality gate there is a consolation band worth at most 0.07, and at or above the starter perplexity ratio there is nothing at all. Only p at or below 1.030 reaches the ramp over s, so a submission that spends its 8 minutes buying perplexity headroom it did not need has bought nothing.

## Traps that catch agents that are not thinking carefully

- Perplexity measurement noise exceeds the 3 percent gate margin. Guard: the gate is a ratio against the bf16 reference measured on the same private slice in the same run, and the slice is sized at 400k tokens or more so the ratio's standard error is well under 0.005.
- Throughput at batch 1 on a 1.5B model is dominated by kernel launch overhead rather than by weight bytes, which would collapse the graded axis. Guard: batch 16 is in the geometric mean precisely to keep the memory-bound regime dominant, and the authoring gate refuses the slot unless the measured T0-to-T1 range separates the reference from the tuned uniform family by the margin the ramp needs.
- The 4.10 budget is satisfiable by one obvious allocation, which would collapse the search. Guard: INT4 group-256 at 4.0625 fits, so the budget alone does not force a mixture; what forces it is that the best uniform allocation has to sit far below the full-reward floor on THROUGHPUT, which is what the uniform-family control on the ladder is there to anchor.
- Recipe re-execution is nondeterministic because GPU reduction order is, so an exact-bytes regeneration check would reject honest submissions. Guard: the regeneration check compares the per-layer format map exactly and the perplexity within 0.5 percent, never the packed bytes, and the runtime pins deterministic reduction for the calibration path.
- The provided runtime becomes the bottleneck, so every allocation measures the same and the tradeoff disappears. Guard: the per-format kernels are measured on the grading host at Phase 1 and the cost table is published with those figures, so the tradeoff is real and disclosed rather than assumed.
- Download a pre-quantized checkpoint -- an AWQ or GPTQ release of this exact model exists -- and submit it. This is the lookup shortcut in its purest form. Guard: the allowlist excludes model hubs by name, the absence check proves no such artifact arrived, and the regeneration check requires the harness's own re-execution of the submitted recipe to reproduce the export's per-layer format map and perplexity. A downloaded artifact has no recipe that regenerates it.
- Apply the published default recipe, uniform INT4 group-128, and stop. Guard: arithmetic, not prohibition. 4 + 16/128 = 4.125 bits is over the 4.10 budget, so it zeroes with reason bit_budget_exceeded, and the decoy vendor_notes tree recommends exactly this by majority.
- Calibrate on the evaluation distribution to buy perplexity headroom. Guard: calibration is C4 and grading is PG-19, two different real corpora; a cross-corpus retention gap above 0.02 zeroes with reason calibration_overfit, and the private PG-19 book selection is never exposed.
- Declare layers quantized in the manifest while serving them from an fp16 buffer, or pack padding to game the byte count. Guard: the average is recomputed from the packed bytes rather than from the manifest, and every layer declared quantized must show a nonzero runtime dequant-invocation counter.
- Declare a stage order that differs from what the recipe executed, or declare an order that is decorative. Guard: the traced stage-entry order is compared against the declared manifest element for element, and re-execution from that manifest has to reproduce the result, so a decorative order fails reproduction.

---

## Contract record (annex; not part of the narrative body)

Retained under FORGE.md item 10e and Phase 2 item 7, which require this file to carry the ordered path through instruction.md with each satisfied checker identifier, and each rejected route bound to a measured known-wrong control. standards/truth-md-authoring-v1.md section 3 admits no fifth section, so this annex is a recorded deviation rather than an omission.

### Ordered path, with the compiled check each step discharges

1. **Price the table before you touch a weight** -- `uniform_int4_group128_prices_above_the_budget`, `every_layer_format_is_inside_the_runtime_envelope`

2. **Establish the baseline in the same run you are graded in** -- `pg19_perplexity_ratio_within_the_gate`, `throughput_is_the_geometric_mean_across_both_batch_sizes`

3. **Spend the 8 minutes on sensitivity, not on completeness** -- `calibration_wall_clock_within_eight_minutes`, `calibration_seconds_declared_per_stage_and_add_up`

4. **Allocate as a knapsack, not as a choice of one format** -- `bits_per_weight_recomputed_from_packed_bytes`, `layer_roster_covers_all_196_linear_projections`

5. **Declare the order you actually execute** -- `traced_stage_order_equals_the_declared_recipe_order`

6. **Pack the bytes you declare and declare the bytes you pack** -- `export_manifest_and_packed_shards_present`, `bits_per_weight_recomputed_from_packed_bytes`

7. **Prove the packed buffer is what gets served** -- `every_declared_quantized_layer_dequantizes_at_decode`

8. **Check the cross-corpus gap before you submit** -- `cross_corpus_retention_gap_within_two_hundredths`

9. **Measure throughput the way the metric is defined** -- `decode_window_is_512_steps_after_64_warmup`, `trial_spread_inside_the_iqr_admissibility_band`

10. **Make the recipe regenerable, and let it be regenerated** -- `reexecution_regenerates_the_format_map_not_the_bytes`, `seed_is_derived_from_the_frozen_content_hash`

11. **Leave the upstream pin alone and stay off the hubs** -- `pinned_checkpoint_digest_unchanged_by_the_recipe`, `no_foreign_quantized_checkpoint_and_no_private_slice_read`

12. **Understand which band you are in** -- `sub_gate_band_credit_stays_below_the_graded_axis`, `speedup_over_bf16_baseline_through_the_ramp`

### Rejected routes, each bound to the control that measures it wrong

- Download a pre-quantized checkpoint -- an AWQ or GPTQ release of this exact model exists publicly -- and submit it as the export.
  Controlled by: The declared egress allowlist names pypi.org and files.pythonhosted.org and excludes the model hubs by name; no_foreign_quantized_checkpoint_and_no_private_slice_read proves no such artifact arrived; and reexecution_regenerates_the_format_map_not_the_bytes requires the harness's own re-execution of the submitted recipe to regenerate the per-layer format map and the perplexity. A downloaded artifact has no recipe that regenerates it.

- Apply the published default recipe, uniform INT4 group-128, and stop.
  Controlled by: Arithmetic, not prohibition. INT4 group-128 symmetric costs 4 + 16/128 = 4.125 bits, above the 4.10 budget, so bits_per_weight_recomputed_from_packed_bytes zeroes it with reason bit_budget_exceeded. The environment/vendor_notes/ tree recommends exactly this by majority, so the popular route is also the measured-wrong route.

- Calibrate on the evaluation distribution to buy perplexity headroom.
  Controlled by: Calibration is C4 and grading is PG-19, two different real corpora. cross_corpus_retention_gap_within_two_hundredths zeroes a gap above 0.02 with reason calibration_overfit, and the private PG-19 book selection is never exposed, so the read itself is recorded as heldout_read.

- Declare layers quantized in the manifest while serving them from an fp16 buffer, or pack padding to make the byte count look compliant.
  Controlled by: bits_per_weight_recomputed_from_packed_bytes recomputes the average from the shard bytes on disk and refuses a declared total that disagrees with them, and every_declared_quantized_layer_dequantizes_at_decode requires a strictly positive dequant invocation counter for every layer the manifest does not call bf16_passthrough.

- Declare a stage order that differs from what the recipe executed, or declare an order that is decorative.
  Controlled by: traced_stage_order_equals_the_declared_recipe_order compares the traced stage-entry sequence against the declared list element for element and fails a declared stage that never appears, and reexecution_regenerates_the_format_map_not_the_bytes requires re-execution from that manifest to reproduce the result.

- Measure sensitivity exhaustively over all 196 linear layers, then move the overrun into the graded phase so the calibration clock looks compliant.
  Controlled by: calibration_wall_clock_within_eight_minutes reads a harness-measured wall clock, not a self-reported one, and requires the run record to keep the calibration clock and the graded clock apart, so work moved across the boundary is still charged where it ran.

- Ship several exports and let the harness pick the fastest one that happens to pass.
  Controlled by: Exactly one declared manifest is regenerated and graded, and every surplus export on disk is a quantized checkpoint of un-recipe origin, which no_foreign_quantized_checkpoint_and_no_private_slice_read reports as an absence violated.

### Control ladder recorded in seed/contract.yaml

- `C4S3-CTL-NOOP-EMPTY` (recorded score 0) -- No export and no recipe at the declared paths. Fires deliverable_absent. Measured on a frozen fixture.

- `C4S3-CTL-NOOP-NOQUANT` (recorded score 0) -- Recipe that quantizes nothing, so the export is 16 bits per weight; the average recomputed from packed bytes fires bit_budget_exceeded. Measured on a frozen fixture.

- `C4S3-CTL-NOOP-STARTER` (recorded score 0) -- Unchanged shipped starter recipe, p = the starter ratio 1.185, so no_gain_over_starter. Measured on a frozen fixture.

- `C4S3-CTL-STALE-SEED` (recorded score 0.056) -- Allocation and clipping constants fitted to the shipped calibration seed and graded under the harness's own seed and private slices. The clipping ranges are wrong off-seed, p = 1.061 lands inside the sub-gate band, and the run earns 0.07 x (1.185 - 1.061) / 0.155 = 0.056 without ever reaching the graded axis. Projected.

- `C4S3-CTL-DECOY-VENDOR` (recorded score 0) -- Adopts the majority recommendation of the vendor_notes A100-era tables and migration guides, uniform INT4 group-128 at 4.125 bits, and fires bit_budget_exceeded. This is the DEC7 binding. Projected.

- `C4S3-CTL-LISTALL` (recorded score 0.043) -- Submission shipping several exports and letting the harness pick a winner. The one declared manifest is a uniform INT4 group-256 export at 4.0625 bits with p = 1.090, landing in the sub-gate band at 0.07 x (1.185 - 1.090) / 0.155 = 0.043, while the surplus exports independently fire the absence check as quantized checkpoints of un-recipe origin. The higher, non-zero branch is reported. Projected.

- `C4S3-CTL-OBVWRONG-PREQUANT` (recorded score 0) -- Downloaded pre-quantized checkpoint served as the export. The absence check finds a quantized checkpoint of un-recipe origin and re-execution fails to regenerate the format map from the declared recipe, so recipe_export_divergence. Projected.

- `C4S3-CTL-OBVWRONG-FP16SERVED` (recorded score 0) -- Layers declared quantized in the manifest but served in fp16 at runtime. The per-layer dequant invocation counter reads zero on those layers, so unquantized_layer_served. This is the GAM1 verifier-aware route. Projected.

- `C4S3-CTL-OBVWRONG-PADPACK` (recorded score 0) -- Padding-inflated packing that makes the manifest report a compliant average while the packed bytes do not. The average is recomputed from the packed bytes, so bit_budget_exceeded. Projected.

- `C4S3-CTL-OBVWRONG-CALIBONEVAL` (recorded score 0) -- Calibration performed on the evaluation corpus rather than on the provided C4 sequences. The absence check records the private-slice read as heldout_read, and the cross-corpus retention gap would fire independently. Projected.

- `C4S3-CTL-SIBLING-OVERFIT` (recorded score 0) -- Recipe whose clipping and smoothing constants are fitted to PG-19-like text, producing a cross-corpus retention gap of 0.071 against the 0.02 bound on the private C4 sibling slice, so calibration_overfit. Projected.

- `C4S3-CTL-UNIFORM-FAMILY-BEST` (recorded score 0.463) -- Slot-specific and the highest required control in the slot; it is the LRN1 anchor. The best uniform-format allocation that fits under 4.10 bits with clipping search but with NO per-layer sensitivity measurement, taken as the maximum over 12 draws: s = 1.38 with p = 1.028, so it clears the quality gate and reaches the graded axis rather than being zeroed by a constraint. This is the control that proves the ramp separates real allocation search from format-uniformity tuning; without it nothing in the control set fires on the graded axis at all. Projected.

- `C4S3-CTL-RECIPE-ORDER` (recorded score 0) -- Slot-specific. A recipe whose manifest declares clipping-range search before rounding while the instrumented runtime traces the reverse order, so it scores well while executing a different order. Fires recipe_order_divergence. Without this control the ordering obligation has no targeted shortcut and is inert under invariant 17. Projected.

- `C4S3-CTL-CALIB-BUDGET` (recorded score 0) -- Slot-specific. A recipe that measures sensitivity exhaustively over all 196 linear layers and overruns the 8-minute calibration budget at 11.4 min. Wall-clock is measured harness-side, so calibration_budget_exceeded. Without this control the budget obligation has no targeted shortcut and is inert. Projected.

### The compiled surface

20 compiled checks carrying 87 weight, of which 14 are knockouts. Every one runs against the export manifest, the packed shard bytes on disk, the submitted recipe or the harness run record.

- `speedup_over_bf16_baseline_through_the_ramp` (throughput_outcome, weight 10, knockout) -- The normalized decode speedup of the quantized export over the bf16 baseline clears the published bar and is mapped through the ramp.
- `bits_per_weight_recomputed_from_packed_bytes` (bit_budget, weight 5, knockout) -- Average bits per weight, recomputed from the shard bytes actually on disk, is at or under the 4.10 budget, and the manifest's own figure is checked against that recomputation rather than believed.
- `every_declared_quantized_layer_dequantizes_at_decode` (serving_effect, weight 5, knockout) -- Every layer the manifest declares quantized was actually served from its packed buffer during the graded decode.
- `export_manifest_and_packed_shards_present` (deliverable_presence, weight 5, knockout) -- The recipe program, its manifest and every packed shard the manifest names are all on disk under the declared export root.
- `layer_roster_covers_all_196_linear_projections` (layer_roster, weight 5, knockout) -- The per-layer format map covers every linear projection of the pinned checkpoint exactly once and names no layer the checkpoint does not have.
- `no_foreign_quantized_checkpoint_and_no_private_slice_read` (provenance_absence, weight 5, knockout) -- Nothing arrived from outside the recipe: no downloaded quantized checkpoint, no model-hub egress, no read of a private slice and no write into the checker tree.
- `pg19_perplexity_ratio_within_the_gate` (quality_gate, weight 5, knockout) -- Word-level perplexity of the quantized export on the private PG-19 slice is no worse than 1.030 times the bf16 reference measured on the same slice in the same run.
- `reexecution_regenerates_the_format_map_not_the_bytes` (recipe_regeneration, weight 5, knockout) -- The harness's own re-execution of the submitted recipe reproduces the export's per-layer format map and its perplexity, and is not asked to reproduce its bytes.
- `calibration_wall_clock_within_eight_minutes` (calibration_budget, weight 4, knockout) -- The recipe finished inside the 8-minute calibration budget on the grading host, on the harness's clock and not the agent's.
- `cross_corpus_retention_gap_within_two_hundredths` (calibration_transfer, weight 4, knockout) -- Quality retention on the graded PG-19 slice and on the private C4 sibling slice agree, so a recipe tuned to one corpus is visible rather than assumed away.
- `every_layer_format_is_inside_the_runtime_envelope` (format_envelope, weight 4, knockout) -- Every layer's weight format and group size is a row of the runtime's bound format contract.
- `pinned_checkpoint_digest_unchanged_by_the_recipe` (upstream_pin, weight 4, knockout) -- The recipe converted the pinned upstream checkpoint and did not replace it.
- `throughput_is_the_geometric_mean_across_both_batch_sizes` (throughput_definition, weight 4, knockout) -- The graded throughput is the geometric mean of the batch-1 and batch-16 medians, recomputed here from the trial table.
- `traced_stage_order_equals_the_declared_recipe_order` (stage_ordering, weight 4, knockout) -- The stage-entry order the instrumented runtime traced is the order the manifest declared, and every declared stage was actually entered.
- `calibration_seconds_declared_per_stage_and_add_up` (budget_accounting, weight 3) -- The agent declares its own calibration budget accounting, stage by stage, and the declaration is arithmetically coherent.
- `decode_window_is_512_steps_after_64_warmup` (measurement_protocol, weight 3) -- Every timed trial ran the published decode window and timed only the measured part of it.
- `seed_is_derived_from_the_frozen_content_hash` (replay_determinism, weight 3) -- The seed the recipe ran under is a pure function of the frozen bundle bytes, so the harness re-execution uses the same seed by construction rather than by convention.
- `sub_gate_band_credit_stays_below_the_graded_axis` (reward_banding, weight 3) -- The recorded reward band agrees with the recomputed perplexity ratio, and a submission that only improves perplexity can never out-earn one that improves throughput.
- `trial_spread_inside_the_iqr_admissibility_band` (measurement_stability, weight 3) -- The five trials at each batch size are tight enough that the median is a measurement rather than a draw.
- `uniform_int4_group128_prices_above_the_budget` (format_pricing, weight 3) -- The published format cost table prices each format from its own payload, scale and zero-point fields, and the price it puts on the vendor notes' majority recommendation is above the budget.

### Behavioural rubrics, graded downstream and never inside r_det

- `sensitivity_evidence_accumulates_across_attempts` -- Does the transcript show per-layer sensitivity evidence carried forward between attempts, or is the 196-layer ranking rebuilt from nothing every time? The 8-minute budget cannot measure every layer in one attempt, so a recipe that never accumulates is a recipe that never gets past the subset it can afford in a single run.

- `allocation_is_argued_from_measured_sensitivity` -- Is the choice of which projections receive FP8 and which receive INT3 argued from the agent's own measured sensitivity ranking, or adopted from the environment/vendor_notes/ majority? Per layer sensitivity is disclosed nowhere in the bundle and no combination of the vendor notes determines it, so an allocation with no measurement behind it is a guess wearing a table.

- `declared_stage_order_is_explained_not_decorated` -- Does the agent say why clipping-range search precedes rounding and why group folding comes last, and does that explanation match the order it actually declared? An order the agent cannot explain is an order it did not choose, and C-C4 and C-C5 are what turn that from a stylistic complaint into a measured one.

### Deliverable manifest

/submission/quant_export/manifest.json (json): an object carrying recipe_stages (the declared stage order as a list), layers (one record per linear projection with name, weight_count, format, group_size, packed_bytes and shard), shards (the packed files with their byte counts), declared_bits_per_weight (self-reported and never trusted), calibration_seconds (the agent's own per-stage accounting) and upstream_checkpoint (the pinned revision and weight digest the recipe converted). Companions the checks also read: /submission/quant_export/recipe.py, /submission/quant_export/shards/.

<!-- AELLO-CANARY-BLOCK
  slot0: AELLO-CANARY-984f0bb9464dd324952dbd7d9fad7601
  slot1: AELLO-CANARY-6bd7f7dd136f19f5856e9f97a07752ba
  slot2: AELLO-CANARY-6a1fe153309429d4a4f5408c6b2efb52
  slot3: AELLO-CANARY-ad3cfdc08b790d8bc1f5c6b3a110df31
-->
