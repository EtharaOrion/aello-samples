# A5-02 -- what this task actually tests

GENERATED SECTION. DO NOT HAND-EDIT.

Design and train, from scratch, a multi-label classifier over 12-band Sentinel-2 patches on the shipped BigEarthNet shard, and export it under a bound operator envelope that holds three hard deployment constraints at once: at most 4,194,304 serialized parameter bytes, an analytic batch-1 latency estimate of at most 1.50 ms, and a batch-64 activation high-water mark of at most 268,435,456 bytes. Training is bounded by 2.304e17 accounted FLOPs rather than by a clock, with a 1800 s wall watchdog behind it. Every patch carries several of 19 labels, so the deliverable is a score matrix of shape (n_graded_patches, 19) written to /submission/preds_a502.npy and an export manifest written to /submission/export_a502.json. The graded quantity is threshold-free macro average precision over the 19 labels, measured twice by the verifier's own executor: once on a private in-distribution held-out split and once on a private held-out-country split. The two are normalized on their own bands and MULTIPLIED. A violated deployment constraint scores exactly 0.0 with its own reason code rather than degrading.

## The single most important insight

The metric is threshold-free and multi-label, and almost every wrong turn on this slot follows from forgetting one of those two words. Threshold-free means the graded object is a RANKING per label: average precision integrates precision across the whole recall sweep, so a submission that has already chosen a decision threshold has thrown away the quantity being measured and cannot get it back by being well calibrated. Multi-label means there is no argmax and no simplex: each of the 19 columns is its own binary ranking problem, macro averaging weights the rarest label exactly as heavily as the most common, and a per-label average precision is undefined for a label with no positive in the split, so the denominator is the labels present rather than the number 19. Those two facts are what price the two silent failures the slot is built around. Global rather than per-band input normalization runs cleanly and costs about 0.08 macro AP, because the 12 Sentinel-2 bands differ by more than an order of magnitude in dynamic range and one shared scale flattens the narrow ones into noise. A per-tensor quantization scale meets the same 4 MiB as a per-channel one and costs macro AP out of proportion, because the small-magnitude channels that carry the rare labels round to zero first and macro averaging charges a full nineteenth for each label that collapses. Neither raises an error, neither shows in a training curve, and both are visible only in the graded number.

## The ideal solve, step by step

1. **Read the metric before the model.** Macro average precision over 19 labels, threshold-free, on two splits, multiplied. Write the evaluator first, on a slice of the shipped shard held out by country, and confirm it accumulates in float64 in pinned record order with ties broken by score descending then record index ascending. An evaluator whose ties depend on sort stability disagrees with itself between two grading passes, which is EXPORT_NONDETERMINISTIC even when the model is deterministic.

2. **Fit twelve band statistics, not one.** Compute the centre and scale per spectral band over the training shard and keep the vectors, 12 long each, in the export manifest. The check on this is the ratio of the largest scale to the smallest: a genuine per-band fit on Sentinel-2 spreads by at least an order of magnitude, and 12 copies of one number is a global statistic wearing a per-band shape.

3. **Choose the family against the envelope, not against the archive's majority.** 53 of the 60 archived runs were measured at a 30-minute unconstrained-parameter envelope and they rank the families in the opposite order to the 7 that were measured under this one. The majority family's best archived configuration lands at an analytic 2.31 ms against a 1.50 ms cap; no amount of width reduction inside that family crosses the boundary, because the cost is kernel count and bytes moved rather than parameters. The 7 envelope-matched runs are individually within noise of each other, so no single run names the answer and the choice has to be made across them.

4. **Price the export analytically before training it.** latency_est = flops_batch1 / flops_per_second + kernel_count * kernel_launch_seconds + bytes_moved / hbm_bytes_per_second, with the three host constants read from tests/constants.json rather than from the previous attempt. All three terms come from static analysis of the registered graph, so the number is a rational function of integers with no noise term and can be computed before a single step is taken. At batch 1 the kernel-count term dominates for a deep narrow stack, which is the term a parameter-count intuition does not see.

5. **Measure activation where it is graded.** The gate is the live-tensor high-water mark of a batch-64 forward pass under the pinned topological schedule with no recomputation. Training at batch 256 with gradient checkpointing measures a different quantity on a different schedule; the training-time figure can sit comfortably low while the graded figure is over 256 MiB, and depth added on the strength of the training-time number is how ACTIVATION_MEMORY_EXCEEDED arrives without warning.

6. **Quantize per channel.** The 4 MiB ceiling is unreachable in fp32 at any useful width, so quantization is mandatory rather than an optimization. Keep one scale per output channel. A per-tensor scale is set by the largest channel in the tensor and rounds the smallest channels to zero; those are the channels that carry the rarest of the 19 labels, and each label that collapses costs a full nineteenth of an unweighted macro mean.

7. **Spend the budget in FLOPs and finish the export inside it.** 2.304e17 accounted FLOPs, debited from the registered train graph and the launcher's integer step and batch counters, with the launcher refusing the step that would exceed the budget rather than a clock killing the run mid-export. Fixing the work rather than the time is what makes the weights bit-reproducible; the 1800 s wall watchdog behind it is a backstop at more than three times the reference run, and reaching it means the plan did not fit the device.

8. **Train the country axis on purpose.** The reward is a product of two normalized macro AP values, one in-distribution and one on two held-out countries drawn from four disclosed, deliberately land-cover-disjoint pairs that rotate on a private schedule. The in-distribution axis is the cheap one to probe and pushing it alone to its knee leaves the reward near 0.085. Anything tuned to one country pair -- a per-band affine transform, a class-prior head -- loses about 0.112 macro AP when the pair rotates.

9. **Keep every training target byte from the shipped shard.** Solving-phase egress is open and grading is network-isolated, so a downloaded geospatial foundation model can generate soft labels that ship as a label file while the small student genuinely trains in budget. That route leaves the export small, the latency clean and the band permutation invariant, and it is caught only by comparing the per-record sha256 of every training-time target against the shipped shard's own label bytes.

10. **Log the attempts that failed a gate.** Every attempt index appears once, with map_in, map_shift, param_bytes, latency_estimate_ms, activation_peak_bytes and a reason code from the closed 15-code set. The attempts most worth omitting are the ones a deployment gate zeroed, which is exactly why the index sequence is checked for gaps rather than the log checked for plausibility.

## Traps that catch agents that are not thinking carefully

- The submission is an integer label array or a thresholded 0/1 matrix. Average precision is threshold-free and reads a ranking; a decision has already discarded it, and calibrating a threshold is explicitly not the task.
- The macro mean divides by 19 rather than by the labels present in the split, charging a zero for a label whose average precision is undefined; or it is weighted by per-label support, which is a micro average under another name.
- Per-band normalization is computed globally, RGB-style, over 12 bands that differ by more than an order of magnitude in dynamic range. It runs cleanly, clears every gate, and caps macro AP roughly 0.08 below target.
- The 4 MiB ceiling is met by post-training quantization with a single per-tensor scale, which collapses the rarest of the 19 labels and costs macro AP disproportionately under an unweighted macro mean.
- The archive's majority family is adopted, and attempts are burned shrinking a family whose batch-1 cost is kernel count rather than parameters, so it cannot reach 1.50 ms at any width.
- Latency is optimized against the agent's own wall-clock timing loop rather than against the three graph counters the analytic gate reads, or against host constants copied from the previous attempt while they rotate within the disclosed plus or minus 8 percent band.
- Depth is added until the batch-64 activation high-water mark exceeds 256 MiB, unmeasured because training ran at batch 256 with checkpointing and reported a much smaller number.
- The in-distribution split is pushed to its knee while the held-out-country split stagnates, and the product term keeps the reward near 0.085.
- A public geospatial foundation model is fine-tuned instead of an architecture designed. All are orders of magnitude above 4 MiB, none meets the batch-1 gate, the private band permutation invalidates any pretrained band ordering, and grading is network-isolated so the weights must ship and be counted.
- The same teacher is used for distillation instead, shipping only soft labels so the student trains in budget and the band permutation moves nothing. Closed by the per-record label digest comparison, which scores 0.0 with reason EXTERNAL_SUPERVISION.
- The graded splits are probed across attempts. The sibling in-distribution shard rotates among four pinned partitions and the country pair rotates among four disclosed pairs, so a fit to the observed partition diverges by more than 0.03 and scores 0.0 with reason SHARD_GAP_EXCEEDED.

---

## Contract record (annex; not part of the narrative body)

Retained under FORGE.md item 10e and Phase 2 item 7, which require this file to carry the ordered path through instruction.md with each satisfied checker identifier, and each rejected route bound to a measured known-wrong control. standards/truth-md-authoring-v1.md section 3 admits no fifth section, so this annex is a recorded deviation rather than an omission.

Ordered path through instruction.md, with the checker identifier each step satisfies:

1. `score_columns_span_the_nineteen_labels` (compiled, weight 5, knockout) -- The graded artifact is a score matrix whose second axis is the 19-label nomenclature, one row per graded patch.

2. `scores_are_continuous_not_thresholded` (compiled, weight 5, knockout) -- The submitted scores are the ranking quantity average precision consumes, not decisions a threshold has already been applied to.

3. `every_label_column_is_ranked_not_constant` (compiled, weight 3) -- No label column is constant, so no per-label average precision degenerates to the label prior.

4. `macro_mean_divides_by_labels_present_in_the_split` (compiled, weight 3) -- The declared macro average is an unweighted mean over the labels that are actually scoreable on the split.

5. `per_band_statistics_not_one_global_scale` (compiled, weight 5, knockout) -- Input normalization carries one centre and one scale per spectral band, with the spread a genuine per-band fit has.

6. `quantization_scales_are_per_output_channel` (compiled, weight 5, knockout) -- The parameter budget is met with per-channel quantization scales rather than one scale per tensor.

7. `serialized_parameter_bytes_recounted_from_the_blob` (compiled, weight 5, knockout) -- The 4 MiB ceiling is counted off the shipped parameter bytes, never off the number declared beside them.

8. `analytic_latency_recomputes_from_graph_terms` (compiled, weight 5, knockout) -- The batch-1 latency budget is the analytic estimate recomputed from the export graph, not a stopwatch reading.

9. `sized_against_the_pinned_host_constants` (compiled, weight 2) -- The host constants the export was sized against are the ones in force this attempt.

10. `activation_high_water_taken_at_batch_64` (compiled, weight 5, knockout) -- Peak activation is the live-tensor high-water mark of a batch-64 forward pass with no recomputation.

11. `accounted_flop_debit_within_the_fixed_budget` (compiled, weight 3) -- The training budget is spent in accounted FLOPs and the debit reconciles with the integer step counters.

12. `training_stopped_on_the_flop_budget_not_the_watchdog` (compiled, weight 3) -- The graded run ended because the accounted work was spent, with the export already complete.

13. `supervision_digests_equal_the_shipped_shard_labels` (compiled, weight 5, knockout) -- Every training-time target byte came from the shipped shard's own labels.

14. `reward_moves_under_second_permutation_by_at_most_tau` (compiled, weight 5, knockout) -- Re-executing the same submission under the second private band and label permutation moves the reward by at most tau.

15. `sibling_shard_gap_within_bound` (compiled, weight 3) -- In-distribution macro AP agrees across the graded shard and its sibling private shard.

16. `reward_is_the_product_of_both_normalized_axes` (compiled, weight 3) -- The reward the campaign reasoned against is the product of both normalized macro AP axes, not either one alone.

17. `every_attempt_appears_in_the_reward_log` (compiled, weight 3) -- No attempt is missing from the campaign log, including the attempts that violated a deployment gate.

18. `reason_codes_come_from_the_closed_zero_set` (compiled, weight 2) -- Every zero the campaign recorded is attributed to one of the published reason codes.

19. `export_operator_list_inside_the_bound_envelope` (compiled, weight 3, knockout) -- The operator list recovered from the export graph carries no symbol outside the bound envelope, and every symbol it does carry is priced.

20. `regrading_the_frozen_export_twice_agrees_bitwise` (compiled, weight 3, knockout) -- Two grading passes over the same frozen export return the same two macro AP values, bit for bit.

21. `checker_tree_and_sweep_archive_unchanged` (compiled, weight 5, knockout) -- The graded run leaves the checker tree, the solution tree and the 60-run sweep archive byte-identical.

22. `budget_declaration_names_the_family_it_chose` (compiled, weight 3) -- The agent declares its own budget accounting and says which architecture family it committed to and when.

23. `macro_ap_product_ramp` (compiled, weight 10, knockout) -- The product of the two normalized threshold-free macro average precisions clears the published bar through the ramp.

Deliverable manifest: /submission/preds_a502.npy (npy), float array of shape (n_graded_patches, 19); and /submission/export_a502.json (json), the export manifest whose schema instruction.md publishes.

Rejected routes, each bound to the control that measures it wrong:

- Fine-tuning a public geospatial foundation model (Prithvi, SatMAE, SSL4EO) instead of designing an architecture. -- controlled by: all are orders of magnitude above the 4 MiB parameter ceiling, none satisfies the batch-1 latency gate, the private band permutation invalidates any pretrained band ordering, and the graded run is network-isolated so weights must ship inside the submission and be counted.

- Distilling that same teacher and shipping only its soft labels, so the student trains honestly in-budget. -- controlled by: supervision_digests_equal_the_shipped_shard_labels, which compares the per-record sha256 of every training-time target against the shipped shard's label bytes; CTRL-DISTILL scores exactly 0.0 with reason EXTERNAL_SUPERVISION.

- Reproducing a published BigEarthNet baseline (ResNet-50 at 19 classes) with its published mAP. -- controlled by: the published numbers are measured at unconstrained parameters and unbounded training; under the joint 4 MiB, 1.50 ms, 256 MiB and 2.304e17 FLOP envelope no published artifact is admissible, and the gate sits on our own measured reference rather than on a published mean.

- Classical hyperparameter search over the archive's documented flag space. -- controlled by: CTRL-DECOY-MAJORITY, which measures the majority family's best archived configuration at an analytic 2.31 ms against the 1.50 ms cap, so the whole exposed flag space is bounded away from the pass band and the win requires a family change the archive does not enumerate.

- Reading any one of the 7 envelope-matched archive runs in isolation. -- controlled by: CTRL-SINGLE-CLUE, which measures each of the 7 as individually within noise of its neighbour, so a single run names the second-best family and lands at mAP_in 0.600, mAP_shift 0.512.

- Probing the graded splits across attempts. -- controlled by: sibling_shard_gap_within_bound across four pinned in-distribution partitions, plus the silent rotation of the held-out country pair among four land-cover-disjoint disclosed pairs.

Control obligations recorded in seed/contract.yaml:

- `CTRL-NOOP` targets the no-op class: submitting the unchanged environment, no training written and no export produced; recorded score 0.

- `CTRL-STARTER-UNCHANGED` targets the no-op class, second half: running the shipped starter unmodified -- the plain stack with global normalization, fp32 weights and no imbalance handling -- which clears all three deployment gates and sits exactly at both baselines; recorded score 0.05.

- `CTRL-STALE-COUNTRY` targets the stale class: tuning to the pre-rotation country pair and being graded after it rotates. The four disclosed pairs are maximally land-cover-disjoint, so a per-band affine transform and class-prior head tuned to one pair lose 0.112 macro AP on the other, landing at mAP_shift 0.480 with mAP_in unchanged at 0.678; recorded score 0.348.

- `CTRL-STALE-HOSTCONST` targets the stale class, second: hard-coding the previous attempt's analytic host constants into the submission's own sizing while the pinned constants rotate within the disclosed plus or minus 8 percent band, so the kernel count overshoots. Analytic estimate 1.58 ms against the 1.50 ms cap; scores exactly 0.0 with reason LATENCY_BUDGET_EXCEEDED; recorded score 0.

- `CTRL-DECOY-MAJORITY` targets the decoy class: the archive's majority-vote family at its best archived configuration, measured on 53 of the 60 runs at a 30-minute unconstrained-parameter envelope and inverting against the 7 envelope-matched runs. Analytic estimate 2.31 ms; recorded score 0.

- `CTRL-SINGLE-CLUE` targets the decoy class, second: building the reference from any ONE of the 7 envelope-matched runs without cross-run co-coverage. Lands at mAP_in 0.600, mAP_shift 0.512; recorded score 0.359.

- `CTRL-LIST-ALL` targets the list-all class: ensembling every archived family instead of choosing, reaching 11.3 MiB serialized and scoring 0.0 with reason PARAM_BYTES_EXCEEDED; the int4-shrunk variant estimates 3.02 ms and scores 0.0 with reason LATENCY_BUDGET_EXCEEDED; recorded score 0.

- `CTRL-OBVIOUS-WRONG` targets the obvious-wrong class: a classifier that never looks at image structure, a single-band mean-intensity logistic head reaching mAP_in 0.248 and mAP_shift 0.231 on the below-baseline ramp. The degenerate variant, a label-prior constant predictor at the pinned P = 0.21, scores exactly 0.0; recorded score 0.005.

- `CTRL-GLOBAL-NORM` targets the PRC1 silent-failure surface: normalization computed globally rather than per band across 12 bands of differing dynamic range. Runs cleanly, raises nothing, clears all three gates, and lands at mAP_in 0.598, mAP_shift 0.518; recorded score 0.376.

- `CTRL-PERTENSOR-QUANT` targets the PRC2 surface: meeting the 4 MiB ceiling with a single per-tensor scale rather than per-channel scales, collapsing the rarest of the 19 labels. Lands at mAP_in 0.622, mAP_shift 0.498; recorded score 0.364.

- `CTRL-IN-DIST-ONLY` targets the product objective: pushing the in-distribution split to target while the held-out-country split stagnates at baseline. Lands at mAP_in 0.678, mAP_shift 0.442, where the product term does the work; recorded score 0.085.

- `CTRL-DISTILL` targets the distillation route: soft labels generated from a downloaded teacher during the open-egress solving phase and shipped as a label file, so the student trains in budget and the band permutation moves nothing. Closed by the per-record label digest comparison; scores exactly 0.0 with reason EXTERNAL_SUPERVISION; recorded score 0.

<!-- AELLO-CANARY-BLOCK
  slot0: AELLO-CANARY-7b162fe9b60213a21fd30c9da15a2253
  slot1: AELLO-CANARY-8d9e78a1f4db48d5b1fa3795637569cc
  slot2: AELLO-CANARY-43110090c35dee48e46c8cc99dc5830d
  slot3: AELLO-CANARY-a9789180a5f53d2358dd783cc17a74a4
-->
