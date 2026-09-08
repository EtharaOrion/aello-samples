# AELLO-C6-S2 -- what this task actually tests

GENERATED SECTION. DO NOT HAND-EDIT.

Build a pipeline that, in one graded run on one H100, emits for every solution in a private held-out set BOTH a per-step label vector over {good, neutral, bad} AND an integer first-error index, using the exact sentinel -1 when the solution contains no bad step. The two outputs are coupled: the first error is the first step labelled bad, and a submission whose index contradicts its own vector is rejected outright rather than graded down. Score rises as the graded run gets faster once the joint quality gate clears -- macro-F1 over the three human step classes at or above T_F1 and exact first-error index match at or above T_LOC. Two hard constraints hold throughout: the emitted neutral rate inside every disclosed difficulty stratum stays within plus or minus 0.03 of the human neutral rate, and the emitted index equals the index of the first bad label in the emitted vector. The graded artifact is an object with an ids list and a matching outputs list at /submission/c6s2.json.

## The single most important insight

The middle class is the task. PRM800K's human labels are literally three-valued and neutral is an annotated intermediate state that is neither progress nor error, so the dominant failure is a pipeline that collapses it into one of the decisive classes. That collapse is nearly invisible in binary step accuracy and it destroys localisation in a direction you can predict: fold neutral into good and the first-error index arrives late, fold it into bad and it arrives early. Both hard constraints exist to make the collapse loud -- the per-stratum band catches the rate, and the prefix invariant catches the index that the collapse moved. Everything that follows is downstream of resolving the ambiguous class honestly, once, in a decode that produces both outputs at the same time.

## The ideal solve, step by step

1. **Re-derive the step boundaries before anything else** The graded shard's steps are re-segmented under a private whitespace and LaTeX-spacing normalisation that instruction.md discloses in prose and nothing announces at run time. Apply it first and check your step count per solution against your own re-segmentation of the shipped fold; every downstream count is keyed by step position, so a segmentation fitted to the shipped byte form mislabels every step after the first drift and shifts the index with it.

2. **Fit a three-class head, not a scalar with two cuts** The labelled phase-2 fold ships with the three human classes intact. Train the head to emit all three. A scalar process reward with a global threshold is the shape of every published PRM and it is the shape this task rejects: it has no place to put an annotated intermediate state, so the neutral rate it emits is whatever falls between two cuts rather than a calibrated class.

3. **Calibrate the neutral rate per stratum on your own local fold** The band is plus or minus 0.03 INSIDE each disclosed difficulty stratum, and the graded shard's stratum mixture is not the shipped split's. One global threshold that matches the aggregate rate misses inside the levels; fit one decision threshold per stratum on the local fold, and check every stratum, including the thin ones where a zero emitted neutral rate would pass the band while having collapsed the class.

4. **Decode once, monotonically, and let both outputs fall out of the same pass** Walk the steps in order, emit the first index whose bad evidence crosses that stratum's threshold, and derive the label vector from that same walk. Prefix consistency then holds structurally rather than by luck. The alternative -- per-step argmax, then a separately argmaxed index, then a reconciliation pass -- scores respectably on macro-F1 and lands thousands of divergences, and the invariant is a hard zero rather than a deduction.

5. **Type the sentinel** Emit the integer -1, and only for a vector that genuinely holds no bad step. It is a class on the localisation axis, matched by the human -1 and by nothing else. Treating it as a wildcard turns the whole no-error population into free score and is the exact route CTRL_S2_ALL_SENTINEL is measured wrong on.

6. **Check the two subgroups before you look at the aggregate** Deep errors -- at least twelve steps with the first error at index eight or later -- and correct-final-answer-with-a-bad-step are separately floored at 0.44 and 0.48. Both are hard zeros. A localiser fitted on short solutions passes the aggregate and fails the first; a pipeline that reads the final answer instead of the process emits -1 across the second and fails it entirely.

7. **Keep the sibling shard out of the loop** Fifty attempts with reward history carried between them is enough to fit the graded shard through its own feedback. The sibling shard is scored the same way and the joint-quality gap is capped at 0.04, so a threshold set tuned against observed reward opens a gap it cannot close.

8. **Put every stage inside the timed region and then make it fast** Segmentation, encoding, three-class fitting, per-stratum calibration, localisation decode and export all run inside the one timed invocation, and the seconds are the harness's, not yours. Above the joint quality gate the score is wall clock alone: batch the inference, fix the pad length, and do not pay for a per-step forward pass you can share.

9. **Write the run record the verifier actually grades** The stage spans, the replay digest, the guarded-path list, your own budget declaration and a reward_log whose macro-F1, localisation rate and seconds match what the run produced. The log is compared against the verifier's own recomputation and a disagreement is a fabrication, not a rounding.

## Traps that catch agents that are not thinking carefully

- The neutral class is collapsed into good or bad. Binary step accuracy stays respectable, the per-stratum neutral band is violated outright, and the score is exactly zero.
- Steps are scored independently and the first-error index is argmaxed separately, so the emitted index contradicts the emitted vector and the prefix-ordering invariant fires at pc above zero.
- A public scalar process reward model is thresholded globally. It passes aggregate macro-F1 and fails the per-stratum calibration band, because a scalar has nowhere to put the intermediate class.
- The sentinel is treated as a wildcard rather than a class, so every no-error solution reads as a match and the localisation rate measures nothing.
- Step indices are emitted one-based against a zero-based manifest. Every index is off by one, macro-F1 is untouched, and exact match collapses with no traceback anywhere.
- First-error localisation is fitted on short solutions and collapses on the long-solution subgroup where the error sits deep in a haystack of correct distractor steps.
- The final answer is read instead of the process, so every correct-answer-with-a-bad-step solution is emitted as -1 and the subgroup floor of 0.48 fires.
- Per-stratum thresholds are tuned against the graded shard's observed reward across attempts; the sibling shard's joint quality does not follow and the 0.04 gap cap fires.
- The pipeline is accurate but unbudgeted and overruns the slow end of the ramp on the full held-out step count, or moves calibration outside the timed region and loses stage closure instead.

---

## Contract record (annex; not part of the narrative body)

Retained under FORGE.md item 10e and Phase 2 item 7, which require this file to carry the ordered path through instruction.md with each satisfied checker identifier, and each rejected route bound to a measured known-wrong control. standards/truth-md-authoring-v1.md section 3 admits no fifth section, so this annex is a recorded deviation rather than an omission.

Ordered path through instruction.md, with the compiled check each step satisfies. Every check named here runs against delivered bytes; none of them is a self-report and none of them consults a run-record key that nothing writes.

1. `both_outputs_present_for_every_graded_solution` (weight 5, knockout) -- the coupled pair is written at /submission/c6s2.json, both halves per solution.
2. `ids_and_step_counts_match_the_graded_manifest` (weight 5, knockout) -- the id set and every vector length agree with the private re-segmentation.
3. `labels_drawn_from_the_three_human_classes` (weight 5, knockout) -- every label is good, neutral or bad, and the emitted alphabet is recorded.
4. `first_error_index_equals_first_bad_label_in_its_own_vector` (weight 5, knockout) -- pc equals 0.
5. `no_error_sentinel_is_typed_minus_one_not_a_wildcard` (weight 4) -- -1 is an exact class and every other index is in range.
6. `constant_label_or_constant_index_submission_rejected` (weight 4) -- the all-good, all-sentinel route is rejected on the face of the bytes.
7. `per_stratum_neutral_rate_inside_the_band` (weight 5, knockout) -- nb at or below 0.03 in every stratum.
8. `every_stratum_carries_emitted_neutral_support` (weight 4) -- no stratum with a dead neutral class, and the collapse direction reported.
9. `neutral_row_and_column_of_the_confusion_matrix_are_live` (weight 4) -- macro-F1 is a mean over three live classes.
10. `index_exact_match_scores_minus_one_as_its_own_class` (weight 4) -- the four-way pair accounting.
11. `global_index_offset_rejected` (weight 3) -- a whole-shard off-by-one is named rather than absorbed.
12. `long_solution_first_error_subgroup_floor` (weight 5, knockout) -- 0.44 on the deep-error subgroup.
13. `correct_answer_bad_step_subgroup_floor` (weight 5, knockout) -- 0.48 where the answer corroborates the wrong conclusion.
14. `sibling_shard_joint_quality_gap` (weight 4) -- 0.04 between the two private shards.
15. `counts_invariant_under_the_presentation_permutation` (weight 4) -- graded counts run over canonical id order.
16. `human_labels_and_stratum_map_not_held_agent_side` (weight 5, knockout) -- no guarded write and no private copy at any of three granularities.
17. `six_named_stages_inside_one_timed_invocation` (weight 3) -- stage closure and harness-measured seconds.
18. `reward_log_matches_recomputed_f1_and_localization` (weight 4, knockout) -- the log is faithful.
19. `both_coupled_artefacts_replay_byte_identically` (weight 3) -- the whole artifact reproduces from the recorded seed.
20. `staged_r_det_speed_ramp_behind_the_joint_quality_gate` (weight 10, knockout) -- the staged value, gated on the four measured constants being null.

Rejected routes, each bound to the control that measures it wrong:

- A downloaded scalar process reward model with one global cut -- CTRL_S2_PUBLIC_PRM_GLOBAL_THRESHOLD, projected macro-F1 0.507 with a worst per-stratum neutral deviation of 0.114, rejected by `per_stratum_neutral_rate_inside_the_band`.
- Independent per-step argmax with a separately argmaxed index -- CTRL_S2_INDEPENDENT_ARGMAX, projected 2,904 divergences in 10,000 with a macro-F1 of 0.571 that would otherwise clear T_F1, rejected by `first_error_index_equals_first_bad_label_in_its_own_vector`.
- All-good labels with -1 everywhere -- CTRL_S2_LIST_ALL, emitted neutral rate 0.000 against a human rate near 0.31, rejected by `constant_label_or_constant_index_submission_rejected` and by the band.
- Thresholds and segmentation carried over from the shipped split -- CTRL_S2_STALE_PRIOR, projected worst deviation 0.081, rejected by the band and by `ids_and_step_counts_match_the_graded_manifest`.
- The first-error index predicted from step count alone -- CTRL_S2_LENGTH_INDEX_PREDICTOR, projected macro-F1 0.371 and localisation 0.244, landing on the sub-gate ramp at 0.0126.
- Thresholds tuned against the graded shard's own reward over fifty attempts -- CTRL_S2_SHARD_FITTED, projected sibling gap 0.071 against the 0.04 cap.
- The reference with batch size 1 -- CTRL_S2_UNBATCHED_REFERENCE, identical quality, 3,940 s against the slow end of the ramp, proving the graded axis is genuinely wall clock.
- The reference with its training schedule truncated to a fifth -- CTRL_S2_UNDERTRAINED, every hard constraint holding and both quality gates missing, landing on the sub-gate ramp at 0.0589.
- The shipped scaffold run verbatim -- CTRL_S2_NOOP, which writes neither output and scores exactly zero.

Deliverable manifest: /submission/c6s2.json (json), an object with an ids list and a matching outputs list, each output carrying a labels sequence over the three human classes and an integer first_error_index whose no-error value is exactly -1.

<!-- AELLO-CANARY-BLOCK
  slot0: AELLO-CANARY-6fba8de0ad64e1b823dba2be01e8ffac
  slot1: AELLO-CANARY-f316b670f6f37e03b147995e2ec96c47
  slot2: AELLO-CANARY-0fe6e7abe0d04af8a82cc4703cc2353d
  slot3: AELLO-CANARY-a056eaed8430320e4f458335f5f1f722
-->
