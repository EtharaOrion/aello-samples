# C2-S2 -- what this task actually tests

GENERATED SECTION. DO NOT HAND-EDIT.

The environment ships eight pinned stage tools under environment/stage/ -- render, sft, merge, seed, rollout, score, update and export -- each read-only, each digest-locked, each documenting its complete interface under --help, together with a suboptimal starter orchestration script environment/starter/train.sh that runs all eight to completion and reaches roughly 0.19 end-state accuracy in roughly 2900 s. The deliverable is not a training script. It is an orchestration: the agent chooses the invocation order, binds twelve declared interface arguments across seven link boundaries, and pipes each stage's artifacts into the next. Not one byte under environment/stage/ may change. The graded artifact is the exported model together with the chain manifest written to /submission/c2s2.json, which records per stage invocation the digest of every artifact it consumed and every artifact it produced, the arguments it was called with, and the sandbox epoch it ran under. Grading is a composite over two axes, published verbatim in instruction.md: chain end-state accuracy over 1200 private executable tool-call episodes, and harness-measured wall clock of the grader's single timed invocation. Both axes are published, both weights are fixed, and neither is the agent's to choose.

## The single most important insight

Every stage exits 0 on a mis-bound input. That single fact is the task. A wrong binding does not raise, does not warn and does not leave a mark in a log; it produces an artifact that satisfies the next stage's syntactic precondition and violates its semantic one, and the chain runs happily to an export that loads. So exit status carries no information, and neither does a clean run: roughly 1.05e5 argument combinations times at least forty admissible stage orderings all execute to completion, and only a handful of them are semantically valid. The second fact is that the feedback cannot tell you which link broke. The twelve single-link mis-bindings each land below the lowest target on their own, and all twelve land inside a band 0.07 wide, so the one scalar the refinement loop carries between attempts is the same number whichever argument is wrong. Fifty graded evaluations against roughly 4.2e6 realizable configurations is not a search when the signal does not localize. What replaces the missing signal is reading the interfaces: each stage's --help states what it consumes, what convention it applies internally, and what it emits, and every intermediate artifact carries a manifest with a tokenizer fingerprint, a template id, a mask convention, an epoch id and a producer digest. Nothing here is undecidable from inside the container. The agent that wins reconciles adjacent stages' declared conventions before running anything; the agent that loses runs the chain, sees a plausible number, and tries a different learning rate.

## The ideal solve, step by step

1. **Pin one sandbox epoch before anything else runs.** Call seed exactly once and hoist it ahead of render, ahead of the warm start and ahead of every rollout. The epoch is mutable cross-stage state: a second seed call placed anywhere before score rebases the world, the score stage accepts a rollout carrying the superseded epoch_id, exits 0 and emits no warning, and every reward from that point is computed against a state no rollout ever observed. The rewards do not look broken -- they look uniformly a little low, which is exactly what an underfit policy looks like. Bind --epoch-binding to inherit and let every downstream record carry the one epoch the seed produced.

2. **Reconcile the mask convention across the render-to-sft boundary before training a single step.** The sft stage shifts labels internally. Handing it a mask under the pre-shift convention, which is what the starter does, lands the mask one position off and silently trains on the user turns and the tool-result turns as though the model had written them. No error is raised and the loss curve is healthy the whole way down, because a model can fit user text perfectly well. Bind --mask-convention postshift. This is the single binding whose fault is invisible everywhere except in end-state accuracy on episodes the agent never sees.

3. **Pin the generation prompt to one explicit value at both ends of the render-to-rollout link.** The starter leaves --generation-prompt on auto at render and on at rollout, so every training prompt differs from every inference prompt by one token of the shipped Qwen2.5-1.5B-Instruct tool-call template. One token is enough: the model learns to emit its first call after a prefix it will never see again. auto is not a value, it is a deferral to whichever stage resolves it first, and the two stages resolve it differently. Choose a member of the domain and bind the same one at both invocations.

4. **Align padding across the rollout-to-update link, because the KL term is computed positionally.** Rollout emits left-padded sequences. An update stage bound to right padding reads the reference logprobs at shifted positions, so the KL penalty is computed between a distribution and a shifted copy of itself. The loss curve looks ordinary, the policy drifts, and end-state accuracy lands below the gate. Capture reference logprobs pre-update at the pad side the rollout actually emitted, and bind --pad-side to that side rather than to the update stage's default.

5. **Merge before rollout, and carry the render spec digest into the export.** The vLLM engine has to load merged weights whose template and tool-token ids match the render spec, so merge belongs before rollout rather than after the loop. Set --embed-resize to tool-tokens consistently at render, merge and rollout, or the tool-call special tokens map to unrelated ids the moment the adapter is merged. Export with --export-form merged rather than adapter and carry the render spec digest into the export record: an adapter export is rejected outright, and a merged export carrying some other spec digest means the grader's renderer and the exported model disagree about the template, which no stage in the chain is positioned to notice.

6. **Down-weight the triplicated wrong-order chains before fitting anything to them.** For a subset of the shipped training episodes the split carries three unmarked copies of a call chain that is correct in content and wrong in ORDER, against one copy of the correct ordering. A frequency-weighted fit learns the wrong ordering and per-call accuracy does not move at all, because every call in the majority copies is individually right. The signature is a divergence between per-call accuracy and end-state accuracy: the first stays high, the second collapses, because the graded episodes execute in a stateful sandbox where order decides the end state. Detect the triplication by grouping episodes on their call multiset and bind --decoy-downweight rather than trusting the majority.

7. **Build the GRPO advantage baseline inside each episode's rollout group.** Score is a pinned stage and its executable reward is not the agent's to redefine, so the only lever left on the update side is what the advantage is normalized against. Normalizing across the batch mixes episodes whose reward scales differ by construction -- a six-call travel episode and a two-call message episode are not comparable -- and the policy drifts toward whichever class happens to be easy, with no error raised. Bind --advantage-baseline to the rollout group, and choose the reference model dtype independently of the bf16 the rest of the chain runs in, then show the KL term stays finite at the value chosen.

8. **Write the chain manifest as the thing the verifier will recompute, not as a report.** Closure is recomputed from artifact digests, per artifact NAME: every consumed digest has to be the most recent one emitted under that name by a stage the published artifact-producer map entitles to emit it. Per name rather than per acyclic predecessor, because the GRPO cycle legitimately re-produces merged_weights at update and hands it back to rollout. The most-recent rule is what catches a score stage run against a previous iteration's rollout file, which the stage accepts and exits 0 on. Record every invocation, in the order it ran, with its stage digest, its epoch id, the arguments it was called with, and both digest lists. Anything the orchestration did that the manifest does not describe -- a cached rollout, a hand-run merge, a second seed -- is either invisible to the grader or breaks closure, and neither outcome is one worth discovering at grading.

9. **Buy the wall back from the loop structure, not from a faster kernel.** The starter finishes in roughly 2900 s for roughly 0.19 accuracy; the reference finishes in roughly 2000 s at roughly 0.46. The 900 s is not in any single stage. It comes from batching the rollout stage over episodes rather than walking them, and from overlapping score with the next rollout, both of which are ordering decisions the partial order permits and the starter does not take. The speed axis is a log ramp that saturates at 2700 s and reads zero at 3600 s, so seconds recovered below the knee are worth nothing and seconds spent above it are worth a great deal; measure the timed window before spending it on more GRPO iterations.

## Traps that catch agents that are not thinking carefully

- Reading a clean run as a correct one. All eight stages exit 0 under every one of the 1.05e5 argument bindings and every one of the at least forty admissible orderings. A green pipeline is the null result here, not the signal.
- Treating the scalar reward as a debugger. The twelve single-link ablations lie inside a band 0.07 wide, so the number returned after a failed attempt is the same number whichever link is mis-bound. Trying twelve one-argument variants across the attempt budget spends the budget and learns nothing; reading twelve --help pages costs one attempt's worth of wall and resolves the bindings outright.
- Optimizing per-call accuracy. There is no per-call partial credit anywhere in the grading. Emitting every plausible call in the schema raises recall and mutates backend state, so end-state equality collapses while the number the agent can compute for itself goes up. The two metrics diverge by at least 0.10 on the decoy-trusting route while staying within 0.03 of each other per call.
- Memorizing the six shipped API classes. Two of the eight backend classes are absent from the environment entirely and their function schemas are seen for the first time at grading; the transfer stratum is 300 of the 1200 graded episodes and its floor caps the quality fraction at 0.60 when it fails. A policy that clears the overall bar on the shipped classes and collapses on transfer is capped, not credited.
- Hardcoding literals seen in the shipped instances. Every private episode's backend initial state is re-instanced under the identical class schemas, so file trees, ticket identifiers, portfolio holdings and vehicle states all differ from anything shipped. A memorized gold chain carries the wrong literals and fails end-state equality on every private episode while executing without raising.
- Ignoring the haystack stratum's decoys. Two hundred graded episodes pad the tool schema with forty or more near-name functions -- send_message against send_message_to_user against message_send, and equivalents across all eight classes. The model picks the wrong one at its own call site, the call executes, and the end state is wrong.
- Wrapping a stage instead of composing it. A shim, a patched copy or a re-implemented sandbox is rejected on the pinned per-file digests, and a re-implemented sandbox produces no epoch id at all, so the epoch binding fails as well. Composition order is the lever; the interface is the adversary rather than a suggestion.
- Spending the timed window on more GRPO iterations than the log ramp pays for. Above 3600 s plus 300 s of grace the run scores exactly 0.0 under budget_exceeded, which is not a slightly worse score. Below the 2700 s knee the speed term is already saturated and further seconds buy nothing.
- Looking for the private episodes. The tree is not mounted in the graded container at all, so reaching for it is not forbidden-but-tempting, it is unreachable; a process opening a path under it produces a mutually exclusive wrong deliverable at exactly 0.0 rather than a judged opinion about intent.

---

## Contract record (annex; not part of the narrative body)

Retained under FORGE.md item 10e and Phase 2 item 7, which require this file to carry the ordered path through instruction.md with each satisfied checker identifier, and each rejected route bound to a measured known-wrong control. standards/truth-md-authoring-v1.md section 3 admits no fifth section, so this annex is a recorded deviation rather than an omission.

Ordered path through instruction.md, with the checker identifier each step satisfies:

1. `chain_end_state_accuracy_ramp` (compiled, weight 10, knockout) -- Chain end-state accuracy over the 1200 private episodes clears the published bar through the ramp, and prices the quality axis of the published composite.

2. `chain_manifest_records_every_stage_invocation` (compiled, weight 5, knockout) -- The orchestration emits a chain manifest describing every invocation of the eight pinned stage tools, alongside the bindings, the digests and the declared composite.

3. `consumed_digest_resolves_to_an_earlier_producing_stage` (compiled, weight 5, knockout) -- Every consumed artifact digest closes against the most recent digest emitted under that artifact name by a stage declared to produce it.

4. `each_single_link_misbinding_alone_falls_below_t1` (compiled, weight 3) -- No one of the twelve bound arguments is individually optional.

5. `end_state_digest_equality_with_no_per_call_credit` (compiled, weight 5, knockout) -- An episode counts only on exact backend end-state equality, and per-call accuracy is never a term in the numerator.

6. `every_invocation_names_consumed_and_produced_digests` (compiled, weight 5, knockout) -- Each stage invocation records the digest of every artifact it consumed and every artifact it produced.

7. `export_form_merged_carries_the_render_spec_digest` (compiled, weight 5, knockout) -- The export is a merged model and it carries the digest of the render spec the chain was trained under.

8. `generation_prompt_pinned_to_one_value_at_render_and_rollout` (compiled, weight 3) -- The generation-prompt flag holds one value across the render link and the rollout link rather than being left to each stage's default.

9. `grader_reason_codes_equal_the_validate_export_codes` (compiled, weight 3) -- The failure vocabulary the agent debugs against in-environment is the same list the grader can emit.

10. `haystack_and_transfer_floors_cap_quality_at_060` (compiled, weight 3) -- Clearing the overall bar on the six shipped API classes does not lift the cap that the two never-shipped classes impose.

11. `invoked_stage_digest_equals_the_pinned_tool_digest` (compiled, weight 5, knockout) -- The eight stage tools are composed, never rewritten, wrapped or shimmed.

12. `mask_convention_postshift_because_sft_shifts_labels_itself` (compiled, weight 3) -- The loss mask is handed to the sft stage under the convention that stage actually applies.

13. `no_private_episode_path_opened_and_no_gold_digest_carried` (compiled, weight 5, knockout) -- The graded container never reaches the private episode tree, and the submission carries none of its digests.

14. `one_seed_call_pins_the_epoch_for_the_whole_run` (compiled, weight 5, knockout) -- The sandbox epoch is pinned once, ahead of everything, and never rebased mid-chain.

15. `pad_side_agrees_between_rollout_and_update` (compiled, weight 3) -- Reference logprobs are consumed at the padding side the rollout stage emitted them on.

16. `per_stage_seconds_declared_for_all_eight_stages` (compiled, weight 3) -- The agent says where its wall went, stage by stage across all eight.

17. `realized_sequence_linearizes_the_declared_partial_order` (compiled, weight 5, knockout) -- The order the eight stages actually ran in is an admissible linearization of the partial order instruction.md declares.

18. `replay_on_a_fresh_epoch_reproduces_the_end_state_digests` (compiled, weight 3) -- Replaying the graded export's emitted call chains twice against a freshly seeded epoch yields byte-identical backend end-state digests.

19. `score_epoch_equals_the_rollout_and_seed_epoch` (compiled, weight 5, knockout) -- Every scored record is bound to the epoch of the rollout it scores and of the seed that produced that world state.

20. `strata_counts_partition_the_1200_private_episodes` (compiled, weight 3) -- The three private strata partition the graded split exactly, and the counts are summed rather than taken from a stated total.

21. `timed_window_is_host_measured_not_self_reported` (compiled, weight 3) -- The seconds that price the speed axis are the harness's, and every stage invocation falls inside that one window.

22. `twelve_ablation_rewards_lie_inside_a_007_band` (compiled, weight 3) -- The scalar the refinement loop carries between attempts cannot say which link broke.

23. `twelve_interface_arguments_bound_inside_published_domains` (compiled, weight 5, knockout) -- All twelve declared interface arguments are bound, each to a value from the domain instruction.md publishes for it.

24. `two_axis_composite_uses_the_published_weights_verbatim` (compiled, weight 3) -- The composite over the two axes is the one instruction.md publishes verbatim; the weighting is not the agent's to choose.

Rejected routes, each bound to a measured known-wrong control:

- `empty_submission_noop` targets no-op: unchanged environment with no train.sh; recorded score 0.

- `stale_epoch_control` targets stale: calling seed a second time before score, so the score stage consumes a rollout carrying a superseded epoch_id; every stage exits 0 and emits no warning, and the run trips epoch_unbound; recorded score 0.

- `stale_score_input` targets stale: running the pinned score stage against a previous iteration's rollout file, which the stage accepts and exits 0 on, breaking the chain_manifest digest closure; recorded score 0.

- `literal_state_hardcode` targets stale: a policy emitting calls with literals observed in the shipped instances, which fails end-state equality on every re-instanced private backend state; recorded score 0.

- `emit_all_calls_control` targets list-all: emitting every plausible call in the schema to maximize per-call recall; the extra calls mutate backend state so end-state equality collapses; recorded score 0.

- `gold_leak_attack` targets verifier-aware attack: opening a path under the private episode tree to reach the gold end-state digests; recorded score 0.

- `majority_order_decoy` targets decoy: trusting the triplicated unmarked call chains that are correct in content and wrong in ORDER, so a frequency-weighted fit learns the wrong ordering while per-call accuracy stays within 0.03; recorded score 0.265.

- `starter_defaults_control` targets obvious-wrong: the shipped starter run unmodified with four of the twelve interface arguments mis-bound and every stage exiting 0; recorded score 0.288.

- `exposed_constants` targets disclosure: holding T, the stratum floors, the complete reward map, the reason-code list, the declared stage partial order and the full domain of all twelve bound arguments, without training and without re-binding the starter; recorded score 0.288.

- `pad_side_control` targets obvious-wrong: consuming left-padded rollout logprobs in an update stage that assumes right-padding, so the KL term is computed over shifted positions with no error raised; recorded score 0.353.

- `haystack_decoy` targets distractor: the 200-episode stratum whose tool schema carries 40 or more near-name decoy functions, taken without any schema-disambiguation handling; recorded score 0.382.

- `clean_exit_decoy` targets the twelve single-link argument mis-bindings applied one at a time to the reference; every stage exits 0 with clean logs and each lands in acc [0.200, 0.219], reported here at the band top; recorded score 0.409.

- `mask_convention_control` targets obvious-wrong: passing the loss mask under the pre-shift convention into a stage that shifts labels internally, silently training on the user and tool-result turns; recorded score 0.412.

- `shipped_class_memorizer` targets memorization: a policy fitted to the six shipped API classes, which collapses on the transfer stratum over the two never-shipped classes and trips the transfer floor cap; recorded score 0.432.

<!-- AELLO-CANARY-BLOCK
  slot0: AELLO-CANARY-323a294673512fc8b207f58340b7b58f
  slot1: AELLO-CANARY-3f898a18d5e6f83c266c22f0b561dd0e
  slot2: AELLO-CANARY-1ac3fab5ecd8397d1398bde2c191f6c6
  slot3: AELLO-CANARY-e00cbd212d4e05bf3576f0b24bc52caa
-->
