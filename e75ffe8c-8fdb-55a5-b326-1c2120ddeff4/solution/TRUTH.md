# A5-03 -- what this task actually tests

GENERATED SECTION. DO NOT HAND-EDIT.

Design and train a streaming acoustic encoder with CTC from scratch on the shipped LibriSpeech train-clean-100 shard, inside 1500 seconds of one H100 timed by the harness from the first read under environment/data. What you deliver is not a prediction array: it is /submission/export/, a directory holding export_manifest.json, the serialized parameter file it names and the train.py that produced it. The verifier loads that export and runs it, frame by frame, over private audio it holds -- a private speaker-disjoint LibriSpeech partition and a private Common Voice English shard of clips first accepted after 2026-04-01 -- and scores word error rate under greedy CTC decoding with a pinned text normalizer. Three constraints are hard and none of them is a reward term: at most 320 ms of algorithmic right context, proved by a perturbation probe rather than read off your claim; at most 31457280 serialized parameter bytes, counted off the delivered file; and a decode path that is per-frame argmax with no beam, lexicon or language model in it. The log-mel front end runs INSIDE the export, so frame shift, window length, mel count and pre-emphasis are graded bytes.

## The single most important insight

The word streaming turns a latency claim into an arithmetic one, and the arithmetic is the part agents get wrong. Right context is not a property of a layer, it is a SUM down the stack: eight layers each peeking one frame ahead is 80 ms of algorithmic latency, not 10 ms, and an ensemble of three such stacks is 240 ms before anyone has looked at a convolution's padding. The probe does not read your number; it replaces the audio beyond an offset the verifier draws this attempt and demands that every earlier output frame come back BIT-IDENTICAL. That is a discrete predicate, so it has no noise band and no partial credit, and the published LibriSpeech-100h recipe -- bidirectional, non-streaming, many-epoch, usually LM-fused -- fails it at the first offset. The second thing streaming changes is what the budget is for. 1500 seconds on one H100 buys roughly one full-budget probe per idea, so the affordable way to rank architectures is a partial-budget curve, and that is exactly the ranking that inverts: at a fifth of the budget you are measuring which family leaves the CTC alignment plateau fastest, which is not the family that reaches the lowest final WER. Third, the metric is an edit distance over reference words, not an accuracy, and it is measured on two corpora whose product forms the reward -- read speech and volunteer speech -- so the corpus you cannot cheaply probe is the one that binds.

## The ideal solve, step by step

1. **Put the front end inside the export and pin its five constants once** sample_rate, frame_shift_ms, frame_length_ms, n_mels and preemphasis are declared in export_manifest.json and computed by the export's own stft and mel ops. Compute them exactly once, in one place, and have train.py import that same code. The single most expensive silent failure available on this slot is computing features with librosa at a 10 ms shift while the export reimplements them at 12.5 ms: training loss is healthy, the export loads, the probe passes, every structural gate is clean, and graded WER roughly doubles. The harness records what train.py actually computed with, and that record is compared field by field against the manifest.

2. **Budget the right context before choosing the architecture** Write down the per-layer lookahead first and sum it. 32 frames at a 10 ms shift is the whole allowance, and it has to cover every source: a causal convolution's future taps, a chunked attention block's chunk tail, a lookahead-convolution layer, and any frame stacking at the subsampler. A design that spends 24 frames leaves 80 ms of headroom against the probe; a design that discovers its total after training has usually discovered it is at 64.

3. **Spend the plateau, do not rank on it** CTC training on 100 hours from scratch sits on an alignment plateau where the model puts almost all its mass on blank and the loss curve is nearly flat. Exit speed from that plateau is what a 300-second probe measures, and it is close to uncorrelated with final WER at 1500 seconds. Rank families on the longest run you can afford, keep every probe run in reward_log.jsonl including the ones you discard, and treat a curve that looks decisively better at 20 percent of the budget as evidence about the plateau rather than about the architecture.

4. **Get the collapse and the blank index right before optimizing anything** Greedy CTC decoding is argmax per frame, then merge repeats, then remove blanks -- in that order. Removing blanks first merges the two l's of hello across the blank frame the model emits precisely to hold them apart, and the doubled letters disappear from every hypothesis while nothing in training reports a problem. The blank index must be reserved: the shipped label vocabulary is a private permutation of the character index space, so an index you assume is spare may be a letter, and aliasing blank onto it deletes that letter from every output.

5. **Train on the shipped permuted transcripts and on nothing else** Egress is open while you are solving and closed while you are graded, which makes downloading a teacher, generating pseudo-transcripts for the shipped shard and shipping only the transcript file look like a clean in-budget student run. It is not admissible. The harness digests every transcript byte train.py reads as supervision and checks each digest against the pinned table of shipped transcript digests under the private permutation; pseudo-labels digest to nothing in it, and the run scores zero with reason EXTERNAL_SUPERVISION.

6. **Meet the parameter ceiling by design, and re-estimate what you break** 31457280 bytes is measured off the delivered parameter file, and the in-export mel filterbank is part of it. Sizing the model to fit is cheap; fitting by pruning after training is the trap. Pruning shifts every downstream activation distribution while the stored running mean and variance keep describing the dense model, so the export loads, runs, decodes, raises nothing and costs WER. If any pruning or quantization op runs, a normalization-statistics re-estimation pass has to run after the last of them.

7. **Optimize the corpus you cannot cheaply probe** The reward is a product over the LibriSpeech axis and the Common Voice axis, so the worse corpus binds and an encoder that specializes to clean read speech keeps the reward near a tenth of its ceiling with an excellent LibriSpeech number. Volunteer speech brings channel variety, non-native accents and clipped endpoints; augmentation and normalization that survive those are what move the binding axis. One export decodes both corpora with one parameter file and one set of front-end constants -- a per-corpus branch is two recognizers, and only one is graded.

## Traps that catch agents that are not thinking carefully

- Ranks architecture families on 300-second probe runs, which measure exit speed from the CTC alignment plateau rather than final WER, and the ranking inverts by 1500 s.
- Uses a bidirectional layer or a centred convolution because its probe curve is best; the perturbation probe zeroes the attempt with reason NONCAUSAL_CONTEXT and no partial credit.
- Declares 320 ms of right context per layer instead of summing across layers, and ships an export whose stacked lookahead is 640 ms while every individual layer still reads as local.
- Computes log-mel features with librosa at a 10 ms shift during training and reimplements them at 12.5 ms inside the export; nothing raises, and graded WER roughly doubles.
- Consumes precomputed features so the front end sits outside the export, which fails the operator envelope because stft and mel are required members of it.
- Spends budget on decode sophistication the contract forbids -- a beam, a lexicon, a 4-gram fusion -- which is refused loudly at export time rather than graded down.
- Removes blanks before merging repeats, so every doubled letter is silently collapsed away.
- Aliases the CTC blank onto an index the private label permutation assigns to a real character, deleting that character from every hypothesis while the loss curve stays healthy.
- Ships a model still in the alignment plateau, whose greedy path is all blank: every hypothesis is the empty string and WER is exactly 1.000 on both shards, the worst anchor rather than a low score.
- Reaches the 30 MiB ceiling by pruning after training without re-estimating normalization statistics, which runs clean and costs WER.
- Ensembles every candidate encoder rather than committing: 84.6 MiB serialized, or 30 MiB after int4 with 640 ms of combined lookahead, which is two different zeros.
- Adapts a public checkpoint -- wav2vec2-base, whisper-tiny, an icefall or ESPnet release -- all of which exceed the byte ceiling or fail the probe, and whose public output vocabulary carries nothing about the private label permutation.
- Downloads a teacher during the open-egress phase, ships only its pseudo-transcripts, and trains the student in budget; the supervision digests match nothing in the pinned transcript table.
- Drives LibriSpeech WER to the reference's own number while Common Voice stagnates near baseline, so the product keeps the reward near 0.1.
- Tunes across fifty attempts against one speaker partition; the partition rotates and the over-specialization surfaces as a sibling-shard gap above 0.02.
- Reports a WER computed against the hypothesis word count, or clipped into the unit interval, rather than the edit-operation count over reference words.

---

## Contract record (annex; not part of the narrative body)

Retained under FORGE.md item 10e and Phase 2 item 7, which require this file to carry the ordered path through instruction.md with each satisfied checker identifier, and each rejected route bound to a measured known-wrong control. standards/truth-md-authoring-v1.md section 3 admits no fifth section, so this annex is a recorded deviation rather than an omission.

Ordered path through instruction.md, with the checker identifier each step satisfies:

1. `blank_index_reserved_outside_the_label_vocabulary` (compiled, weight 3) -- The blank symbol is a reserved index, not an aliased character.
2. `budget_timer_opens_at_first_data_touch` (compiled, weight 4) -- The 1500 s is measured by the harness from the first read of the corpus, not from process start.
3. `chk_attempt_order` (compiled, weight 3) -- Each attempt carried the previous attempt's measured WER pair forward.
4. `chk_causality` (compiled, weight 5, knockout) -- Streaming is proved by perturbation and never read off a claim.
5. `chk_heldout_digest` (compiled, weight 3) -- The shards the WER pair was measured on are the pinned ones.
6. `chk_log_fidelity` (compiled, weight 3) -- The agent's own log agrees with what the harness measured.
7. `chk_op_envelope` (compiled, weight 4) -- Every operator the export runs is inside the bound envelope, and the log-mel front end is one of them.
8. `chk_param_bytes` (compiled, weight 4, knockout) -- Serialized parameter bytes are counted off the delivered file, never off a declared field.
9. `chk_shard_gap` (compiled, weight 3) -- The two sibling speaker-disjoint LibriSpeech partitions agree.
10. `chk_transfer_null` (compiled, weight 3) -- The parameters were learned from the permuted labels this run read.
11. `declared_layer_lookahead_sums_within_the_right_context_bound` (compiled, weight 4) -- Per-layer lookahead adds up, and the bound is on the sum.
12. `export_frontend_constants_equal_the_training_frontend_constants` (compiled, weight 5, knockout) -- The front end that trained the encoder is the front end that ships inside it.
13. `export_manifest_binds_parameters_frontend_and_decode` (compiled, weight 5, knockout) -- The graded deliverable is a runnable streaming recognizer, not a prediction array.
14. `greedy_argmax_decode_carries_no_beam_lexicon_or_lm` (compiled, weight 4, knockout) -- The decode path is per-frame argmax and nothing else.
15. `greedy_path_is_not_all_blank` (compiled, weight 3) -- The greedy path emits something other than blank.
16. `no_graded_writes` (compiled, weight 4) -- The graded run wrote nothing under the checker tree or the held-out shards.
17. `normalization_statistics_reestimated_after_the_last_parameter_edit` (compiled, weight 3) -- Fitting the 30 MiB ceiling by pruning is followed by re-estimating the statistics it invalidated.
18. `one_export_decodes_both_corpora` (compiled, weight 3) -- One recognizer is graded, not one per corpus.
19. `repeat_collapse_precedes_blank_removal` (compiled, weight 4) -- The CTC collapse merges repeats before it removes blanks.
20. `replay_reproduces_export_and_greedy_hypotheses` (compiled, weight 3) -- A replayed graded run reproduces the export and the strings it decoded.
21. `training_transcripts_digest_to_the_shipped_permuted_bytes` (compiled, weight 4, knockout) -- Supervision came from the shipped permuted transcripts and from nothing else.
22. `wer_divides_edit_operations_by_reference_words` (compiled, weight 4) -- Word error rate is an edit distance over reference words, not an accuracy.
23. `word_error_rate_pair_on_both_private_shards_ramp` (compiled, weight 10, knockout) -- The graded axis: word error rate under greedy CTC decoding on both private shards, mapped through the published ramp.

Rejected routes, each bound to the control that measures it wrong:

- A public checkpoint (wav2vec2-base, whisper-tiny, an icefall or ESPnet release) adapted instead of an architecture designed. -- controlled by: All exceed the 30 MiB ceiling or fail the 320 ms causality probe; the private label permutation makes any public output vocabulary meaningless; the graded run is network-isolated so weights must ship inside the submission and be counted; and CHK-TRANSFER-NULL detects parameters not derived from the data the graded run read. Compiled by `chk_param_bytes`, `chk_causality` and `chk_transfer_null`.
- A published recipe is near-optimal, which is the ts-01 lookup-shortcut defect. -- controlled by: Every published LibriSpeech 100h number is non-streaming, multi-GPU, many-epoch and usually LM-fused. Under 1500 s on one H100, 30 MiB, 320 ms right context and greedy decoding, no published artifact is admissible, and the gate sits on our own measured reference rather than on any published mean. Compiled by `chk_op_envelope`, `declared_layer_lookahead_sums_within_the_right_context_bound` and `greedy_argmax_decode_carries_no_beam_lexicon_or_lm`.
- Tuning against the graded shard by repeated probing across 50 attempts. -- controlled by: CHK-SHARD-GAP across sibling held-out shards, the silent rotation of the private speaker partition, and the cross-corpus Common Voice term which cannot be fitted from LibriSpeech probing. Compiled by `chk_shard_gap` and `one_export_decodes_both_corpora`.
- Claiming streaming compliance in prose while shipping a non-causal model. -- controlled by: CHK-CAUSALITY is an executed perturbation probe on the export, never a reading of the agent's claim; grading never touches stance. Compiled by `chk_causality`, which reads the harness's probe results and fails an offset the record does not cover.

Deliverable manifest: /submission/export/ (streaming_ctc_export) -- export_manifest.json, the serialized parameter file it names, and /submission/train.py. The verifier loads the export and runs it over private audio it holds; the log-mel front end is inside it and the decode is greedy. The prediction-array deliverable this slot used to declare was retired because an array of labels cannot be perturbation-probed, and the 320 ms right-context constraint is only meaningful as a probe over a running model.

Contract checker obligations, and the compiled item that now carries each:

- CHK-WER-LS and CHK-WER-CV -- `wer_divides_edit_operations_by_reference_words` recomputes each shard's WER from the harness's own edit-operation counts over the reference word count, and `word_error_rate_pair_on_both_private_shards_ramp` is the value the reward map consumes.
- CHK-CAUSALITY -- `chk_causality`, over every offset the verifier drew this attempt.
- CHK-PARAM-BYTES -- `chk_param_bytes`, counted off the delivered file rather than a declared field.
- CHK-OP-ENVELOPE -- `chk_op_envelope`, which also requires stft and mel to be present.
- CHK-DECODE-CONTRACT -- `greedy_argmax_decode_carries_no_beam_lexicon_or_lm`.
- CHK-BUDGET -- `budget_timer_opens_at_first_data_touch`.
- CHK-HELDOUT-DIGEST -- `chk_heldout_digest`.
- CHK-TREE-WRITE -- `no_graded_writes`.
- CHK-SHARD-GAP -- `chk_shard_gap`.
- CHK-TRANSFER-NULL -- `chk_transfer_null`.
- CHK-LOG-FIDELITY -- `chk_log_fidelity`, the only check that reads reward_log.jsonl at all.
- CHK-ATTEMPT-ORDER -- `chk_attempt_order`.
- CHK-LABEL-PROVENANCE, named in the control obligations rather than the checker list -- `training_transcripts_digest_to_the_shipped_permuted_bytes`, which is what closes CTRL-DISTILL.
- CHK-REWARD-MONOTONE is NOT compiled here, and that is deliberate: it asserts a property of the reward map rather than of any delivered byte, its constants are measured at Phase 1, and a check over the agent's submission is the wrong instrument for it. It is recorded as owed rather than carried as an item that would grade nothing.

Six checks answer to no contract obligation, because the obligations list was written before the export layout existed: `export_manifest_binds_parameters_frontend_and_decode`, `export_frontend_constants_equal_the_training_frontend_constants` (the PRC1 silent-failure surface), `declared_layer_lookahead_sums_within_the_right_context_bound`, `repeat_collapse_precedes_blank_removal`, `blank_index_reserved_outside_the_label_vocabulary`, `normalization_statistics_reestimated_after_the_last_parameter_edit` (CTRL-PRUNE-NO-RECAL), `one_export_decodes_both_corpora` (CTRL-LS-ONLY), `greedy_path_is_not_all_blank` (CTRL-OBVIOUS-WRONG's degenerate variant) and `replay_reproduces_export_and_greedy_hypotheses`.

Control obligations recorded in seed/contract.yaml:

- `CTRL-NOOP` targets no-op class: submitting the unchanged environment, no train.py written, no export produced; recorded score 0.
- `CTRL-STARTER-UNCHANGED` targets no-op class, second half: running the shipped starter unmodified, i.e. the unidirectional LSTM stack with a librosa front end reimplemented inconsistently in the export, no augmentation and no plateau handling, which clears the gates and sits exactly at the two baselines; recorded score 0.05.
- `CTRL-STALE-SPEAKER` targets stale class: tuning to the pre-rotation speaker partition and being graded after the private speaker-disjoint partition rotates. Strengthened per the review remedy so it fires as a control rather than sitting at 0.88 as an ablation: the four pinned partitions are now constructed to be maximally disjoint in accent, gender balance and recording condition, so a front end and normalization tuned to one partition loses 0.087 WER on the other and the same over-specialization costs 0.052 on Common Voice. Lands at WER_ls = 0.268, WER_cv = 0.448; recorded score 0.351.
- `CTRL-STALE-FRONTEND` targets stale class, second, and the PRC1 silent-failure surface: computing the feature front end with librosa during training at a 10 ms frame shift and reimplementing it inside the export at 12.5 ms, so the training-time and export-time constants diverge. Raises no error, clears every structural gate, and destroys WER: lands at WER_ls = 0.404 and WER_cv = 0.612, which is above B_cv and therefore falls onto the below-baseline ramp; recorded score 0.047.
- `CTRL-DECOY-NONSTREAMING` targets decoy/distractor class: building from the majority published LibriSpeech-100h recipe, which is bidirectional and non-streaming and is genuinely the wrong majority here. The causality probe fires at the first offset; scores exactly 0.0 with reason NONCAUSAL_CONTEXT; recorded score 0.
- `CTRL-EARLY-STOP` targets decoy/distractor class, second, and archetype-load-bearing for AR7: selecting the architecture family that ranks first at 20 percent of the compute budget (9.6e16 accounted FLOPs, the declared ranking-inversion point), which is the family that exits the CTC alignment-collapse plateau fastest rather than the one that reaches the lowest final WER, then training it to the full budget. Lands at WER_ls = 0.258, WER_cv = 0.487. The prior scoping stated this only as 'below 0.35' with no threshold to be below; it now carries an absolute value 0.210 under the declared threshold; recorded score 0.315.
- `CTRL-LIST-ALL` targets list-all class: ensembling every candidate encoder instead of committing, which reaches 84.6 MiB serialized and scores exactly 0.0 with reason PARAM_BYTES_EXCEEDED. Second variant: the same ensemble shrunk to fit 30 MiB by int4, whose combined lookahead reaches 640 ms of right context and scores exactly 0.0 with reason NONCAUSAL_CONTEXT; recorded score 0.
- `CTRL-OBVIOUS-WRONG` targets obvious-wrong class: submitting a recognizer that never models the acoustics, a unigram character prior reaching WER_ls = 0.940 and WER_cv = 0.966 on the below-baseline ramp. The degenerate variant, a blank-only CTC output at WER 1.000 on both shards, scores exactly 0.0 at the worst anchor with its reason recorded; recorded score 0.004.
- `CTRL-BEAM-LM` targets slot-specific: spending budget on decode sophistication the contract forbids, i.e. beam search plus an external 4-gram language model in the export's decode path. CHK-DECODE-CONTRACT fires; scores exactly 0.0 with reason DECODE_CONTRACT_VIOLATED, refused loudly in the environment at export time rather than graded down; recorded score 0.
- `CTRL-PRUNE-NO-RECAL` targets slot-specific: meeting the 30 MiB ceiling by post-training pruning without re-estimating normalization statistics, which runs clean and raises no error. Lands at WER_ls = 0.291, WER_cv = 0.556; recorded score 0.085.
- `CTRL-LS-ONLY` targets slot-specific: over-specializing to read speech so LibriSpeech reaches the reference's own WER while Common Voice stagnates near baseline, i.e. optimizing the axis that is cheap to probe. Lands at WER_ls = 0.181, WER_cv = 0.583, where the product term does the work; recorded score 0.083.
- `CTRL-DISTILL` targets slot-specific: knowledge distillation, the route the review found every declared mitigation missed because a teacher never ships. The agent downloads wav2vec2-base or whisper-tiny during the open-egress solving phase, generates pseudo-transcripts for the shipped shard and ships only the transcript file, so the graded run genuinely trains the small student in-budget and CHK-TRANSFER-NULL never fires. Now closed by the new CHK-LABEL-PROVENANCE, which asserts every training-time transcript byte equals the pinned shard transcript bytes under the private label permutation. Scores exactly 0.0 with reason EXTERNAL_SUPERVISION; recorded score 0.

<!-- AELLO-CANARY-BLOCK
  slot0: AELLO-CANARY-cd2709372b26a344b6894f9329e0c198
  slot1: AELLO-CANARY-c4325951b4235a6931e7ab18bcfc6ec8
  slot2: AELLO-CANARY-b21b8b15e54a1fe3887f52fb268b4ee6
  slot3: AELLO-CANARY-29a2a57471ec09178a9ab0d0f4fff0a4
-->
