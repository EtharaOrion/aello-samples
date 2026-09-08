# AELLO-C8-S7 -- what this task actually tests

GENERATED SECTION. DO NOT HAND-EDIT.

maximise keyword accuracy on a private split while every inference stays inside a bound per-utterance latency. The graded artifact is one integer keyword index per graded utterance -- 3500 rows, 100 for each of the 35 Speech Commands v0.02 keywords -- written to /submission/preds_c8s7.npy. The graded quantity is top-1 accuracy over that private split, forced to exactly zero when any one of the 3500 inferences the harness times exceeds 20.0 ms.

## The single most important insight

Two objectives are bound to one forward pass, and the binding is what makes this hard. Accuracy wants a wider window, more mel bands, a raw-waveform branch beside the spectrogram branch, and a fusion deep enough that the two views inform each other. Every one of those costs milliseconds on the same inference the gate measures, and the gate is a maximum: it is not enough to be fast on average, or fast on 3499 of 3500 utterances. So the design question is not how accurate a model can be, nor how fast, but which of the front-end levers buys the most accuracy per millisecond -- and that is a question about the mel resolution, the hop, and where the two branches join, not about model size. The second thing that matters is that both numbers have to describe the same inference. A latency measured on one configuration and an accuracy measured on another is two measurements of two models, and the checks bind them together with a digest taken on both passes.

## The ideal solve, step by step

1. **Pin the label mapping before training anything** Class index 0 is backward and class index 34 is zero, the 35 keyword folders in lexicographic order, and the run record has to say so. This is the one mistake on this corpus that hides completely: a model trained against its own ordering emits a well-formed vector of integers, all in range, that scores near 1/35, and nothing in the artifact distinguishes it from a model that simply did not learn. Build the encoder from the published roster rather than from whatever order the loader walks, and declare it.

2. **Choose the front end against the gate, not after it** The window, the hop and the mel band count are the levers, and they set both the accuracy and most of the cost. Frames per utterance is 1 + (16000 - window) // hop, so halving the hop doubles the frames and roughly doubles the per-utterance convolution cost; the mel band count sets the width of every one of those frames. Pick the operating point by measuring accuracy per millisecond across the box, not by maximising accuracy and then hoping the gate is loose. The arithmetic is checked: a declared frame count that does not follow from the declared window and hop describes a front end that did not run.

3. **Make the two branches share a representation before the decision** A waveform branch and a spectrogram branch that meet only at the logits are two models and a vote. Neither branch can use what the other resolved -- the waveform branch's fine time structure never reaches the spectrogram branch's frequency detail -- and the vote costs the full inference time of both. Joining earlier, at a shared feature stage, is what makes the second branch worth the milliseconds it spends; it is also the choice the contract records as separately ablatable, and its worth is measured on the Phase 2 ladder rather than asserted here.

4. **Time every utterance separately, and let the harness do it** The gate is applied per inference, so the measurement is per inference: 3500 timings, one per utterance, taken by the harness's clock. Batching the whole split into one forward pass and dividing by 3500 amortises away precisely the cost the bound exists to constrain -- the per-utterance overhead a deployed spotter actually pays -- and a timer the solve process holds is red line 2 regardless of how carefully it is written.

5. **Spend the noise lever on the training split only** Mixing the six vendored background clips into training audio between 0 and 20 dB is the robustness lever on this corpus and it is free at inference time. Mixing them into the graded utterances is not augmentation, it is rewriting the graded input: it raises accuracy and lowers latency at once by changing what is being classified, the archive digest moves, and red line 1 zeroes the attempt.

6. **Declare the budget and replay the labels** Train seconds, the number of utterances the design was timed against, and attempts used are the agent's own accounting, and the second of those is where this slot's honest work shows: a front end tuned against a dozen clips has not been measured against a gate it must clear on 3500. Then re-run under the recorded seed -- the label vector must come back byte-identical under the same front-end digest, even though the milliseconds will not.

## Traps that catch agents that are not thinking carefully

- Fuses at the logit level and the two branches never share representation -- the waveform branch then pays its full inference cost against the gate while contributing only a vote, which is the worst of both objectives.
- Meets accuracy by exceeding the latency bound, and the gate zeroes the attempt. The gate is read before the labels are compared, so 3499 inferences inside the bound do not pay for the one outside it.
- Reads the gate as an average, or as a high percentile. A mean over 3500 utterances hides an arbitrarily slow inference; a 99th-percentile gate passes 35 of them by construction.
- Times a batched forward over the whole split and divides by the row count, which amortises away the per-utterance cost the bound is written to constrain.
- Times a cheap front end and scores an expensive one. One digest is taken on the timed pass and one on the scored pass, and they must be equal and must be the digest of the declared block.
- Trains against its own label ordering, emits 3500 in-range integers that score near chance, and reports a modelling failure that was a mapping failure.
- Declares a hop wider than its window, so a keyword onset can fall entirely between frames, or more mel bands than the window has positive-frequency bins, so the extra filters cost latency and carry nothing.
- a pretrained audio encoder would shortcut the fusion design Mitigation: no model-zoo host is reachable at solve time; any backbone must be vendored

---

## Contract record (annex; not part of the narrative body)

Retained under FORGE.md item 10e and Phase 2 item 7, which require this file to carry the ordered path through instruction.md with each satisfied checker identifier, and each rejected route bound to a measured known-wrong control. standards/truth-md-authoring-v1.md section 3 admits no fifth section, so this annex is a recorded deviation rather than an omission.

Ordered path through instruction.md, with the checker identifier each step satisfies:

1. `class_index_maps_to_the_pinned_keyword_roster` (compiled, weight 5, knockout) -- The run
declares which keyword each class index means, and it is the pinned roster in the pinned order.

2. `front_end_frame_count_follows_from_window_and_hop` (compiled, weight 3) -- The declared window,
hop and mel band count are arithmetically possible for a one-second 16 kHz utterance.

3. `timed_front_end_is_the_scoring_front_end` (compiled, weight 5, knockout) -- One front-end
configuration served the timed pass and the scored pass, and it is the one the record declares.

4. `every_graded_utterance_timed_by_the_harness` (compiled, weight 5, knockout) -- The harness
holds one measured latency for every graded utterance and for nothing else.

5. `latency_measured_by_the_harness_clock_not_the_agents` (compiled, weight 5, knockout) -- The
graded latency comes from the harness's timer and no agent-held clock is offered in its place.

6. `latency_gate_is_a_per_utterance_maximum_not_a_mean` (compiled, weight 5, knockout) -- The gate
is recomputed as a maximum over utterances and the run's own verdict agrees with it.

7. `keyword_label_vector_lands` (compiled, weight 5, knockout) -- The keyword decision vector lands
at the declared path as a one-dimensional integer array.

8. `one_label_per_graded_utterance` (compiled, weight 5, knockout) -- Exactly one decision per
graded utterance, neither short nor long.

9. `labels_inside_the_thirty_five_keyword_index` (compiled, weight 5, knockout) -- Every decision
names a keyword the graded split contains.

10. `no_keyword_claims_more_than_the_balanced_share_cap` (compiled, weight 3) -- No keyword claims
a share of the decisions a 1.00:1 balanced split cannot support.

11. `noise_augmentation_confined_to_the_training_split` (compiled, weight 3) -- Background mixing
is declared, uses the vendored clips, and touches training audio only.

12. `graded_audio_and_checker_tree_unwritten` (compiled, weight 5, knockout) -- No guarded root is
written and the graded utterance archive is byte-identical after the run.

13. `replay_reproduces_the_label_vector_under_the_recorded_front_end` (compiled, weight 3) -- A
second run under the recorded seed reproduces the labels under the same front end.

14. `budget_declared_in_training_seconds_and_timed_utterances` (compiled, weight 3) -- The agent
declares its own budget in the terms this task spends it.

15. `latency_gated_keyword_accuracy_ramp` (compiled, weight 10, knockout) -- Top-1 keyword accuracy
clears the published bar through the ramp, with the per-utterance gate applied first.

Rejected routes, each bound to the control that measures it wrong:

- A pretrained audio encoder would shortcut the fusion design -- controlled by: no model-zoo host
is reachable at solve time; any backbone must be vendored.

- Fusing at the logit level, so the two branches never share a representation -- controlled by:
CTL-SPECONLY, which drops the waveform branch entirely and measures what the second branch was
actually worth when it only votes.

- Buying accuracy past the latency bound -- controlled by: CTL-OVERBUDGET, a model that exceeds
the per-utterance bound, which the gate must zero rather than score.

- Reading the gate as a mean or a percentile over the 3500 utterances -- controlled by the
checker's own recomputation of both statistics, which reports the substitution under its own name
rather than failing silently.

Deliverable manifest: /submission/preds_c8s7.npy (npy), one integer keyword index per graded
utterance, 3500 rows in the pinned roster order.

Control obligations recorded in seed/contract.yaml:

- `CTL-NOOP` targets the shipped starter verbatim; recorded score None.

- `CTL-SPECONLY` targets drop the waveform branch; recorded score None.

- `CTL-OVERBUDGET` targets a model that exceeds the latency bound; recorded score None.

<!-- AELLO-CANARY-BLOCK
  slot0: AELLO-CANARY-34477cdc8d45a4268664ce6af225566f
  slot1: AELLO-CANARY-be3116e30ccdfbd449346d06d270ee88
  slot2: AELLO-CANARY-aa4237d8ef3a4e17d01dff3c00bf57f1
  slot3: AELLO-CANARY-5994bef66a56e9cfd85895e8bafffa3f
-->
