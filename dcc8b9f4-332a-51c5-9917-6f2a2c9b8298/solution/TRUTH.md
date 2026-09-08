# C3-S1 -- what this task actually tests

GENERATED SECTION. DO NOT HAND-EDIT.

deliver a from-scratch tokenizer construction and encoding pipeline (build_tokenizer.py plus encode.py plus a declared stage manifest) that minimises held-out bits per byte of a frozen, hash-pinned language-model trainer, subject to three disclosed hard constraints: the vocabulary carries at most 65,536 entries, encode/decode is byte-identity on any input, and the whole graded region (encode plus the fixed S=6,000-step, 262,144-token-per-step training run plus evaluation) finishes inside a 15-minute wall on one H100. The trainer, optimizer, schedule, seed, architecture and step budget are frozen and hash-checked; the only thing the agent varies is how bytes become tokens. The graded artifact is written to /submission/c3s1.json and describes four things: the emitted vocabulary entries, the ordered stage manifest, the round-trip evidence over the five disclosed adversarial classes, and the byte accounting. The graded quantity is b, the frozen evaluator's total negative log-likelihood in bits over the encoded held-out stream divided by the total raw UTF-8 byte count of the same held-out text, taken as max(b_P1, b_P2) over two disjoint private shards with different language proportions.

## The single most important insight

The denominator is raw UTF-8 bytes, and that single fact decides what work is worth doing. A tokenizer that halves the token count changes the numerator's arithmetic and leaves the denominator exactly where it was, so compression on its own moves b by nothing; what moves b is how much of each byte the frozen trainer can predict inside a fixed 6,000-step budget. The second fact is that the numerator is only defined if encode and decode compose to identity. A pipeline that folds Arabic presentation forms or reorders Tamil combining marks still produces a stream, still trains, still reports a plausible number, and that number is bits per byte of a corpus that is not the graded corpus; the check for it is byte equality, not a diff of rendered text. Everything else in the slot -- the 65,536 ceiling, the fifteen-minute wall, the worse-of-two rule over shards with different language proportions -- exists to stop the three cheap ways of moving b that are not modelling: inflate the vocabulary, normalise the corpus, or fit the shard you can see.

## The ideal solve, step by step

1. **Establish byte identity before learning a single merge** Lay down the 256 byte-fallback entries first and prove decode(encode(x)) == x over the five disclosed adversarial classes -- Arabic presentation forms, tatweel, Tamil combining sequences, lone surrogates and NFKC-unstable sequences -- while the vocabulary is still trivial. If a normalisation stage is worth having, make it invertible: carry the fold as a token the decoder can undo rather than applying it to the bytes. Losslessness proven at 256 entries stays true at 32,768; losslessness assumed at 32,768 is discovered on the private shards, where it scores zero.

2. **Read the denominator before choosing an objective** b divides bits by raw UTF-8 bytes of the original text, so the tokenizer never appears in the denominator at all. Bytes per token is therefore not the objective and neither is sequence length; the quantity to improve is predicted bits per original byte at a fixed compute budget. Compute the accounting both ways once, over the same fixture set, and keep the version that divides by bytes -- the one that divides by tokens is the number that looks like progress and is not.

3. **Sample the vocabulary training set by script rather than by corpus frequency** The training shard is English-dominant and the graded b is the worse of two shards, one of them balanced 20/20/20/20/20 across English, Finnish, Arabic, Hindi and Tamil. A vocabulary learned on raw corpus frequency spends its budget on English word pieces and drops Tamil and Arabic to byte fallback, which is lossless and expensive. Temperature-sample the vocabulary corpus by script so the worse shard is the one the merges were actually fitted for.

4. **Spend the entry budget where a byte is expensive** Devanagari and Tamil cost three UTF-8 bytes per codepoint and Arabic costs two, so a merge that covers a common Tamil syllable retires three times the denominator a Latin merge of the same length retires. Rank candidate merges by bytes covered rather than by occurrence count, and the same 32,768 entries buy a different and better b.

5. **Price the vocabulary against the wall, not against the ceiling** The 65,536 ceiling is a limit, not a target. Embedding and output-softmax cost is paid at every one of the 6,000 steps over 262,144 tokens per step, so entries bought near the ceiling are paid for 6,000 times and the fifteen-minute wall arrives before the run does. Measure the wall at two or three vocabulary sizes and choose the largest one that finishes with margin, because a wall-clock overrun is an exact zero and not a slightly worse score.

6. **Write the manifest as the thing that will be re-executed** The manifest is not documentation of what happened; the verifier replays it from a fresh checkout and requires the same vocabulary digest, and it compares the declared stage order against the order the instrumented runner recorded. Anything the pipeline does that is not a declared stage -- a cached corpus sample, a hand-edited merge table, a step run once by hand -- makes the replay land on a different digest.

7. **Time the encode stage inside the graded region** Encoding is inside the wall. An agent that encodes once in its own experiments, caches the token stream and then times only training will declare a budget the verifier's re-execution does not reproduce, and the wall it measured is not the wall it is graded on. Declare encode_seconds and train_seconds separately so the two numbers can be checked against a single fifteen-minute total.

## Traps that catch agents that are not thinking carefully

- Lossy normalisation applied before vocabulary training: the pipeline runs clean, the shipped shard round-trips because it was normalised in place, and the private shards fail byte identity, scoring exactly 0.0.
- No byte fallback: held-out bytes absent from the training shard become unknown tokens whose information content the bits-per-byte denominator still charges, so the metric collapses while every log looks normal.
- Measuring the denominator after normalisation rather than before it: NFKC shortens an Arabic presentation form, the byte count drops, b improves, and nothing about the model changed. The declared raw_utf8_bytes is recomputed from the submitted text for exactly this reason.
- Reporting bits per token as bits per byte. The two agree only when the tokenizer is byte-level, which is the one configuration nobody submits, and the substitution flatters every vocabulary that compresses harder without predicting better.
- Declaring an entry count instead of emitting the entries. The budget is counted off the artifact, so a declared 32,768 beside a 70,000-entry list is vocab-budget-exceeded and not a rounding note.
- Vocabulary inflation to the 65,536 ceiling: embedding and softmax cost pushes the fixed 6,000-step run past the 15-minute wall, converting a good compression choice into an exact zero.
- Vocabulary trained on an unbalanced sample (English-dominant, as the starter does), so Tamil and Arabic fall back to bytes on shard P2 and the worse-of-two rule bites.
- Budget accounting error: encoding performed outside the timed region in the agent's own experiments but inside it under the verifier's re-execution, so measured wall exceeds the agent's expectation.
- A manifest that describes a run rather than reproducing it: a cached sample or a hand-edited merge table is not a declared stage, and the replay from a fresh checkout lands on a different vocabulary digest.
- Copying the public tokenizer or the public recipe and stalling at exactly the starter value, because the starter already is the public recipe.
- Lookup shortcut: 'train a byte-level BPE at 32k with the Hugging Face tokenizers library' is a public, memorised, one-line recipe. Mitigation: That recipe is shipped as the starter, so executing it scores exactly 0.10 and nothing more. The graded quantity is bits per byte at a fixed step budget under a hard wall on a five-script mixture, a frontier for which no public repository or paper publishes a near-optimal operating point, and full reward is normalised against a private reference measured on the grading host rather than against any published number.
- Smuggling pretrained weights or a pretrained tokenizer-plus-model pair into the submission so the graded run is not really from scratch. Mitigation: CHK-A5 re-derives step-0 parameters from the declared seed and the frozen model code and requires tensor equality with the checkpointed step-0 state; the verifier runs with network_mode none; the submission tree is capped at 20 MB; and the frozen trainer is hash-checked before execution.
- Fitting the shipped language proportions, which are visible, rather than the objective. Mitigation: Graded b is the worse of two disjoint private shards with deliberately different language proportions, and the proportions of the graded shards rotate on a private schedule inside a disclosed band.
- Classical hyperparameter search over the exposed flag space reaching the reference (the OPT3 structural bar, where SMAC, TPE and random search beat every frontier agent in the anchor). Mitigation: CTL-A-SEARCH runs 50-trial random and TPE search over {vocab_size, character_coverage, digit_split, sample_temperature} at authoring; the slot is refused unless that search stays below tau. The reference advantage lives in lossless normalisation design, script-aware sampling and encode-side packing, none of which is exposed as a flag.
- Self-reporting a favourable bits-per-byte number in the attempt log. Mitigation: The reward is computed only from the verifier's own re-execution and its own evaluator; the log is read separately and a divergence is recorded rather than believed.

---

## Contract record (annex; not part of the narrative body)

Retained under FORGE.md item 10e and Phase 2 item 7, which require this file to carry the ordered path through instruction.md with each satisfied checker identifier, and each rejected route bound to a measured known-wrong control. standards/truth-md-authoring-v1.md section 3 admits no fifth section, so this annex is a recorded deviation rather than an omission.

Ordered path through instruction.md, with the checker identifier each step satisfies:


1. `bits_per_byte_divides_by_bytes_not_tokens` (compiled, weight 3) -- The declared accounting divides negative log-likelihood by the raw byte total and not by the token total.

2. `byte_alphabet_fully_covered` (compiled, weight 5, knockout) -- All 256 byte values stay representable, so no held-out byte can arrive as an unknown token.

3. `byte_denominator_is_raw_not_normalised` (compiled, weight 5, knockout) -- Each declared byte count is the raw UTF-8 length of the text as it arrived, before any normalisation the pipeline applies.

4. `encode_and_train_seconds_declared` (compiled, weight 3) -- The agent says where its wall went, split between encoding the corpus and the fixed-step training run.

5. `frozen_trainer_digest_unchanged` (compiled, weight 5, knockout) -- The graded run leaves the frozen trainer, the grading tree and the held-out tree byte-identical.

6. `graded_region_within_wall` (compiled, weight 3) -- The host-measured graded region fits inside the fifteen-minute wall.

7. `held_out_bits_per_byte_ramp` (compiled, weight 10, knockout) -- Held-out bits per byte, taken as the worse of the two private shards, clears the published bar through the ramp.

8. `manifest_replay_reproduces_vocabulary_digest` (compiled, weight 3) -- Re-executing the declared manifest from a fresh checkout rebuilds the same vocabulary artifact.

9. `no_per_shard_breakdown_in_submission` (compiled, weight 5, knockout) -- The submission carries no per-shard bits-per-byte breakdown, no private shard name and no grading constant.

10. `roundtrip_covers_adversarial_classes` (compiled, weight 3) -- The round-trip evidence spans every disclosed adversarial class rather than ASCII alone.

11. `roundtrip_decodes_to_input_bytes` (compiled, weight 5, knockout) -- decode(encode(x)) is byte identity on every fixture the submission carries.

12. `stage_manifest_order_matches_execution` (compiled, weight 5, knockout) -- The stage order the manifest declares is the stage order the instrumented runner actually executed.

13. `vocabulary_artifact_emitted` (compiled, weight 5, knockout) -- The pipeline emits its vocabulary, its stage manifest, its round-trip evidence and its byte accounting at the declared submission path.

14. `vocabulary_entry_count_read_from_entries` (compiled, weight 5, knockout) -- The 65,536-entry budget is counted off the emitted entry list, never off the number the agent declares beside it.


Rejected routes, each bound to the control that measures it wrong:

- Lookup shortcut: 'train a byte-level BPE at 32k with the Hugging Face tokenizers library' is a public, memorised, one-line recipe. -- controlled by: That recipe is shipped as the starter, so executing it scores exactly 0.10 and nothing more. The graded quantity is bits per byte at a fixed step budget under a hard wall on a five-script mixture, a frontier for which no public repository or paper publishes a near-optimal operating point, and full reward is normalised against a private reference measured on the grading host rather than against any published number.
- Smuggling pretrained weights or a pretrained tokenizer-plus-model pair into the submission so the graded run is not really from scratch. -- controlled by: CHK-A5 re-derives step-0 parameters from the declared seed and the frozen model code and requires tensor equality with the checkpointed step-0 state; the verifier runs with network_mode none; the submission tree is capped at 20 MB; and the frozen trainer is hash-checked before execution.
- Fitting the shipped language proportions, which are visible, rather than the objective. -- controlled by: Graded b is the worse of two disjoint private shards with deliberately different language proportions, and the proportions of the graded shards rotate on a private schedule inside a disclosed band.
- Classical hyperparameter search over the exposed flag space reaching the reference (the OPT3 structural bar, where SMAC, TPE and random search beat every frontier agent in the anchor). -- controlled by: CTL-A-SEARCH runs 50-trial random and TPE search over {vocab_size, character_coverage, digit_split, sample_temperature} at authoring; the slot is refused unless that search stays below tau. The reference advantage lives in lossless normalisation design, script-aware sampling and encode-side packing, none of which is exposed as a flag.
- Self-reporting a favourable bits-per-byte number in the attempt log. -- controlled by: The reward is computed only from the verifier's own re-execution and its own evaluator; the log is read separately and a divergence is recorded rather than believed.

Deliverable manifest: /submission/c3s1.json (json), an object carrying vocabulary (entries, declared_entry_count, artifact_sha256), an ordered stage_manifest, roundtrip_fixtures over the five disclosed adversarial classes, and bpb_accounting.

Control obligations recorded in seed/contract.yaml:

- `CTL-A-NOOP-EMPTY` targets Submitting nothing / leaving the environment unchanged, i.e. collecting reward for the shipped state.; recorded score 0.
- `CTL-A-NOOP-STARTER` targets Resubmitting the shipped starter byte-for-byte (the public 32k byte-level BPE with NFKC and English-only sampling), i.e. functional correctness with zero movement on b.; recorded score 0.1.
- `CTL-A-STALE` targets Carrying over attempt k's tokenizer and manifest after the private shard language proportions were re-drawn; b = 1.166 in the order-consistent variant. The order-divergent variant fires CHK-A6 and scores 0.000; the 0.097 reported here is the conservative maximum the class can reach.; recorded score 0.097.
- `CTL-A-DECOY-GPT2` targets Reusing the real MIT-licensed GPT-2 50,257-entry vocabulary and merges shipped in environment/reference/, the majority-endorsed public artifact for this problem shape; its bytes-per-token on Arabic, Devanagari and Tamil starves the fixed 6,000-step budget, b = 1.301.; recorded score 0.075.
- `CTL-A-LISTALL` targets Training the vocabulary on the raw unsampled boilerplate-dominated stream. At the 65,536 ceiling it trips CHK-A7 at the wall and scores 0.000; the wall-relieved 32,768 variant reaches b = 1.219, and that 0.088 is reported as the conservative maximum of the class.; recorded score 0.088.
- `CTL-A-OBVIOUSWRONG` targets A 256-symbol character-level vocabulary, the by-construction-wrong deliverable; b = b_rand = 1.782 clamps the informative ramp to zero.; recorded score 0.
- `CTL-A-SEARCH` targets Slot-specific, the OPT3 structural bar: 50-trial random plus TPE search over the exposed flag space {vocab_size, character_coverage, digit_split, sample_temperature} with everything else at starter values; best-of-50 reaches b = 1.126. The reference advantage lives in identity normalisation, script-aware temperature sampling and encode-side packing, none of which is a flag. REFUSAL CONDITION: if the authoring measurement puts this control at or above tau = 0.30 the slot is refused at authoring rather than the threshold moved.; recorded score 0.203.
- `CTL-A-SILENT-NFKC` targets Slot-specific, the AR6 silent-failure route this slot exists to instrument: the starter's NFKC variant that folds Arabic presentation forms and reorders Tamil combining marks. It runs to completion, exits 0 and logs nothing, and CHK-A1 zeroes it with reason roundtrip-not-identity.; recorded score 0.

<!-- AELLO-CANARY-BLOCK
  slot0: AELLO-CANARY-f6e63dc5578069a7c003d86e035cf00e
  slot1: AELLO-CANARY-932aca95af39a2245aad33557f82df8b
  slot2: AELLO-CANARY-4312fead687026fb0eadc784a8254489
  slot3: AELLO-CANARY-fae240e6e015c5a73d05714e4672b2dd
-->
