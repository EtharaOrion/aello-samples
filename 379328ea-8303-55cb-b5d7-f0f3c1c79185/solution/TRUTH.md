# C3-S2 -- what this task actually tests

GENERATED SECTION. DO NOT HAND-EDIT.

from a fixed 12 GB raw web-text slice of allenai/c4 en.noclean, build a curation pipeline from scratch that emits a training corpus and a machine-readable recipe manifest, such that the frozen, hash-pinned trainer consuming the first N tokens of your corpus IN THE ORDER YOU EMIT THEM minimises held-out bits per byte on two private evaluation shards drawn from sources other than the training slice. The tokenizer, trainer, optimizer, schedule, seed, architecture and 6,000-step budget are frozen and hash-checked, so the graded object is the CORPUS and the RECIPE, never a model. Selection, deduplication, filtering, decontamination placement and emission order are the entire design space; N is drawn from a private schedule inside the disclosed band [560M, 600M] tokens, so the recipe must be robust to where the single pass stops. The graded artifact is written to /submission/c3s2.json and carries the emitted record ids and their byte digests in emission order, the ordered recipe_manifest, the corpus_sha256 that manifest must re-execute to, one (source_id, offset, length) span per emitted record, and the consumed token count. The graded quantity is b = max(b_P1, b_P2), held-out bits per byte from the frozen evaluator taken as the WORSE of the two private shards, P1 encyclopedic and P2 scientific.

## The single most important insight

Two things carry this slot and neither is a threshold. FIRST: where decontamination sits in the operator order. Applied to raw bytes it reports zero fingerprint hits and leaves normalised 13-grams of the held-out shards sitting in the emitted corpus; applied after byte normalisation it removes them. Both pipelines exit 0 with healthy logs, both produce corpora that agree on document count, mean length, token count, duplicate rate and language distribution, and nothing in the intermediate state separates them -- the difference surfaces only as an exact 0.0 with reason eval-leak, after a full build-and-train cycle. SECOND: b is the WORSE of two shards, so every bit bought on the encyclopedic shard at the scientific shard's expense is worth exactly nothing, and a corpus tuned to look like one held-out source is a mutually exclusive wrong deliverable rather than a partial win. Everything else -- thresholds, badwords strength, minhash parameters -- is the published flag space that a 50-trial TPE search already exhausts.

## The ideal solve, step by step

1. **Byte normalisation and encoding repair, first in the order** Normalisation runs before anything that matches text, because every downstream matcher -- exact dedup, the fingerprint filter, near-dup shingling -- compares strings, and a matcher fed unrepaired mojibake compares the wrong strings. Its separable worth is measured on the Phase 2 ladder over the full lever power set, never asserted here.

2. **Exact duplicate removal at document granularity** Whole-document duplicates are removed while the corpus is still the full slice, so the survivor set is chosen by content rather than by where the token budget happens to cut. Its separable worth is measured on the Phase 2 ladder, never asserted here.

3. **Held-out fingerprint decontamination applied to the NORMALISED text** The one-way sorted hash set of normalised 13-grams is applied after step 1 and not before it. This placement is load-bearing and is the single most common wrong choice on this slot; CTL-B-PLACEMENT is the inverted order and it scores exactly 0.0. Its separable worth is measured on the Phase 2 ladder, never asserted here.

4. **Near-duplicate removal at paragraph granularity with measured parameters** Granularity and threshold are measured on this slice rather than copied from the published MinHash settings, because the published settings were fitted to a different corpus and a different budget. Its separable worth is measured on the Phase 2 ladder, never asserted here.

5. **Length-and-repetition filter tuned on the raw slice itself** The filter is calibrated against the distribution actually present in the 12 GB noclean shard, and calibrated to leave more than the largest N in the band, since over-filtering makes the single pass underfill. Its separable worth is measured on the Phase 2 ladder, never asserted here.

6. **An emission curriculum that interleaves rather than sorts** The frozen trainer sees the corpus exactly once, in the emitted order, so ordering is a lever and not a formatting detail; interleaving the surviving high-information documents gives the single pass a balanced stream from step one rather than a sorted one. Its separable worth is measured on the Phase 2 ladder, never asserted here.

7. **Truncation-safe emission across the whole [560M, 600M] band** N is drawn privately and never announced, so the emitted prefix must be a usable corpus at every stopping point in the band rather than only at the one the last attempt happened to see. Its separable worth is measured on the Phase 2 ladder, never asserted here.

8. **Record the manifest as the executed order, not as an after-the-fact description** The verifier re-executes recipe_manifest from a fresh checkout and requires the corpus digest to reproduce, so every stage must be runnable and deterministic: keyed order-independent hashing, PYTHONHASHSEED pinned, no hash-order-dependent iteration on the graded path, near-dup ties broken by (source_id, offset).

9. **Carry the span triples and the two wall-clock regions with the submission** Every emitted record ships its (source_id, offset, length) triple so the coverage proof and the 2 percent byte-level sample can verify that nothing was rewritten, and the agent-authored declaration accounts for the build and train regions the harness measures against the 1200-second and 900-second walls.

## Traps that catch agents that are not thinking carefully

- Order collapse: deduplication applied after truncation to the token budget, so the surviving duplicates were already fixed by the truncation point and dedup does nothing measurable.
- Decontamination applied before normalisation, so normalised held-out 13-grams survive in the emitted corpus and CHK-B5 scores the attempt exactly 0.0 while the agent's own pre-normalisation check reported zero hits.
- Recipe manifest drift: the emitted corpus reflects an interactive pipeline the manifest no longer describes, so verifier re-execution produces a different digest and scores 0.0 with reason recipe-corpus-divergence.
- Over-filtering: a high-quality threshold that leaves fewer than the required token budget, so the single pass repeats data or underfills, and bits per byte degrades below the starter.
- Emission order ignored: documents emitted in raw shard order, leaving the entire curriculum dimension of the design space unexplored while the agent spends the attempt budget on thresholds.
- Reproducing the published C4 recipe and stalling at exactly 0.10, because the published recipe is the starter.
- Lookup shortcut: the C4 cleaning recipe, the Gopher quality rules, MinHash deduplication parameters and the FineWeb filter cascade are all published, memorised and directly applicable to this corpus. Mitigation: The published heuristic pipeline is shipped as the starter and scores exactly 0.10, so retrieving and pasting the public recipe is worth one tenth of the range and no more. The graded quantity is worse-of-two held-out bits per byte at a frozen trainer, a fixed token budget and a private truncation point, for which no published recipe exists; and the emission-order dimension, which carries a measured share of the reference delta, appears in no public curation recipe at all.
- Fitting the evaluation by seeding held-out-like content into the training corpus. Mitigation: CHK-B5 zeroes any corpus containing a normalised 13-gram present in the held-out fingerprint set, and CHK-B10 requires every emitted byte to trace to an exact (source_id, offset, length) span of a provided raw document, so paraphrase-based evasion is structurally impossible: the agent may select, order and truncate, but never rewrite.
- The one-way fingerprint file lets the agent identify which held-out documents exist and target them. Mitigation: It is a hash set, not text; identification only permits inclusion, which CHK-B5 zeroes, and the span-containment invariant blocks reconstruction. Shard P2 is drawn from a source whose distribution differs from P1 so a P1-targeted strategy loses on the worse-of-two rule.
- Classical search over the exposed thresholds reaching the reference, the OPT3 structural bar under which SMAC, TPE and random search beat every frontier agent. Mitigation: CTL-B-SEARCH runs random and TPE search over the five exposed flags at authoring and the slot is refused unless it stays below tau. The reference advantage lives in operator order, decontamination placement and emission curriculum, which are structural choices with no flag representation.
- Self-reported metrics in the attempt log. Mitigation: The reward comes only from the verifier's own re-execution of the recipe and its own evaluator; the log is read separately and divergence is recorded.

---

## Contract record (annex; not part of the narrative body)

Retained under FORGE.md item 10e and Phase 2 item 7, which require this file to carry the ordered path through instruction.md with each satisfied checker identifier, and each rejected route bound to a measured known-wrong control. standards/truth-md-authoring-v1.md section 3 admits no fifth section, so this annex is a recorded deviation rather than an omission.

Ordered path through instruction.md, with the checker identifier each step satisfies:


1. `consumed_prefix_within_drawn_budget` (compiled, weight 5) -- The consumed token prefix fits the privately drawn budget N.

2. `curation_wallclock_within_dual_walls` (compiled, weight 3) -- Both host-measured wall-clock regions sit inside their own ceiling.

3. `decontamination_after_normalisation` (compiled, weight 5) -- Held-out fingerprint decontamination is applied to normalised text, not to raw bytes.

4. `dedup_before_token_truncation` (compiled, weight 3) -- Deduplication runs while there is still a corpus to deduplicate, that is, before truncation to the token budget fixes the survivor set.

5. `emission_order_is_consumption_order` (compiled, weight 3) -- The order the corpus is emitted in is the order the frozen trainer consumes it in, so emission order is part of the graded artifact.

6. `emitted_bytes_are_raw_document_spans` (compiled, weight 5, knockout) -- Every emitted byte traces to an exact span of a provided raw document, so the agent may select, order and truncate but never rewrite.

7. `executed_operator_sequence_declared` (compiled, weight 3) -- The agent declares the operator sequence it actually ran and the budget it accounts for, and that declaration matches the manifest it submitted.

8. `frozen_trainer_and_heldout_digests_unchanged` (compiled, weight 5, knockout) -- The graded run leaves the frozen trainer, the held-out tree, the fingerprint file and the grading tree byte-identical.

9. `heldout_13gram_containment_zero` (compiled, weight 5, knockout) -- No normalised 13-gram of the emitted corpus is present in the held-out fingerprint set.

10. `manifest_operators_carry_parameters` (compiled, weight 3) -- The recipe manifest is a machine-readable ordered operator list rather than a prose description of a pipeline.

11. `manifest_reexecutes_to_emitted_digest` (compiled, weight 5, knockout) -- Re-executing the declared manifest over the pinned raw slice reproduces the emitted corpus byte for byte.

12. `recipe_manifest_and_corpus_digest_present` (compiled, weight 5, knockout) -- The graded artifact lands carrying both halves of the deliverable: the recipe and the digest of the corpus that recipe emitted.

13. `worse_shard_bits_per_byte_ramp` (compiled, weight 10, knockout) -- The worse-shard bits per byte clears the published bar and is mapped through the descending ramp.

14. `worse_shard_is_the_graded_value` (compiled, weight 3) -- The graded scalar is the worse of the two private shards, not their mean and not the better one.


Rejected routes, each bound to the control that measures it wrong:

- Lookup shortcut: the C4 cleaning recipe, the Gopher quality rules, MinHash deduplication parameters and the FineWeb filter cascade are all published, memorised and directly applicable to this corpus. -- controlled by: The published heuristic pipeline is shipped as the starter and scores exactly 0.10, so retrieving and pasting the public recipe is worth one tenth of the range and no more. The graded quantity is worse-of-two held-out bits per byte at a frozen trainer, a fixed token budget and a private truncation point, for which no published recipe exists; and the emission-order dimension, which carries a measured share of the reference delta, appears in no public curation recipe at all.
- Fitting the evaluation by seeding held-out-like content into the training corpus. -- controlled by: CHK-B5 zeroes any corpus containing a normalised 13-gram present in the held-out fingerprint set, and CHK-B10 requires every emitted byte to trace to an exact (source_id, offset, length) span of a provided raw document, so paraphrase-based evasion is structurally impossible: the agent may select, order and truncate, but never rewrite.
- The one-way fingerprint file lets the agent identify which held-out documents exist and target them. -- controlled by: It is a hash set, not text; identification only permits inclusion, which CHK-B5 zeroes, and the span-containment invariant blocks reconstruction. Shard P2 is drawn from a source whose distribution differs from P1 so a P1-targeted strategy loses on the worse-of-two rule.
- Classical search over the exposed thresholds reaching the reference, the OPT3 structural bar under which SMAC, TPE and random search beat every frontier agent. -- controlled by: CTL-B-SEARCH runs random and TPE search over the five exposed flags at authoring and the slot is refused unless it stays below tau. The reference advantage lives in operator order, decontamination placement and emission curriculum, which are structural choices with no flag representation.
- Self-reported metrics in the attempt log. -- controlled by: The reward comes only from the verifier's own re-execution of the recipe and its own evaluator; the log is read separately and divergence is recorded.

Deliverable manifest: /submission/c3s2.json (json), an object with an ids list and a matching outputs list.

Control obligations recorded in seed/contract.yaml:

- `CTL-B-NOOP-EMPTY` targets Submitting nothing / leaving the environment unchanged.; recorded score 0.
- `CTL-B-NOOP-STARTER` targets Resubmitting the shipped starter byte-for-byte (published C4 heuristic pipeline), i.e. a legal, lossless, in-budget run that moves b not at all.; recorded score 0.1.
- `CTL-B-STALE` targets Carrying attempt k's recipe manifest and emitted corpus forward after N was re-drawn inside the [560M, 600M] band; b = 1.342 in the order-consistent variant. The order-divergent variant fires CHK-B1 and scores 0.000; 0.092 is the conservative maximum of the class.; recorded score 0.092.
- `CTL-B-DECOY-C4DOC` targets Following the shipped real published C4 heuristic description and badwords list faithfully, i.e. trusting the majority-endorsed public artifact; it reproduces the starter, b = 1.329.; recorded score 0.099.
- `CTL-B-LISTALL` targets Emitting the entire raw slice unfiltered and truncating at N. The primary variant trips CHK-B5 on Wikipedia text present in the crawl and scores 0.000; the leak-relieved variant (decontamination only, no other curation) reaches b = 1.398, and that 0.062 is reported as the conservative maximum.; recorded score 0.062.
- `CTL-B-OBVIOUSWRONG` targets A uniformly random N-token sample of the raw slice, the by-construction-wrong deliverable; b = b_rand = 1.512 clamps the informative ramp to zero.; recorded score 0.
- `CTL-B-SEARCH` targets Slot-specific: 50-trial random plus TPE search over the exposed operator parameters {dedup n-gram width, near-dup threshold, minimum document length, repetition threshold, badwords strength} with the published operator ORDER held fixed; best-of-50 reaches b = 1.313. The reference advantage lives in operator order, decontamination placement and emission curriculum, none of which is a scalar flag. REFUSAL CONDITION: if this control measures at or above tau = 0.30 at authoring, the slot is refused rather than the threshold moved.; recorded score 0.209.
- `CTL-B-PLACEMENT` targets Slot-specific, the single most common wrong choice and the AR7 ambiguous-intermediate-state route: decontamination placed BEFORE byte normalisation, so normalised held-out 13-grams survive the filter. The pipeline exits 0 with healthy logs and CHK-B5 zeroes it with reason eval-leak.; recorded score 0.

<!-- AELLO-CANARY-BLOCK
  slot0: AELLO-CANARY-6b43a83e0d9ceb85305f340ced6b5f99
  slot1: AELLO-CANARY-9383ea62e6f2331240c3312fe7e79f81
  slot2: AELLO-CANARY-6541eb481d5bfc618e97be0c9968f0a8
  slot3: AELLO-CANARY-825ff8b67f70266cb9b5c33c45596132
-->
