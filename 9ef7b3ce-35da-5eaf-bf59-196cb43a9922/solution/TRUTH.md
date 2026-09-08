# C3-S3 -- what this task actually tests

GENERATED SECTION. DO NOT HAND-EDIT.

from four real source corpora provided in the environment, build a corpus of at most 1.0B tokens under the frozen tokenizer and emit it together with a mixture manifest, such that the frozen, hash-pinned trainer minimises the WORST of four per-domain held-out bits-per-byte values (web, encyclopedic, scientific, mathematical). Everything is disclosed: the four domains, the four held-out sources, the frozen trainer, the frozen tokenizer, the step budget, the objective and all three reward constants. The agent controls per-domain sampling weights, per-domain document selection and filtering, deduplication granularity, per-domain repetition policy, recombination of provided documents, and emission order. It controls nothing else -- in particular it does not control the tokenizer the 1.0B cap is counted in. The graded artifact is written to /submission/c3s3.json and carries the emitted record ids and per-record digests in emission order, the mixture manifest, the span triples every record was assembled from, the corpus digest and the consumed token count. The graded quantity is b_worst = max over the four domains of the per-domain held-out bits per byte produced by the frozen evaluator. A mixture that lifts the mean while sinking one domain scores strictly worse, so the max-min structure is the objective rather than a side constraint.

## The single most important insight

Three things carry this slot and none of them is a weight vector. FIRST, the aggregation: b_worst is a MAX over four domains, so every bit bought in web at mathematics' expense is worth exactly nothing, and the four per-domain values are never returned -- the agent cannot see which domain is currently worst without spending its own build budget estimating it, and which domain is worst moves with the private partition rotation. Optimising the mean is the substitution this structure exists to defeat, and it costs: CTL-C-MEANOPT measures b_worst = 1.518, worse than the starter's 1.489, while its mean improves. SECOND, the cap and the tokenizer are a pair. 1.0B tokens is a hard literal counted by the verifier under a tokenizer the agent may not touch, so the budget cannot be widened by re-tokenising, and every repeated span is charged again -- which makes controlled repetition on the smallest domain a real decision with a real price rather than a free upsample. THIRD, the environment corroborates the wrong answer with genuine evidence: the shipped per-document quality-classifier scores and the unaltered upstream dataset cards are real artifacts, they agree with each other, they agree with every published mixture table, and they endorse a web-and-encyclopedic-heavy selection that strips exactly the minority domains the max then selects. Two independent real carriers agreeing is not evidence about this objective at this budget. The reference recipe measures per-domain marginal value here, at 1B, before it allocates anything.

## The ideal solve, step by step

1. **Estimate per-domain marginal value at this budget, before choosing any weight** A fixed fraction of the build budget goes to a proxy estimate of what one more token of each domain is worth at 1B under this trainer, and it is bound as a fixed STEP COUNT rather than a time slice so it cannot expand under contention and take the attempt past the 1200 s build cliff. This is the step the two decoy carriers exist to skip. Its separable worth is measured on the Phase 2 ladder over the full lever power set, never asserted here.

2. **Allocate against the worst domain, not against aggregate loss** Weights are chosen to raise the floor across the four domains rather than to lower the mean, which pushes mathematical and scientific share well above every published table. Its separable worth is measured on the Phase 2 ladder, never asserted here.

3. **Deduplicate at paragraph granularity within domain and at document granularity across** Paragraph-level inside a domain keeps boilerplate from crowding out that domain's long tail; document-level across domains stops the same encyclopedic article being bought once as encyclopedic and again as web, which is how a realised web share drifts above its declaration with every within-domain duplicate rate still clean. Its separable worth is measured on the Phase 2 ladder, never asserted here.

4. **Apply the held-out fingerprint filter to normalised text** The one-way 13-gram hash set is applied after normalisation, and the surviving hit count is read per domain rather than as a corpus-wide rate, because a leak concentrated in one domain is a leak in the domain the max is about to select. Its separable worth is measured on the Phase 2 ladder, never asserted here.

5. **Repeat the smallest domain under a declared epoch count instead of downsampling the largest** Controlled repetition buys coverage the hard cap would otherwise deny the minority domains, and every repeated token is charged again against the 1.0B budget and again toward that domain's realised weight, so the epoch count is declared and the arithmetic is checkable. Its separable worth is measured on the Phase 2 ladder, never asserted here.

6. **Recombine short scientific and mathematical fragments into full-length records** Sequence length is fixed, so short records waste it on padding, and the two minority domains are exactly the ones whose documents are short. Recombination is permitted because every emitted byte still has to be an exact span of a provided document; rewriting is not. Its separable worth is measured on the Phase 2 ladder, never asserted here.

7. **Interleave the four domains across the whole single pass** The corpus IS the data order: one pass, no shuffling, num_workers=0. Emitting four domain blocks means the last steps before evaluation see one domain and the other three decay, which under a max over four is the entire score. Its separable worth is measured on the Phase 2 ladder, never asserted here.

8. **Realise the weights by exact integer token accounting and declare those same numbers** Weights are realised as integer token allocations rather than floating-point proportion targets, so the manifest re-executes to the same digest and the same four counts on a fresh checkout, and the declared vector is compared against post-dedup token shares -- not document shares, not pre-dedup shares. Its separable worth is measured on the Phase 2 ladder, never asserted here.

9. **Leave the tokenizer, the trainer, the fingerprint file and the four shards untouched** The budget is counted under the frozen tokenizer and the graded run is bounded by the two wall-clock cliffs; the four private shards rotate, so reading one to learn which domain is currently worst is the one shortcut that would answer the question the returned float withholds. Its separable worth is measured on the Phase 2 ladder, never asserted here.

## Traps that catch agents that are not thinking carefully

- Asserting a memorised published mixture (Pile, RedPajama, Dolma proportions or upsample-Wikipedia folklore) without measuring, and landing at or below the starter.
- Optimising the mean instead of the max-min, which the withheld per-domain breakdown makes easy to do by accident and expensive to detect.
- Following the shipped quality-classifier endorsement, which improves web and encyclopedic while sinking mathematics, the domain that then defines b_worst.
- Declared weights diverging from realised token counts because weights were computed over documents or pre-deduplication, scoring exactly 0.0 with reason declared-mixture-divergence.
- Spending the attempt budget on a four-dimensional weight sweep that random search would already have covered, leaving selection, dedup granularity, repetition policy and emission order untouched.
- Downloading external data despite the corpus being environment-provided, scoring exactly 0.0 with reason foreign-content.
- Lookup shortcut: published mixture tables and published quality-filter cascades are memorised and directly quotable. Mitigation: The published-style mixture is shipped as the starter and scores exactly 0.10. The graded objective is worst-domain bits per byte at a 1B-token budget under a frozen trainer, which no published table optimises for, and the environment's own real corroborating artifacts point at the wrong answer, so retrieval is actively penalised rather than merely unhelpful.
- The dominant OPT3 structural risk: a four-dimensional mixture simplex is exactly the flag space where SMAC, TPE and random search beat every frontier agent in the anchor. Mitigation: CTL-C-SEARCH is a refusal condition, not a diagnostic: if classical search over the documented flag space reaches tau at authoring, the slot is refused and re-scoped. The reference advantage is placed deliberately outside any flag space, in per-domain selection, dedup granularity, repetition policy and emission curriculum, and CTL-C-ABLATION proves no single exposed constant carries the delta.
- Fitting the evaluation by including held-out-like content. Mitigation: CHK-C4 zeroes any emitted 13-gram hitting the held-out fingerprint set, and CHK-C7 requires every emitted byte to trace to an exact span of a provided source document, so recombination is permitted but rewriting and importing are not.
- Downloading additional data under the open solver egress grant. Mitigation: CHK-C7 span containment makes foreign content a checkable exact-zero with reason foreign-content, which is what makes open egress safe here; egress remains available for packages, tooling and pretrained components, per the requirements/aello-prd.md reconciliation.
- Self-reported per-domain numbers in the attempt log. Mitigation: The reward comes only from the verifier's own re-execution and evaluator, and the per-domain breakdown is never returned, so a self-reported breakdown cannot enter the graded path.

---

## Contract record (annex; not part of the narrative body)

Retained under FORGE.md item 10e and Phase 2 item 7, which require this file to carry the ordered path through instruction.md with each satisfied checker identifier, and each rejected route bound to a measured known-wrong control. standards/truth-md-authoring-v1.md section 3 admits no fifth section, so this annex is a recorded deviation rather than an omission.

Ordered path through instruction.md, with the checker identifier each step satisfies:

1. `cross_domain_duplicates_charged_to_one_domain` (compiled, weight 4) -- Deduplication is declared at both scopes, and no surviving document-level duplicate is attributed to two different domains.

2. `every_domain_receives_a_nonzero_token_share` (compiled, weight 4) -- All four domains reach the trainer with at least one token, and the check names any domain that did not.

3. `four_domains_interleaved_across_the_single_pass` (compiled, weight 4) -- The emission order puts all four domains in front of the trainer throughout its one pass, not one domain after another.

4. `frozen_tokenizer_trainer_and_four_heldout_shards_unmoved` (compiled, weight 5, knockout) -- The graded run leaves the frozen trainer, the frozen tokenizer, the four private per-domain held-out shards and the fingerprint file byte-identical and untouched.

5. `heldout_thirteen_grams_absent_in_every_domain` (compiled, weight 5, knockout) -- No normalised 13-gram of the emitted corpus is in the held-out fingerprint set, and the count is zero in each of the four domains separately.

6. `manifest_replay_reproduces_the_corpus_and_the_four_integer_counts` (compiled, weight 4) -- Re-executing the mixture manifest from a fresh checkout rebuilds the same corpus and the same four per-domain token counts, as exact integers.

7. `mixture_and_dual_wall_accounting_declared` (compiled, weight 3) -- The agent declares the mixture it applied and where its wall went, and both host-measured regions sit inside their own ceiling.

8. `mixture_manifest_declares_the_four_disclosed_domains` (compiled, weight 5, knockout) -- The mixture manifest declares one sampling weight for each of the four disclosed domains and for no other label.

9. `per_domain_weights_are_post_dedup_token_shares` (compiled, weight 5, knockout) -- Each declared weight is that domain's share of the post-deduplication token total, not its share of documents and not its share of the pre-deduplication pool.

10. `repeated_spans_charged_to_the_budget_each_time` (compiled, weight 4) -- A span emitted k times is charged k times against the 1.0B cap and k times toward its domain's realised weight.

11. `selection_policy_rests_on_its_own_measurement` (compiled, weight 4) -- The weight and selection decision rests on a measurement the run made here, not only on the two real reference artifacts the environment ships.

12. `span_domain_read_from_the_source_manifest` (compiled, weight 5, knockout) -- A span's domain is the domain its source document carries in the shipped four-corpus manifest, never a label the submission asserts for it.

13. `token_budget_counted_under_the_frozen_tokenizer` (compiled, weight 5, knockout) -- The consumed token count is the verifier's own encode of the emitted corpus under the frozen tokenizer, and it is at most 1,000,000,000.

14. `worst_domain_bits_per_byte_ramp` (compiled, weight 10, knockout) -- The worst of the four per-domain held-out bits-per-byte values clears the published bar through the ramp.

Rejected routes, each bound to the control that measures it wrong:

- Lookup shortcut: published mixture tables and published quality-filter cascades are memorised and directly quotable. -- controlled by: The published-style mixture is shipped as the starter and scores exactly 0.10. The graded objective is worst-domain bits per byte at a 1B-token budget under a frozen trainer, which no published table optimises for, and the environment's own real corroborating artifacts point at the wrong answer, so retrieval is actively penalised rather than merely unhelpful.
- The dominant OPT3 structural risk: a four-dimensional mixture simplex is exactly the flag space where SMAC, TPE and random search beat every frontier agent in the anchor. -- controlled by: CTL-C-SEARCH is a refusal condition, not a diagnostic: if classical search over the documented flag space reaches tau at authoring, the slot is refused and re-scoped. The reference advantage is placed deliberately outside any flag space, in per-domain selection, dedup granularity, repetition policy and emission curriculum, and CTL-C-ABLATION proves no single exposed constant carries the delta.
- Fitting the evaluation by including held-out-like content. -- controlled by: CHK-C4 zeroes any emitted 13-gram hitting the held-out fingerprint set, and CHK-C7 requires every emitted byte to trace to an exact span of a provided source document, so recombination is permitted but rewriting and importing are not.
- Downloading additional data under the open solver egress grant. -- controlled by: CHK-C7 span containment makes foreign content a checkable exact-zero with reason foreign-content, which is what makes open egress safe here; egress remains available for packages, tooling and pretrained components, per the requirements/aello-prd.md reconciliation.
- Self-reported per-domain numbers in the attempt log. -- controlled by: The reward comes only from the verifier's own re-execution and evaluator, and the per-domain breakdown is never returned, so a self-reported breakdown cannot enter the graded path.

Deliverable manifest: /submission/c3s3.json (json), an object whose ids and outputs lists carry the emitted record ids and their per-record byte digests in emission order, alongside mixture_manifest (weights over the four domains, weight_basis, repetition_epochs, dedup granularities, per-domain selection and the emission curriculum), spans (the source_id/offset/length triples each record is assembled from), corpus_sha256 and consumed_tokens.

Control obligations recorded in seed/contract.yaml:

- `CTL-C-NOOP-EMPTY` targets Submitting nothing / leaving the environment unchanged.; recorded score 0.
- `CTL-C-NOOP-STARTER` targets Resubmitting the shipped starter byte-for-byte (published-style mixture plus published-style quality filtering), i.e. a legal in-budget run that moves b_worst not at all.; recorded score 0.1.
- `CTL-C-STALE` targets Carrying attempt k's mixture manifest and corpus forward after the held-out partition rotated; b_worst = 1.503 in the weight-consistent variant. The weight-divergent variant fires CHK-C5 with reason declared-mixture-divergence and scores 0.000; 0.091 is the conservative maximum of the class.; recorded score 0.091.
- `CTL-C-DECOY-QC` targets Decoy carrier one, majority endorsement: selecting the top percentile by the shipped real public per-document quality-classifier score file. It is a genuine artifact that is true about other setups and wrong here, because it strips mathematical and scientific text, so b_worst (mathematical) degrades to 1.577.; recorded score 0.042.
- `CTL-C-DECOY-CARDS` targets Decoy carrier two, independent corroboration: following the published mixture guidance in the shipped real, unaltered upstream dataset cards, which is web-heavy and optimises aggregate rather than worst-domain loss; b_worst = 1.548. Two independent real carriers agreeing is the trap this control instruments.; recorded score 0.061.
- `CTL-C-LISTALL` targets Emitting all four sources whole in natural order and truncating at 1.0B tokens, so the minority domains collapse; b_worst = 1.598.; recorded score 0.028.
- `CTL-C-OBVIOUSWRONG` targets A 100 percent web mixture, the by-construction-wrong deliverable; b_worst (mathematical) = 1.884, above b_rand = 1.641, so the informative ramp clamps to zero.; recorded score 0.
- `CTL-C-SEARCH` targets Slot-specific and the highest-risk control in this slot: 50-trial random plus TPE search over the four-simplex mixture weights alone, with selection, dedup granularity, repetition policy, recombination and emission curriculum all left at starter values; best-of-50 reaches b_worst = 1.472. REFUSAL CONDITION: compute_envelope records that the four-simplex alone is searchable at this attempt budget, so if this control measures at or above tau = 0.30 at authoring the slot is refused at authoring rather than the threshold moved.; recorded score 0.224.
- `CTL-C-MEANOPT` targets Slot-specific, the objective-substitution shortcut the max-min structure exists to defeat: optimising MEAN bits per byte across the four domains instead of the worst. The mean improves and mathematical sinks, so b_worst = 1.518, worse than the starter's 1.489.; recorded score 0.081.

<!-- AELLO-CANARY-BLOCK
  slot0: AELLO-CANARY-f42c199cd5a0d3d34bcae0194d0a49cb
  slot1: AELLO-CANARY-e5dda9fc7062a2acc5607acd68841fc0
  slot2: AELLO-CANARY-c334cff533ec729899ef99136cfa6e30
  slot3: AELLO-CANARY-2f96edbdd8a3d045c2102804507a2679
-->
