# AELLO-C6-S3 ground truth

GENERATED SECTION. DO NOT HAND-EDIT.

This task asks for a judge that picks which of two assistant replies a person preferred, using real human preference comparisons from the helpful portion of Anthropic's hh-rlhf collection. The unusual part is that the judge may decline to answer any item it is unsure about. It is scored on how much of six thousand hidden items it does answer, on the worse of two hidden batches. That coverage counts only if mistakes among the answered items stay at or below one in five, in every kind of conversation, within a fixed time ceiling.

## The single most important insight

Most agents will read this as an accuracy problem and try to make the comparer as correct as possible. That instinct is not wrong, but it is incomplete. Even a properly fine-tuned scorer that answers every item gets about three in ten wrong, far above the one-in-five cap. No amount of ordinary accuracy work rescues a policy of answering everything. What is really graded is a bar: how sure the judge must be before it speaks. The realisation that unlocks the task is that this bar cannot be one number. The three helpful sub-collections differ in difficulty, and the two hidden batches mix them at undisclosed proportions unlike the visible ones. A single shared bar holds where it was fitted, then quietly breaches the cap on a batch tilted toward harder material.

## The ideal solve, step by step

1. **Work out what actually buys score before building any judge** The score is carried by coverage, the share of hidden items the judge is willing to answer, on the worse of the two batches. About eighty-five percent of the reward rides on coverage and the rest on finishing quickly. Every constraint is a gate rather than a slope: miss one and the score is exactly zero. Silence costs no penalty and earns no credit, so the aim is the least silence that keeps every gate shut. The pitfall is polishing the shipped length-based judge, which answers everything and errs on about forty-four items in a hundred. Reading the scoring shape first gives every later choice one test: does this buy answers without breaking a cap?

2. **Learn how the graded conversations differ from the ones you can see** The visible file holds twenty-two thousand labelled comparisons, each tagged with its helpful sub-collection. The six thousand graded items carry no winner and no tag. Their dialogues are written out differently too, with speaker names and spacing unlike the training text, and their order is privately shuffled. A careful agent normalises both forms: take each turn's speaker from its place in the alternating conversation, not from the printed tag, and collapse the spacing. The pitfall avoided is fitting to the visible punctuation, which looks strong in rehearsal and drifts silently when graded. Normalising is what makes a confidence number measured at home mean the same thing on the hidden batches.

3. **Divide the evidence once, before anything is fitted** Two separate things must be learned here: how to compare two replies, and how sure that comparison must be before it is trusted. A model is over-confident on the pairs it trained on, so a bar set there runs far too loose. Measured that way, both hidden batches land near twenty mistakes per hundred answers, above the cap. So cut the labelled pairs once, up front, into three parts that never mix. One trains the comparer, one learns which sub-collection a dialogue came from, and one is held back purely for setting the bar. Cutting before any fitting begins is what makes the later risk estimate honest.

4. **Make confidence mean something before trying to calibrate it** A selective judge needs a confidence signal that truly ranks its own answers, so the most confident ones really are the most reliable. The cheap signal fails this. When abstention rests on reply length alone and is calibrated properly, no bar survives and the rule falls silent everywhere. So the comparison itself must be learned. One practical route takes the small open-weights conversational model shipped with the task, keeps nearly all of it frozen, and trains only a thin set of added parameters plus one scoring output. Scoring each reply alone and taking the difference keeps the judgment symmetric: the sign gives the verdict and the size gives the confidence.

5. **Set a separate confidence bar for each kind of conversation** The three helpful sub-collections are not equally hard, and the cap is checked inside each one separately, at a slightly looser one mistake in four. A single pooled bar is what a standard calibration package hands you. Measured, it answers about half of each batch at roughly twenty-one mistakes per hundred answers, and a small breach still scores zero. A graded item never says which sub-collection it came from, so the agent infers it from the fold reserved for that job, then sets a bar inside each inferred group. Because that inference is imperfect, aim a little inside the cap rather than exactly at it. The trap refused here is averaging those group bars into one number at the visible proportions, which assumes the hidden mixtures match.

6. **Spend the one graded run's minutes where they change the score** There is a single graded run on one accelerator, a hard ceiling of six thousand seconds, and full speed credit under eighteen hundred. The tempting shortcut is to buy accuracy by pulling down a large public preference model. That route was measured and it loses everything to the ceiling. A modest fine-tune on part of the labelled pairs is the better trade, with scoring done in batches and sequences grouped by length. Remember that the held-back pairs and both hidden batches must be scored inside the same run, not just the training set. The reference finishes near twelve hundred seconds, so the budget is generous to a right-sized plan and unforgiving of an oversized one.

7. **Check the rule on every slice the grader will check** The cap is not tested once. It is tested on each hidden batch, inside each of the three sub-collections, and on the dialogues that run to several turns. Reproduce all of those checks on the held-back pairs first, because a rule can look fine overall and fail on the smallest group. The slip to avoid is the wrong denominator: the rate that matters is mistakes among answered items, not among all items. That slip raises no error and leaves a healthy-looking calibration, yet it answers about seventy-eight in a hundred at roughly twenty-six mistakes per hundred answers. Done correctly the numbers show headroom, with the hardest sub-collection near seventeen mistakes against a cap of twenty-five.

8. **Balance the two batches, then report only what the run produced** The two hidden batches will not answer at the same rate, because their mixtures differ. The reference answers nine hundred and sixty-six of three thousand items on one, and eight hundred and seventy-seven on the other. Only five answered items per hundred of difference are allowed, so drop the least confident answers on the more generous batch until the gap closes. That reads only the run's own verdict counts, never a hint about correctness. It leaves a worst-batch coverage just over twenty-nine in a hundred, above the twenty-six needed for full credit. The artefact must be exact in form, and every figure the agent reports must be one this run genuinely produced.

## Traps that catch agents that are not thinking carefully

- Polishing the shipped length-based judge and then answering every item. The length rule errs on about forty-four in a hundred, and even a fine-tuned comparer still errs on about thirty, both far above the one-in-five cap.
- Taking the single pooled bar an off-the-shelf calibration package returns. It answers about half of each hidden batch at roughly twenty-one mistakes per hundred answers, and that small breach scores zero, because the bar matches the visible mixture and neither graded one.
- Setting the confidence bar on the same pairs the comparer trained on. The model is over-confident on its own training examples, so the bar comes out too loose and both hidden batches finish near twenty mistakes per hundred answers.
- Counting calibration mistakes against every calibration item rather than against the answered ones. Nothing crashes and the calibration looks healthy, but the rule then answers about seventy-eight in a hundred at roughly twenty-six mistakes per hundred answers.
- Tuning the bar against the graded reward, or trying to recover the private winners and mixture proportions from the test directory. A bar fitted to one batch's grades was caught by the coverage-gap check at about six answered items per hundred against a cap of five.
- Buying safety with silence. Abstaining everywhere scores zero for no coverage. A correct rule calibrated at half the intended risk level answers only about twenty-two and nineteen in a hundred, reaching about forty-five percent of the score.

---

## Contract record (annex; not part of the narrative body)

Retained under FORGE.md item 10e and Phase 2 item 7, which require this file to carry the ordered path through instruction.md with each satisfied checker identifier, and each rejected route bound to a measured known-wrong control, and to reconcile with the checker set and the deliverable manifest by identifier set equality in both directions. standards/truth-md-authoring-v1.md section 3 admits no section beyond the four above, so this annex is a recorded deviation rather than an omission, and the section 4 word count is measured over the narrative body alone.

GENERATED SECTION. DO NOT HAND-EDIT. Every number below is a frozen literal of `solution/grounding.yaml`, rendered by `solution/recompute.py`.

### The task

Build and run, inside one graded run on one H100, a selective pairwise preference judge over the three helpful sub-distributions of Anthropic's hh-rlhf corpus. The judge emits A, B or the exact sentinel ABSTAIN for every item of two private shards whose winner labels and sub-distribution mixtures it never sees. The graded axis is quality_under_budget: the score is carried by the answered fraction of the WORST shard, and it is exactly zero unless the selective risk holds a distribution-free cap on each shard, within each helpful sub-distribution, and on the long-dialogue subgroup, with the two shards' answered fractions inside a gap cap.


### Bound constants

- `ALPHA` = 0.2
- `STRATUM_SLACK` = 0.05
- `COV_FLOOR` = 0.15
- `COV_TARGET` = 0.26
- `COV_GAP_CAP` = 0.05
- `W_COV` = 0.85
- `W_SPEED` = 0.15
- `T_MIN` = 1800.0
- `T_MAX` = 6000.0

### The reference and why it reaches full reward

The reference at `solution/reference.py` splits the shipped labelled pairs by one seeded permutation into three disjoint folds before any fit begins: a FIT fold for the scorer, a DOMAIN fold for the sub-distribution classifier, and a CONFORMAL fold for the calibration. It fine-tunes Qwen2.5-1.5B-Instruct as a Bradley-Terry pairwise scorer with a rank-16 LoRA adapter on every linear projection plus the scalar score head; the verdict is the sign of an antisymmetric score difference and the selective confidence is its magnitude. A multinomial head fitted on the DOMAIN fold infers each graded item's helpful sub-distribution, which is never shipped with the item. Mondrian split-conformal calibration then runs separately inside each inferred stratum on the CONFORMAL fold, taking the largest answered prefix whose observed error count still admits the finite-sample quantile index ceil((m+1)(1-alpha)) at a nominal level tightened by a declared margin for inferred-stratum noise under shift. A final equalisation pass drops the least-confident answered items on the higher-coverage shard until the between-shard coverage gap is inside its cap, reading only the run's own verdict counts.


Measured on the authoring host over the two frozen private shards:

- worst-shard coverage: **0.2923**, the graded quantity
- shard p1 coverage: **0.322**, answered fraction
- shard p2 coverage: **0.2923**, answered fraction
- between-shard coverage gap: **0.0297**, at or below COV_GAP_CAP
- shard p1 selective risk: **0.1449**, at or below ALPHA
- shard p2 selective risk: **0.1608**, at or below ALPHA
- worst helpful sub-distribution risk: **0.1714**, at or below ALPHA + STRATUM_SLACK
- long-dialogue subgroup risk: **0.1642**, at or below ALPHA + STRATUM_SLACK
- elapsed seconds: **1214**, at or below T_MIN, so the speed clamp saturates

With every Stage-0 and Stage-1 condition clear, the coverage clamp saturated and the speed clamp saturated, `S_outcome` is 1.0, the compiled structural knockouts pass so `GATE` is 1, and `r_det = GATE * S_outcome * RubricFactor = 1.0`, the bound full reward.


### Ordered path through instruction.md

1. Split the shipped pairs into disjoint fit, domain and conformal folds before any fit begins — satisfies `calibration_fold_disjoint`.
2. Fine-tune the pairwise scorer on the fit fold alone and take the verdict as the sign of the antisymmetric score difference — satisfies `verdicts_wellformed`.
3. Infer each graded item's helpful sub-distribution with a head fitted on the domain fold, because the graded mixture is private — satisfies `abstain_sentinel_exact`.
4. Calibrate a per-stratum abstention threshold on the conformal fold at the finite-sample quantile index — satisfies `verdicts_complete`.
5. Equalise the two shards' answered fractions inside the coverage-gap cap using own verdict counts only — satisfies `thresholds_held_back`.
6. Write one verdict in the exact closed vocabulary for every item of both shards — satisfies `verdicts_present`.

### Rejected routes, each bound to a measured control

- **`CTRL_S3_NOOP`** (no-op class) — The shipped scaffold is left unrun, so no verdict artefact exists for either shard. Measured `r_det` **0.0**, reason `empty-submission`.
- **`CTRL_S3_STARTER_LIST_ALL`** (obvious-wrong class, the shipped starter) — environment/starter/judge.py verbatim: a length-margin prior applied to every item with no abstention. Answering everything makes the risk cap binding and a length prior does not hold it. Measured `r_det` **0.0**, reason `risk-cap-exceeded`.
- **`CTRL_S3_ABSTAIN_ALL`** (trivial-safety class) — Abstain on every item, which makes the risk cap vacuously true and buys nothing. Measured `r_det` **0.0**, reason `no-coverage`.
- **`CTRL_S3_LIST_ALL_SCORED`** (list-all class) — The reference's own fine-tuned scorer with the selective rule deleted: answer every item. This is the strongest available answer-everything route and it still fails the cap. Measured `r_det` **0.0**, reason `risk-cap-exceeded`.
- **`CTRL_S3_POOLED_CONFORMAL`** (decoy/distractor class, the library default) — Plain POOLED split conformal with one global threshold, which is what a public conformal package supplies. It is exchangeable with neither shard because the graded mixtures differ from the shipped one, so the Mondrian construction is demonstrably load-bearing. Measured `r_det` **0.0**, reason `risk-cap-exceeded`.
- **`CTRL_S3_STALE_THRESHOLD`** (stale class, bound to the mixture silent mutation) — Per-stratum thresholds blended into ONE threshold at the SHIPPED mixture proportions, i.e. assuming the graded mixture equals the shipped one, which it does not on either shard. Measured `r_det` **0.0**, reason `shift-instability`.
- **`CTRL_S3_LENGTH_ONLY_SELECTOR`** (single-clue class, bound to lever SCT1) — Abstention driven by absolute response-length difference alone, conformally calibrated exactly as the reference calibrates. No threshold on the calibration fold clears the level at all, so the rule abstains on everything. Measured `r_det` **0.0**, reason `no-coverage`.
- **`CTRL_S3_SHARD1_FITTED`** (reward-signal-as-oracle class) — The abstention threshold fitted against shard p1's private labels over many attempts and applied to both shards, which the between-shard coverage-gap cap detects. Measured `r_det` **0.0**, reason `shift-instability`.
- **`CTRL_S3_CALIBRATE_ON_FIT_FOLD`** (leak class, the contract's third intended failure mode) — Calibration run on the FIT fold the scorer was trained on. The scorer is over-confident on its own training pairs, so the threshold is far too loose and both shards blow the cap. Measured `r_det` **0.0**, reason `risk-cap-exceeded`.
- **`CTRL_S3_UNCONDITIONAL_DENOMINATOR`** (silent-arithmetic class, bound to lever PRC1) — The conformal threshold chosen so that errors-among-answered over ALL calibration items clears alpha, instead of over the ANSWERED items. No traceback, a healthy-looking calibration, and the failure is visible only in measured risk. Measured `r_det` **0.0**, reason `risk-cap-exceeded`.
- **`CTRL_S3_TIMID`** (ramp-resolution class, the tightest instrument in the ladder) — A correctly constructed Mondrian rule calibrated far too conservatively at nominal level 0.10. Every risk cap, stratum cap and coverage-gap cap holds and it is fast, but it answers only 65 percent of what the reference answers. It proves the reward map is non-constant across genuine solutions and that safety bought with abstention does not reach the solved line. Nominal levels below 0.08 are not usable as a more conservative control because the surviving answered set inside the smallest stratum becomes small enough to breach the per-stratum cap, which was measured rather than assumed. Measured `r_det` **0.4514**, reason `graded`.
- **`CTRL_S3_SLOW_REFERENCE`** (timed-axis class) — The reference's own verdicts with the graded run past the T_MAX ceiling. Proves the fixed budget is a real constraint and prices the download-a-large-public-reward-model route inside the timed region. Measured `r_det` **0.0**, reason `budget-exceeded`.
- **`CTRL_S3_OFF_BY_ONE_QUANTILE`** (MEASURED NON-DISCRIMINATING, recorded rather than claimed as a control) — The canonical wrong constant: floor(m*(1-alpha)) substituted for the finite-sample index ceil((m+1)*(1-alpha)), Mondrian otherwise correct. At this calibration size the two indices differ by at most one unit of error budget out of several hundred admitted errors, so the route is not a shortcut: it reaches full reward. The contract projected it at global risk 0.208 against alpha 0.20; measurement contradicts that projection. It is recorded as a finding for reconciliation rather than shipped as a control that must fail, and lever PRC1 is carried instead by CTRL_S3_UNCONDITIONAL_DENOMINATOR, a silent conformal arithmetic error that IS discriminating. Measured `r_det` **1.0**, reason `graded`.

### Residuals recorded rather than hidden

- The elapsed second count enters the reward through the submission's own `elapsed_s` when the harness writes no `run_meta.json`, which it does not today. The speed band carries only W_SPEED of the reward and the honest-metrics council rubric is its Bucket-N backstop, but a host-side timing carrier is owed and is recorded as a named gap rather than papered over.
- The sub-distribution classifier is imperfect, so the Mondrian guarantee is exchangeable in the INFERRED stratum rather than the true one. The reference absorbs that with a declared calibration margin and the realised per-stratum risks are reported against the cap rather than assumed to hold.
- The reference was measured on a shared authoring host carrying other tenants, so its elapsed seconds are an upper bound; an exclusive grading host widens the timed margin.

### Deliverable and checker map

The single deliverable is `/submission/verdicts.json`. The compiled structural checks (`verdicts_present`, `verdicts_wellformed`, `abstain_sentinel_exact`, `verdicts_complete`) gate and factor `r_det`; the outcome measure recomputes every coverage and risk quantity as an integer count over an integer denominator from the private label vectors and the private mixture map, then applies the reward map. The four council rubrics judge the trajectory downstream and may only lower the composed score.

<!-- AELLO-CANARY-BLOCK
  slot0: AELLO-CANARY-5a2a398da0b1caaab4680dfe60f25158
  slot1: AELLO-CANARY-e63e4a2ffec892f6e66052747d2ca73b
  slot2: AELLO-CANARY-d5486f12956ce7d142627e039e421dc1
  slot3: AELLO-CANARY-28761ed455783590174668c81e473ff2
-->
