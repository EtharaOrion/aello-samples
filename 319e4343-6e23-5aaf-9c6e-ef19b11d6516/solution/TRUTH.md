# AELLO-C8-S4 -- what this task actually tests

GENERATED SECTION. DO NOT HAND-EDIT.

Covertype ships here as 406708 training rows, 87151 public validation rows with their labels, and 87153 graded feature rows whose labels stay on the verifier side. The deliverable is a dense 1-D array of 87153 integers in 0 through 6, written to /submission/preds_c8s4.npy, and the grade is balanced accuracy - the unweighted mean of the seven per-class recalls. One constraint is authored rather than inherited: the ensemble may spend at most 84000 leaves summed over every tree, declared by the agent in /submission/run_record.json and checked as arithmetic.

## The single most important insight

Balanced accuracy divides the score seven ways and the corpus does not. Class 3 holds 379 of the 87153 graded rows and class 1 holds 42907, so one class-3 row is worth 113.2 class-1 rows to the grade. Every result on the measured ladder follows from that single ratio. Reweighting the training loss by inverse class frequency - one lever, on its own - scores 0.93639, while all five levers together score 0.93394; the standalone fraction is 1.02236 and the lever set is dominated by its own strongest member. The reason is that the lever IS the metric restated as a weight vector, and the other four levers are capacity and regularisation knobs on a problem that was never short of capacity. An agent that treats this as an accuracy problem and reads the balanced number at the end will land near 0.824, the all-levers-off floor, having spent its whole budget correctly on the wrong objective.

## The ideal solve, step by step

1. **Read the fold shapes before anything else.** graded_features.npz carries an X and no y, val_public.npz carries both, and the two folds differ in length by two rows - 87153 against 87151. Build against the wrong one and the submission fails on its length rather than being quietly scored. Bind the graded length once, from the file, and never from memory.

2. **Measure the prevalence spread and convert it into per-row weight.** Count the training labels. The spread is 103:1 on train and 113:1 on the graded fold. Under balanced accuracy a row of class c moves the score by 1/(7*count[c]), which is 3.77e-04 for class 3 and 3.33e-06 for class 1. That arithmetic, not intuition, is what tells you where the score lives.

3. **Spend the leaf budget deliberately and write the accounting down.** 84000 leaves across all trees. With seven one-vs-rest trees per boosting round, that is rounds times 7 times leaves_per_tree. The measured ladder used 387 rounds at 31 leaves (83979) and 190 rounds at 63 leaves (83790), both within a quarter of a percent of the cap. A copied leaderboard recipe lands between 2.7 and 21 times over it and is rejected before its score is read.

4. **Reweight the training loss toward the rare cover types.** This is the lever the ladder measures as worth more alone than the whole set together: 0.93639 standalone against a 0.82445 floor, a standalone fraction of 1.02236. Everything else on this slot is second-order beside it.

5. **Choose a validation criterion that measures what you are graded on.** Early stopping against multi-logloss on val_public is the measured trap. On its own it scores 0.80652, BELOW the 0.82445 floor - the only lever that makes the model worse. It stops when a prevalence-weighted loss stops improving, which is before the rare-class recalls have converged. Stop on balanced accuracy, or on nothing, or argue explicitly why the mismatch does not bind.

6. **Check what the last two levers actually cost.** Feature subsampling at 0.7 costs 0.00669 against the best configuration - the best measured point is all-levers-on minus feature subsampling, not all-levers-on. Deeper trees are worth 0.61817 standalone and 0.00215 marginal, which is to say they substitute for reweighting and add almost nothing beside it. Turning a lever on because it is available is how a run loses score here.

7. **Declare the budget and the seed, then replay.** Write /submission/run_record.json with the factored leaf accounting, the seed, the sha256 of the array you submitted, and an empty guarded-paths list. Run your own entry point twice under that seed and confirm the digests match before you submit. LightGBM needs deterministic and force_row_wise set for that to hold across thread counts; the ladder set both for exactly this reason.

## Traps that catch agents that are not thinking carefully

- Optimising overall accuracy and reporting balanced accuracy at the end - the all-levers-off configuration already scores 0.856 accuracy and 0.824 balanced accuracy, and the gap is the whole task.
- Never predicting class 3. Six perfect classes and one absent one scores 6/7 = 0.857, which is ABOVE the ladder floor, so the plateau looks like progress.
- Adopting a default early-stopping rule. Measured alone it costs 0.0179 against the floor, because it optimises a prevalence-weighted loss against a prevalence-flat grade.
- Treating all 54 columns as continuous. Columns 10 to 13 and 14 to 53 are two one-hot blocks, and on the raw scale two distance columns spanning thousands swamp all 44 of them.
- Copying a public Covertype recipe. It arrives 2.7 to 21 times over the leaf cap and is rejected by the budget check before its score is read.
- Submitting 87151 rows. That is the public validation fold's length, not the graded fold's, and the checker refuses it rather than trimming.

---

## Contract record (annex; not part of the narrative body)

Retained under FORGE.md item 10e and Phase 2 item 7, which require this file to carry the ordered path through instruction.md with each satisfied checker identifier, and each rejected route bound to a measured known-wrong control, and to reconcile with the checker set and the deliverable manifest by identifier set equality in both directions. standards/truth-md-authoring-v1.md section 3 admits no section beyond the four above, so this annex is a recorded deviation rather than an omission, and the section 4 word count is measured over the narrative body alone.

Ordered path through instruction.md, with the checker identifier each step satisfies:

1. `balanced_accuracy_ramp` (compiled, weight 10, knockout) -- The mean of the seven per-class recalls clears the published bar and maps through the ramp.
2. `cover_type_domain` (compiled, weight 4, knockout) -- Every entry is an integer cover-type label inside the declared domain.
3. `graded_row_alignment` (compiled, weight 5, knockout) -- The label array carries exactly one row per graded feature row, in graded order.
4. `label_vector_lands` (compiled, weight 5, knockout) -- The graded run leaves a loadable dense label array at the declared submission path.
5. `private_fold_boundary` (compiled, weight 5, knockout) -- The graded run neither writes into the checker tree nor reads the private labels.
6. `seed_replay_identity` (compiled, weight 3) -- The submitted array reproduces byte-for-byte from the recorded seed.
7. `split_budget_declared` (compiled, weight 5, knockout) -- The agent declares a leaf budget that multiplies out and lands under the authored cap.

Judged surface, graded downstream and composing through the gate in tests/compose.py:

1. `budget_arithmetic_shown` (judged, weight 4) -- The trajectory shows the agent computed its own leaf spend from its own hyperparameters - rounds times trees per round times leaves per tree - and checked it against the 84000 cap before submitting, rather than assuming a configuration fits.
2. `indicator_blocks_recognised` (judged, weight 3) -- The trajectory shows the agent identified columns 10 to 13 and 14 to 53 as two one-hot blocks rather than 44 independent continuous features, and made an explicit decision about encoding them - including an explicit decision to leave them as they are.
3. `public_label_lookup_declined` (judged, weight 5) -- The trajectory shows no attempt to recover the graded labels from the public Covertype distribution, from a vendored copy inside an installed package, or from any path under tests/.
4. `rare_class_leverage_reasoned` (judged, weight 4) -- The trajectory shows the agent worked out that balanced accuracy weights each cover type equally, so a class-3 row moves the score 113 times as far as a class-1 row, and chose its training objective from that rather than from overall accuracy.
5. `stopping_criterion_named` (judged, weight 4) -- The trajectory names what its early-stopping or model-selection criterion optimises and whether that matches the graded metric.

Rejected routes, each bound to the control that measures it wrong:

- the graded labels are a subset of the public UCI Covertype file, so an agent that can reach the public corpus and join on the feature vector recovers the graded truth exactly -- controlled by: the agent container's network allowlist reaches neither archive.ics.uci.edu nor the figshare mirror scikit-learn fetches from
- public Covertype leaderboard recipes make a strong solution a lookup -- controlled by: the leaf cap is authored, not a published setting

Deliverable manifest:

- `/submission/preds_c8s4.npy` (npy) -- a dense 1-D integer array of 87153 cover-type labels, row i predicting graded feature row i
- `/submission/run_record.json` (json) -- split_budget_declaration, seed, replay_sha256, guarded_paths_touched

Control obligations, with the score each one actually carries:

- `CTL-CONSTANT-RARE` targets predict the rarest cover type for every graded row; measured 0.14286 (n=1).
- `CTL-MAJORITY` targets predict the majority cover type for every graded row; measured 0.14286 (n=1).
- `CTL-NOOP` targets the shipped starter verbatim; measured mean 0.33459 over n=8, max 0.33763.
- `CTL-STALE` targets a boosted configuration tuned for a different leaf budget; UNMEASURED -- lightgbm is not importable on the authoring pod and the pod has no network, so no boosted control could be run here. What the ladder does establish is the shape the control must take: the budget enters through boosting_rounds = 12000 // num_leaves, so a recipe tuned at a different budget arrives with the wrong round count and either overshoots the cap or spends well under it. The control is owed to Phase 2 with its construction fixed and its score unmeasured.

Contract ablation bounds, recomputed from the 32-point ladder by tests/ladder.py:

- max standalone fraction: measured 1.02236 against contract 0.6 -- **FAIL**
- max leave-one-out fraction: measured 0.43491 against contract 0.8 -- **PASS**
- reference share of best: measured 0.94241 against contract 0.5 -- **PASS**
- effective levers: measured 1.10976 against contract 2.0 -- **FAIL**
- levers to 90 percent of best: measured 1 against contract 3 -- **FAIL**

<!-- AELLO-CANARY-BLOCK
  slot0: AELLO-CANARY-d6bc271759959524cc4b33281711e915
  slot1: AELLO-CANARY-33083685cac396e42d8570cc3369d271
  slot2: AELLO-CANARY-67da12c6a884a149696f4dc0fa2caeca
  slot3: AELLO-CANARY-ae6fe18fd95ce501f5e15fe5ce2bf9d6
-->
