# AELLO-C8-S13 -- what this task actually tests

GENERATED SECTION. DO NOT HAND-EDIT.

minimise expected calibration error on private corrupted splits while holding clean accuracy inside a bound tolerance. The graded artifact is a float array of 29000 rows by 100 columns written to /submission/confidences.npy, rows non-negative and summing to 1, the first 19000 rows covering the 95-cell corruption grid in pinned order and the last 10000 covering the clean graded fold. The graded quantity is the macro average of the 95 per-cell expected calibration errors, mapped to [0,1] higher-better as 1 minus that mean.

## The single most important insight

Calibration is the one objective an agent can improve by giving up. Flatten every posterior toward 1/100 and the expected calibration error falls, because a predictor that is never confident is never overconfident; the limit of that strategy is CTL-UNIFORM, which scores a near-zero calibration error while predicting nothing. Nothing inside the corrupted objective punishes it, and no amount of hardening the corrupted objective can, because the shortcut is not a defect in the metric but a property of it. That is why this slot grades two blocks in one array and why the clean fold ships unlabelled: the price for flattening is charged on clean accuracy or it is not charged at all. The second thing that matters is that the score is a macro average over 95 cells rather than a pooled figure over 19000 rows, so one catastrophic corruption cannot be diluted by the 94 that went well.

## The ideal solve, step by step

1. **Temperature-scaling fit set composition** Only clean_train and clean_val carry labels, and the run record must name which of them the calibration was fitted on. A temperature fitted on clean data is applied to corrupted data whose confidence distribution has moved, so the composition of the fit set is a real choice and not a formality. The contract records it as separately ablatable; its worth is measured on the Phase 2 ladder over the full lever power set and never asserted here.

2. **Severity-aware recalibration policy** The grid is 19 corruptions by 5 severities and the severity of each graded row is shipped in the corpus, so a single global temperature is a choice rather than a necessity. Whether conditioning on severity, or on a severity estimated from the image alone, is worth its variance is exactly the kind of thing the macro average over cells rewards differently from a pooled figure. The contract records it as separately ablatable; its worth is measured on the Phase 2 ladder and never asserted here.

3. **Ensemble or MC-dropout variance signal** A confidence built from the spread of several forward passes is a different estimator from the softmax maximum of one, and it degrades differently under corruption. It also costs forward passes, which the declaration must account for. The contract records it as separately ablatable; its worth is measured on the Phase 2 ladder and never asserted here.

4. **Clean-accuracy guard against over-smoothing confidence** The lever that decides whether the flattening shortcut is taken deliberately or by accident. Any recalibration that is not argmax-preserving trades clean accuracy for calibration, and the tolerance gate prices that trade; a solve that never measures its own clean accuracy will not know which side of the gate it is on. The contract records it as separately ablatable; its worth is measured on the Phase 2 ladder and never asserted here.

5. **Binning scheme used to fit, distinct from the graded binning** The graded partition is pinned in the checker tree and is deliberately not published. Fitting a histogram to a guessed copy of it is red line 2, and it is detectable because such a fit lands at a calibration error under the pinned partition that binomial noise alone forbids; ck_binpin is the obligation that detects it. The lever is the choice of a fitting binning that is genuinely the agent's own, fitted on data the agent is allowed to hold labels for. The contract records it as separately ablatable; its worth is measured on the Phase 2 ladder and never asserted here.

## Traps that catch agents that are not thinking carefully

- Buys calibration by flattening confidence and fails the accuracy tolerance.
- Fits temperature on the corrupted split it is graded on.
- Submits only the 19000 corrupted rows, which grades the stated objective and silently removes the one gate that prices the flattening shortcut.
- Optimises the pooled 19000-row calibration error and lets one catastrophic corruption hide inside an average the ramp does not read.
- Fits a lookup to the private corrupted fold, which shows up as a calibration error under the pinned partition four orders of magnitude below the binomial noise floor any honest solve carries.
- emitting a uniform posterior trivially lowers calibration error Mitigation: CTL-UNIFORM is a compiled control and the clean-accuracy tolerance gate rejects any submission that buys calibration with accuracy

---

## Contract record (annex; not part of the narrative body)

Retained under FORGE.md item 10e and Phase 2 item 7, which require this file to carry the ordered path through instruction.md with each satisfied checker identifier, and each rejected route bound to a measured known-wrong control. standards/truth-md-authoring-v1.md section 3 admits no fifth section, so this annex is a recorded deviation rather than an omission.

Ordered path through instruction.md, with the checker identifier each step satisfies:

1. `posterior_file_lands` (compiled, weight 5, knockout) -- The graded run leaves a loadable floating-point posterior at the declared submission path.
2. `stacked_block_layout` (compiled, weight 5, knockout) -- The array carries the 19000 corrupted graded rows and the 10000 clean graded rows in the one pinned stacking order.
3. `rows_are_posteriors` (compiled, weight 4, knockout) -- Every row is a probability vector over the 100 fine labels, not merely a vector that sums to one.
4. `calibration_fit_split_declared` (compiled, weight 3) -- The agent declares how many graded forward passes it spent and which split it fitted its calibration on.
5. `heldout_label_isolation` (compiled, weight 5, knockout) -- The agent holds no copy of either private label vector, at any of the three lengths a copy can take here.
6. `graded_fold_boundary` (compiled, weight 5, knockout) -- The graded run writes into neither the checker tree nor the private label store.
7. `pinned_binning_intact` (compiled, weight 3, knockout) -- The partition the score is read under is the one pinned in the bundle.
8. `uniform_posterior_rejected` (compiled, weight 4) -- The compiled rejection of control CTL-UNIFORM.
9. `ck_binpin` (compiled, weight 5, knockout) -- Control CTL-GRADEDBINS is rejected.
10. `ck_binfam` (compiled, weight 2) -- Contract checker obligation CK-BINFAM is discharged.
11. `posterior_replay_identity` (compiled, weight 3) -- The posterior reproduces byte for byte from the recorded seed.
12. `per_cell_ece_recorded` (compiled, weight 3) -- All 95 per-cell calibration errors are computed and the worst cell is named.
13. `clean_accuracy_tolerance` (compiled, weight 5, knockout) -- Clean accuracy stays inside the bound tolerance. Raises ConstantUnmeasured while clean_accuracy_floor is null.
14. `corruption_macro_ece_ramp` (compiled, weight 10, knockout) -- The macro-averaged corrupted calibration error clears the published bar. Raises ConstantUnmeasured while the ramp is null.

Rejected routes, each bound to the control that measures it wrong:

- emitting a uniform posterior trivially lowers calibration error -- controlled by: CTL-UNIFORM is a compiled control and the clean-accuracy tolerance gate rejects any submission that buys calibration with accuracy. The compiled half is `uniform_posterior_rejected`; the priced half is `clean_accuracy_tolerance`.
- fitting calibration on the graded binning scheme -- controlled by: CTL-GRADEDBINS, which seed/contract.yaml requires be REJECTED and which had no enforcing obligation. `ck_binpin` is that obligation, authored here against the recorded checker_obligations_shortfall. The contract still carries the shortfall paragraph until it is amended outside this bundle, so the gap is closed in the checker before it is closed in the record.

Deliverable manifest: /submission/confidences.npy (npy), float array of 29000 rows by 100 columns, rows non-negative and summing to 1, corrupted block first.

Control obligations recorded in seed/contract.yaml:

- `CTL-NOOP` targets the shipped starter verbatim; recorded score None.
- `CTL-UNIFORM` targets emit a uniform posterior everywhere; recorded score None.
- `CTL-GRADEDBINS` targets fit on the graded binning; must be rejected; recorded score None.

<!-- AELLO-CANARY-BLOCK
  slot0: AELLO-CANARY-61cc355fac243dd3dad4afc3b9a84a97
  slot1: AELLO-CANARY-a107ee07d664b4b1f66d41dffb3ee882
  slot2: AELLO-CANARY-306614f257370612315448f37e23f919
  slot3: AELLO-CANARY-21d137251a71485a1d61069e01973a95
-->
