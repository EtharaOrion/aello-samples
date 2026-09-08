# AELLO-C8-S12 -- what this task actually tests

GENERATED SECTION. DO NOT HAND-EDIT.

maximise balanced accuracy on a clean private split when the training labels carry real human noise and the training-set size is bound. The graded artifact is an integer label array, one row per graded instance, written to /submission/preds_c8s12.npy. The graded quantity is mean per-class recall over the clean CIFAR-100 test fold, where chance is one hundredth.

## The single most important insight

The training labels and the graded labels are not drawn from the same distribution. The human annotations that ship are wrong on a large share of the pool and wrong in a structured way, while the score is taken against clean labels, so every quantity the agent can measure on its own data points slightly away from the quantity it is scored on. Training harder closes the gap on the wrong one: an over-parameterised network fits the mislabelled examples last but it does fit them, and the moment its accuracy against the noisy training labels rises past the rate at which humans agreed with the clean label, every further point is a human error it has learned to repeat. The training-set size is bound, so more data cannot dilute the noise, and the noise is not symmetric, so a single scalar rate models the wrong corpus. What is left is deciding which of a fixed pool of images to believe, and doing it with a signal that is not just the training loss and not just the clean answer key.

## The ideal solve, step by step

1. **Sample-selection criterion for the bound training set** Which images of the fixed pool to believe. A lever the contract records as separately ablatable; its worth is measured on the Phase 2 ladder over the full lever power set, never asserted here. Two compiled checks bound it without pricing it -- the retained indices must come from the shipped pool, and the retained set must not be the clean-label ordering.

2. **Loss correction against the estimated noise transition** Correcting the objective toward the clean posterior rather than reweighting samples. A lever the contract records as separately ablatable; its worth is measured on the Phase 2 ladder, never asserted here. What is compiled is only that the transition being corrected against is class-conditional, because CIFAR-100N's errors concentrate inside the coarse superclasses and a symmetric transition corrects toward the wrong place.

3. **Co-teaching or ensemble disagreement signal** A second view that disagrees where the label is wrong. A lever the contract records as separately ablatable; its worth is measured on the Phase 2 ladder, never asserted here. It is also the way past the first trap, since disagreement between two models separates the mislabelled easy example that a single model's loss cannot.

4. **Warm-up schedule before selection begins** How long to train before trusting any per-sample signal. A lever the contract records as separately ablatable; its worth is measured on the Phase 2 ladder, never asserted here. A compiled check requires the warm-up to be non-zero, because selecting at step zero selects on initialisation.

5. **Augmentation strength for retained samples** How hard to augment what survived selection, given that the pool is bound and the retained part of it is smaller still. A lever the contract records as separately ablatable; its worth is measured on the Phase 2 ladder over the full lever power set, never asserted here.

## Traps that catch agents that are not thinking carefully

- Selects on training loss alone and keeps exactly the mislabelled easy examples.
- Discards so much data that the bound training set underfits.
- Fits the noisy training labels past the rate at which the human annotators agreed with the clean ones, and reports a training accuracy that is a measurement of memorised human error.
- Models the noise as one symmetric rate over 100 classes when it concentrates on confusable pairs inside the 20 coarse superclasses.
- the clean CIFAR-100 labels are public and could be reconstructed from the images Mitigation: a compiled check compares the selected set against the clean-label ordering and rejects a selection that matches it beyond chance

---

## Contract record (annex; not part of the narrative body)

Retained under FORGE.md item 10e and Phase 2 item 7, which require this file to carry the ordered path through instruction.md with each satisfied checker identifier, and each rejected route bound to a measured known-wrong control. standards/truth-md-authoring-v1.md section 3 admits no fifth section, so this annex is a recorded deviation rather than an omission.

Ordered path through instruction.md, with the checker identifier each step satisfies:

1. `retention_within_the_bound_train_pool` (compiled, weight 5, knockout) -- The retained subset is drawn from the bound training pool and from nothing else.
2. `selection_begins_after_warmup_not_at_step_zero` (compiled, weight 3) -- Selection waited for a warm-up and used more than the training loss.
3. `retained_set_keeps_every_noisy_class` (compiled, weight 4) -- Selection did not discard any of the 100 classes outright.
4. `noise_model_is_class_conditional_not_symmetric` (compiled, weight 4) -- The declared noise model carries the structure real human annotation has.
5. `noisy_train_accuracy_below_human_agreement_rate` (compiled, weight 4) -- The graded model did not memorise the human annotation's errors.
6. `label_vector_lands` (compiled, weight 4, knockout) -- An integer label vector is written to the declared submission path.
7. `one_label_per_graded_image` (compiled, weight 5, knockout) -- The submission carries exactly one label for every image of the clean graded split.
8. `labels_inside_the_hundred_fine_classes` (compiled, weight 4, knockout) -- Every predicted label is one of the 100 CIFAR-100 fine classes.
9. `prediction_prior_closer_to_balanced_split_than_to_noisy_prior` (compiled, weight 3) -- The submission's class histogram looks like the clean graded split, not like the noisy training prior.
10. `clean_training_labels_unread` (compiled, weight 5, knockout) -- The graded run never opened a clean label for any training image.
11. `retained_subset_not_the_clean_label_ordering` (compiled, weight 5, knockout) -- The retained training subset is not a function of the clean labels.
12. `checker_tree_and_clean_split_untouched` (compiled, weight 4, knockout) -- The graded run wrote nothing into the checker tree or the clean-label staging tree.
13. `clean_split_balanced_accuracy_ramp` (compiled, weight 10, knockout) -- Balanced accuracy over the clean private split clears the published bar through the ramp.

Rejected routes, each bound to the control that measures it wrong:

- the clean CIFAR-100 labels are public and could be reconstructed from the images -- controlled by: CTL-CLEANPEEK, which selects using the clean labels, is rejected by retained_subset_not_the_clean_label_ordering, and whose measured retained purity is what the Phase 2 ladder uses to set the calibrated arm of the detector.

Deliverable manifest: /submission/preds_c8s12.npy (npy), integer label array, one row per graded instance, values in 0..99.

Control obligations recorded in seed/contract.yaml:

- `CTL-NOOP` targets the shipped starter verbatim; recorded score None.
- `CTL-ALLDATA` targets train on every noisy label without selection; recorded score None.
- `CTL-CLEANPEEK` targets select using the clean labels; must be rejected; recorded score None.

<!-- AELLO-CANARY-BLOCK
  slot0: AELLO-CANARY-67655e24f59c37a970327975520caf38
  slot1: AELLO-CANARY-f3e8eca232d018e338784243dcffc30b
  slot2: AELLO-CANARY-ef51c9a8a85c57f764e8ab7422b035bf
  slot3: AELLO-CANARY-52d049979efe780c34f5f68358752138
-->
