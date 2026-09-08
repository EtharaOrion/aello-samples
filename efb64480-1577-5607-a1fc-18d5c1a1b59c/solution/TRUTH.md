# AELLO-C8-S8 -- what this task actually tests

GENERATED SECTION. DO NOT HAND-EDIT.

maximise mean IoU on a private split when only a bound fraction of pixels carry a label. The graded artifact is one integer mask array per graded scene, keyed by ADE20K image id and on that scene's own pixel grid, written to /submission/masks.npz. The graded quantity is mean intersection-over-union over the private split, accumulated per class over the 150 stuff and object classes.

## The single most important insight

The training annotation is sparse and the grading is dense. Every pixel of the private split is scored, and only a bound fraction of the training pixels carries a target, so the whole task is what a model is taught to do with a pixel that has no label. The trap is that the convenient answer -- call it background and train a dense loss -- is available, cheap, and wrong: the unlabelled majority becomes the dominant target, every rare class is taught as background wherever the annotation did not reach it, and mean IoU averages over classes rather than pixels, so the tail it destroys is most of the score. Before any of that, r_det is zero on a red line, and red line 1 forbids READING the private dense labels rather than writing to them, which is the stricter of the two claims and the one a careful-looking run is most likely to break.

## The ideal solve, step by step

1. **Decide what an unlabelled pixel is before deciding anything else** Lever L1 is the partial-label loss masking policy. Either the unlabelled pixels are excluded from the loss by an ignore index, or they are given a target and become the class that dominates training. The run record declares which, and unlabelled_pixels_masked_not_backpropagated reads that declaration, because the choice is invisible in a submitted mask and visible only in how the loss was built. Its worth against the other four levers is measured on the Phase 2 ladder over the full power set and is never asserted here.

2. **Get a target onto the unlabelled pixels from somewhere other than the labels** Lever L2 is consistency regularisation across augmented views -- two views of the same scene must agree where neither carries an annotation. This is the honest way to use the unlabelled majority, and it is available precisely because red line 1 forbids the dishonest one. Its worth is measured on the Phase 2 ladder and is never asserted here.

3. **Sample the tail often enough that it exists in the gradient** Lever L3 is class-balanced sampling over the long tail. Mean IoU averages over the 150 classes, so a class holding a thousandth of the pixels is worth as much as one holding a third of them, and uniform sampling over a sparse annotation may not show a rare class at all inside the step budget. Its worth is measured on the Phase 2 ladder and is never asserted here.

4. **Buy decoder resolution with steps you actually have** Lever L4 is decoder resolution against the step budget. Finer output helps thin structure and boundaries and costs steps, and the second recorded trap is raising it until training truncates. The run record declares planned_steps, completed_steps and the decoder output stride so a truncated run reports as truncated rather than as a weak model. Its worth is measured on the Phase 2 ladder and is never asserted here.

5. **Weight the boundaries, where sparse annotation is thinnest** Lever L5 is boundary-aware loss weighting. Sparse annotation tends to sit in class interiors, so boundaries are exactly where supervision is least present and where IoU is most often lost. Its worth is measured on the Phase 2 ladder and is never asserted here.

6. **Ship a dense mask on every scene's own grid** The deliverable is one integer mask per graded image id at that scene's own height and width, values in 0..149. ADE20K scenes are not a uniform tile size, so one array shape for the split cannot satisfy scene_grid_per_image_id, and nothing interpolates a mis-sized mask into place at grading time. Leaving pixels at the ignore index 255 is legal and concedes them, which is a submission shaped like its own supervision and is the last way to give the metric away.

## Traps that catch agents that are not thinking carefully

- Backpropagates through unlabelled pixels as background and the long tail collapses.
- Raises decoder resolution until the step budget truncates training.
- Submits the sparse annotation's own support and leaves the rest at the ignore index, conceding every pixel that was never labelled.
- Resizes every scene to one working resolution and ships masks on that grid rather than on each scene's own.
- public ADE20K recipes are abundant and could be transferred wholesale Mitigation: the supervision fraction is authored and is not a public setting, so a transferred dense recipe degrades and is caught by the ablation ladder

---

## Contract record (annex; not part of the narrative body)

Retained under FORGE.md item 10e and Phase 2 item 7, which require this file to carry the ordered path through instruction.md with each satisfied checker identifier, and each rejected route bound to a measured known-wrong control. standards/truth-md-authoring-v1.md section 3 admits no fifth section, so this annex is a recorded deviation rather than an omission.

Ordered path through instruction.md, with the checker identifier each step satisfies:

1. `supervision_fraction_declared_and_bounded` (compiled, weight 5, knockout) -- The agent declares the labelled pixels it trained on, and the count does not exceed what ships.
2. `unlabelled_pixels_masked_not_backpropagated` (compiled, weight 4) -- The declared partial-label loss policy masks unlabelled pixels instead of teaching them.
3. `step_budget_not_truncated_by_decoder_resolution` (compiled, weight 3) -- Training reached its declared step count at the decoder resolution the agent chose.
4. `dense_private_labels_unread` (compiled, weight 5, knockout) -- No dense label raster of the private split was opened during the graded run. Red line 1.
5. `checker_tree_and_graded_masks_untouched` (compiled, weight 4, knockout) -- The graded run modifies nothing under tests/ and no byte of the private dense masks. Red line 2.
6. `mask_archive_lands` (compiled, weight 4, knockout) -- A per-scene integer mask archive is written to the declared submission path.
7. `scene_grid_per_image_id` (compiled, weight 5, knockout) -- Each graded ADE20K scene carries exactly one mask, on that scene's own pixel grid.
8. `label_domain_is_the_150_classes` (compiled, weight 4, knockout) -- Every predicted pixel names one of the 150 stuff and object classes, or concedes.
9. `prediction_dense_despite_sparse_supervision` (compiled, weight 4) -- The prediction classifies every pixel, though supervision covered only a bound fraction.
10. `long_tail_not_collapsed_to_modal_class` (compiled, weight 3) -- The prediction distinguishes classes rather than painting the split one colour.
11. `sparse_supervised_miou_ramp` (compiled, weight 10, knockout) -- Mean intersection-over-union over the private split clears the published bar through the ramp.

Rejected routes, each bound to the control or the red line that measures it wrong:

- treat every unlabelled pixel as background and train a dense cross-entropy -- controlled by: CTL-STALE, and refused by `unlabelled_pixels_masked_not_backpropagated`
- public ADE20K recipes are abundant and could be transferred wholesale -- controlled by: the supervision fraction is authored and is not a public setting, so a transferred dense recipe degrades and is caught by the ablation ladder
- raise the decoder to full input resolution and take whatever steps fit -- caught by `step_budget_not_truncated_by_decoder_resolution`
- submit the sparse annotation's own support and leave the rest at the ignore index -- caught by `prediction_dense_despite_sparse_supervision`
- read the private split's dense masks to select a configuration -- red line 1, caught by `dense_private_labels_unread`

Deliverable manifest: /submission/masks.npz (npz), one integer mask array per graded scene, keyed by ADE20K image id, on that scene's own pixel grid, values in 0..149 with 255 reserved for a conceded pixel.

Control obligations recorded in seed/contract.yaml:

- `CTL-NOOP` targets the shipped starter verbatim; recorded score None.
- `CTL-MODAL` targets predict the modal class for every pixel; recorded score None.
- `CTL-STALE` targets a recipe tuned for the dense-label setting; recorded score None.

<!-- AELLO-CANARY-BLOCK
  slot0: AELLO-CANARY-499b6720890d6f693baa338ed2604792
  slot1: AELLO-CANARY-28b9e5a75be70673ff77e11abc1a8559
  slot2: AELLO-CANARY-fe83372c9521a691808f135217540f44
  slot3: AELLO-CANARY-149bdddb0ee8828149dfabe093875559
-->
