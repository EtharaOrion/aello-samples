# AELLO-C8-S6 -- what this task actually tests

GENERATED SECTION. DO NOT HAND-EDIT.

Maximise box-level localisation accuracy on a private split of Oxford-IIIT Pet when only a fixed small fraction of the training images carries a box. The graded artifact is /submission/detections.json: one scored box list per graded image id, each box four absolute pixel coordinates in that image's own native frame. The graded quantity is the correct-localisation rate at IoU 0.5 over the private split -- a per-image binary, decided by the single highest-scoring box, counted over the size of the split.

## The single most important insight

The metric is a cliff at one threshold, and the supervision is a fixed small set of boxes that cannot be enlarged. Both facts push the same way and against detector instinct. Because the rate is a per-image binary read off ONE box, every additional box in a list is inert: it cannot raise the rate, it cannot hedge, and the ranked-list tactics that pay under mean average precision -- widening recall, withholding uncertain boxes to protect precision -- pay nothing here and can only cost the top slot. Because the threshold is 0.5 and not a sweep from 0.50 to 0.95, tightening a box that already overlaps at 0.7 buys exactly zero, while moving a box from 0.45 to 0.51 buys a whole image. And because the boxes are fixed and few, the only way to get more supervision is to manufacture it from the unlabelled majority, which is worth doing and is also the fastest way to poison the training set. So the work is: answer every image, rank one box per image honestly, spend the effort on the images near the threshold rather than on the ones already over it, and gate what you propagate.

## The ideal solve, step by step

1. **Pseudo-box propagation from the labelled fraction** The fixed fraction that ships a box is the only ground truth there is; the rest of the training set is images and class names. Propagation manufactures boxes for the unlabelled majority from what the labelled fraction taught. The record declares which images received one, and those ids must be disjoint from the labelled fraction -- a pseudo-box on an image that already has a real one is the label copied onto itself. A lever the contract records as separately ablatable; its worth is measured on the Phase 2 ladder over the full lever power set, never asserted here.

2. **Class-activation-map thresholding policy** A classification head trained on image-level labels leaves an activation map, and a box is what you get by thresholding it and taking a connected extent. The policy is the whole lever: an aggressive threshold returns the animal's head and misses the body at IoU 0.5, a permissive one returns the animal and the sofa. Neither error is visible in a classification score, and at a single hard threshold the difference between them is the whole graded outcome. A lever the contract records as separately ablatable; its worth is measured on the Phase 2 ladder, never asserted here.

3. **Augmentation that preserves box geometry** Every augmentation that moves pixels moves the box, and the few boxes that ship are the scarcest thing in the run. A crop that cuts the animal, a flip applied to the image and not to the coordinates, or a resize that is not carried into the frame the box is written in silently corrupts the small supervised set that everything else is bootstrapped from. A lever the contract records as separately ablatable; its worth is measured on the Phase 2 ladder, never asserted here.

4. **Confidence gating on propagated boxes** Propagation without a gate compounds: an accepted wrong box becomes training signal that produces the next wrong box, and the run converges on its own mistakes. The gate is declared with its arithmetic -- how many boxes were proposed, how many survived, at what confidence -- so a gate that admitted everything is visible as a number rather than inferred from a disappointing score. A lever the contract records as separately ablatable; its worth is measured on the Phase 2 ladder, never asserted here.

5. **Multi-scale inference for small subjects** A cat occupying an eighth of the frame is the case where a single-scale activation map returns a box that is roughly right and lands at IoU 0.4, which at this threshold scores the same as no box at all. Running inference at more than one scale and merging costs time per graded image and is declared in the budget as the scales actually used. A lever the contract records as separately ablatable; its worth is measured on the Phase 2 ladder, never asserted here.

## Traps that catch agents that are not thinking carefully

- Treats the unlabelled majority as background. The detector learns that most images contain nothing, emits an empty list per image, and the rate is zero -- not because the localisation was poor but because no localisation was attempted.
- Propagates pseudo-boxes with no confidence gate. Every accepted error becomes supervision for the next round and the training set degrades faster than the extra volume helps.
- Submits a ranked list per image and tunes it for recall. Correct-localisation reads the top-scoring box and nothing else, so every box below it is inert and any effort spent ranking them is spent outside the metric.
- Withholds a box on images the model is unsure about. Under mean average precision that protects precision; here the denominator is the whole graded split, so an unanswered image is a miss and a guess is strictly better than silence.
- Optimises overlap it already has. The threshold is a cliff at 0.5: pushing a box from 0.7 to 0.9 is worth nothing, and the only images that pay are the ones sitting just under the line.
- Returns boxes in the 0-to-1 normalised frame, or on the fixed square the network was fed. Oxford-IIIT Pet images are not one size, and the graded boxes are on each image's own native frame.
- Gives every box the same score. The top box is then whichever one the serialiser wrote first, and the graded binary stops being a property of the model.
- Reaches for the trimaps. Oxford-IIIT Pet ships a per-pixel trimap for every image and thresholding one gives a tight box on the whole training set, which would replace the sparse supervision this task is about with dense supervision. They are withheld from the delivered tree and their absence is a compiled check.

---

## Contract record (annex; not part of the narrative body)

Retained under FORGE.md item 10e and Phase 2 item 7, which require this file to carry the ordered path through instruction.md with each satisfied checker identifier, and each rejected route bound to a measured known-wrong control. standards/truth-md-authoring-v1.md section 3 admits no fifth section, so this annex is a recorded deviation rather than an omission.

Ordered path through instruction.md, with the checker identifier each step satisfies:

1. `box_supervision_confined_to_the_labelled_fraction` (compiled, weight 5, knockout) -- The run consumed boxes for the fixed small fraction of training images that ships them, and for no others.
2. `trimaps_withheld_and_unread` (compiled, weight 4, knockout) -- The upstream per-pixel trimaps are absent from the delivered tree and no read of one is declared.
3. `pseudo_boxes_gated_and_disjoint_from_labelled` (compiled, weight 4) -- Propagated pseudo-boxes are confidence-gated and land only on images that carry no box.
4. `detections_manifest_lands` (compiled, weight 5, knockout) -- The scored box lists land at the graded path and every record is a box and a score.
5. `no_abstention_every_graded_image_localised` (compiled, weight 5, knockout) -- Every graded image carries at least one box; abstaining is not a move this metric offers.
6. `boxes_in_native_pixel_frame` (compiled, weight 4, knockout) -- Every box is absolute pixels on that image's own native frame, with positive area, inside the image.
7. `top_scoring_box_is_unambiguous` (compiled, weight 4, knockout) -- Exactly one box per image attains the top score, so the graded binary is a property of the model.
8. `localisation_budget_declared` (compiled, weight 3) -- The agent declares its own labelled-image, time and inference-scale budget, and the IoU it optimised for.
9. `top_box_replay_digest` (compiled, weight 3) -- The seeded replay reproduces the box that decides each image, to the pixel.
10. `private_split_boxes_unread` (compiled, weight 5, knockout) -- No graded image's box was read during the graded run, though its pixels were.
11. `checker_tree_and_box_indexes_untouched` (compiled, weight 5, knockout) -- The graded run writes nothing into the checker tree or onto the shipped box and image indexes.
12. `correct_localisation_rate_at_iou_half` (compiled, weight 10, knockout) -- The correct-localisation rate at IoU 0.5 over the private split clears the published bar through the ramp.

Rejected routes, each bound to the control that measures it wrong:

- Threshold the shipped per-pixel trimaps into boxes for the whole training set, which converts a sparse-box task into a dense-supervision one -- controlled by: the trimap directory is withheld from the delivered tree and `trimaps_withheld_and_unread` checks the tree and the read declaration, not the promise.
- Read the private split's boxes and fit to them -- controlled by: the graded boxes ship only under tests/heldout/, which the separate verifier mounts and red line 1 forbids the graded run to read; `private_split_boxes_unread` fails closed on an absent declaration.
- Widen each image's box list and tune the ranking, as a mean-average-precision submission would -- controlled by: correct-localisation reads the top-scoring box alone, so the added boxes are inert and `top_scoring_box_is_unambiguous` refuses the constant-score list that makes the top slot an artefact of serialisation order.

Deliverable manifest: /submission/detections.json (json), a JSON object whose `detections` member maps every graded image id to a non-empty list of scored box records, each `{"box": [x1, y1, x2, y2], "score": <number>}` in that image's own native pixel frame.

Control obligations recorded in seed/contract.yaml:

- `CTL-NOOP` targets the shipped starter verbatim; measured as the starter answers every graded image with one full-frame box and declares a complete record, so it passes every structural check and measures what the delivery path alone is worth on the rate; recorded score None.
- `CTL-FULLIMAGE` targets predict the whole image as the box; measured as a pet often fills much of an Oxford-IIIT Pet frame, so the whole-image box is not a zero and the ramp has to sit above what it scores; recorded score None.
- `CTL-STALE` targets a recipe tuned for the dense-supervision setting; measured as a detector tuned for the averaged 0.50-to-0.95 sweep on full box supervision, run at this supervision fraction, declaring optimised_for_iou other than 0.5; recorded score None.

<!-- AELLO-CANARY-BLOCK
  slot0: AELLO-CANARY-a0e42c0065f28a3ae7cd80750e6deb25
  slot1: AELLO-CANARY-78fa5ff042735b8899aa461525adfeb5
  slot2: AELLO-CANARY-b9f439e31b6c2b3bf2a0eca0f82e6642
  slot3: AELLO-CANARY-398a99cbdda81fcdf9fc88a19f7daa24
-->
