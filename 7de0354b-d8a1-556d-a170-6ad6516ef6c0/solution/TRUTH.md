# AELLO-C8-S5 -- what this task actually tests

GENERATED SECTION. DO NOT HAND-EDIT.

maximise mean per-class IoU on a private held-out tile set by fusing the twelve Sentinel-2 reflectance bands with the co-registered elevation channel. The graded artifact is one two-dimensional integer land-cover mask per graded tile id, each on that tile's own co-registered spectral grid, written to /submission/masks.npz. The graded quantity is mean per-class intersection-over-union over the private tile set, pooled per class and averaged over the classes that actually occur.

## The single most important insight

Thirteen channels reach the network and only twelve of them are reflectance. The elevation raster shares the spectral grid by construction -- that co-registration is the only reason it can be fused pixel-for-pixel at all -- but it is metres against per-band reflectance, so it is a second modality on a shared grid rather than a thirteenth band, and the two facts pull in opposite directions: the grid invites concatenation and the units forbid it. The metric compounds the pull. Mean PER-CLASS IoU pools each land-cover class across the graded tiles and averages over the classes that occur, so a class absent everywhere is dropped rather than scored perfect, and a rare class covering a few hundred pixels weighs exactly as much as the class covering half the corpus -- which is where the elevation channel earns its place, because the classes that separate on terrain are the small ones. None of that is reachable until every graded tile carries a mask on its own co-registered grid: r_det is zero on any red line and on any knockout failure before a single intersection is counted.

## The ideal solve, step by step

1. **Fusion point: early channel concat vs mid-level feature fusion** Where the elevation channel joins decides whether it is a modality or a band. Early concatenation puts all thirteen channels through one stem and one set of first-layer statistics; mid-level fusion carries an elevation branch of its own and joins it after the spectral stem has already formed features. The run record has to say which, and name the elevation channel as elevation, because the two designs are not distinguishable from the submitted masks alone. A lever the contract records as separately ablatable for this slot; its worth is measured on the Phase 2 ladder over the full lever power set, never asserted here.

2. **Per-band normalisation from vendored band statistics** The twelve bands are acquired at three ground resolutions and span different reflectance ranges; the vendored statistics give a location and scale per band so that B11 and B12 are not normalised by a number computed mostly from the visible bands. This is the lever the compiled per-band check reads a declaration of, and the check refuses a single global statistic rather than scoring it low. A lever the contract records as separately ablatable for this slot; its worth is measured on the Phase 2 ladder over the full lever power set, never asserted here.

3. **Elevation-derived slope/aspect features** Slope and aspect are derivatives of the elevation raster on the same grid, so they cost no new co-registration and carry terrain information the raw height does not: a valley floor and a plateau at the same altitude are the same number until you differentiate. A lever the contract records as separately ablatable for this slot; its worth is measured on the Phase 2 ladder over the full lever power set, never asserted here.

4. **Class-frequency-aware loss for the rare land-cover classes** The graded mean is per class, not per pixel, so a class covering a thousandth of the corpus carries a full term in it. A loss weighted by pixel frequency optimises something the metric does not measure. A lever the contract records as separately ablatable for this slot; its worth is measured on the Phase 2 ladder over the full lever power set, never asserted here.

5. **Tile-overlap inference stitching** Predictions near a tile edge see less context than predictions at its centre, and the stride at which overlapping windows are stitched decides how much of each graded tile is labelled from an edge. The declared stride is bounded by the tile edge because a wider one leaves graded pixels no window ever covered. A lever the contract records as separately ablatable for this slot; its worth is measured on the Phase 2 ladder over the full lever power set, never asserted here.

## Traps that catch agents that are not thinking carefully

- Concatenates elevation as a thirteenth band and the fusion collapses to a single stream.
- Normalises all bands with one global statistic and the SWIR bands vanish.
- Counts a class the graded tiles do not contain as a perfect one: an empty intersection over an empty union is undefined, and scoring it 1.0 pays for classes nobody predicted.
- Submits masks only for the tiles it is confident about. A graded tile with no member contributes its whole label area to the union and nothing to the intersection, so an omitted tile costs more than a wrong one.
- Predicts at a downsampled resolution and upsamples to the elevation raster's native grid instead of the spectral grid the graded labels are on.
- a pretrained Sentinel-2 encoder from a model zoo would shortcut the fusion design Mitigation: solve-time egress is allowlisted to model-API and bootstrap hosts only and no model-zoo host is reachable; any backbone must be vendored

---

## Contract record (annex; not part of the narrative body)

Retained under FORGE.md item 10e and Phase 2 item 7, which require this file to carry the ordered path through instruction.md with each satisfied checker identifier, and each rejected route bound to a measured known-wrong control. standards/truth-md-authoring-v1.md section 3 admits no fifth section, so this annex is a recorded deviation rather than an omission.

Ordered path through instruction.md, with the checker identifier each step satisfies:

1. `tile_budget_declared` (compiled, weight 3) -- The agent declares its own tile and inference-stride budget for the graded run.
2. `thirteen_channel_fusion_order_declared` (compiled, weight 4) -- The run record declares the fusion order over all thirteen input channels and the stage at which the elevation channel joins.
3. `per_band_scaling_not_global` (compiled, weight 4) -- Normalisation is declared per band, and the SWIR bands do not inherit the visible bands' scale.
4. `elevation_heldout_tiles_untrained` (compiled, weight 5, knockout) -- No graded tile's elevation raster or spectral stack entered training.
5. `vendored_encoder_only` (compiled, weight 3) -- Any pretrained encoder is vendored inside the bundle, not fetched.
6. `mask_archive_lands` (compiled, weight 5, knockout) -- The keyed mask archive lands at the graded path and every member is an integer raster.
7. `coregistered_grid_agreement` (compiled, weight 5, knockout) -- Every graded tile carries a mask on the grid its spectral stack and elevation raster share.
8. `land_cover_labels_within_class_index` (compiled, weight 4, knockout) -- Every mask value is a land-cover class the shipped class index names.
9. `mask_member_order_deterministic` (compiled, weight 3) -- The mask archive replays to the same content, member order included.
10. `graded_tile_archives_untouched` (compiled, weight 5, knockout) -- The graded run writes nothing into the checker tree or onto the graded tile archives.
11. `mean_per_class_iou_ramp` (compiled, weight 10, knockout) -- Mean per-class IoU over the private tile set clears the published bar through the ramp.

Rejected routes, each bound to the control that measures it wrong:

- a pretrained Sentinel-2 encoder from a model zoo would shortcut the fusion design -- controlled by: solve-time egress is allowlisted to model-API and bootstrap hosts only and no model-zoo host is reachable; any backbone must be vendored
- dropping the nine non-RGB bands and the elevation channel -- controlled by: CTL-RGBONLY, and by `thirteen_channel_fusion_order_declared`, which refuses a roster that does not name all twelve reflectance bands
- predicting the modal land-cover class everywhere -- controlled by: CTL-SINGLECLASS, and by the per-class pooling in `mean_per_class_iou_ramp`, under which every class but the modal one scores zero

Deliverable manifest: /submission/masks.npz (npz), one two-dimensional integer land-cover mask per graded tile id, each on that tile's own co-registered spectral grid.

Corpus layout, as the compiled checks read it:

- `environment/data/graded_spectral.npz` -- agent-visible, one twelve-band stack per graded tile id.
- `environment/data/graded_elevation.npz` -- agent-visible, one elevation raster per graded tile id, on the same grid.
- `environment/data/band_statistics.json` -- the vendored per-band location and scale.
- `environment/data/class_index.json` -- the land-cover class ids the per-class mean runs over.
- `tests/heldout/graded_masks.npz` -- the private graded labels, verifier-side only.

Control obligations recorded in seed/contract.yaml:

- `CTL-NOOP` targets the shipped starter verbatim; recorded score None.
- `CTL-RGBONLY` targets drop the nine non-RGB bands and the elevation channel; recorded score None.
- `CTL-SINGLECLASS` targets predict the modal land-cover class everywhere; recorded score None.

<!-- AELLO-CANARY-BLOCK
  slot0: AELLO-CANARY-64e40139a41090860fd302035370d48d
  slot1: AELLO-CANARY-5bd0be3a160298993782b68cfcf9f8a5
  slot2: AELLO-CANARY-53a8ccda1f8385db6db18a47947ded69
  slot3: AELLO-CANARY-6e98d3431d8759bf062af90956533987
-->
