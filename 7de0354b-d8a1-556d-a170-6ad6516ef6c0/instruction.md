# AELLO-C8-S5

maximise mean per-class IoU on a private held-out tile set by fusing the twelve Sentinel-2 reflectance bands with the co-registered elevation channel.

## What is graded

Write your graded artifact to **`/submission/masks.npz`**. It must carry one two-dimensional integer land-cover mask per graded tile id, each on that tile's own co-registered spectral grid. A missing, malformed or out-of-domain artifact scores 0.

Your score is `r_det`, computed by the verifier in a separate environment from state it measures itself. The graded quantity is mean per-class intersection-over-union over the private tile set: intersections and unions are pooled per land-cover class across every graded tile, and the mean is taken over the classes whose pooled union is non-empty. Two consequences worth reading twice:

- A class the graded tiles do not contain contributes **no term**. It is never credited as a perfect one, so there is nothing to be gained by predicting classes nobody has.
- A graded tile with **no** submitted member, or one whose mask is not on the graded grid, contributes its whole label area to the union and nothing to the intersection. Omitting a tile costs more than getting it wrong.

`r_det` is zero on any red line and on any knockout failure, before any quality is considered.

## The thirteen channels

Twelve of the thirteen inputs are reflectance and one is not.

- `environment/data/graded_spectral.npz` -- one twelve-band stack per graded tile id, shape `(12, rows, cols)`, band axis in the order `B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12`. That is Sentinel-2's thirteen bands less the cirrus band `B10`, which this corpus does not ship.
- `environment/data/graded_elevation.npz` -- one elevation raster per graded tile id, shape `(rows, cols)`, **co-registered**: same grid, same pixel, metres instead of reflectance.
- `environment/data/band_statistics.json` -- the vendored per-band location and scale. The bands do not share a dynamic range.
- `environment/data/class_index.json` -- the land-cover class ids the per-class mean runs over. Every value in every submitted mask must be one of them.

The private graded masks live under `tests/heldout/` and the graded run must not touch them.

## The run record

Write `/workspace/run_record.json` during the graded run. The verifier reads it and the compiled checks below grade it, so an absent or incomplete record costs score on its own:

| key | what it must carry |
| --- | --- |
| `channel_fusion_order` | the thirteen channels in fusion order: all twelve band names exactly once, plus exactly one channel whose name identifies it as elevation (`ELEV`, `DEM` or `DSM`) |
| `elevation_fusion_stage` | `early_concat` or `mid_level_fusion` -- where the elevation channel joins |
| `band_normalisation` | one `{"mean": .., "std": ..}` per band, twelve entries; the scales may not all be one number, and `B11`/`B12` may not carry a visible band's scale |
| `training_tile_ids` | every tile id whose spectral stack entered training |
| `elevation_read_tile_ids` | every tile id whose elevation raster was read during training -- declared separately, because the two rasters are separately readable |
| `paths_written` | every path the graded run wrote |
| `encoder_provenance` | `"from_scratch"`, or a path inside the delivered tree for a vendored backbone; no remote host or hub identifier |
| `mask_member_order` | the tile ids in the order they were written into the archive; npz bytes depend on that order |
| `replay_content_sha256` | the content digest of the archive the seeded replay produces: sha256 over each member's name, dtype string, shape and raw bytes, in stored order. Not the file digest -- an npz is a zip and its headers carry a timestamp convention that belongs to numpy, not to your run |
| `agent_budget_declaration` | `train_tiles`, `graded_run_seconds`, `inference_stride`; the stride is a positive integer no larger than the tile edge |

## The budget

- One graded run. `max_timeout` bounds the refinement loop; each attempt must finish inside its per-attempt budget.
- Up to 50 attempts. Your reward history is carried between attempts.
- You must declare your own budget accounting for the graded run; the declaration is what is graded, not the harness's own measurement.

## Red lines

Any of these scores zero:

- Any write into tests/ or the graded tile set during the graded run.
- Using the elevation raster's held-out tiles as training signal.

## What ships

- `environment/` -- the corpus and a runnable starter.
- `environment/starter/` -- a correct, unoptimised baseline you may replace entirely. It writes a well-formed archive and a complete run record, so the delivery path is exercised before you change a line of it.

## Traps

- Concatenates elevation as a thirteenth band and the fusion collapses to a single stream.
- Normalises all bands with one global statistic and the SWIR bands vanish.
- Counts a class the graded tiles do not contain as a perfect one: an empty intersection over an empty union is undefined, and scoring it 1.0 pays for classes nobody predicted.
- Submits masks only for the tiles it is confident about. A graded tile with no member contributes its whole label area to the union and nothing to the intersection, so an omitted tile costs more than a wrong one.
- Predicts at a downsampled resolution and upsamples to the elevation raster's native grid instead of the spectral grid the graded labels are on.
- a pretrained Sentinel-2 encoder from a model zoo would shortcut the fusion design Mitigation: solve-time egress is allowlisted to model-API and bootstrap hosts only and no model-zoo host is reachable; any backbone must be vendored
