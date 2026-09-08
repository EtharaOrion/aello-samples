# AELLO-C8-S8

maximise mean IoU on a private split when only a bound fraction of pixels carry a label.

## What is graded

Write your graded artifact to **`/submission/masks.npz`**. It must be one integer mask array per
graded scene, keyed by ADE20K image id. A missing, malformed or out-of-domain artifact scores 0.

The label domain is fixed:

- class ids **0..149**, the 150 ADE20K stuff and object classes;
- **255** is the ignore index. It is legal in your archive and it *concedes* the pixel. Mean IoU is
  taken over every pixel of the private split, not over the labelled ones, so a pixel you leave at
  255 costs you exactly what a wrong class would.

Each mask must be on **its own scene's pixel grid**. ADE20K scenes are not a uniform tile size, so
one array shape for the whole split will not do, and nothing resizes a mis-shaped mask into place
at grading time -- it is counted as a missing prediction.

Your score is `r_det`, computed by the verifier in a separate environment from state it measures
itself. The graded quantity is mean intersection-over-union over the private split, accumulated per
class across every graded scene and averaged over the classes the split contains. It averages over
**classes**, not pixels: a class holding a thousandth of the pixels is worth as much as one holding
a third of them.

`r_det` is zero on any red line and on any knockout failure, before any quality is considered.

## The supervision you get

`environment/data/sparse_labels.npz` is the only annotation that ships. It carries one member per
training image id, on that scene's own grid, with a class id where the annotation labels a pixel and
255 everywhere else. Only a bound fraction of pixels carries a label, and that fraction is authored
for this task rather than taken from any published setting.

The asymmetry is the whole task: **sparse training targets, dense grading**. What you teach the
model to do with a pixel that has no label decides the score.

## What ships

- `environment/data/scenes/` -- the pixels of every scene you may see, the graded ones included.
- `environment/data/sparse_labels.npz` -- the sparse annotation described above.
- `environment/data/graded_scenes.json` -- which image ids are graded and each scene's grid. You are
  told which scenes are graded and at what size; you are never told what is in them.
- `environment/starter/` -- a correct, unoptimised baseline you may replace entirely.

## What your run record must carry

The verifier grades declarations you write, not measurements the harness makes. Write
`/workspace/run_record.json` with:

- `supervision_declaration` -- `{"labelled_pixels_consumed": int, "pixels_total": int}`. The consumed
  count must not exceed the labelled support of the shipped sparse annotation.
- `loss_policy` -- how unlabelled pixels enter the loss. It must name an ignore index or an explicit
  mask over them, and must not name background, void or a zero class as their target.
- `training_schedule` -- `{"planned_steps": int, "completed_steps": int, "decoder_output_stride": int}`.
  A run that stopped early is a truncated run, not a weak model, and this is where that shows.
- `paths_read` -- the list of paths your graded run opened. Red line 1 is checked against this log.
- `guarded_paths_touched` -- paths your run created or modified under the guarded roots.
- `agent_budget_declaration` -- your own budget accounting for the graded run.

A missing declaration is a failed check, not a neutral one.

## The budget

- One graded run. `max_timeout` bounds the refinement loop; each attempt must finish inside its
  per-attempt budget.
- Up to 50 attempts. Your reward history is carried between attempts.
- You must declare your own budget accounting for the graded run; the declaration is what is graded,
  not the harness's own measurement.

## Red lines

Any of these scores zero:

- **Any read of the private split's dense labels.** This forbids the *read*, which is stricter than
  forbidding a write: a run that creates and modifies nothing anywhere and still opens the dense
  masks has broken this red line. It is checked against your own `paths_read` log. The forbidden
  roots are `tests/heldout/` and `environment/data/dense_labels/`.
- Any write into `tests/` during the graded run.

## Traps

- Backpropagates through unlabelled pixels as background and the long tail collapses.
- Raises decoder resolution until the step budget truncates training.
- Submits the sparse annotation's own support and leaves the rest at the ignore index, conceding
  every pixel that was never labelled.
- Resizes every scene to one working resolution and ships masks on that grid rather than on each
  scene's own.
- public ADE20K recipes are abundant and could be transferred wholesale Mitigation: the supervision
  fraction is authored and is not a public setting, so a transferred dense recipe degrades and is
  caught by the ablation ladder
