# AELLO-C8-S12

maximise balanced accuracy on a clean private split when the training labels carry real human noise
and the training-set size is bound.

## What is graded

Write your graded artifact to **`/submission/preds_c8s12.npy`**. It must be an integer label array,
one row per graded instance, in the order the graded images ship. A missing, malformed or
out-of-domain artifact scores 0.

The label domain is fixed:

- class ids **0..99**, the CIFAR-100 fine labels.
- There is no ignore value, no abstention, and no coarse superclass id in the graded domain. A row
  carrying anything but a fine label is out of domain, not a weak prediction.
- The array dtype must be integer. A float array is not rounded into labels for you.

Your score is `r_det`, computed by the verifier in a separate environment from state it measures
itself. The graded quantity is **balanced accuracy** over the clean private split: mean per-class
recall, averaged over the 100 fine classes, where chance is **0.01**. It averages over *classes*,
not over images, so a class you never predict costs you a full hundredth of the score.

`r_det` is zero on any red line and on any knockout failure, before any quality is considered.

## The supervision you get, and why it is not the thing you are scored on

`environment/data/noisy_train_labels.npy` is the only training annotation that ships. It is
CIFAR-100N: **one real human annotation per training image**, not a synthetic corruption. It is
wrong on a large share of the pool, and it is wrong with structure -- the flip rate differs class by
class and the errors concentrate on confusable pairs inside the 20 coarse superclasses, because an
annotator who mislabels a maple tree reaches for another tree. A single symmetric flip rate is not a
coarser model of this noise; it is a different and wrong one.

**The graded split is clean.** The training and graded label distributions therefore differ, and
that gap is the task. Every quantity you can measure on your own data is measured against noisy
targets while the score is taken against clean ones, so training accuracy stops being evidence at
the point where it starts being memorisation.

The **training-set size is bound**. More data is not an available answer. What is left is deciding
which of a fixed pool of images to believe.

## What ships

- `environment/data/train_images.npy` -- the bound training pool's pixels.
- `environment/data/noisy_train_labels.npy` -- the human annotation described above, one row per
  training image. It is the only training label in the image.
- `environment/data/graded_images.npy` -- the pixels of the graded split. You predict on these; you
  are never told what is in them.
- `environment/data/cifar100_coarse_map.json` -- fine label id to coarse superclass id.
- `environment/data/fine_label_names.json` -- the 100 fine class names.
- `environment/starter/` -- a correct, unoptimised baseline you may replace entirely.

## What your run record must carry

The verifier grades declarations you write, not measurements the harness makes. Write
`/workspace/run_record.json` with:

- `selection_declaration` --
  `{"train_pool": int, "retained_indices": [int, ...], "warmup_epochs": int, "selection_signal": str}`.
  `train_pool` must equal the number of rows the shipped annotation carries. `retained_indices` are
  the training indices you kept after selection: unique, non-empty, every one inside the pool.
  `warmup_epochs` must be greater than zero -- selection at step zero selects on initialisation.
  `selection_signal` must name something beyond the training loss (a disagreement between two
  views, a second model, a prediction margin, an agreement history).
- `noise_model_declaration` --
  `{"per_class_flip_rate": [100 floats], "confusion_pairs": [[from, to], ...]}`. The flip rates must
  cover all 100 classes and must not be one scalar repeated. The confusion pairs must land inside a
  coarse superclass more often than a uniform off-diagonal model would.
- `training_outcome` -- `{"noisy_train_accuracy_over_pool": float}`. The accuracy of your final
  model against the **noisy** training labels over the whole pool.
- `paths_read` -- the list of paths your graded run opened. Red line 1 is checked against this log.
- `guarded_paths_touched` -- paths your run created or modified under the guarded roots.

A missing declaration is a failed check, not a neutral one.

## The budget

- One graded run. `max_timeout` bounds the refinement loop; each attempt must finish inside its
  per-attempt budget.
- Up to 50 attempts. Your reward history is carried between attempts.
- You must declare your own budget accounting for the graded run; the declaration is what is graded,
  not the harness's own measurement.

## Red lines

Any of these scores zero:

- **Any read of the clean labels for any training image.** This forbids the *read*, which is
  stricter than forbidding a write: a run that creates and modifies nothing anywhere and still opens
  a clean training label has broken it. The forbidden roots are `tests/heldout/` and
  `environment/data/clean_labels/`, and it is checked against your own `paths_read` log.
- Any write into `tests/` during the graded run.

Separately from the red lines, your retained subset is measured against the verifier's own record of
which human annotations were correct. A selection that is a function of the clean labels reaches
perfect purity, which no learned selector does, and it is rejected. Selecting *well* is the task;
enriching hard for correctly-labelled images is expected and is not what this check looks at.

## Traps

- Selects on training loss alone and keeps exactly the mislabelled easy examples -- the ones an
  over-parameterised network fits first and most confidently.
- Discards so much of the bound pool that it underfits, or drops a fine class outright and forfeits
  its whole recall term.
- Fits the noisy labels past the rate at which the human annotators agreed with the clean ones. Past
  that point every further point of training accuracy is a human error the model learned to repeat.
- Models the noise as one symmetric rate over 100 classes when it concentrates on confusable pairs
  inside the 20 superclasses, and corrects the loss toward the wrong place.
- Predicts with the shape of the noisy training prior rather than of the balanced graded split.
