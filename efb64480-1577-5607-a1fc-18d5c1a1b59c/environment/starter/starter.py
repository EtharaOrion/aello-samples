"""Correct, unoptimised baseline for AELLO-C8-S8. Replace it entirely if you want.

It exists to prove the delivery path end to end: it reads the sparse annotation that ships,
produces a well-formed mask for every graded scene on that scene's own grid, and writes the run
record the verifier grades. It is deliberately weak -- it learns nothing and predicts the single
class that the sparse annotation labels most often -- and that weakness is the point: it is the
floor the ramp is anchored to, not a model.

What it demonstrates that is worth keeping:

  * it never opens anything under tests/heldout/ or environment/data/dense_labels/, and it logs
    every path it does open, because red line 1 forbids the READ and is checked against that log;
  * it emits a mask for every graded image id at that image's own height and width, because
    ADE20K grids are heterogeneous and a mask on the wrong grid is a missing prediction;
  * it leaves no pixel at the ignore index, because grading is dense while supervision is sparse
    and a conceded pixel scores exactly as a wrong one does.

What it does not do, and what you are being asked to do instead: decide what an unlabelled pixel
means, get a target onto it from somewhere other than the labels, and reach the long tail.
"""
import json
import os

import numpy as np

SUBMISSION = '/submission/masks.npz'
DATA = '/workspace/environment/data'
RECORD = '/workspace/run_record.json'
IGNORE = 255
NUM_CLASSES = 150

# Every path this baseline opens, appended as it opens it. Red line 1 is checked against this log,
# so the log is built by the code that does the reading rather than reconstructed afterwards.
READS = []


def read(path):
    READS.append(path)
    return path


def graded_scenes():
    """Image ids to predict and each scene's own (height, width). Agent-visible by design."""
    path = os.path.join(DATA, "graded_scenes.json")
    if not os.path.exists(path):
        return {}
    with open(read(path)) as handle:
        return {str(k): (int(v[0]), int(v[1])) for k, v in json.load(handle).items()}


def sparse_annotation():
    """The only annotation that ships: a class id on a bound fraction of pixels, 255 elsewhere."""
    path = os.path.join(DATA, "sparse_labels.npz")
    if not os.path.exists(path):
        return None
    return np.load(read(path), allow_pickle=False)


def modal_class_and_support(annotation):
    """The most frequently labelled class, and how many labelled pixels ship in total.

    Counted over the labelled support only. Counting the ignore index would make 255 the modal
    value everywhere, which is the same mistake as treating unlabelled pixels as a class.
    """
    counts = np.zeros(NUM_CLASSES, dtype="int64")
    support = 0
    if annotation is not None:
        for name in annotation.files:
            labels = np.asarray(annotation[name])
            labelled = labels[labels != IGNORE]
            support += int(labelled.size)
            if labelled.size:
                counts += np.bincount(labelled.ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
    return int(counts.argmax()) if counts.any() else 0, support


def main():
    os.makedirs(os.path.dirname(SUBMISSION), exist_ok=True)
    annotation = sparse_annotation()
    modal, support = modal_class_and_support(annotation)
    scenes = graded_scenes()

    # One dense mask per graded scene, on that scene's own grid, entirely inside 0..149.
    masks = {name: np.full(shape, modal, dtype="int16") for name, shape in scenes.items()}
    if not masks:
        # The shard has not landed. Emit one well-formed member so the delivery path is still
        # exercised; it will fail scene_grid_per_image_id, which is the honest report.
        masks = {"scene_0": np.full((2, 2), modal, dtype="int16")}
    np.savez(SUBMISSION, **masks)

    with open(RECORD, "w") as handle:
        json.dump({
            "agent_budget_declaration": {"attempts_used": 1, "seconds_declared": 0.0},
            "supervision_declaration": {
                # This baseline reads every labelled pixel once and trains on none of them.
                "labelled_pixels_consumed": support,
                "pixels_total": int(sum(h * w for h, w in scenes.values())) or 4,
            },
            "loss_policy": "no training performed; unlabelled pixels carry ignore_index 255 and "
                           "are masked out of every loss this baseline does not compute",
            "training_schedule": {"planned_steps": 0, "completed_steps": 0,
                                  "decoder_output_stride": 1},
            "paths_read": list(READS),
            "guarded_paths_touched": [],
        }, handle, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
