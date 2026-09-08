"""Correct, unoptimised baseline for AELLO-C8-S12. Replace it entirely if you want.

It exists to prove the delivery path end to end: it reads the human annotation that ships, emits a
well-formed integer label for every graded image, and writes the run record the verifier grades. It
is deliberately weak -- it learns nothing and predicts the single class the human annotators used
most often -- and that weakness is the point: it is the floor the ramp is anchored to, not a model.

What it demonstrates that is worth keeping:

  * it never opens anything under tests/heldout/ or environment/data/clean_labels/, and it logs
    every path it does open, because red line 1 forbids the READ and is checked against that log;
  * it emits exactly one label per graded image, every one inside 0..99;
  * it reports its accuracy against the NOISY training labels honestly. A constant predictor scores
    about a hundredth there, far under the rate at which the annotators agreed with the clean label,
    which is the ceiling a model that learned the clean concept can reach.

What it deliberately does NOT do, and what you are being asked to do instead:

  * it performs no selection at all -- it retains the whole bound pool, which is the CTL-ALLDATA
    control, so `warmup_epochs` is 0 and `selection_signal` names nothing beyond the loss. It fails
    `selection_begins_after_warmup_not_at_step_zero` for that reason, and the failure is the honest
    report of a baseline that selects nothing rather than a defect to paper over;
  * it omits `noise_model_declaration` entirely. Estimating a class-conditional transition over
    real human noise is the task, and declaring a symmetric rate here to make a check pass would be
    shipping the wrong model as the starting point.
"""
import json
import os

import numpy as np

SUBMISSION = '/submission/preds_c8s12.npy'
DATA = '/workspace/environment/data'
RECORD = '/workspace/run_record.json'
NUM_CLASSES = 100

# Every path this baseline opens, appended as it opens it. Red line 1 is checked against this log,
# so the log is built by the code that does the reading rather than reconstructed afterwards.
READS = []


def read(path):
    READS.append(path)
    return path


def load(name):
    """A shipped corpus array, or None when the shard has not landed."""
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        return None
    return np.load(read(path), allow_pickle=False)


def main():
    os.makedirs(os.path.dirname(SUBMISSION), exist_ok=True)
    noisy = load("noisy_train_labels.npy")
    graded = load("graded_images.npy")

    # The class the human annotators reached for most often. Counted over the noisy labels, which
    # are the only training labels in this image; nothing here consults a clean one.
    if noisy is None:
        modal, pool = 0, 0
        noisy_train_accuracy = 0.0
    else:
        flat = np.asarray(noisy).ravel().astype("int64")
        pool = int(flat.size)
        counts = np.bincount(flat, minlength=NUM_CLASSES)[:NUM_CLASSES]
        modal = int(counts.argmax())
        noisy_train_accuracy = float(counts[modal] / flat.size) if flat.size else 0.0

    # One label per graded image, every one inside the fine label domain. When the shard has not
    # landed there is nothing to size the vector against, so a single row ships and
    # one_label_per_graded_image fails, which is the honest report.
    rows = int(np.asarray(graded).shape[0]) if graded is not None else 1
    np.save(SUBMISSION, np.full((rows,), modal, dtype="int64"))

    with open(RECORD, "w") as handle:
        json.dump({
            "agent_budget_declaration": {"attempts_used": 1, "seconds_declared": 0.0},
            "selection_declaration": {
                # No selection: the whole bound pool is retained, which is CTL-ALLDATA.
                "train_pool": pool,
                "retained_indices": list(range(pool)),
                "warmup_epochs": 0,
                "selection_signal": "none; this baseline performs no selection",
            },
            "training_outcome": {"noisy_train_accuracy_over_pool": noisy_train_accuracy},
            "paths_read": list(READS),
            "guarded_paths_touched": [],
        }, handle, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
