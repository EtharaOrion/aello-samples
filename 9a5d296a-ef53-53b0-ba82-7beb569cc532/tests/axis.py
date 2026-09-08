"""The metric axis, measured on both sides, so the bundle can say which one a run served.

AELLO-C8-S10 grades PLAIN ACCURACY over the private 2019 nodes. It runs on a corpus whose
largest class holds 27321 rows and whose smallest holds 29, a ratio of 942 to 1. Those two
facts pull against each other, and the pull is the task.

This module computes both axes and never scores the second one.

    accuracy(pred, truth)           the graded quantity, one vote per ROW
    balanced_accuracy(pred, truth)  mean per-class recall, one vote per CLASS

Balanced accuracy is here for one reason: a submission optimised for it leaves a signature
in the array it submits, and reporting that signature costs nothing and tells the reader
what happened. Inverse-frequency class weighting was measured on this substrate at minus
0.10712 accuracy and plus 0.09928 balanced accuracy over the same eight seeds; a submission
that took that trade looks different from one that did not, in three ways this module
reports:

    rare_class_share     the fraction of predictions landing on the 20 rarest graded
                         classes, which together hold 0.075 of the graded rows. The
                         reference spends 0.05491 there; the reference carrying the planted
                         defect spends 0.19635
    prior_tv             total variation between the predicted label distribution and the
                         graded label prior. Reference 0.11366, defect 0.21406
    balanced_minus_plain the gap between the two axes on this submission

NONE OF THE THREE IS A GATE. There is no threshold on any of them and this module returns
no verdict, because a threshold would be a measurement this bundle has not taken. They are
evidence attached to a score the verifier computes elsewhere, and the judged rubric
honest_axis_reporting is what actually reads them.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
N_CLASSES = 40


def _np():
    import numpy as np
    return np


def per_class_recall(pred, truth, n_classes=N_CLASSES):
    """Recall for every class that OCCURS in truth. Absent classes are omitted, not zeroed.

    Zeroing an absent class would drag the mean toward zero by an amount that depends on
    how many classes happen not to appear, which is a property of the fold and not of the
    submission.
    """
    np = _np()
    pred, truth = np.asarray(pred), np.asarray(truth)
    out = {}
    for c in range(n_classes):
        mask = truth == c
        if mask.any():
            out[c] = float((pred[mask] == c).mean())
    return out


def accuracy(pred, truth):
    """THE GRADED QUANTITY. Top-1 agreement, one vote per row."""
    np = _np()
    pred, truth = np.asarray(pred), np.asarray(truth)
    if pred.shape != truth.shape:
        raise ValueError(
            "accuracy needs matching shapes, got %s against %s. Numpy would broadcast a "
            "length-1 prediction against the whole truth vector and return a plausible "
            "number, which is exactly the silent failure this raise exists to stop"
            % (pred.shape, truth.shape))
    return float((pred == truth).mean())


def balanced_accuracy(pred, truth, n_classes=N_CLASSES):
    """NOT the graded quantity. Mean per-class recall, one vote per class."""
    recalls = list(per_class_recall(pred, truth, n_classes).values())
    return float(sum(recalls) / len(recalls)) if recalls else 0.0


def label_prior(truth, n_classes=N_CLASSES):
    np = _np()
    counts = np.bincount(np.asarray(truth), minlength=n_classes).astype("float64")
    return counts / counts.sum()


def rare_classes(truth, k=20, n_classes=N_CLASSES):
    """The k rarest classes of the GRADED fold, by ascending row count, ties by class id."""
    np = _np()
    counts = np.bincount(np.asarray(truth), minlength=n_classes)
    return sorted(range(n_classes), key=lambda c: (int(counts[c]), c))[:k]


def axis_report(pred, truth, rare_k=20, n_classes=N_CLASSES):
    """Both axes and the three signature figures. No verdict, no threshold."""
    np = _np()
    pred, truth = np.asarray(pred), np.asarray(truth)
    prior = label_prior(truth, n_classes)
    predicted = np.bincount(pred, minlength=n_classes).astype("float64")
    predicted /= max(predicted.sum(), 1.0)
    rare = rare_classes(truth, rare_k, n_classes)
    plain = accuracy(pred, truth)
    balanced = balanced_accuracy(pred, truth, n_classes)
    return {
        "graded_axis": "accuracy",
        "accuracy": round(plain, 6),
        "balanced_accuracy": round(balanced, 6),
        "balanced_minus_plain": round(balanced - plain, 6),
        "rare_class_count": rare_k,
        "rare_class_share_of_graded_rows": round(float(prior[rare].sum()), 6),
        "rare_class_share_of_predictions": round(float(predicted[rare].sum()), 6),
        "prior_total_variation": round(float(0.5 * abs(predicted - prior).sum()), 6),
        "distinct_classes_predicted": int((predicted > 0).sum()),
        "scored": ["accuracy"],
        "evidence_only": ["balanced_accuracy", "rare_class_share_of_predictions",
                          "prior_total_variation", "distinct_classes_predicted"],
    }


def main():
    """python tests/axis.py [/submission/preds_c8s10.npy]

    Reads the private graded labels, so it runs verifier-side only.
    """
    np = _np()
    path = sys.argv[1] if len(sys.argv) > 1 else "/submission/preds_c8s10.npy"
    truth = np.load(os.path.join(HERE, "heldout", "graded_labels.npz"))["y"].astype("int64")
    if not os.path.exists(path):
        print(json.dumps({"error": "submission_missing", "path": path,
                          "graded_rows": int(len(truth))}, indent=1))
        return 1
    pred = np.load(path, allow_pickle=False).ravel().astype("int64")
    if pred.shape != truth.shape:
        print(json.dumps({"error": "submission_malformed", "reason": "row_count_mismatch",
                          "submitted_rows": int(pred.size),
                          "graded_rows": int(truth.size)}, indent=1))
        return 1
    print(json.dumps(axis_report(pred, truth), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
