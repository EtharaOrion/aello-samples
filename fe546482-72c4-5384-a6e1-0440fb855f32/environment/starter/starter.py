"""Correct, unoptimised baseline for AELLO-C8-S6. Replace it entirely if you want.

It exists to prove the delivery path end to end: it reads the shipped indexes, answers EVERY
graded image with one box in that image's own native frame, writes a complete run record, and
declares the digest of its own decision. It is deliberately weak on the graded metric -- the
whole-image box is the CTL-FULLIMAGE control and beating it is what you are being asked to do.

It reads no box it was not given: `labelled_boxes.json` names the fixed small fraction, and that
is the only file it opens boxes from.
"""
import hashlib
import json
import math
import os

SUBMISSION = "/submission/detections.json"
RUN_RECORD = "/workspace/run_record.json"
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
GRADED_IMAGES = os.path.join(DATA, "graded_images.json")
TRAINING_IMAGES = os.path.join(DATA, "training_images.json")
LABELLED_BOXES = os.path.join(DATA, "labelled_boxes.json")


def read_index(path):
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        doc = json.load(handle)
    return doc if isinstance(doc, dict) else {}


def full_frame_boxes(graded):
    """One box per graded image: the whole image, scored 1.0. Weak, well-formed, and answered.

    Every graded image gets an entry because the rate's denominator is the split -- an image left
    out is a miss, and there is nothing to be gained by abstaining.
    """
    detections = {}
    for image in sorted(graded):
        meta = graded[image]
        width = float(meta.get("width", 1))
        height = float(meta.get("height", 1))
        detections[image] = [{"box": [0.0, 0.0, width, height], "score": 1.0}]
    return detections


def decision_digest(detections):
    """sha256 over the top box per image, floored to the pixel -- the digest the record declares."""
    digest = hashlib.sha256()
    for image in sorted(detections):
        boxes = detections[image]
        if not boxes:
            digest.update(("%s|none\n" % image).encode("utf-8"))
            continue
        best = max(record["score"] for record in boxes)
        winners = [record["box"] for record in boxes if record["score"] == best]
        if len(winners) != 1:
            digest.update(("%s|none\n" % image).encode("utf-8"))
            continue
        box = winners[0]
        digest.update(("%s|%d,%d,%d,%d\n" % (image, math.floor(box[0]), math.floor(box[1]),
                                             math.floor(box[2]), math.floor(box[3])))
                      .encode("utf-8"))
    return digest.hexdigest()


def main():
    graded = read_index(GRADED_IMAGES)
    training = read_index(TRAINING_IMAGES)
    labelled = read_index(LABELLED_BOXES)

    detections = full_frame_boxes(graded)
    os.makedirs(os.path.dirname(SUBMISSION), exist_ok=True)
    with open(SUBMISSION, "w") as handle:
        json.dump({"detections": detections}, handle, indent=1, sort_keys=True)

    record = {
        "box_supervised_image_ids": sorted(labelled),
        "labelled_fraction": (len(labelled) / float(len(training))) if training else 0.0,
        "pseudo_box_image_ids": [],
        "pseudo_box_accounting": {"proposed": 0, "retained": 0, "confidence_gate": 0.5},
        "unlabelled_images_treated_as": "ignored",
        "paths_read": [GRADED_IMAGES, TRAINING_IMAGES, LABELLED_BOXES],
        "paths_written": [SUBMISSION, RUN_RECORD],
        "replay_top_box_sha256": decision_digest(detections),
        "agent_budget_declaration": {
            "labelled_images_used": len(labelled),
            "graded_run_seconds": 0.0,
            "inference_scales": [1.0],
            "optimised_for_iou": 0.5,
        },
    }
    os.makedirs(os.path.dirname(RUN_RECORD), exist_ok=True)
    with open(RUN_RECORD, "w") as handle:
        json.dump(record, handle, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
