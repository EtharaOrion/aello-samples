"""Correct, unoptimised baseline recipe for AELLO-C8-S3. Replace it entirely if you want.

It exists to prove the delivery path end to end: the verifier CALLS build(pool_index, out_dir),
this writes a readable shard manifest under out_dir, and it declares the op sequence it executed.
It is deliberately weak on the graded metric -- uniform sampling with no burst dedup, no
rebalancing toward the rare classes and no illumination-aware statistics -- and that is what you
are being asked to improve. It is not a model; the trainer that consumes this shard is frozen.
"""
import csv
import json
import os

SUBMISSION = '/submission/recipe.py'
MANIFEST = 'shard_manifest.json'
BOUNDARY_FIELD = 'capture_timestamp'


def build(pool_index, out_dir):
    """Select images from the served pool and emit a shard the frozen trainer can read.

    pool_index -- path to the authoritative index: image_id, capture_timestamp, site_id,
                  burst_id, label. Address it by FIELD; the served index is a rotated
                  subsample, so row offsets mean nothing.
    out_dir    -- where the shard and its manifest go.
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(pool_index, newline="") as handle:
        rows = list(csv.DictReader(handle))

    executed = []
    rows = _drop_post_boundary(rows, _boundary(pool_index))
    executed.append("temporal_filter")
    samples = [{"source_id": r["image_id"], "label": r["label"],
                "path": os.path.join("images", r["image_id"] + ".jpg")} for r in rows]
    executed.append("decode")

    manifest = {"samples": samples, "sample_count": len(samples),
                "declared_op_sequence": executed}
    with open(os.path.join(out_dir, MANIFEST), "w") as handle:
        json.dump(manifest, handle, indent=1)
    return manifest


def _boundary(pool_index):
    """The declared boundary ships beside the index. Never guess it from the data."""
    path = os.path.join(os.path.dirname(pool_index), "capture_boundary.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle).get("boundary")


def _drop_post_boundary(rows, boundary):
    """Red line 1. One frame at or after the boundary scores exactly zero, so filter first."""
    if boundary is None:
        return rows
    return [r for r in rows if r[BOUNDARY_FIELD] < boundary]
