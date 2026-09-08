"""Verifier for AELLO-C8-S8: runs the compiled checks and composes r_det.

Runs inside the separate verifier environment, on CPU and with no network. It holds the private
split's dense masks, which the graded run is forbidden to read, and it measures its own state:
no quality value the agent reports is ever consulted.

While the ramp constants are null the outcome check raises ConstantUnmeasured, and this reports
r_det null with gradeable false. That is not a score of zero. A zero asserts the agent failed;
null asserts this slot is not yet measurable, and collapsing the two is the one error the
null-constant convention exists to prevent.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_output as T  # noqa: E402


def main():
    results, knockout_failed, unmeasured = [], False, []
    for c in T.CHECKS:
        try:
            ok, ev = c["fn"]()
        except T.ConstantUnmeasured as exc:
            unmeasured.append({"id": c["id"], "reason": str(exc)})
            continue
        results.append({"id": c["id"], "passed": bool(ok), "weight": c["weight"],
                        "knockout": c["knockout"], "evidence": ev})
        if c["knockout"] and not ok:
            knockout_failed = True

    out = {"slot_id": 'AELLO-C8-S8', "checks": results, "unmeasured": unmeasured,
           "summary": summarise(results)}
    if unmeasured:
        # Not a score of zero: a score of zero asserts the agent failed. This asserts the SLOT is
        # not yet gradeable, which is a different claim and must not be collapsed into the first.
        out["r_det"] = None
        out["gradeable"] = False
        out["reason"] = "ramp constants are null at Phase 0; see solution/grounding.yaml"
    else:
        out["gradeable"] = True
        out["r_det"] = 0.0 if knockout_failed else None
    json.dump(out, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0



def summarise(results):
    """Report the sparse-supervision facts a reader of this slot needs at a glance.

    Not a generic pass count. The three things that decide a submission here are whether the mask
    set matches the graded scene grids, how much of the grid the agent conceded to the ignore
    sentinel rather than classifying, and how many of the 150 classes the archive distinguishes at
    all. Each is lifted from the check that measured it, so a run that failed for a structural
    reason does not read as a run that merely scored badly.
    """
    values = {}
    for r in results:
        values[r["id"]] = (r.get("evidence") or {}).get("values", {})
    grids = values.get("scene_grid_per_image_id", {})
    conceded = values.get("prediction_dense_despite_sparse_supervision", {})
    return {"checks_passed": sum(1 for r in results if r["passed"]),
            "checks_total": len(results),
            "graded_scenes": grids.get("graded_scenes"),
            "submitted_scenes": grids.get("submitted_scenes"),
            "off_scene_grid": grids.get("off_scene_grid"),
            "conceded_share": conceded.get("conceded_share"),
            "distinct_classes_predicted":
                values.get("long_tail_not_collapsed_to_modal_class", {})
                .get("distinct_classes_predicted"),
            "red_lines": {
                "dense_labels_read":
                    values.get("dense_private_labels_unread", {}).get("private_label_reads"),
                "guarded_paths_written":
                    values.get("checker_tree_and_graded_masks_untouched", {}).get("touched")}}


if __name__ == "__main__":
    raise SystemExit(main())
