"""Verifier for AELLO-C8-S13: runs the compiled checks and composes r_det.

Runs inside the separate verifier environment. Measures its own state; never reads a quality
value the agent reports.
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

    out = {"slot_id": 'AELLO-C8-S13', "checks": results, "unmeasured": unmeasured,
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
    """Coverage, plus the three gates a reader of this record asks about first.

    The gate names are this slot's own item ids. A summary that named a gate no item carries
    would report None forever and read as "not measured" rather than as "misspelled", so the
    ids here are the ids solution/grounding.yaml declares.
    """
    covered = sum(1 for r in results if r["passed"])
    gate = lambda name: next((r["passed"] for r in results if r["id"] == name), None)
    worst = next((r["evidence"]["values"].get("worst_cell") for r in results
                  if r["id"] == "per_cell_ece_recorded"), None)
    return {"checks_passed": covered, "checks_total": len(results),
            "layout_gate": gate("stacked_block_layout"),
            "posterior_gate": gate("rows_are_posteriors"),
            "graded_binning_gate": gate("ck_binpin"),
            "worst_cell": worst}


if __name__ == "__main__":
    raise SystemExit(main())
