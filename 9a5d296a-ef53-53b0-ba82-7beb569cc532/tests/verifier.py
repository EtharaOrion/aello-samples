"""Verifier for AELLO-C8-S10: runs the compiled checks and composes r_det.

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

    out = {"slot_id": 'AELLO-C8-S10', "checks": results, "unmeasured": unmeasured,
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
    """A dense deliverable reports coverage as a row count against the graded manifest."""
    covered = sum(1 for r in results if r["passed"])
    return {"checks_passed": covered, "checks_total": len(results),
            "shape_gate": next((r["passed"] for r in results
                                if r["id"] == "deliverable_shape"), None)}


if __name__ == "__main__":
    raise SystemExit(main())
