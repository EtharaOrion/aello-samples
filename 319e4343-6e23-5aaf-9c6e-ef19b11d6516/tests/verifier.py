"""Verifier for AELLO-C8-S4: runs the compiled checks and composes r_det.

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
            # MUST be caught before the general handler. Python takes the first matching clause,
            # so ordering these the other way round routed an unmeasured constant into the generic
            # branch, and re-raising from inside it propagated out of the try entirely rather than
            # falling through to this one.
            unmeasured.append({"id": c["id"], "reason": str(exc)})
            continue
        except Exception as exc:
            # One broken or inapplicable check must not take the whole grading run down. A check
            # that raises is a FAILED check with its reason recorded, not a crashed verifier: this
            # slot's per-class recall raises FileNotFoundError when no submission exists, which is
            # exactly the case deliverable_present is there to report.
            results.append({"id": c["id"], "passed": False, "weight": c["weight"],
                            "knockout": c["knockout"],
                            "evidence": {"values": {"reason": "check_raised",
                                                    "error": "%s: %s" % (type(exc).__name__, exc)}}})
            if c["knockout"]:
                knockout_failed = True
            continue
        results.append({"id": c["id"], "passed": bool(ok), "weight": c["weight"],
                        "knockout": c["knockout"], "evidence": ev})
        if c["knockout"] and not ok:
            knockout_failed = True

    out = {"slot_id": 'AELLO-C8-S4', "checks": results, "unmeasured": unmeasured,
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
