"""Verifier for AELLO-C6-S2: runs the compiled checks and composes r_det.

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

    out = {"slot_id": 'AELLO-C6-S2', "checks": results, "unmeasured": unmeasured,
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
    """Coverage, plus the two gates that decide whether anything downstream is meaningful.

    coupling_gate asks whether the two emitted outputs agree with each other, because a submission
    whose index contradicts its own label vector is not a weaker answer to this task, it is a
    different artifact -- every rate computed over it would describe a deliverable nobody asked
    for. calibration_gate asks whether the middle class survived, because macro-F1 over three
    classes and macro-F1 over two wearing the same name are not comparable numbers.

    Both are reported as None when their check did not run, never as False. A summary that reports
    a gate it did not measure as a failure is a measurement the verifier did not make.
    """
    failed = [r["id"] for r in results if not r["passed"]]

    def gate(check_id):
        return next((r["passed"] for r in results if r["id"] == check_id), None)

    return {"checks_passed": len(results) - len(failed), "checks_total": len(results),
            "coupling_gate": gate("first_error_index_equals_first_bad_label_in_its_own_vector"),
            "calibration_gate": gate("per_stratum_neutral_rate_inside_the_band"),
            "failed_ids": sorted(failed)}


if __name__ == "__main__":
    raise SystemExit(main())
