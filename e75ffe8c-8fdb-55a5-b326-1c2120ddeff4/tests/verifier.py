"""Verifier for A5-03: runs the compiled checks and composes r_det.

Runs inside the separate verifier environment, CPU-only and network-isolated. Measures its own
state: it loads the delivered export, runs it over the private audio it holds, draws this
attempt's causality probe offsets, and computes word error rate itself under the pinned text
normalizer. The only agent-authored quantity it reads at all is reward_log.jsonl, and it reads
that solely to contradict it.
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

    out = {"slot_id": 'A5-03', "checks": results, "unmeasured": unmeasured,
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
    """Coverage, plus the three gates that decide whether anything downstream is meaningful.

    `deliverable_shape` was retired with the prediction-array deliverable, and a summary that
    still named a check no longer in CHECKS reported None forever while looking like a
    measurement. The questions that replace it are this slot's own: did an export arrive that the
    verifier could load, did it stay causal under perturbation, and did it fit the byte ceiling.
    """
    covered = sum(1 for r in results if r["passed"])

    def gate(check_id):
        return next((r["passed"] for r in results if r["id"] == check_id), None)

    return {"checks_passed": covered, "checks_total": len(results),
            "export_gate": gate("export_manifest_binds_parameters_frontend_and_decode"),
            "streaming_gate": gate("chk_causality"),
            "parameter_ceiling_gate": gate("chk_param_bytes")}


if __name__ == "__main__":
    raise SystemExit(main())
