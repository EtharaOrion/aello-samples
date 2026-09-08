"""Verifier for C4-S3: runs the compiled checks and composes r_det.

Runs inside the separate verifier environment, CPU-only and with no network. It measures its own
state: the average bits per weight comes from the size of the shard files on disk, the throughput
comes from the harness trial table, and the recipe wall clock comes from the harness clock. No
quantity the recipe reports about itself is read as a measurement of that quantity.
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

    out = {"slot_id": 'C4-S3', "checks": results, "unmeasured": unmeasured,
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
    """A red-lined slot reports WHICH gate fired, not only how many checks failed.

    Every knockout here carries a machine-readable reason -- bit_budget_exceeded,
    calibration_budget_exceeded, format_envelope_violation, calibration_overfit,
    recipe_order_divergence, recipe_export_divergence, unquantized_layer_served, heldout_read,
    deliverable_absent -- and r_det is zero on any one of them before any speed is considered, so
    the reason is the finding and the count is the footnote.
    """
    failed = [r for r in results if not r["passed"]]
    reasons = {}
    for r in failed:
        reason = (r.get("evidence") or {}).get("values", {}).get("reason")
        if reason:
            reasons[r["id"]] = reason
    return {"checks_passed": len(results) - len(failed), "checks_total": len(results),
            "knockouts_failed": sorted(r["id"] for r in failed if r["knockout"]),
            "reasons": reasons}


if __name__ == "__main__":
    raise SystemExit(main())
