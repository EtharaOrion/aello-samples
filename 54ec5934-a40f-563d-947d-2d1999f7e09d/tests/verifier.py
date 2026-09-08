"""Verifier for C7-S2: runs the compiled checks and composes r_det.

Runs inside the separate verifier environment. Every quantity it reports it measured itself:
the private-split success rate comes from WebArena's own evaluators over verifier-owned site
fixtures, and the per-instance budget numbers come from host-side per-episode metering. Nothing
here reads a success, a latency or an action count that the scaffold reported about itself.
"""
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_output as T  # noqa: E402

SLOT_ID = 'C7-S2'


def main():
    results, knockout_failed, unmeasured = [], False, []
    for c in T.CHECKS:
        try:
            ok, ev = c["fn"]()
        except T.ConstantUnmeasured as exc:
            unmeasured.append({"id": c["id"], "reason": str(exc)})
            continue
        results.append({"id": c["id"], "passed": bool(ok), "weight": c["weight"],
                        "kind": c["kind"], "knockout": c["knockout"], "evidence": ev})
        if c["knockout"] and not ok:
            knockout_failed = True

    out = {"slot_id": SLOT_ID, "checks": results, "unmeasured": unmeasured,
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
    """Report which gate fired, not only how many.

    The three failures this slot exists to separate are answered separately: an episode set the
    external evaluator could not read, a per-instance budget the run did not hold, and a fixture
    or grader byte that moved. Collapsing them into one count loses the diagnosis.
    """
    failed = sorted(r["id"] for r in results if not r["passed"])
    return {"checks_passed": len(results) - len(failed), "checks_total": len(results),
            "knockouts_failed": sorted(r["id"] for r in results
                                       if not r["passed"] and r["knockout"]),
            "evaluator_legibility": sorted(i for i in failed if "evaluator" in i or "legible" in i),
            "per_instance_budget": sorted(i for i in failed if "budget" in i),
            "fixture_and_grader_integrity": sorted(i for i in failed
                                                   if i in ("reset_metered_out_of_graded_clock",
                                                            "external_grader_bytes_unmodified",
                                                            "target_site_within_retained_roster")),
            "failed_ids": failed}


if __name__ == "__main__":
    raise SystemExit(main())
