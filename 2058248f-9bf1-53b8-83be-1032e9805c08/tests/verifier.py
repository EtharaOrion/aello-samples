"""Verifier for A5-02: runs the compiled checks and composes r_det.

Runs inside the separate verifier environment with no network and no GPU. Every quantity it
reports it measured itself over frozen bytes; nothing here reads a macro average precision, a
latency or a byte count that the agent declared as its score.

The report separates three states that a single number would collapse. A check that FAILED
examined the submission and found it wanting. A check that is UNMEASURED could not examine it,
because a Phase 1 constant is still null or because this executor cannot reach a pinned
dependency. A KNOCKOUT failure zeroes r_det outright, before any quality is considered. Only the
first of the three is a statement about the agent.
"""
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_output as checks  # noqa: E402

# The three deployment constraints gate admissibility rather than quality, so their checks are
# reported apart from the rest: headroom on any of them earns nothing and a violation of any one
# of them is worth exactly as much as a violation of all three.
ADMISSIBILITY_GATES = ("activation_high_water_taken_at_batch_64",
                       "analytic_latency_recomputes_from_graph_terms",
                       "serialized_parameter_bytes_recounted_from_the_blob")
# The two assertions that decide whether the submitted object is the object the metric reads.
SCORE_MATRIX_GATES = ("score_columns_span_the_nineteen_labels",
                      "scores_are_continuous_not_thresholded")


def run_one(entry):
    ok, evidence = entry["fn"]()
    return {"id": entry["id"], "passed": bool(ok), "weight": entry["weight"],
            "knockout": entry["knockout"], "evaluation_target": entry["target"],
            "evidence": evidence}


def refusal_reason(rows):
    """The first machine-readable zero reason any failing check produced, or None.

    The red lines are loud by design: a violated deployment constraint carries its own code from
    the closed set and is never a silent grade-down.
    """
    for row in rows:
        if row["passed"]:
            continue
        code = (row["evidence"].get("values") or {}).get("reason")
        if code in checks.ZERO_REASONS:
            return code
    return None


def summarise(rows):
    by_id = dict((row["id"], row["passed"]) for row in rows)
    return {
        "checks_passed": sum(1 for row in rows if row["passed"]),
        "checks_total": len(rows),
        "admissibility_gates": dict((name, by_id.get(name)) for name in ADMISSIBILITY_GATES),
        "score_matrix_gates": dict((name, by_id.get(name)) for name in SCORE_MATRIX_GATES),
        "knockouts_failed": sorted(row["id"] for row in rows
                                   if row["knockout"] and not row["passed"]),
    }


def main():
    rows, unmeasured = [], []
    for entry in checks.CHECKS:
        try:
            rows.append(run_one(entry))
        except checks.ConstantUnmeasured as exc:
            unmeasured.append({"id": entry["id"], "reason": str(exc)})
    report = {"slot_id": "A5-02", "checks": rows, "unmeasured": unmeasured,
              "summary": summarise(rows), "zero_reason": refusal_reason(rows)}
    if unmeasured:
        # NOT a score of zero. Zero asserts the agent failed; this asserts the SLOT is not yet
        # measurable, which is a different claim and must not be collapsed into the first.
        report["gradeable"] = False
        report["r_det"] = None
        report["reason"] = ("the two macro average precision bands and the three analytic host "
                            "constants are null at Phase 0; see solution/grounding.yaml")
    else:
        report["gradeable"] = True
        report["r_det"] = 0.0 if report["summary"]["knockouts_failed"] else None
    json.dump(report, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
