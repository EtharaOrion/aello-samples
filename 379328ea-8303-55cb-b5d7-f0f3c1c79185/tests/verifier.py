"""Verifier for C3-S2: runs the compiled curation checks and composes r_det.

Runs inside the separate verifier environment -- no network, no GPU, fresh checkout. It measures
its own state. The two bits-per-byte values come from the frozen evaluator's own logits over the
verifier's own held-out bytes, the token count from its own pass under the frozen tokenizer, and
the 13-gram containment from its own scan over the full emitted corpus. No number the agent
reports enters the reward.

CHK-B9 bounds what may cross back to the agent: exactly one float and at most one reason code,
with no per-stage, per-operator or per-shard diagnostic. The per-check evidence below is the
AUTHORING view and stays on this side of that boundary; `closure()` composes the payload that
does not.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_output as T  # noqa: E402

SLOT_ID = "C3-S2"


def red_line_reason(results):
    """The ONE reason slug that crosses the CHK-B9 boundary.

    Ordered by the contract's own zero list rather than by check id: a corpus that both leaks a
    held-out 13-gram and misses a wall reports eval-leak, the failure that made the run
    inadmissible, not whichever check happened to sort first. A reason a check names that is not
    in the closed set is reported only after every closed-set slug has been ruled out.
    """
    named = {r["evidence"]["values"].get("reason") for r in results if not r["passed"]}
    for slug in T.ZERO_REASONS:
        if slug in named:
            return slug
    outside = sorted(s for s in named if s)
    return outside[0] if outside else None


def closure(r_det, reason):
    """The agent-visible payload. One float, at most one reason code, nothing else."""
    payload = {"reward": r_det}
    if reason is not None:
        payload["reason"] = reason
    return payload


def summarise(results):
    """Which keyed gates fired, not only how many: the checks name distinct red lines."""
    failed = [r["id"] for r in results if not r["passed"]]
    return {"checks_passed": len(results) - len(failed), "checks_total": len(results),
            "failed_ids": sorted(failed)}


def main():
    results, knockout_failed, unmeasured = [], False, []
    for c in T.CHECKS:
        try:
            ok, ev = c["fn"]()
        # ConstantUnmeasured is caught HERE, ahead of any general handler. Routed into a generic
        # branch instead, an unmeasured constant reads as an ordinary failure and the slot
        # reports a measured zero it never measured.
        except T.ConstantUnmeasured as exc:
            unmeasured.append({"id": c["id"], "reason": str(exc)})
            continue
        results.append({"id": c["id"], "passed": bool(ok), "weight": c["weight"],
                        "knockout": c["knockout"], "evidence": ev})
        if c["knockout"] and not ok:
            knockout_failed = True

    reason = red_line_reason(results)
    out = {"slot_id": SLOT_ID, "checks": results, "unmeasured": unmeasured,
           "summary": summarise(results)}
    if unmeasured:
        # Not a score of zero: a score of zero asserts the agent failed. This asserts the SLOT is
        # not yet gradeable, which is a different claim and must not be collapsed into the first.
        # No reason slug crosses either -- naming eval-leak here would report a red line on a run
        # whose reward was never computed.
        out["r_det"] = None
        out["gradeable"] = False
        out["reason"] = "ramp constants are null at Phase 0; see solution/grounding.yaml"
        out["agent_visible_closure"] = closure(None, None)
    else:
        out["gradeable"] = True
        out["r_det"] = 0.0 if knockout_failed else None
        out["agent_visible_closure"] = closure(out["r_det"], reason)
    json.dump(out, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
