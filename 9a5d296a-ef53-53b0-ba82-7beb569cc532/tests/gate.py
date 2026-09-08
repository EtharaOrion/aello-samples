"""Compose the judged conjunction onto r_det -- and refuse to, while it cannot be done.

WHAT THIS BUNDLE EMITS AND WHAT IT DOES NOT. tests/verifier.py writes r_det from the
compiled checks and the measured accuracy alone. It never sees a judged verdict: the six
trajectory rubrics in tests/rubrics.jsonl are graded downstream, outside this bundle, after
the verifier has exited. So the bundle cannot apply the gate itself, and this module is the
composition the CONSUMER applies once those verdicts exist.

THE RULE, bound in tests/constants.json:

    every enumerated judged rubric passes  ->  r_final = r_det
    anything else                          ->  r_final = min(r_det, reward_gate_ceiling)

"Anything else" is deliberately wide, and COVERAGE is part of it. The judged set this bundle
expects is enumerated in tests/rubrics.json, so the expected ids are available at the point
of use. A verdict collection that does not cover them -- empty, short, missing an id, or
uncheckable because the enumeration itself will not parse -- is a FAILURE of the conjunction
and caps. That is the case a broken grader actually produces, and all() over an empty
collection is True, so a gate that only inspected the verdicts it was handed would pass
exactly when it had been handed none.

WHY THIS MODULE CURRENTLY REFUSES. A council emits DISCRETE FRACTIONS averaged over a panel,
not booleans, while the conjunction consumes booleans. reward_gate_pass_threshold is the
conversion, and in this bundle it is NULL, because it is one of the four constants Phase 0
does not author. There is a tempting default -- 0.5 -- and taking it would be the wrong
thing to do: the same bundle and the same council fractions would then compose to different
rewards depending on who ran the composition and what their environment happened to hold.
So compose() raises ConstantUnmeasured while the threshold is null, and the caller learns
that the composition is not yet defined rather than receiving a number that looks defined.

Boolean verdicts are a different case and are handled: a collection of real booleans needs
no conversion, so compose() completes on them even while the threshold is null. That is not
a loophole. It is the exact boundary of what these bytes can settle.

    python tests/gate.py --r-det 0.72 --verdicts pass,pass,fail
    python tests/gate.py --r-det 0.72 --verdicts-json verdicts.json
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


class ConstantUnmeasured(Exception):
    """Raised when the composition needs a constant Phase 0 deliberately left null."""


def bound_gate():
    """(form, ceiling, pass_threshold). The threshold may be None and that is meaningful."""
    with open(os.path.join(HERE, "constants.json")) as handle:
        constants = json.load(handle)
    threshold = constants.get("reward_gate_pass_threshold")
    return (constants["reward_gate_form"], float(constants["reward_gate_ceiling"]),
            None if threshold is None else float(threshold))


def enumerated_rubrics():
    """The judged rubric ids this bundle enumerates. None when it cannot be established.

    A gate that cannot learn what it is required to cover must not report a pass, so the
    unreadable case returns None and the caller treats it as a failure.
    """
    try:
        with open(os.path.join(HERE, "rubrics.json")) as handle:
            ids = [item["id"] for item in json.load(handle)["rubrics"]]
        return ids or None
    except Exception:
        return None


def coverage(verdicts, expected=None):
    """Resolve a verdict collection against the enumerated judged set.

    Returns (values, covered, reason). `verdicts` may map rubric id to verdict, or be a
    sequence in the enumerated order. Coverage is settled before any verdict VALUE is read.
    """
    expected = enumerated_rubrics() if expected is None else expected
    if not expected:
        return [], False, "judged_set_unreadable"
    if hasattr(verdicts, "keys"):
        present = {str(k) for k in verdicts.keys()}
        missing = [i for i in expected if i not in present]
        values = [verdicts.get(i) for i in expected]
        if missing:
            return values, False, "verdicts_missing_for:%s" % ",".join(missing)
        return values, True, "covered"
    values = list(verdicts)
    if len(values) < len(expected):
        return values, False, ("verdict_count_%d_below_enumerated_%d"
                               % (len(values), len(expected)))
    return values, True, "covered"


def passes(value, threshold):
    """Is one judged verdict a pass? None is unavailable and fails closed.

    A real boolean needs no threshold. A fraction does, and raises while it is unmeasured.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    if threshold is None:
        raise ConstantUnmeasured(
            "AELLO-C8-S10: reward_gate_pass_threshold is null, so the council fraction %r "
            "cannot be converted to a pass or a fail. Phase 0 authors no measurement; the "
            "Phase 2 measurement wave writes this constant and the composition becomes "
            "defined then. Defaulting to 0.5 here would make the same bundle and the same "
            "verdicts compose to different rewards for different readers." % (value,))
    return numeric >= threshold


def compose(r_det, verdicts, expected=None, form=None, ceiling=None, threshold=None):
    """Apply the judged residue to a deterministic reward. It can never raise the value."""
    bound_form, bound_ceiling, bound_threshold = bound_gate()
    form = bound_form if form is None else form
    ceiling = bound_ceiling if ceiling is None else ceiling
    threshold = bound_threshold if threshold is None else threshold
    if not 0.0 <= r_det <= 1.0:
        raise ValueError("deterministic reward %r is outside the closed unit interval" % (r_det,))
    if form != "gate":
        raise ValueError("this bundle binds the gate form, not %r" % (form,))
    if not 0.0 <= ceiling < 1.0:
        raise ValueError(
            "gate ceiling %r must sit strictly below 1.0, otherwise a failed rubric leaves "
            "full reward reachable and the gate does not bind" % (ceiling,))
    if threshold is not None and not 0.0 < threshold <= 1.0:
        raise ValueError("pass threshold %r must sit in (0, 1]" % (threshold,))
    values, covered, _ = coverage(verdicts, expected)
    all_pass = covered and all(passes(v, threshold) for v in values)
    r_final = r_det if all_pass else min(r_det, ceiling)
    if r_final > r_det:                      # checked, not argued
        raise ValueError("the judged residue raised the reward from %r to %r" % (r_det, r_final))
    return r_final


def parse_token(token):
    text = str(token).strip().lower()
    if text in ("pass", "true", "1", "yes"):
        return True
    if text in ("fail", "false", "0", "no"):
        return False
    try:
        return float(text)
    except ValueError:
        return None                          # unavailable: fail-closed


def main():
    parser = argparse.ArgumentParser(
        description="Apply this bundle's bound judged-rubric gate to a deterministic reward.")
    parser.add_argument("--r-det", type=float, required=True,
                        help="the deterministic reward tests/verifier.py wrote")
    parser.add_argument("--verdicts", default="",
                        help="comma-separated judged verdicts: pass, fail, a fraction, or absent")
    parser.add_argument("--verdicts-json", default=None,
                        help="a JSON file mapping rubric id to verdict, or a JSON list")
    args = parser.parse_args()
    expected = enumerated_rubrics()
    if args.verdicts_json:
        with open(args.verdicts_json) as handle:
            document = json.load(handle)
        raw = document if isinstance(document, dict) else list(document)
    else:
        raw = [t for t in args.verdicts.split(",") if t.strip()]
    if hasattr(raw, "keys"):
        verdicts = {k: parse_token(v) for k, v in raw.items()}
    else:
        verdicts = [parse_token(v) for v in raw]
    values, covered, reason = coverage(verdicts, expected)
    form, ceiling, threshold = bound_gate()
    report = {"r_det": args.r_det, "enumerated_judged": len(expected or []),
              "judged_verdicts": len(values), "covered": covered, "reason": reason,
              "form": form, "ceiling": ceiling, "pass_threshold": threshold}
    try:
        report["r_final"] = compose(args.r_det, verdicts, expected=expected)
        report["composed"] = True
    except ConstantUnmeasured as exc:
        report["composed"] = False
        report["r_final"] = None
        report["refusal"] = str(exc)
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if report["composed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
