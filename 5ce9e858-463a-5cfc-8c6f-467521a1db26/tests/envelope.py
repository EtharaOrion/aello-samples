"""Audit the three budgets this slot binds, and report the one that does not currently close.

THREE DISTINCT QUANTITIES, never aliases of one another. Conflating them is the failure this
module exists to prevent, and task.toml carries the same warning at the point of declaration:

  budget_hours          a per-ATTEMPT completion bound. What one attempt must finish inside.
  max_timeout           the refinement loop's terminator ACROSS attempts. An authored pacing
                        choice, not a measurement of anything.
  lookback              a per-FORECAST conditioning bound. How far back in the covariate stream
                        a prediction for a given row may reach. This is the only one of the
                        three that is a property of the task rather than of the schedule, and it
                        is the only one this module can check against a submission.

WHAT IT CHECKS.

  1. The submitted conditioning manifest declares a lookback at or under the bound, and declares
     that no fit row was drawn from at or beyond the graded window's first row.
  2. Every per-block entry sits inside the block it describes. This is the manifest half of the
     causality rule; the compiled checker enforces the same relation and this module is the tool
     an agent can run on itself before submitting.
  3. The curve-admissibility ratio, max_timeout_seconds / per_attempt_seconds, which the
     contract's desk_checks require to reach at least 25.

WHAT IT REPORTS RATHER THAN FIXES. Check 3 does not currently pass. task.toml declares
max_timeout 8.0 hours and per_attempt_seconds 1800.0, which is a ratio of 16.0 against a
required 25. Both numbers are authored pacing choices rather than measurements, so the shortfall
could be closed by moving either of them, and this module deliberately does not: raising
max_timeout to 12.5 hours or cutting per_attempt_seconds to 1152 would each produce a compliant
ratio and neither would be evidence about anything. The shortfall is a scheduling decision that
belongs to the wave that measures per_attempt_seconds on the grading host, and it is recorded as
an open residual in solution/grounding.yaml under residuals rather than closed by arithmetic
here. The module returns a non-zero exit for it, so it stays visible.

    python tests/envelope.py                 # budgets only
    python tests/envelope.py /submission     # additionally audit a submitted manifest
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BUNDLE, "environment"))
import ettm1_window as W  # noqa: E402

REQUIRED_CURVE_RATIO = 25.0
MANIFEST_NAME = "conditioning.json"


def task_budgets():
    """Read the three declared budgets out of task.toml without importing a TOML parser.

    The delivered verifier image installs numpy and pyyaml and nothing else, and Python 3.11
    carries tomllib while 3.10 does not; this module has to run under whichever interpreter the
    reader has, so the three scalars are lifted by name. They are simple `key = number` lines
    at the top level of the [metadata] table, and a value this parser cannot find is reported as
    missing rather than guessed.
    """
    wanted = {"max_timeout": None, "budget_hours": None, "per_attempt_seconds": None,
              "max_attempts": None}
    path = os.path.join(BUNDLE, "task.toml")
    if not os.path.exists(path):
        return wanted
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            key = key.strip()
            if key in wanted and wanted[key] is None:
                try:
                    wanted[key] = float(value.strip())
                except ValueError:
                    pass
    return wanted


def curve_admissibility(budgets):
    """max_timeout_seconds / per_attempt_seconds, against the contract's floor of 25."""
    timeout, attempt = budgets.get("max_timeout"), budgets.get("per_attempt_seconds")
    if timeout is None or not attempt:
        return {"ratio": None, "required": REQUIRED_CURVE_RATIO, "ok": False,
                "reason": "budgets_unreadable"}
    ratio = (timeout * 3600.0) / attempt
    return {"ratio": round(ratio, 6), "required": REQUIRED_CURVE_RATIO,
            "max_timeout_seconds": timeout * 3600.0, "per_attempt_seconds": attempt,
            "ok": ratio >= REQUIRED_CURVE_RATIO,
            "reason": None if ratio >= REQUIRED_CURVE_RATIO else "curve_ratio_below_25"}


def read_manifest(submission):
    path = os.path.join(submission, MANIFEST_NAME)
    if not os.path.exists(path):
        return None, "manifest_missing"
    try:
        with open(path) as handle:
            return json.load(handle), None
    except Exception as error:
        return None, "manifest_unparseable:%s" % (error,)


def audit_manifest(manifest):
    """Every way a conditioning declaration can be wrong, named separately."""
    problems = []
    rows = manifest.get("block_last_context_row")
    if not isinstance(rows, list):
        problems.append("block_last_context_row_absent")
    elif len(rows) != W.BLOCKS:
        problems.append("block_count_%d_not_%d" % (len(rows), W.BLOCKS))
    else:
        offenders = [[b, r] for b, r in enumerate(rows)
                     if not isinstance(r, int) or isinstance(r, bool)
                     or not W.manifest_row_admissible(b, r)]
        if offenders:
            problems.append("blocks_reaching_beyond_their_own_horizon:%s" % (offenders[:8],))
    lookback = manifest.get("lookback_rows")
    if not isinstance(lookback, int) or isinstance(lookback, bool) or lookback < 0:
        problems.append("lookback_rows_absent_or_negative")
    elif lookback > W.LOOKBACK:
        problems.append("lookback_%d_over_budget_%d" % (lookback, W.LOOKBACK))
    end = manifest.get("fit_rows_end")
    if not isinstance(end, int) or isinstance(end, bool):
        problems.append("fit_rows_end_absent")
    elif end > W.HISTORY_ROWS:
        problems.append("fit_rows_end_%d_reaches_into_the_graded_window" % (end,))
    return problems


def main():
    budgets = task_budgets()
    report = {"budgets": budgets, "lookback_budget_rows": W.LOOKBACK,
              "horizon_rows": W.HORIZON, "blocks": W.BLOCKS,
              "curve_admissibility": curve_admissibility(budgets)}
    failures = [] if report["curve_admissibility"]["ok"] else ["curve_admissibility"]
    if len(sys.argv) > 1:
        manifest, error = read_manifest(sys.argv[1])
        if manifest is None:
            report["manifest"] = {"ok": False, "problems": [error]}
            failures.append("manifest")
        else:
            problems = audit_manifest(manifest)
            report["manifest"] = {"ok": not problems, "problems": problems,
                                  "declared_lookback": manifest.get("lookback_rows"),
                                  "declared_fit_rows_end": manifest.get("fit_rows_end")}
            if problems:
                failures.append("manifest")
    report["failures"] = failures
    print(json.dumps(report, indent=1, sort_keys=True))
    if failures:
        print("OPEN: %s" % ", ".join(failures), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
