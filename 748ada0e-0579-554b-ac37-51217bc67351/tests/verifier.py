"""Grade a submission by WIRING the bundle to the eval engine.

WHAT THIS VERIFIER WRITES IS r_det, THE DETERMINISTIC SCORE (Bucket D).

    r_det = GATE_det * S_outcome * RubricFactor_det

GATE_det and RubricFactor_det are computed from the compiled checks in test_output.py
alone. The judged trajectory rubrics in tests/rubrics.jsonl are graded DOWNSTREAM by an
LLM trajectory grader and compose after r_det under the bound gate form:
they may cap the composed result and may never raise it, and their absence here must not
zero an honest score. An earlier revision of this file passed council_scores=None into the
aggregate, which failed the council knockout closed and wrote score 0.000000 for every
submission including a perfect one. Judged rubrics therefore never enter this aggregate.

WHAT THIS BUNDLE SUPPLIES: S_outcome, the measured base.

    S_outcome = clamp((balanced_accuracy - FLOOR) / (KNEE - FLOOR), 0, 1)

FLOOR sits just above the shipped starter's best seed, so an unchanged starter scores 0 on
every seed (per attempt and not in expectation). KNEE sits above the
measured ceiling by a stated margin, so beating our best still pays. The computation is
PURE: it reads prediction files and constants, and re-running it on the same bytes is
bit-identical.

OUTCOME STATUS, and the boundary between an agent's zero and an infrastructure fault:
  ok            a real measurement was taken (any value, including 0.0)
  agent_miss    the submission is missing or malformed: absent file, wrong shape, wrong
                dtype, out-of-range labels. This is an honest agent zero and is DELIVERED.
  infra_failure the grader itself could not do its job: graded labels unreadable, engine
                import broken. This is withheld upstream, never delivered as a zero.
An earlier revision classified every measurement exception as infra_failure, which made a
wrong-shape prediction file vanish from the score history instead of scoring 0.
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONST = json.load(open(os.path.join(HERE, "constants.json")))
INSTANCES = CONST["instances"]
N_CLASSES = CONST["n_classes"]
LOG_DIR = os.environ.get("HARBOR_VERIFIER_LOG_DIR", "/logs/verifier")
RUBRIC_FLOOR = 0.5


class SubmissionError(Exception):
    """Malformed submission: an agent miss, not an infrastructure fault."""


def balanced_accuracy(pred, y, k):
    """Mean per-class recall. Chance is 1/k regardless of class imbalance."""
    present = [c for c in range(k) if (y == c).any()]
    return float(np.mean([float((pred[y == c] == c).mean()) for c in present]))


def ramp(acc, floor, knee):
    if knee <= floor:
        raise ValueError("degenerate ramp: knee %r <= floor %r" % (knee, floor))
    return max(0.0, min(1.0, (acc - floor) / (knee - floor)))


def load_predictions(path, expected_shape):
    if not os.path.exists(path):
        raise SubmissionError("MISSING_PREDICTIONS: %s" % path)
    try:
        p = np.load(path)
    except Exception as exc:
        raise SubmissionError("UNREADABLE_PREDICTIONS: %s" % exc)
    if not np.issubdtype(p.dtype, np.integer):
        raise SubmissionError("NON_INTEGER_PREDICTIONS: dtype %s" % p.dtype)
    if p.shape != expected_shape:
        raise SubmissionError("WRONG_SHAPE: predictions %s, graded split %s"
                              % (p.shape, expected_shape))
    p = p.astype(np.int64)
    if int(p.min()) < 0 or int(p.max()) >= N_CLASSES:
        raise SubmissionError("LABEL_OUT_OF_RANGE: [%d, %d]" % (p.min(), p.max()))
    return p


def measure(sub):
    """S_outcome plus its audit detail. Pure over the submitted bytes."""
    graded = np.load(os.path.join(HERE, "heldout", "graded_test.npz"))
    y = graded["y"].astype(np.int64)
    per = {}
    for inst in INSTANCES:
        p = load_predictions(os.path.join(sub, "preds_%s.npy" % inst), y.shape)
        acc = balanced_accuracy(p, y, N_CLASSES)
        per[inst] = {"balanced_accuracy": acc,
                     "floor": CONST["floor"][inst], "knee": CONST["knee"][inst],
                     "S_outcome": ramp(acc, CONST["floor"][inst], CONST["knee"][inst])}
    s = sum(v["S_outcome"] for v in per.values()) / len(per)
    return s, {"per_instance": per, "S_outcome": s}


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else "/submission"
    os.environ.setdefault("SUBMISSION", sub)
    os.makedirs(LOG_DIR, exist_ok=True)

    from aello_eval import RubricResult, aggregate, load_check_results, write_reward

    # Deterministic layer. The plugin flushes check_results.json at session end; the
    # markers in test_output.py are the single source of truth for id, weight, knockout
    # and kind. load_check_results takes the FILE path: an earlier revision passed the
    # directory, which silently returned [] and dropped every compiled check from the
    # aggregate.
    subprocess.run(["pytest", "-q", "--no-header", "-p", "aello_eval.checks",
                    os.path.join(HERE, "test_output.py")],
                   cwd=LOG_DIR, env=dict(os.environ, HARBOR_VERIFIER_LOG_DIR=LOG_DIR))
    check_results = load_check_results(os.path.join(LOG_DIR, "check_results.json")) or []

    try:
        outcome_score, outcome_detail = measure(sub)
        status = "ok"
    except SubmissionError as exc:               # the agent's fault: an honest zero
        outcome_score, outcome_detail = 0.0, {"reason": str(exc)}
        status = "agent_miss"
    except Exception as exc:                     # the grader's fault: never a zero
        outcome_score, outcome_detail = 0.0, {"reason": "%s: %s"
                                              % (type(exc).__name__, exc)}
        status = "infra_failure"

    if not check_results:
        # The deterministic layer failed to run or flush: grading infrastructure fault.
        outcome_detail["check_layer"] = "check_results_missing_or_empty"
        status = "infra_failure"

    # An agent's honest zero and an infrastructure fault must never look like the same
    # number, so the outcome DECLARES which it is and the scaffold withholds on the latter.
    outcome_detail["outcome_status"] = status
    outcome_detail["s_outcome"] = outcome_score
    json.dump(outcome_detail, open(os.path.join(LOG_DIR, "outcome.json"), "w"), indent=1)

    # r_det: deterministic checks ONLY. Judged rubrics (tests/rubrics.jsonl) are graded by
    # the downstream trajectory grader and compose after this number; their absence here
    # is by design, not an outage.
    det = [RubricResult(id=str(c["id"]), score=float(c.get("score", 0.0)),
                        weight=int(c.get("weight", 1)),
                        knockout=bool(c.get("knockout", False)),
                        grader="pytest", kind=str(c.get("kind", "process")))
           for c in check_results]
    reward, detail = aggregate(outcome_score=outcome_score, rubrics=det,
                               floor=RUBRIC_FLOOR)
    detail["outcome_detail"] = outcome_detail
    detail["reward_is"] = "r_det (deterministic); judged rubrics compose downstream"
    write_reward(reward, detail, LOG_DIR)
    print(json.dumps({"reward": round(reward, 10), "S_outcome": round(outcome_score, 10),
                      "gate": detail["gate"], "outcome_status": status}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
