"""Grade a submission by WIRING the bundle to the aello_eval engine.

WHAT THIS VERIFIER WRITES IS r_det, THE DETERMINISTIC REWARD (Bucket D).

    r_det = GATE_det * S_outcome * RubricFactor_det

GATE_det and RubricFactor_det are computed from the compiled checks in test_output.py
alone. The judged trajectory rubrics in tests/rubrics.jsonl are graded DOWNSTREAM by an
LLM trajectory grader and compose after r_det under the bound gate form (seed/reward.py):
they may cap the composed result and may never raise it, and their absence here must not
zero an honest score. An earlier revision of this file passed council_scores=None into the
aggregate, which failed the council knockout closed and wrote reward 0.000000 for every
submission including a perfect one. Judged rubrics therefore never enter this aggregate.

WHAT THIS BUNDLE SUPPLIES: S_outcome, the measured base.

    S_outcome = clamp((balanced_accuracy - FLOOR) / (KNEE - FLOOR), 0, 1)

FLOOR sits just above the shipped starter's best seed, so an unchanged starter scores 0 on
every seed (FORGE.md:274, per attempt and not in expectation). KNEE sits above the
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
wrong-shape prediction file vanish from the reward history instead of scoring 0.
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
    """Malformed submission: an agent miss, not an infrastructure fault.

    `absent` marks the submission artifact not existing AT ALL, as distinct from existing and
    being malformed. The two produce the same reward and must be told apart: an absent artifact
    is also what an agent that never ran produces, and that is infrastructure, not an agent zero.
    """

    def __init__(self, message, absent=False):
        super().__init__(message)
        self.absent = absent


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
        raise SubmissionError("MISSING_PREDICTIONS: %s" % path, absent=True)
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


def _expected_compiled():
    """Compiled-check id -> (weight, knockout) DECLARED in the verifier-owned test_output.py.
    Binds the expected target set so a check MISSING from the runner log is a FAILURE, never a
    silent drop that raises the reward (G-VER-INVARIANCE). test_output.py is sealed in the
    verifier image, so the submission cannot shrink this set."""
    import ast
    src = open(os.path.join(HERE, "test_output.py")).read()
    out = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and getattr(dec.func, "id", "") == "check":
                    kw = {k.arg: k.value for k in dec.keywords}
                    cid = kw["id"].value
                    w = kw["weight"].value if "weight" in kw else 1
                    ko = kw["knockout"].value if "knockout" in kw else False
                    out[cid] = (int(w), bool(ko))
    return out



def agent_authored_count():
    """Files in the collected agent tree byte-distinct from every shipped source digest.

    This is the definition tests/test_output.py already uses three times over, and
    solution/grounding.yaml states: agent-authored means byte-distinct from every digest in
    tests/constants.json shipped_source_digests. Returns None when the tree was never
    collected, which is unknowable rather than zero.
    """
    import hashlib
    base = os.environ.get("AELLO_AGENT_SRC", "/workspace/agent_src")
    if not os.path.isdir(base):
        return None
    pristine = set(CONST.get("shipped_source_digests") or ())
    seen, authored = 0, 0
    for root, _, files in os.walk(base):
        for name in sorted(files):
            try:
                blob = open(os.path.join(root, name), "rb").read()
            except OSError:
                continue
            seen += 1
            if hashlib.sha256(blob).hexdigest() not in pristine:
                authored += 1
    return None if seen == 0 else authored


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else "/submission"
    os.environ.setdefault("AELLO_SUBMISSION", sub)
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
        # An ABSENT submission is what BOTH a crashed agent and an agent that never ran
        # produce, and the predicate above cannot tell them apart. Measured 2026-08-24: nine
        # consecutive iterations recorded agent_turns=1 tool_calls=0 files=0 against a 429 from
        # the agent bridge, and every one was delivered into the reward history as a valid 0.0.
        # Discriminate on evidence this verifier can actually see -- the collected agent tree,
        # which arrives as a declared artifact. A tree that was collected and holds NO
        # agent-authored file means the agent never acted.
        #
        # Deliberately narrow, in three ways. A MALFORMED submission proves the agent acted and
        # stays agent_miss, so the wrong-shape regression recorded in this file's header cannot
        # return. A tree that was never collected, or is empty, falls back to agent_miss rather
        # than withholding, because a control run places a submission with no agent phase and
        # withholding would deny the exactly-zero the empty-submission red line and the
        # known-wrong controls are required to produce. And the count is recorded either way,
        # so the branch is auditable from the delivered bytes.
        if getattr(exc, "absent", False):
            n_authored = agent_authored_count()
            outcome_detail["agent_authored_files"] = n_authored
            if n_authored == 0:
                status = "infra_failure"
                outcome_detail["withheld_reason"] = (
                    "submission absent and the collected agent tree holds no agent-authored "
                    "file: the agent never acted, so this is an infrastructure fault and not "
                    "an agent zero")
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
    # Reconcile the runner log against the DECLARED compiled set: a target missing from the
    # log is injected as a failure (score 0), never silently dropped. A dropped target can no
    # longer raise the reward, and a missing knockout gates the whole reward to 0.
    expected = _expected_compiled()
    got = {str(c["id"]): c for c in check_results}
    missing = sorted(set(expected) - set(got))
    if missing:
        outcome_detail["missing_targets"] = missing
    det = []
    for cid, (w, ko) in expected.items():
        if cid in got:
            c = got[cid]
            det.append(RubricResult(id=cid, score=float(c.get("score", 0.0)),
                                    weight=int(c.get("weight", w)),
                                    knockout=bool(c.get("knockout", ko)),
                                    grader="pytest", kind=str(c.get("kind", "process"))))
        else:
            det.append(RubricResult(id=cid, score=0.0, weight=w, knockout=ko,
                                    grader="pytest", kind="process"))
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
