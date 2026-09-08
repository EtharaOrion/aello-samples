"""Grade an AELLO-C6-S3 submission by WIRING the bundle to the aello_eval engine.

WHAT THIS VERIFIER WRITES IS r_det, THE DETERMINISTIC REWARD (Bucket D):

    r_det = GATE_det * S_outcome * RubricFactor_det

GATE_det and RubricFactor_det come from the compiled checks in test_output.py alone. The
judged trajectory rubrics in tests/rubrics.jsonl are graded DOWNSTREAM by the LLM trajectory
grader and compose after r_det under the bound gate form: they may cap the composed result and
never raise it, and their absence here must not zero an honest score, so judged rubrics never
enter this aggregate.

S_outcome is the selective-judge reward map the bundle supplies. Stage 0 (integrity) and
Stage 1 (hard constraints) each zero the outcome with ONE machine-readable reason: absent or
malformed artefact, a verdict outside the closed vocabulary {A, B, ABSTAIN}, the time
ceiling, the per-shard selective risk cap, any per-sub-distribution risk cap, the
long-dialogue subgroup cap, the between-shard coverage-gap cap, zero coverage, or coverage
below the floor on either shard. Stage 2 is the reward map: a coverage ramp from COV_FLOOR to
COV_TARGET over the WORST shard carrying W_COV, plus a log-time band carrying W_SPEED. Every
quantity is an integer answered or error count over an integer denominator recomputed here from
the private label vectors and the private mixture map, or harness-measured integer seconds; the
computation is PURE and bit-identical on re-run.

Outcome status separates an agent's honest zero (agent_miss: absent or malformed artefact) from
a grader fault (infra_failure: private labels unreadable, engine import broken), which is
withheld upstream and never delivered as a zero.
"""
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONST = json.load(open(os.path.join(HERE, "constants.json")))
LOG_DIR = os.environ.get("HARBOR_VERIFIER_LOG_DIR", "/logs/verifier")
RUBRIC_FLOOR = 0.5
VOCAB = ("A", "B", "ABSTAIN")
SUBS = ("helpful-base", "helpful-online", "helpful-rejection-sampled")
SHARDS = ("p1", "p2")


class SubmissionError(Exception):
    """Malformed submission: an agent miss, not an infrastructure fault.

    `absent` marks the submission artifact not existing AT ALL, as distinct from existing and
    being malformed. The two produce the same reward and must be told apart: an absent artifact
    is also what an agent that never ran produces, and that is infrastructure, not an agent zero.
    """

    def __init__(self, message, absent=False):
        super().__init__(message)
        self.absent = absent


def _load(path):
    with open(path) as handle:
        return json.load(handle)


def _ratio(num, den):
    return (num / den) if den else 0.0


def measure(sub_dir):
    """Return (S_outcome, detail). Raises SubmissionError on an agent-visible malformation."""
    verdict_path = os.path.join(sub_dir, "verdicts.json")
    if not os.path.exists(verdict_path):
        raise SubmissionError("empty-submission", absent=True)
    try:
        sub = _load(verdict_path)
        ids, verdicts = sub["ids"], sub["verdicts"]
    except Exception:
        raise SubmissionError("malformed-verdict")
    if not isinstance(ids, list) or not isinstance(verdicts, list) or len(ids) != len(verdicts):
        raise SubmissionError("malformed-verdict")
    if any((not isinstance(v, str)) or v not in VOCAB for v in verdicts):
        raise SubmissionError("malformed-verdict")
    if len(set(ids)) != len(ids):
        raise SubmissionError("malformed-verdict")
    vmap = dict(zip(ids, verdicts))

    labels = {s: _load(os.path.join(HERE, "heldout", "%s_labels.json" % s))["labels"] for s in SHARDS}
    need = set()
    for s in SHARDS:
        need |= set(labels[s])
    if need - set(vmap):
        raise SubmissionError("malformed-verdict")

    # Integer counts. Everything downstream is a ratio of two of these.
    per_shard = {}
    sub_ans = {d: 0 for d in SUBS}
    sub_err = {d: 0 for d in SUBS}
    long_ans = long_err = 0
    for s in SHARDS:
        n = len(labels[s])
        ans = err = 0
        for iid, meta in labels[s].items():
            v = vmap[iid]
            if v == "ABSTAIN":
                continue
            wrong = int(v != meta["winner"])
            ans += 1
            err += wrong
            sub_ans[meta["subset"]] += 1
            sub_err[meta["subset"]] += wrong
            if meta["long"]:
                long_ans += 1
                long_err += wrong
        per_shard[s] = {"n": n, "answered": ans, "errors": err,
                        "cov": _ratio(ans, n), "risk": _ratio(err, ans)}

    cov1, cov2 = per_shard["p1"]["cov"], per_shard["p2"]["cov"]
    cov = min(cov1, cov2)
    gap = abs(cov1 - cov2)
    risk_by_sub = {d: _ratio(sub_err[d], sub_ans[d]) for d in SUBS}
    risk_long = _ratio(long_err, long_ans)

    # elapsed_s is AGENT-WRITTEN, and its absence used to be the best possible move. Defaulting
    # to 0.0 made t = 0, and max(1, 0) drove the log ramp to its upper clamp: omitting the field
    # scored 1.0000, tying the fastest legal run, beating an honest worst-case report by the whole
    # ramp, and passing the budget knockout because 0 > T_MAX is false. Silence was dominant and
    # indistinguishable from an agent that never read instruction.md. The default now sits at the
    # WORST LEGAL REPORT, T_MAX, so silence can never beat a disclosure. A non-finite or
    # non-positive value is treated as absent for the same reason -- a negative elapsed would
    # otherwise re-enter through max(1, t).
    #
    # Where both agent-writable sources carry a value they must AGREE, the same shape as the
    # optional self-report cross-check above: disagreement is log-unfaithful and scores 0.0.
    #
    # RESIDUAL, recorded rather than hidden: neither source is host-authoritative. The verifier
    # cannot recompute wall clock from the submission, so this BOUNDS a dishonest report at the
    # worst legal value and detects self-contradiction, but it does not detect a uniformly
    # understated one. A host-recorded duration surfaced to the verifier is owed.
    def _elapsed_report(value):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        if f != f or f in (float("inf"), float("-inf")) or f <= 0.0:
            return None
        return f

    meta_path = os.path.join(sub_dir, "run_meta.json")
    _reports = []
    if os.path.exists(meta_path):
        _v = _elapsed_report(_load(meta_path).get("elapsed_s"))
        if _v is not None:
            _reports.append(("run_meta.json", _v))
    _v = _elapsed_report(sub.get("elapsed_s"))
    if _v is not None:
        _reports.append(("submission", _v))

    _elapsed_unfaithful = (len(_reports) == 2
                           and abs(_reports[0][1] - _reports[1][1]) > 1e-9)
    if not _reports:
        elapsed = float(CONST["T_MAX"])
        _elapsed_source = "absent__defaulted_to_worst_legal_report"
    else:
        elapsed = _reports[0][1]
        _elapsed_source = _reports[0][0]
    t = int(math.ceil(elapsed))

    detail = {"cov_p1": cov1, "cov_p2": cov2, "cov": cov, "gap": gap,
              "risk_p1": per_shard["p1"]["risk"], "risk_p2": per_shard["p2"]["risk"],
              "risk_by_sub": risk_by_sub, "risk_long": risk_long,
              "counts": {"shards": per_shard, "sub_answered": sub_ans, "sub_errors": sub_err,
                         "long_answered": long_ans, "long_errors": long_err},
              "t_s": t}

    detail["elapsed_source"] = _elapsed_source
    if _elapsed_unfaithful:
        detail["reason"] = "elapsed-unfaithful"
        return 0.0, detail

    # Optional self-report: if present it must agree with the host recomputation.
    rep = sub.get("reported")
    if isinstance(rep, dict):
        for k, v in (("cov_p1", cov1), ("cov_p2", cov2)):
            if k in rep and abs(float(rep[k]) - v) > 1e-9:
                detail["reason"] = "log-unfaithful"
                return 0.0, detail

    alpha, slack = CONST["ALPHA"], CONST["STRATUM_SLACK"]
    stage1 = ((t > CONST["T_MAX"], "budget-exceeded"),
              (per_shard["p1"]["risk"] > alpha or per_shard["p2"]["risk"] > alpha, "risk-cap-exceeded"),
              (any(risk_by_sub[d] > alpha + slack for d in SUBS), "stratum-risk-exceeded"),
              (risk_long > alpha + slack, "long-dialogue-stratum-risk-exceeded"),
              (gap > CONST["COV_GAP_CAP"], "shift-instability"),
              (per_shard["p1"]["answered"] == 0 or per_shard["p2"]["answered"] == 0, "no-coverage"),
              (cov < CONST["COV_FLOOR"], "below-coverage-floor"))
    for cond, reason in stage1:
        if cond:
            detail["reason"] = reason
            return 0.0, detail

    cov_term = max(0.0, min(1.0, (cov - CONST["COV_FLOOR"]) / (CONST["COV_TARGET"] - CONST["COV_FLOOR"])))
    speed_term = max(0.0, min(1.0, (math.log(CONST["T_MAX"]) - math.log(max(1, t)))
                              / (math.log(CONST["T_MAX"]) - math.log(CONST["T_MIN"]))))
    s = CONST["W_COV"] * cov_term + CONST["W_SPEED"] * speed_term
    detail["cov_term"] = cov_term
    detail["speed_term"] = speed_term
    detail["reason"] = "graded"
    return s, detail


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

    subprocess.run(["pytest", "-q", "--no-header", "-p", "aello_eval.checks",
                    os.path.join(HERE, "test_output.py")],
                   cwd=LOG_DIR, env=dict(os.environ, HARBOR_VERIFIER_LOG_DIR=LOG_DIR))
    check_results = load_check_results(os.path.join(LOG_DIR, "check_results.json")) or []

    try:
        outcome_score, outcome_detail = measure(sub)
        status = "ok"
    except SubmissionError as exc:
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
    except Exception as exc:
        outcome_score, outcome_detail = 0.0, {"reason": "%s: %s" % (type(exc).__name__, exc)}
        status = "infra_failure"

    if not check_results:
        outcome_detail["check_layer"] = "check_results_missing_or_empty"
        status = "infra_failure"

    outcome_detail["outcome_status"] = status
    outcome_detail["s_outcome"] = outcome_score
    json.dump(outcome_detail, open(os.path.join(LOG_DIR, "outcome.json"), "w"), indent=1)

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
    reward, detail = aggregate(outcome_score=outcome_score, rubrics=det, floor=RUBRIC_FLOOR)
    detail["outcome_detail"] = outcome_detail
    detail["reward_is"] = "r_det (deterministic); judged rubrics compose downstream"
    write_reward(reward, detail, LOG_DIR)
    print(json.dumps({"reward": round(reward, 10), "S_outcome": round(outcome_score, 10),
                      "gate": detail["gate"], "outcome_status": status,
                      **{k: outcome_detail.get(k) for k in ("cov_p1", "cov_p2", "risk_p1", "risk_p2",
                                                            "risk_by_sub", "risk_long", "gap", "t_s", "reason")}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
