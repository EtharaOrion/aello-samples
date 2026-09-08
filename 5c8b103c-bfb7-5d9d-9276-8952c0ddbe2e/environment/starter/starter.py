"""Correct, unoptimised baseline for AELLO-C6-S2. Replace it entirely if you want.

It exists to prove the delivery path end to end: it reads what ships, emits the COUPLED pair the
task grades -- a per-step label sequence and a first-error index that agree with each other by
construction -- writes them at the graded path, and leaves a complete run record behind.

It is deliberately weak on the graded metric, and weak in the specific way the task is about: its
decode never emits the middle class. Every step is good or bad, so the per-stratum neutral band is
violated on the first stratum that has any human neutral mass at all, and the run scores exactly
zero however good its binary step accuracy looks. Resolving that ambiguous class honestly is the
task; this file is the scaffold, not a starting point on the metric.
"""
import hashlib
import json
import os
import time

SUBMISSION = '/submission/c6s2.json'
RUN_RECORD = '/workspace/run_record.json'
REWARD_LOG = '/workspace/reward_log.jsonl'
DATA = '/workspace/environment/data'
SHARDS = ("graded_solutions.jsonl", "sibling_solutions.jsonl")
STAGES = ("segmentation", "encoding", "three_class_fitting", "per_stratum_calibration",
          "localization_decode", "export")


def read_solutions():
    """Every solution of both private shards, in the order the corpus presents them.

    That order is privately permuted per graded run. It is never used as a feature here and it
    must not become one: the graded counts are taken in canonical sorted id order, so a positional
    prior earns nothing and a positional key in the output is rejected outright.
    """
    rows = []
    for name in SHARDS:
        path = os.path.join(DATA, name)
        if not os.path.exists(path):
            continue
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def decode(solution):
    """One monotone walk producing BOTH outputs, so they cannot disagree.

    The rule is deliberately crude -- the last step is called bad when the final answer misses
    gold -- but the SHAPE is the shape the task wants: walk the steps in order, stop at the first
    step the rule condemns, and derive the label vector from that same walk. The index is read off
    the walk rather than argmaxed separately and reconciled afterwards.
    """
    steps = solution.get("steps") or []
    condemned = (solution.get("final_answer") != solution.get("gold_answer")
                 and len(steps) > 0)
    cut = len(steps) - 1 if condemned else -1
    labels = ["bad" if i == cut else "good" for i in range(len(steps))]
    return labels, cut


def main():
    began = time.time_ns()
    spans, rows = {}, read_solutions()
    spans["segmentation"] = [began, time.time_ns()]
    spans["encoding"] = [time.time_ns(), time.time_ns()]
    spans["three_class_fitting"] = [time.time_ns(), time.time_ns()]
    spans["per_stratum_calibration"] = [time.time_ns(), time.time_ns()]

    start = time.time_ns()
    ids, outputs = [], []
    for solution in rows:
        labels, cut = decode(solution)
        ids.append(solution.get("solution_id"))
        outputs.append({"labels": labels, "first_error_index": cut})
    spans["localization_decode"] = [start, time.time_ns()]

    start = time.time_ns()
    os.makedirs(os.path.dirname(SUBMISSION), exist_ok=True)
    with open(SUBMISSION, "w") as handle:
        json.dump({"ids": ids, "outputs": outputs}, handle)
    ended = time.time_ns()
    spans["export"] = [start, ended]

    digest = hashlib.sha256(open(SUBMISSION, "rb").read()).hexdigest()
    with open(RUN_RECORD, "w") as handle:
        json.dump({"first_data_touch_ns": began,
                   "artefacts_written_ns": ended,
                   "elapsed_seconds": -(-(ended - began) // 10 ** 9),
                   "stage_spans": dict((name, spans[name]) for name in STAGES),
                   # This run is deterministic and single-pass, so its own replay is itself. A
                   # pipeline with any sampling in it must actually re-run and digest the second
                   # artifact rather than copying this line.
                   "replay_artifact_sha256": digest,
                   "reward_log": os.path.basename(REWARD_LOG),
                   "guarded_paths_touched": [],
                   "agent_budget_declaration": {"attempts_used": 1,
                                                "solutions_decoded": len(ids),
                                                "seconds_declared": (ended - began) / 1e9}},
                  handle, indent=1)
    with open(REWARD_LOG, "w") as handle:
        # No macro-F1 and no localisation rate are written here, because this run measured
        # neither and the compiled check compares what is written against the verifier's own
        # recomputation. An honest incomplete log is a missing field; a plausible number this
        # run did not produce is a fabrication and a red line.
        handle.write(json.dumps({"attempt": 1, "solutions_decoded": len(ids),
                                 "elapsed_seconds": -(-(ended - began) // 10 ** 9)}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
