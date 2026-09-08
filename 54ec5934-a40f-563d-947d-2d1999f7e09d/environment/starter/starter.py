"""Correct, unoptimised baseline for C7-S2. Replace it entirely if you want.

It exists to prove the delivery path end to end: it walks the instances the graded run hands it,
writes one terminal record per instance in the shape that instance's evaluator family reads, and
declares its own per-instance budget. It is deliberately weak on the graded metric -- it opens
each instance's start URL, reads nothing, and terminates -- and that is what you are asked to
improve.

The shape matters more than it looks. Every field written below is a field an external evaluator
or the per-instance budget gate reads; a scaffold that navigates brilliantly and then omits one
of them scores the same as a scaffold that never ran.
"""
import json
import os

SUBMISSION = '/submission/c7s2.json'
RUN_RECORD = '/workspace/run_record.json'
MANIFEST = '/workspace/graded_instances.json'

# The one field each WebArena evaluator family reads out of a terminal record. This is a fact
# about the vendored evaluator, not a choice this file makes.
EVALUATOR_FIELD = {
    "exact_match": "answer",
    "must_include": "answer",
    "fuzzy_match": "answer",
    "url_match": "final_url",
    "program_html": "site_end_state_hash",
}

# Ceilings at or inside the published per-instance gate of 30 actions and 240 s.
ACTION_CEILING = 30
LATENCY_CEILING_S = 240.0
CONCURRENCY = 1


def graded_instances():
    """The instances this graded run is scored on, as the harness handed them over.

    Never a list this file authored: the private split is re-bound after the bundle is built, so
    an id set chosen here would name instances that are not the graded ones.
    """
    if not os.path.exists(MANIFEST):
        return []
    with open(MANIFEST) as handle:
        doc = json.load(handle)
    rows = doc.get("instances") if isinstance(doc, dict) else doc
    return rows if isinstance(rows, list) else []


def episode(instance):
    """One trivially weak episode: open the start URL, take no action, terminate.

    It still writes the field its evaluator family reads, because an episode with nothing to read
    is unscoreable rather than merely unsuccessful.
    """
    field = EVALUATOR_FIELD.get(instance.get("evaluator_family"), "answer")
    reached = {"answer": "", "final_url": instance.get("start_url", ""),
               "site_end_state_hash": instance.get("initial_state_hash", "")}[field]
    return {field: reached, "actions": 0, "elapsed_s": 0.0,
            "terminated_by": "goal_confirmed"}


def main():
    instances = graded_instances()
    ids = [str(row.get("instance_id")) for row in instances]
    outputs = [episode(row) for row in instances]

    os.makedirs(os.path.dirname(SUBMISSION), exist_ok=True)
    with open(SUBMISSION, "w") as handle:
        json.dump({"ids": ids, "outputs": outputs}, handle)

    with open(RUN_RECORD, "w") as handle:
        json.dump({"agent_budget_declaration": {
            "per_instance_action_ceiling": ACTION_CEILING,
            "per_instance_latency_ceiling_s": LATENCY_CEILING_S,
            "episode_concurrency": CONCURRENCY,
        }}, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
