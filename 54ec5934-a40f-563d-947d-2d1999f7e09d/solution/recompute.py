"""Derivation for C7-S2: every canonical artifact descends from solution/grounding.yaml.

This slot grades a scaffold against an evaluator it does not own. The generator therefore emits
checks that measure LEGIBILITY to that evaluator and adherence to the PER-INSTANCE budget gate,
and it refuses to emit a compiled item for which no body was authored: a compiled item with no
body used to fall through to a generic obligation lookup that consulted a run-record field
nothing ever writes, which reports a verdict about nothing at all.
"""
import json
import os
import sys
import textwrap

import yaml

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)
# generator_common.py is VENDORED beside this file, not reached for outside the bundle. A staged
# tree could resolve seed/stage/<slug>/../../build; a DELIVERED tree at dataset/<uuid>/ cannot,
# and reaching outside would make the delivered bundle non-self-contained -- it would import in
# the stage and fail everywhere it actually ships.

ITEM_KEYS = ('id', 'dimension', 'weight', 'evaluation_target', 'criterion', 'judgment', 'knockout', 'mode')
SLOT_ID = 'C7-S2'
TITLE = "# C7-S2 -- what this task actually tests"
OUTCOME_OPENER = 'c = require_measured("floor", "knee", "dichotomizing_threshold")'


def read_identity(here):
    """Identity is an INPUT: the freeze computes it over the hashed tree and binds it into the
    carrier, which is hash-excluded, so reading it back keeps generation acyclic."""
    path = os.path.join(here, "provenance.yaml")
    if os.path.exists(path):
        doc = yaml.safe_load(open(path)) or {}
        if isinstance(doc.get("identity"), dict):
            return doc["identity"]
    return {"canonical_content_hash": None, "uuid": None,
            "normalization_domain": "aello.canary.norm/v1",
            "derivation": "uuid5(FORGE_TASK_NAMESPACE, canonical_content_hash)"}


def read_screening(here):
    """The screening instant and result are MEASURED OVER the frozen tree, so they are read back
    the way identity is; the carrier is hash-excluded, so binding them moves nothing."""
    path = os.path.join(here, "provenance.yaml")
    if os.path.exists(path):
        doc = yaml.safe_load(open(path)) or {}
        got = {k: doc[k] for k in ("screening_measured_at", "screening_expires_at",
                                   "screening_result") if k in doc}
        if got:
            return got
    return {"screening_measured_at": None, "screening_expires_at": None,
            "screening_result": None}


def body(source):
    """Normalise an authored check body onto the indentation build_tests splices it into.

    Bodies are authored as ordinary readable Python here rather than as escaped one-liners,
    because a body nobody can read is a body nobody audits, and every one of them runs against
    delivered bytes.
    """
    lines = textwrap.dedent(source).strip("\n").splitlines()
    return "\n".join(("    " + line).rstrip() for line in lines)


def build_tests(g, prelude, bodies):
    """Assemble the compiled-check module.

    The check's kind is the item's own outcome class, so a reader of the emitted module can see
    whether a check asserts an effect, an absence, an invariant or a value without going back to
    the rubric carrier.
    """
    chunks = [prelude]
    for raw in sorted(g["items"], key=lambda i: i["id"]):
        if raw["mode"] != "compiled":
            continue
        chunks.append(
            '\n@check(id="%s", weight=%d, knockout=%s, kind="%s")\ndef test_%s():\n%s\n'
            % (raw["id"], raw["weight"], raw["knockout"], raw["outcome_class"], raw["id"],
               bodies[raw["id"]]))
    return "".join(chunks)


def build_truth(g, title):
    """Render solution/TRUTH.md from the frozen literals of solution/grounding.yaml."""
    n = g["truth_narrative"]
    lines = [title, "", "GENERATED SECTION. DO NOT HAND-EDIT.", "", n["opening"], "",
             "## The single most important insight", "", n["insight"], "",
             "## The ideal solve, step by step", ""]
    for i, step in enumerate(n["steps"], 1):
        lines += ["%d. **%s** %s" % (i, step["heading"], step["paragraph"]), ""]
    lines += ["## Traps that catch agents that are not thinking carefully", ""]
    lines += ["- " + t for t in n["traps"]]
    lines += ["", "---", "", "## Contract record (annex; not part of the narrative body)", "",
              "Retained under FORGE.md item 10e and Phase 2 item 7, which require this file to "
              "carry the ordered path through instruction.md with each satisfied checker "
              "identifier, and each rejected route bound to a measured known-wrong control. "
              "standards/truth-md-authoring-v1.md section 3 admits no fifth section, so this "
              "annex is a recorded deviation rather than an omission.", "", g["truth_annex"], ""]
    return "\n".join(lines) + "\n"


def build_rubrics(g, item_keys):
    """The 9g rubric carrier. The item schema is CLOSED at eight keys, so item 10f's outcome
    classification rides as a greppable prefix of the judgment rather than as a ninth key."""
    items = []
    for raw in g["items"]:
        item = {k: raw[k] for k in item_keys}
        item["judgment"] = "%s: %s" % (raw["outcome_class"], raw["judgment"])
        items.append(item)
    compiled = sum(i["weight"] for i in items if i["mode"] == "compiled")
    total = sum(i["weight"] for i in items)
    return {"$schema": "forge.rubric/v1",
            "banner": "GENERATED SECTION. DO NOT HAND-EDIT.",
            "generator": "solution/recompute.py",
            "compilation_floor": g["compilation_floor"],
            "compiled_weight_share": round(compiled / total, 6) if total else 0.0,
            "evaluation_target_vocabulary": sorted(g["evaluation_target_vocabulary"]),
            "items": sorted(items, key=lambda i: i["id"])}


def load_grounding():
    with open(os.path.join(HERE, "grounding.yaml")) as fh:
        return yaml.safe_load(fh)


def build_provenance(g, identity):
    p = g["provenance"]
    out = {
        "banner": "GENERATED SECTION. DO NOT HAND-EDIT.",
        "generator": "solution/recompute.py from solution/grounding.yaml",
        "schema_version": "1.0",
        "slot_id": g["slot_id"],
        "identity": identity,
        "corpus": p["corpus"],
        "anchors": p["anchors"],
        "shards": p["shards"],
        "narrative": p["narrative"],
        "supersedes": p["supersedes"],
        "upstream_provenance": p["upstream_record"],
        "reward_composition": p["reward_composition"],
        "screening_interval_days": p["screening_interval_days"],
        "screening_detector_version": p["screening_detector_version"],
        "derivation_instant": p["derivation_instant"],
        "screening_roots": p["screening_roots"],
        "empty_submission_result": p["empty_submission_result"],
        "resolved_closure": p["resolved_closure"],
        "applicability": p["applicability"],
        "measurement_tier": g["measurement_tier"],
        "gradeable": g["gradeable"],
        "knee_anchor_status": g["knee_anchor_status"],
        "timed_axis": True,
    }
    out.update(read_screening(HERE))
    return out


def validate(grounding):
    """Well-formedness this slot's own shape requires, checked before anything is emitted."""
    problems = []
    items = grounding["items"]
    ids = [i["id"] for i in items]
    if len(set(ids)) != len(ids):
        problems.append("two items share an id, so one would overwrite the other's check")
    if "run_record" not in {i["evaluation_target"] for i in items}:
        problems.append("the per-instance budget gate and the reset exclusion are harness-metered, "
                        "so at least one item must grade against the run record rather than "
                        "against anything the scaffold hands in")
    outcome = [i for i in items if i["dimension"] == "outcome"]
    if len(outcome) != 1:
        problems.append("expected exactly one outcome item, found %d" % len(outcome))
    return problems


BODIES = {

    "graded_instance_ids_exact": body('''
        if not os.path.exists(SUBMISSION):
            return False, evidence(reason="submission_missing", path=SUBMISSION)
        try:
            doc = load_submission()
        except Exception as exc:
            return False, evidence(reason="submission_malformed", error=str(exc))
        ids = doc.get("ids")
        if not isinstance(ids, list):
            return False, evidence(reason="ids_missing_or_not_a_list")
        if len(set(map(str, ids))) != len(ids):
            return False, evidence(reason="duplicate_instance_ids", submitted=len(ids))
        want = graded_instance_ids()
        if want is None:
            return False, evidence(reason="graded_manifest_unavailable",
                                   path=heldout(PRIVATE_SPLIT))
        got = set(map(str, ids))
        return (got == want), evidence(submitted=len(ids), graded=len(want),
                                       invented=sorted(got - want)[:5],
                                       dropped=sorted(want - got)[:5],
                                       reason=None if got == want
                                       else "instance_id_set_is_not_the_graded_split")
    '''),

    "evaluator_legible_terminal_record": body('''
        try:
            rows = terminal_records()
        except Exception as exc:
            return False, evidence(reason="submission_malformed", error=str(exc))
        if rows is None:
            return False, evidence(reason="graded_manifest_unavailable",
                                   path=heldout(PRIVATE_SPLIT))
        illegible = []
        for iid, family, record in rows:
            field = EVALUATOR_FIELD.get(family)
            if field is None:
                illegible.append([iid, "unknown_evaluator_family:%s" % family])
            elif not isinstance(record, dict):
                illegible.append([iid, "terminal_record_is_not_an_object"])
            elif record.get(field) in (None, "", [], {}):
                illegible.append([iid, "missing_field:%s" % field])
        return (not illegible), evidence(records=len(rows), illegible=len(illegible),
                                         families=sorted(set(f for _, f, _ in rows)),
                                         first=illegible[:5],
                                         reason=None if not illegible
                                         else "record_is_not_legible_to_its_evaluator")
    '''),

    "budget_exhausted_episode_still_legible": body('''
        try:
            rows = terminal_records()
        except Exception as exc:
            return False, evidence(reason="submission_malformed", error=str(exc))
        if rows is None:
            return False, evidence(reason="graded_manifest_unavailable",
                                   path=heldout(PRIVATE_SPLIT))
        stopped, holes = 0, []
        for iid, family, record in rows:
            if not isinstance(record, dict):
                continue
            if record.get("terminated_by") not in BUDGET_STOPS:
                continue
            stopped += 1
            field = EVALUATOR_FIELD.get(family)
            wanted = [field, "actions", "elapsed_s"]
            absent = [str(k) for k in wanted if k is None or record.get(k) in (None, "")]
            if absent:
                holes.append([iid, absent])
        return (not holes), evidence(budget_stopped=stopped, without_record=len(holes),
                                     stops=sorted(BUDGET_STOPS), first=holes[:5],
                                     reason=None if not holes
                                     else "budget_stopped_episode_left_nothing_to_score")
    '''),

    "template_rebinding_not_collapsed": body('''
        try:
            rows = terminal_records()
        except Exception as exc:
            return False, evidence(reason="submission_malformed", error=str(exc))
        if rows is None:
            return False, evidence(reason="graded_manifest_unavailable",
                                   path=heldout(PRIVATE_SPLIT))
        templates = graded_field("intent_template_id")
        groups = {}
        for iid, family, record in rows:
            template = templates.get(iid)
            if template is None:
                continue
            answer = record.get(EVALUATOR_FIELD.get(family)) if isinstance(record, dict) else None
            groups.setdefault(str(template), []).append(json.dumps(answer, sort_keys=True))
        shared = dict((t, v) for t, v in groups.items() if len(v) > 1)
        collapsed = sorted(t for t, v in shared.items() if len(set(v)) == 1)
        return (not collapsed), evidence(templates=len(groups), shared_templates=len(shared),
                                         collapsed_groups=len(collapsed), first=collapsed[:5],
                                         reason=None if not collapsed
                                         else "one_answer_reused_across_a_template_group")
    '''),

    "per_instance_budget_gate_held": body('''
        episodes = harness_episodes()
        if not episodes:
            return False, evidence(target="run_record", reason="no_per_episode_metering")
        over_actions, over_latency, unmetered, spans = [], [], [], []
        for ep in episodes:
            iid = str(ep.get("instance_id"))
            actions, seconds = number(ep, "actions"), number(ep, "wall_clock_s")
            if actions is None or seconds is None:
                unmetered.append(iid)
                continue
            spans.append(seconds)
            if actions > MAX_ACTIONS:
                over_actions.append(iid)
            if seconds > MAX_EPISODE_S:
                over_latency.append(iid)
        tail = percentile(spans, 95)
        ok = (not over_actions and not over_latency and not unmetered
              and tail is not None and tail <= P95_EPISODE_S)
        return ok, evidence(target="run_record", episodes=len(episodes), p95_wall_clock_s=tail,
                            gate={"actions": MAX_ACTIONS, "episode_s": MAX_EPISODE_S,
                                  "p95_s": P95_EPISODE_S},
                            over_action_ceiling=over_actions[:5],
                            over_latency_ceiling=over_latency[:5], unmetered=unmetered[:5],
                            reason=None if ok else "per_instance_budget_exceeded")
    '''),

    "reset_metered_out_of_graded_clock": body('''
        record = run_record()
        episodes = harness_episodes()
        if not episodes:
            return False, evidence(target="run_record", reason="no_per_episode_metering")
        pinned = record.get("reset_fixture_snapshot_digest")
        if not isinstance(pinned, dict):
            return False, evidence(target="run_record", reason="no_pinned_fixture_digests")
        skipped, tampered, metered = [], [], 0.0
        for ep in episodes:
            iid = str(ep.get("instance_id"))
            span = number(ep, "site_reset_s")
            if span is None or span <= 0.0:
                skipped.append(iid)
            else:
                metered += span
            want = pinned.get(str(ep.get("target_site")))
            if want is None or ep.get("pre_episode_snapshot_digest") != want:
                tampered.append(iid)
        excluded = number(record, "site_reset_excluded_s")
        if excluded is None:
            return False, evidence(target="run_record", reason="reset_time_not_excluded")
        drift = abs(excluded - metered)
        ok = not skipped and not tampered and drift <= RESET_TOTAL_TOLERANCE_S
        return ok, evidence(target="run_record", episodes=len(episodes),
                            metered_reset_s=round(metered, 4), excluded_reset_s=excluded,
                            drift_s=round(drift, 4), reset_skipped=skipped[:5],
                            snapshot_mismatch=tampered[:5],
                            reason=None if ok else "reset_skipped_or_not_excluded_from_the_clock")
    '''),

    "target_site_within_retained_roster": body('''
        episodes = harness_episodes()
        if not episodes:
            return False, evidence(target="run_record", reason="no_per_episode_metering")
        off_roster, collateral = [], []
        for ep in episodes:
            iid = str(ep.get("instance_id"))
            site = str(ep.get("target_site"))
            if site not in RETAINED_SITES:
                off_roster.append([iid, site])
            mutated = ep.get("sites_mutated") or []
            elsewhere = sorted(set(str(s) for s in mutated) - {site})
            if elsewhere:
                collateral.append([iid, elsewhere[:3]])
        ok = not off_roster and not collateral
        return ok, evidence(target="run_record", episodes=len(episodes),
                            roster=sorted(RETAINED_SITES), excluded=sorted(EXCLUDED_SITES),
                            off_roster=off_roster[:5], collateral_writes=collateral[:5],
                            reason=None if ok
                            else "site_outside_the_retained_roster_or_a_collateral_write")
    '''),

    "external_grader_bytes_unmodified": body('''
        record = run_record()
        pinned = record.get("pinned_upstream_digests")
        observed = record.get("post_run_digests")
        if not isinstance(pinned, dict) or not isinstance(observed, dict):
            return False, evidence(target="run_record", reason="no_grader_digest_record")
        changed, unrecorded = [], []
        for tree in EXTERNAL_GRADER_TREES:
            want, got = pinned.get(tree), observed.get(tree)
            if want is None or got is None:
                unrecorded.append(tree)
            elif want != got:
                changed.append(tree)
        ok = not changed and not unrecorded
        return ok, evidence(target="run_record", trees=list(EXTERNAL_GRADER_TREES),
                            changed=changed, unrecorded=unrecorded,
                            reason=None if ok else "external_grader_bytes_moved")
    '''),

    "scaffold_rebuilt_its_own_artifacts": body('''
        policy = os.path.join(HERE, "..", "agent", "policy.py")
        if not os.path.exists(policy):
            return False, evidence(target="submitted_source", reason="policy_module_absent",
                                   path="agent/policy.py")
        starter = os.path.join(HERE, "..", "environment", "starter", "starter.py")
        if os.path.exists(starter) and digest(policy) == digest(starter):
            return False, evidence(target="submitted_source", reason="unchanged_starter",
                                   path="agent/policy.py")
        built = run_record().get("graded_run_built_artifacts")
        built = built if isinstance(built, dict) else {}
        prebaked = []
        for name in INDUCED_ARTIFACTS:
            entry = built.get(name)
            if not isinstance(entry, dict) or not entry.get("created_during_graded_run"):
                prebaked.append(name)
        return (not prebaked), evidence(target="submitted_source",
                                        induced=list(INDUCED_ARTIFACTS),
                                        shipped_prebaked=prebaked,
                                        reason=None if not prebaked
                                        else "artifact_was_not_induced_inside_the_graded_run")
    '''),

    "per_instance_budget_declared": body('''
        declaration = run_record().get("agent_budget_declaration")
        if declaration is None:
            return False, evidence(target="run_record", reason="no_agent_declaration")
        try:
            declared = (json.loads(declaration) if isinstance(declaration, str)
                        else dict(declaration))
        except Exception as exc:
            return False, evidence(target="run_record", reason="declaration_unparseable",
                                   error=str(exc))
        absent = [k for k in DECLARED_BUDGET_KEYS if k not in declared]
        if absent:
            return False, evidence(target="run_record", declared=sorted(declared), missing=absent,
                                   reason="declaration_is_not_per_instance")
        actions = number(declared, "per_instance_action_ceiling")
        seconds = number(declared, "per_instance_latency_ceiling_s")
        if actions is None or seconds is None:
            return False, evidence(target="run_record", reason="declared_ceilings_not_numeric")
        ok = 0 < actions <= MAX_ACTIONS and 0 < seconds <= MAX_EPISODE_S
        return ok, evidence(target="run_record", declared_action_ceiling=actions,
                            declared_latency_ceiling_s=seconds,
                            gate={"actions": MAX_ACTIONS, "episode_s": MAX_EPISODE_S},
                            reason=None if ok else "declared_ceiling_outside_the_published_gate")
    '''),

    "evaluator_success_ramp": body('''
        # require_measured FIRST, before a single submitted byte is read. Reading the artifact
        # first lets a missing-file error mask the unmeasured signal: the verifier then reports
        # gradeable=true with r_det=0.0, which says the agent scored zero when the truth is that
        # the slot is not yet measurable. Those are different claims and only one of them is true.
        c = require_measured("floor", "knee", "dichotomizing_threshold")
        success = measure()
        span = c["knee"] - c["floor"]
        s = 0.0 if span <= 0 else max(0.0, min(1.0, (success - c["floor"]) / span))
        return success >= c["dichotomizing_threshold"], evidence(private_split_success=success,
                                                                 s_outcome=s)
    '''),
}


PRELUDE = '''"""Compiled checks for %(slot_id)s. GENERATED by solution/recompute.py. DO NOT HAND-EDIT.

Every check runs against delivered bytes. Two sources are authoritative and neither belongs to
the scaffold: tests/heldout/ holds the verifier's own manifest for the 300 private instances,
and ../run_record.json holds what the harness metered per episode. The graded quantity is
%(metric)s.

evaluator_success_ramp raises ConstantUnmeasured while tests/constants.json carries nulls, and
it does so before it touches the submission.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SUBMISSION = %(path)r
CHECKS = []

PRIVATE_SPLIT = "private_split_300.json"
MAX_ACTIONS = 30
MAX_EPISODE_S = 240.0
P95_EPISODE_S = 180.0
RESET_TOTAL_TOLERANCE_S = 0.5
BUDGET_STOPS = ("action_budget", "time_budget")
# The five retained sites. Postmill/Reddit is excluded by name: its Pushshift-derived content
# carries no redistributable licence, and the exclusion is a declared scope narrowing.
RETAINED_SITES = ("gitlab", "map", "shopping", "shopping_admin", "wikipedia")
EXCLUDED_SITES = ("reddit",)
# The one field each WebArena evaluator family reads out of a terminal record. The evaluator is
# external and unmodifiable, so this mapping is a fact about the grader, not a choice.
EVALUATOR_FIELD = {
    "exact_match": "answer",
    "must_include": "answer",
    "fuzzy_match": "answer",
    "url_match": "final_url",
    "program_html": "site_end_state_hash",
}
EXTERNAL_GRADER_TREES = ("webarena_evaluators", "task_json", "reward_code")
INDUCED_ARTIFACTS = ("workflow_memory", "observation_id_map")
DECLARED_BUDGET_KEYS = ("per_instance_action_ceiling", "per_instance_latency_ceiling_s",
                        "episode_concurrency")


class ConstantUnmeasured(Exception):
    """Raised when a check needs a ramp constant that Phase 0 deliberately left null."""


def check(id, weight, knockout, kind):
    def deco(fn):
        CHECKS.append({"id": id, "weight": weight, "knockout": knockout, "kind": kind, "fn": fn})
        return fn
    return deco


def constants():
    with open(os.path.join(HERE, "constants.json")) as fh:
        return json.load(fh)


def require_measured(*names):
    c = constants()
    missing = [n for n in names if c.get(n) is None]
    if missing:
        raise ConstantUnmeasured(
            "%(slot_id)s: " + ", ".join(missing) + " are null. Phase 0 authors no measurement; "
            "the Phase 2 measurement wave writes these and this check becomes gradeable then.")
    return c


def evidence(**kw):
    return {"evaluation_target": kw.pop("target", "submitted_artifact"), "values": kw}
'''


TAIL = '''
def heldout(name):
    return os.path.join(HERE, "heldout", name)


def graded_split():
    """The verifier's own manifest for the 300 private instances, never the agent's copy.

    Returns None when the manifest is absent, so a check reports the missing manifest instead of
    silently grading the scaffold against its own idea of what the split contains.
    """
    path = heldout(PRIVATE_SPLIT)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        doc = json.load(handle)
    rows = doc.get("instances") if isinstance(doc, dict) else doc
    return rows if isinstance(rows, list) else None


def graded_instance_ids():
    rows = graded_split()
    if rows is None:
        return None
    return set(str(row.get("instance_id")) for row in rows)


def graded_field(name):
    """One verifier-held column of the private split, keyed by instance id."""
    return dict((str(row.get("instance_id")), row.get(name)) for row in (graded_split() or []))


def load_submission():
    """The delivered run ledger: an object with an ids list and matching per-episode lists."""
    with open(SUBMISSION) as handle:
        doc = json.load(handle)
    if not isinstance(doc, dict):
        raise ValueError("the deliverable must be a JSON object")
    return doc


def terminal_records():
    """(instance id, verifier-held evaluator family, terminal record) for each graded instance.

    The family comes from the manifest rather than from the submission: which evaluator scores an
    instance is the grader's fact, and a scaffold that declared its own family would be choosing
    the field it is then checked against.
    """
    doc = load_submission()
    ids, outputs = doc.get("ids"), doc.get("outputs")
    if not isinstance(ids, list) or not isinstance(outputs, list):
        raise ValueError("the ledger must carry an ids list and an outputs list")
    if len(ids) != len(outputs):
        raise ValueError("outputs must align one-to-one with ids")
    families = graded_field("evaluator_family")
    if not families:
        return None
    rows = []
    for iid, record in zip(ids, outputs):
        key = str(iid)
        if key in families:
            rows.append((key, families[key], record))
    return rows


def run_record():
    path = os.path.join(HERE, "..", "run_record.json")
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def harness_episodes():
    """Per-episode metering the harness performed itself.

    Never reward_log.jsonl: the per-instance gate is the one thing a scaffold has a direct
    incentive to under-report, so the number that binds it is measured host-side.
    """
    rows = run_record().get("harness_metered_episodes")
    return rows if isinstance(rows, list) else []


def number(row, key):
    """Numeric field or None. A field that is missing or unparseable is not a zero."""
    try:
        return float(row.get(key))
    except (TypeError, ValueError):
        return None


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * (pct / 100.0)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def digest(path):
    import hashlib
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def measure():
    """Private-split task success exactly as WebArena's own evaluators tallied it.

    Read from the harness tally over verifier-owned site fixtures. The scaffold neither computes
    nor reports this number: the evaluator is external to the submission by construction, and a
    value taken from the agent's ledger would be the agent grading itself. The graded-phase wall
    clock is required alongside it because the ramp above the success gate rides on that clock,
    and a tally without it cannot be turned into a reward.
    """
    record = run_record()
    tally = record.get("evaluator_success_tally")
    if not isinstance(tally, dict):
        raise ConstantUnmeasured("the harness recorded no evaluator success tally")
    solved, scored = tally.get("solved"), tally.get("scored")
    if solved is None or not scored:
        raise ConstantUnmeasured("the evaluator tally carries no solved/scored pair")
    if record.get("graded_phase_wall_clock_s") is None:
        raise ConstantUnmeasured("the graded-phase wall clock the above-gate ramp rides on was "
                                 "not metered")
    return float(solved) / float(scored)
'''


def main():
    g = load_grounding()
    # The family validator runs FIRST and refuses rather than emitting: a generator that writes a
    # carrier it knows is malformed hands the drift check a moving target.
    problems = validate(g)
    if problems:
        raise SystemExit("C7-S2 grounding is not well-formed: " + "; ".join(problems))

    bodies = dict(BODIES)
    # REFUSE, never fall through. The previous generator handed an un-bodied compiled item a
    # generic obligation lookup against run_record["obligations"][ID]; nothing writes that key,
    # so the emitted check consulted nothing and still reported a verdict.
    for it in g["items"]:
        if it["mode"] == "compiled" and it["id"] not in bodies:
            raise SystemExit("C7-S2: no body is written for compiled item %r. A compiled item "
                             "without a body is a check about nothing." % it["id"])
    orphans = sorted(set(bodies) - {i["id"] for i in g["items"]})
    if orphans:
        raise SystemExit("C7-S2: bodies written for items grounding does not carry: %s"
                         % ", ".join(orphans))
    outcome = [i["id"] for i in g["items"] if i["dimension"] == "outcome"][0]
    statements = [ln.strip() for ln in bodies[outcome].splitlines()
                  if ln.strip() and not ln.strip().startswith("#")]
    if not statements or not statements[0].startswith(OUTCOME_OPENER):
        raise SystemExit("C7-S2: the outcome item must call require_measured before it reads any "
                         "submitted byte. Reading the artifact first lets a missing-file error "
                         "mask the unmeasured signal, and the verifier then reports "
                         "gradeable=true with r_det=0.0.")

    dl = g["deliverable"]
    metric = "Task success rate on the 300-instance private held-out split under WebArena's own programmatic evaluators, gated by a per-episode latency and action-count SLO, with graded-phase wall-clock seconds as the optimized axis"
    subst = {"slot_id": SLOT_ID, "path": dl["path"], "kind": dl["kind"],
             "shape": dl["shape"], "metric": metric}

    tests_dir = os.path.join(BUNDLE, "tests")
    os.makedirs(tests_dir, exist_ok=True)

    # solution/TRUTH.md and solution/rubrics.json are PRIVATE_CARRIERS: seed/identity.py plants a
    # canary block into them at freeze, and canary normalisation is what keeps planting from
    # moving identity. A generator that rewrote them plainly would DROP the planted tripwire, and
    # the next content_hash would differ from the frozen one. So the planted block is carried
    # across the regeneration rather than recreated.
    truth_path = os.path.join(HERE, "TRUTH.md")
    truth = build_truth(g, TITLE).rstrip("\n") + "\n"
    keep = None
    if os.path.exists(truth_path):
        import re as _re
        m = _re.search(r"<!-- AELLO-CANARY-BLOCK.*?-->\n?", open(truth_path).read(), _re.S)
        keep = m.group(0) if m else None
    with open(truth_path, "w") as fh:
        fh.write(truth + ("\n" + keep if keep else ""))

    rub_path = os.path.join(HERE, "rubrics.json")
    rub = build_rubrics(g, ITEM_KEYS)
    if os.path.exists(rub_path):
        try:
            prev = json.load(open(rub_path))
            if "canary" in prev:
                rub["canary"] = prev["canary"]
        except Exception:
            pass
    with open(rub_path, "w") as fh:
        json.dump(rub, fh, indent=1, sort_keys=True)
        fh.write("\n")
    with open(os.path.join(tests_dir, "rubrics.json"), "w") as fh:
        json.dump(build_rubrics(g, ITEM_KEYS), fh, indent=1, sort_keys=True)
        fh.write("\n")
    with open(os.path.join(tests_dir, "test_output.py"), "w") as fh:
        # The tail arrives FINAL. It is already resolved and legitimately carries percent signs
        # of its own, so formatting it a second time here raised TypeError on the first slot that
        # carried one. Only the prelude is substituted.
        fh.write(build_tests(g, PRELUDE % subst, bodies) + TAIL)
    with open(os.path.join(tests_dir, "constants.json"), "w") as fh:
        json.dump(g["constants"], fh, indent=1, sort_keys=True)
        fh.write("\n")
    # ORDER MATTERS. `with open(path, "w")` truncates BEFORE its body's arguments are evaluated,
    # so calling read_identity() inside the with-statement reads the file this line just emptied
    # and writes nulls over the frozen identity. Read first, then open.
    prov = build_provenance(g, read_identity(HERE))
    with open(os.path.join(HERE, "provenance.yaml"), "w") as fh:
        # MATCH seed/build/screen_bind.py and the frozen generators exactly: sort_keys=True,
        # default_flow_style=False, width=100. The screen writes this carrier after the freeze,
        # so a generator using a different dump convention reorders the keys on the next
        # regeneration and reports as drift even though no value changed.
        yaml.safe_dump(prov, fh, sort_keys=True, default_flow_style=False, width=100,
                       allow_unicode=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
