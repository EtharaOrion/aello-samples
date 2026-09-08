"""Derivation for C2-S2: every canonical artifact descends from solution/grounding.yaml.

The graded object here is an ORCHESTRATION over eight pinned, digest-locked stage tools, so the
compiled checks this file emits are about composition: invocation order against a declared
partial order, digest closure across seven link boundaries, one sandbox epoch pinned for the
whole run, twelve interface arguments bound inside published domains, and a composite over two
axes whose weights instruction.md publishes verbatim. build_provenance stays here, per family,
for the reason recorded in seed/build/generator_common.py.
"""
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)
# generator_common.py is VENDORED beside this file, not reached for outside the bundle. A staged
# tree could resolve seed/stage/<slug>/../../build; a DELIVERED tree at dataset/<uuid>/ cannot,
# and reaching outside would make the delivered bundle non-self-contained -- it would import in
# the stage and fail everywhere it actually ships.


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


def build_tests(g, prelude, bodies):
    """Assemble the compiled-check module."""
    chunks = [prelude]
    for raw in sorted(g["items"], key=lambda i: i["id"]):
        if raw["mode"] != "compiled":
            continue
        chunks.append(
            '\n@check(id="%s", weight=%d, knockout=%s, kind="process")\ndef test_%s():\n%s\n'
            % (raw["id"], raw["weight"], raw["knockout"], raw["id"], bodies[raw["id"]]))
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


ITEM_KEYS = ('id', 'dimension', 'weight', 'evaluation_target', 'criterion', 'judgment', 'knockout', 'mode')
SLOT_ID = 'C2-S2'
TITLE = "# C2-S2 -- what this task actually tests"

# The outcome item, named once. Its body must open on require_measured, and main() refuses if it
# does not: a check that reads the submission first turns a missing chain manifest into a scored
# zero and hides the fact that no ramp constant has been measured yet.
OUTCOME_ITEM = "chain_end_state_accuracy_ramp"

# Facts without which this slot's rubric cannot tell a composed chain from a chain that merely
# ran. Digest closure, epoch binding, the pinned tool digests and the twelve bound arguments are
# the four the archetype rests on; a rubric that drops any of them grades an eight-stage
# composition by whether all eight stages exited 0, which every wrong composition also does.
REQUIRED_ASSERTIONS = ("consumed_digest_resolves_to_an_earlier_producing_stage",
                       "invoked_stage_digest_equals_the_pinned_tool_digest",
                       "score_epoch_equals_the_rollout_and_seed_epoch",
                       "twelve_interface_arguments_bound_inside_published_domains")
RAMP_CONSTANTS = ("dichotomizing_threshold", "floor", "knee", "reward_gate_pass_threshold")


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
    """Refuse to emit a rubric this slot's own mechanism would not be measured by.

    A timed axis is graded on harness measurement and never on a self-reported duration, so some
    item has to name the run record. The four required assertions above are the composition
    facts; without them the carrier grades a deliverable's existence and shape, which every one
    of the twelve single-link mis-bindings also satisfies.
    """
    problems = []
    targets = {i["evaluation_target"] for i in grounding["items"]}
    if "run_record" not in targets:
        problems.append("a timed slot must grade against the harness run record")
    carried = {i["id"] for i in grounding["items"]}
    for name in REQUIRED_ASSERTIONS:
        if name not in carried:
            problems.append("no item asserts %s" % name)
    if OUTCOME_ITEM not in carried:
        problems.append("no item carries the outcome ramp %s" % OUTCOME_ITEM)
    uncompiled = sorted(i["id"] for i in grounding["items"] if i["mode"] != "compiled")
    if uncompiled:
        problems.append("compilation_floor is 1.0 but these items are not compiled: "
                        + ", ".join(uncompiled))
    for name in RAMP_CONSTANTS:
        if grounding["constants"][name] is not None:
            problems.append("%s is authored at Phase 0, which the convention forbids" % name)
    return problems


def first_statement(body):
    """The first line of a body that is neither blank nor a comment."""
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return ""


def main():
    g = load_grounding()
    # The family validator runs FIRST and refuses rather than emitting: a generator that writes a
    # carrier it knows is malformed hands the drift check a moving target.
    problems = validate(g)
    if problems:
        raise SystemExit("C2-S2 grounding is not well-formed: " + "; ".join(problems))
    dl = g["deliverable"]
    metric = "Chain end-state accuracy over 1200 private executable tool-call episodes, priced against harness-measured wall clock through the two-axis composite instruction.md publishes verbatim: r_det = quality_fraction * (0.55 + 0.45 * speed_fraction). An episode counts only when the emitted call chain executes to completion in the pinned sandbox, the backend end-state digest equals gold's exactly, and every order-dependent pair appears in gold's relative order. There is no per-call partial credit"
    subst = {"slot_id": SLOT_ID, "path": dl["path"], "kind": dl["kind"],
             "shape": dl["shape"], "metric": metric}

    # BODIES ARE WRITTEN FOR THIS SLOT. Each one reads a fact only an eight-stage digest-locked
    # composition has: which pinned tool ran at which position, what each invocation consumed and
    # produced, which sandbox epoch a scored record is bound to, how the twelve interface
    # arguments were bound across the seven link boundaries, and whether the composite the
    # submission restates is the one instruction.md publishes. The generic spine that stood here
    # graded the existence and shape of a deliverable and looked up an obligation record by name;
    # every one of the twelve single-link mis-bindings passes a check like that.
    bodies = {
        "chain_end_state_accuracy_ramp": (
            '    c = require_measured("floor", "knee", "dichotomizing_threshold")\n'
            '    # require_measured is the FIRST statement, before any read of the submission or the run\n'
            '    # record. Reading the chain manifest first lets a missing-file error mask the unmeasured\n'
            '    # signal: the verifier then reports gradeable=true with r_det=0.0, which collapses "not\n'
            '    # yet measurable" into "the agent scored zero". Those are different claims.\n'
            '    accuracy = measure()\n'
            '    span = c["knee"] - c["floor"]\n'
            '    s = 0.0 if span <= 0 else max(0.0, min(1.0, (accuracy - c["floor"]) / span))\n'
            '    reach = T - BASE_ACCURACY\n'
            '    quality = 0.0 if reach <= 0 else max(0.0, min(1.0, (accuracy - BASE_ACCURACY) / reach))\n'
            '    return accuracy >= c["dichotomizing_threshold"], evidence(\n'
            '        target="episode_ledger", chain_end_state_accuracy=accuracy,\n'
            '        quality_fraction=quality, s_outcome=s)'
        ),

        "chain_manifest_records_every_stage_invocation": (
            '    if not os.path.exists(SUBMISSION):\n'
            '        return False, evidence(reason="submission_missing", path=SUBMISSION)\n'
            '    try:\n'
            '        doc = load_submission()\n'
            '    except Exception as exc:\n'
            '        return False, evidence(reason="manifest_malformed", error=str(exc))\n'
            '    absent = [name for name in SECTIONS if name not in doc]\n'
            '    chain = doc.get("chain_manifest")\n'
            '    listed = isinstance(chain, list) and len(chain) > 0\n'
            '    ok = (not absent) and listed\n'
            '    return ok, evidence(path=SUBMISSION, carried=sorted(doc), absent=absent,\n'
            '                        invocations=len(chain) if isinstance(chain, list) else None,\n'
            '                        reason=None if ok else "manifest_malformed")'
        ),

        "consumed_digest_resolves_to_an_earlier_producing_stage": (
            '    try:\n'
            '        records = invocations()\n'
            '    except Exception as exc:\n'
            '        return False, evidence(target="chain_manifest", reason="manifest_malformed",\n'
            '                               error=str(exc))\n'
            '    producers = artifact_producers()\n'
            '    if producers is None:\n'
            '        return False, evidence(target="chain_manifest",\n'
            '                               reason="artifact_producers_unpublished")\n'
            '    # latest[artifact name] = (digest, producing stage) as the chain runs forward. Closure is\n'
            '    # per ARTIFACT NAME and against the MOST RECENT digest under it, not against an acyclic\n'
            '    # predecessor: the GRPO cycle re-enters rollout, score and update many times and update\n'
            '    # legitimately re-produces merged_weights for the next rollout. What is never legitimate\n'
            '    # is eating a digest a later invocation has already superseded -- which is exactly what\n'
            "    # the pinned score stage does when it is pointed at a previous iteration's rollout file,\n"
            '    # accepting it and exiting 0.\n'
            '    latest, broken = {}, []\n'
            '    for index, record in enumerate(records):\n'
            '        stage = record.get("stage")\n'
            '        for entry in (record.get("consumed") or []):\n'
            '            if not isinstance(entry, dict):\n'
            '                broken.append({"invocation": index, "why": "consumed_entry_malformed"})\n'
            '                continue\n'
            '            name, digest = entry.get("artifact"), entry.get("sha256")\n'
            '            if name not in producers:\n'
            '                broken.append({"invocation": index, "stage": stage, "artifact": name,\n'
            '                               "why": "artifact_not_declared"})\n'
            '                continue\n'
            '            if name not in latest:\n'
            '                broken.append({"invocation": index, "stage": stage, "artifact": name,\n'
            '                               "why": "no_earlier_producer"})\n'
            '                continue\n'
            '            produced_digest, producer = latest[name]\n'
            '            if producer not in producers[name]:\n'
            '                broken.append({"invocation": index, "stage": stage, "artifact": name,\n'
            '                               "producer": producer,\n'
            '                               "why": "producer_not_declared_for_this_artifact"})\n'
            '            elif digest != produced_digest:\n'
            '                broken.append({"invocation": index, "stage": stage, "artifact": name,\n'
            '                               "why": "superseded_digest_consumed"})\n'
            '        for entry in (record.get("produced") or []):\n'
            '            if isinstance(entry, dict) and is_digest(entry.get("sha256")):\n'
            '                latest[entry.get("artifact")] = (entry["sha256"], stage)\n'
            '    return (not broken), evidence(target="chain_manifest", offending=broken[:6],\n'
            '                                  artifacts_produced=len(latest),\n'
            '                                  reason=None if not broken else "chain_closure_broken")'
        ),

        "each_single_link_misbinding_alone_falls_below_t1": (
            '    table = run_record().get("link_ablation_accuracy")\n'
            '    if not isinstance(table, dict) or not table:\n'
            '        return False, evidence(target="run_record", reason="link_ablation_unrecorded")\n'
            '    absent = [name for name in BOUND_ARGUMENTS if name not in table]\n'
            '    try:\n'
            '        measured = {name: float(table[name]) for name in BOUND_ARGUMENTS if name in table}\n'
            '    except Exception as exc:\n'
            '        return False, evidence(target="run_record", reason="link_ablation_unparseable",\n'
            '                               error=str(exc))\n'
            '    optional = sorted(name for name, value in measured.items() if value >= T1)\n'
            '    ok = (not absent) and (not optional) and len(measured) == len(BOUND_ARGUMENTS)\n'
            '    return ok, evidence(target="run_record", arguments_unablated=absent,\n'
            '                        links_above_t1=optional, t1=T1, ablations=len(measured),\n'
            '                        reason=None if ok else "link_not_individually_binding")'
        ),

        "end_state_digest_equality_with_no_per_call_credit": (
            '    book = ledger()\n'
            '    if book is None:\n'
            '        return False, evidence(target="episode_ledger", reason="episode_ledger_unrecorded")\n'
            '    record = run_record()\n'
            '    basis = record.get("counting_basis")\n'
            '    per_call = record.get("per_call_accuracy")\n'
            '    faults = []\n'
            '    if basis != COUNTING_BASIS:\n'
            '        faults.append({"why": "counting_basis_is_not_end_state_equality", "declared": basis})\n'
            '    if not isinstance(per_call, (int, float)) or isinstance(per_call, bool):\n'
            '        faults.append({"why": "per_call_accuracy_not_recorded_separately"})\n'
            '    for name in sorted(STRATUM_TOTALS):\n'
            '        row = book.get(name) if isinstance(book.get(name), dict) else {}\n'
            '        counted = row.get("counted")\n'
            '        equal = row.get("end_state_equal")\n'
            '        ran = row.get("executed_to_completion")\n'
            '        violations = row.get("order_violations")\n'
            '        if not all(isinstance(v, int) and not isinstance(v, bool)\n'
            '                   for v in (counted, equal, ran, violations)):\n'
            '            faults.append({"stratum": name, "why": "accounting_incomplete"})\n'
            '            continue\n'
            '        if not (ran <= STRATUM_TOTALS[name] and equal <= ran):\n'
            '            faults.append({"stratum": name, "why": "more_equal_than_executed"})\n'
            '        if counted != equal - violations:\n'
            '            faults.append({"stratum": name, "why": "counted_ignores_order_dependent_pairs",\n'
            '                           "counted": counted, "end_state_equal": equal,\n'
            '                           "order_violations": violations})\n'
            '    ok = not faults\n'
            '    return ok, evidence(target="episode_ledger", counting_basis=basis,\n'
            '                        per_call_accuracy=per_call, offending=faults[:6],\n'
            '                        reason=None if ok else "per_call_credit_leaked_into_the_numerator")'
        ),

        "every_invocation_names_consumed_and_produced_digests": (
            '    try:\n'
            '        records = invocations()\n'
            '    except Exception as exc:\n'
            '        return False, evidence(target="chain_manifest", reason="manifest_malformed",\n'
            '                               error=str(exc))\n'
            '    faults = []\n'
            '    for index, record in enumerate(records):\n'
            '        absent = [key for key in INVOCATION_KEYS if key not in record]\n'
            '        if absent:\n'
            '            faults.append({"invocation": index, "absent_keys": absent})\n'
            '            continue\n'
            '        if not isinstance(record.get("bindings"), dict):\n'
            '            faults.append({"invocation": index, "why": "bindings_not_a_map"})\n'
            '        for side in ("consumed", "produced"):\n'
            '            entries = record.get(side)\n'
            '            if not isinstance(entries, list):\n'
            '                faults.append({"invocation": index, "side": side, "why": "not_a_list"})\n'
            '                continue\n'
            '            for entry in entries:\n'
            '                if not isinstance(entry, dict) or "artifact" not in entry \\\n'
            '                        or not is_digest(entry.get("sha256")):\n'
            '                    faults.append({"invocation": index, "side": side,\n'
            '                                   "why": "not_an_artifact_digest_pair"})\n'
            '        if not (record.get("produced") or []):\n'
            '            faults.append({"invocation": index, "side": "produced",\n'
            '                           "why": "produced_nothing"})\n'
            '        if not (record.get("consumed") or []) and record.get("stage") != "seed":\n'
            '            faults.append({"invocation": index, "side": "consumed",\n'
            '                           "why": "only_seed_may_consume_nothing"})\n'
            '    return (not faults), evidence(target="chain_manifest", invocations=len(records),\n'
            '                                  offending=faults[:6],\n'
            '                                  reason=None if not faults else "manifest_malformed")'
        ),

        "export_form_merged_carries_the_render_spec_digest": (
            '    try:\n'
            '        export = section("export_record")\n'
            '        records = invocations()\n'
            '    except Exception as exc:\n'
            '        return False, evidence(reason="manifest_malformed", error=str(exc))\n'
            '    form = export.get("export_form")\n'
            '    spec = export.get("render_spec_digest")\n'
            '    rendered = {entry.get("sha256") for record in records if record.get("stage") == "render"\n'
            '                for entry in (record.get("produced") or []) if isinstance(entry, dict)}\n'
            '    merged = form == "merged"\n'
            '    carried = is_digest(spec) and spec in rendered\n'
            '    ok = merged and carried\n'
            '    return ok, evidence(export_form=form, render_spec_digest=spec,\n'
            '                        render_produced=len(rendered),\n'
            '                        reason=None if ok else\n'
            '                        ("export_form_rejected" if not merged\n'
            '                         else "render_spec_digest_unresolved"))'
        ),

        "generation_prompt_pinned_to_one_value_at_render_and_rollout": (
            '    try:\n'
            '        bound = section("argument_bindings").get("--generation-prompt")\n'
            '        records = invocations()\n'
            '    except Exception as exc:\n'
            '        return False, evidence(target="argument_bindings", reason="manifest_malformed",\n'
            '                               error=str(exc))\n'
            '    at = {}\n'
            '    for record in records:\n'
            '        stage = record.get("stage")\n'
            '        if stage in ("render", "rollout"):\n'
            '            at.setdefault(stage, set()).add(binding_at(record, "--generation-prompt"))\n'
            '    rendered = at.get("render", set())\n'
            '    rolled = at.get("rollout", set())\n'
            '    explicit = bound is not None and bound != "auto"\n'
            '    pinned = len(rendered) == 1 and rendered == rolled and rendered == {bound}\n'
            '    ok = explicit and pinned\n'
            '    return ok, evidence(target="argument_bindings", bound=bound,\n'
            '                        at_render=sorted(map(str, rendered)),\n'
            '                        at_rollout=sorted(map(str, rolled)),\n'
            '                        reason=None if ok\n'
            '                        else "generation_prompt_differs_across_the_render_rollout_link")'
        ),

        "grader_reason_codes_equal_the_validate_export_codes": (
            '    try:\n'
            '        in_environment = section("export_record").get("validate_export_reason_codes")\n'
            '    except Exception as exc:\n'
            '        return False, evidence(target="run_record", reason="manifest_malformed",\n'
            '                               error=str(exc))\n'
            '    at_grading = run_record().get("grader_reason_codes")\n'
            '    if not isinstance(in_environment, list) or not isinstance(at_grading, list):\n'
            '        return False, evidence(target="run_record", reason="reason_codes_unrecorded")\n'
            '    emitted = {str(code) for code in in_environment}\n'
            '    graded = {str(code) for code in at_grading}\n'
            '    unpublished = sorted((emitted | graded) - set(ADMISSIBILITY_REASON_CODES))\n'
            '    ok = bool(emitted) and emitted == graded and not unpublished\n'
            '    return ok, evidence(target="run_record", only_in_environment=sorted(emitted - graded),\n'
            '                        only_at_grading=sorted(graded - emitted),\n'
            '                        outside_the_published_list=unpublished,\n'
            '                        reason=None if ok else "admissibility_vocabulary_diverged")'
        ),

        "haystack_and_transfer_floors_cap_quality_at_060": (
            '    book = ledger()\n'
            '    if book is None:\n'
            '        return False, evidence(target="episode_ledger", reason="episode_ledger_unrecorded")\n'
            '    haystack = stratum_accuracy(book, "haystack")\n'
            '    transfer = stratum_accuracy(book, "transfer")\n'
            '    if haystack is None or transfer is None:\n'
            '        return False, evidence(target="episode_ledger", acc_haystack=haystack,\n'
            '                               acc_transfer=transfer, reason="stratum_counts_unreadable")\n'
            '    haystack_floor = T - HAYSTACK_MARGIN\n'
            '    transfer_floor = T - TRANSFER_MARGIN\n'
            '    ok = haystack >= haystack_floor and transfer >= transfer_floor\n'
            '    return ok, evidence(target="episode_ledger", acc_haystack=haystack,\n'
            '                        acc_transfer=transfer, haystack_floor=haystack_floor,\n'
            '                        transfer_floor=transfer_floor,\n'
            '                        quality_cap_when_failed=STRATUM_CAP,\n'
            '                        reason=None if ok else "stratum_floor_failed")'
        ),

        "invoked_stage_digest_equals_the_pinned_tool_digest": (
            '    try:\n'
            '        declared = section("stage_digests")\n'
            '        records = invocations()\n'
            '    except Exception as exc:\n'
            '        return False, evidence(target="stage_tree", reason="manifest_malformed",\n'
            '                               error=str(exc))\n'
            '    pinned = run_record().get("pinned_stage_digests")\n'
            '    if not isinstance(pinned, dict) or not pinned:\n'
            '        return False, evidence(target="stage_tree", reason="pinned_stage_digests_unrecorded")\n'
            '    absent = [stage for stage in STAGES if stage not in declared]\n'
            '    added = sorted(set(map(str, declared)) - set(STAGES))\n'
            '    drifted = sorted(stage for stage in STAGES\n'
            '                     if stage in declared and stage in pinned\n'
            '                     and declared[stage] != pinned[stage])\n'
            '    unpinned = [stage for stage in STAGES if stage not in pinned]\n'
            '    shimmed = sorted({record.get("stage") for record in records\n'
            '                      if record.get("stage") in pinned\n'
            '                      and record.get("stage_digest") != pinned[record.get("stage")]})\n'
            '    ok = not (absent or added or drifted or unpinned or shimmed)\n'
            '    return ok, evidence(target="stage_tree", tools_absent=absent, tools_added=added,\n'
            '                        digests_drifted=drifted, tools_unpinned=unpinned,\n'
            '                        invoked_off_pin=shimmed,\n'
            '                        reason=None if ok else "stage_tree_written")'
        ),

        "mask_convention_postshift_because_sft_shifts_labels_itself": (
            '    try:\n'
            '        bound = section("argument_bindings").get("--mask-convention")\n'
            '    except Exception as exc:\n'
            '        return False, evidence(target="argument_bindings", reason="manifest_malformed",\n'
            '                               error=str(exc))\n'
            '    ok = bound == "postshift"\n'
            '    return ok, evidence(target="argument_bindings", bound=bound, required="postshift",\n'
            '                        stage_that_shifts_labels_internally="sft",\n'
            '                        reason=None if ok\n'
            '                        else "preshift_mask_into_a_stage_that_shifts")'
        ),

        "no_private_episode_path_opened_and_no_gold_digest_carried": (
            '    record = run_record()\n'
            '    prefix = record.get("private_episode_tree") or PRIVATE_EPISODE_TREE\n'
            '    mounts = record.get("mount_manifest")\n'
            '    opened = record.get("opened_paths_under_private_tree")\n'
            '    if not isinstance(mounts, list) or not isinstance(opened, list):\n'
            '        return False, evidence(target="run_record", reason="mount_manifest_unrecorded")\n'
            '    mounted = sorted(str(p) for p in mounts if str(p).startswith(str(prefix)))\n'
            '    reached = sorted(str(p) for p in opened)\n'
            '    try:\n'
            '        words, numbers = harvest(load_submission(), set(), set())\n'
            '    except Exception as exc:\n'
            '        return False, evidence(target="run_record", reason="manifest_malformed",\n'
            '                               error=str(exc))\n'
            '    gold = {str(d).lower() for d in (record.get("gold_end_state_digests") or [])}\n'
            '    leaked = sorted(word for word in words if word in gold)\n'
            '    pinned = sorted(str(name) for name, value in constants().items()\n'
            '                    if isinstance(value, (int, float)) and not isinstance(value, bool)\n'
            '                    and float(value) in numbers)\n'
            '    ok = not (mounted or reached or leaked or pinned)\n'
            '    return ok, evidence(target="run_record", private_tree=str(prefix),\n'
            '                        mounted_under_private_tree=mounted[:4], opened=reached[:4],\n'
            '                        gold_digests_in_submission=len(leaked),\n'
            '                        pinned_constants_in_submission=pinned,\n'
            '                        reason=None if ok else "boundary_violation")'
        ),

        "one_seed_call_pins_the_epoch_for_the_whole_run": (
            '    try:\n'
            '        records = invocations()\n'
            '    except Exception as exc:\n'
            '        return False, evidence(target="chain_manifest", reason="manifest_malformed",\n'
            '                               error=str(exc))\n'
            '    seeds = [i for i, r in enumerate(records) if r.get("stage") == "seed"]\n'
            '    bound = [i for i, r in enumerate(records) if r.get("stage") in ("rollout", "score")]\n'
            '    if not bound:\n'
            '        return False, evidence(target="chain_manifest", seed_invocations=len(seeds),\n'
            '                               reason="no_rollout_or_score_to_bind")\n'
            '    once = len(seeds) == 1\n'
            '    ahead = bool(seeds) and max(seeds) < min(bound)\n'
            '    ok = once and ahead\n'
            '    return ok, evidence(target="chain_manifest", seed_invocations=seeds,\n'
            '                        first_bound_stage=min(bound), pinned_once=once, hoisted=ahead,\n'
            '                        reason=None if ok else "epoch_rebased_mid_chain")'
        ),

        "pad_side_agrees_between_rollout_and_update": (
            '    try:\n'
            '        bound = section("argument_bindings").get("--pad-side")\n'
            '        records = invocations()\n'
            '    except Exception as exc:\n'
            '        return False, evidence(target="argument_bindings", reason="manifest_malformed",\n'
            '                               error=str(exc))\n'
            '    emitted = {binding_at(r, "--pad-side") for r in records if r.get("stage") == "rollout"}\n'
            '    consumed = {binding_at(r, "--pad-side") for r in records if r.get("stage") == "update"}\n'
            '    ok = (len(emitted) == 1 and emitted == consumed and None not in emitted\n'
            '          and emitted == {bound})\n'
            '    return ok, evidence(target="argument_bindings", bound=bound,\n'
            '                        rollout_emitted=sorted(map(str, emitted)),\n'
            '                        update_assumed=sorted(map(str, consumed)),\n'
            '                        reason=None if ok else "pad_side_differs_across_the_kl_link")'
        ),

        "per_stage_seconds_declared_for_all_eight_stages": (
            '    try:\n'
            '        seconds = section("stage_seconds")\n'
            '    except Exception as exc:\n'
            '        return False, evidence(reason="no_agent_declaration", error=str(exc))\n'
            '    absent = [stage for stage in STAGES if stage not in seconds]\n'
            '    unparseable = []\n'
            '    for stage in STAGES:\n'
            '        if stage not in seconds:\n'
            '            continue\n'
            '        value = seconds[stage]\n'
            '        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:\n'
            '            unparseable.append(stage)\n'
            '    ok = not absent and not unparseable\n'
            '    return ok, evidence(stages_undeclared=absent, stages_unparseable=unparseable,\n'
            '                        declared_total_s=sum(v for v in seconds.values()\n'
            '                                             if isinstance(v, (int, float))\n'
            '                                             and not isinstance(v, bool)),\n'
            '                        reason=None if ok else\n'
            '                        ("no_agent_declaration" if absent else "declaration_unparseable"))'
        ),

        "realized_sequence_linearizes_the_declared_partial_order": (
            '    try:\n'
            '        records = invocations()\n'
            '    except Exception as exc:\n'
            '        return False, evidence(target="chain_manifest", reason="manifest_malformed",\n'
            '                               error=str(exc))\n'
            '    edges = declared_edges()\n'
            '    if edges is None:\n'
            '        return False, evidence(target="chain_manifest",\n'
            '                               reason="declared_partial_order_unpublished")\n'
            '    first = {}\n'
            '    for index, record in enumerate(records):\n'
            '        first.setdefault(record.get("stage"), index)\n'
            '    absent = [stage for stage in STAGES if stage not in first]\n'
            '    # First occurrence, not every occurrence: the GRPO cycle re-enters rollout, score and\n'
            '    # update many times, and a linearization of a partial order constrains where a stage may\n'
            '    # START, not how often it may repeat once its predecessors have run.\n'
            '    violated = [{"edge": "%s->%s" % (a, b), "first_seen": [first[a], first[b]]}\n'
            '                for a, b in edges if a in first and b in first and first[a] > first[b]]\n'
            '    ok = (not absent) and (not violated)\n'
            '    return ok, evidence(target="chain_manifest", stages_absent=absent, violated=violated,\n'
            '                        realized=[r.get("stage") for r in records][:12],\n'
            '                        reason=None if ok else "stage_order_not_a_linearization")'
        ),

        "replay_on_a_fresh_epoch_reproduces_the_end_state_digests": (
            '    record = run_record()\n'
            '    replay = record.get("sandbox_replay")\n'
            '    if not isinstance(replay, dict) or not replay:\n'
            '        return False, evidence(target="run_record", reason="sandbox_replay_unrecorded")\n'
            '    first = replay.get("first_pass_digest")\n'
            '    second = replay.get("second_pass_digest")\n'
            '    if not (is_digest(first) and is_digest(second)):\n'
            '        return False, evidence(target="run_record", reason="replay_digests_unreadable",\n'
            '                               first=first, second=second)\n'
            '    stable = first == second\n'
            '    # A FRESHLY seeded epoch, not the one the graded run held. Replaying inside the graded\n'
            '    # epoch re-reads a world the run already mutated and would agree with itself for the\n'
            '    # wrong reason.\n'
            '    graded_epoch = replay.get("graded_epoch_id")\n'
            '    replay_epoch = replay.get("epoch_id")\n'
            '    fresh = bool(replay_epoch) and replay_epoch != graded_epoch\n'
            '    covered = replay.get("episodes_replayed")\n'
            '    complete = covered == EPISODE_TOTAL\n'
            '    ok = stable and fresh and complete\n'
            '    return ok, evidence(target="run_record", first_pass_digest=first,\n'
            '                        second_pass_digest=second, replay_epoch=replay_epoch,\n'
            '                        graded_epoch=graded_epoch, episodes_replayed=covered,\n'
            '                        reason=None if ok else\n'
            '                        ("end_state_digest_flapped" if not stable else\n'
            '                         "replay_epoch_not_fresh" if not fresh\n'
            '                         else "replay_incomplete"))'
        ),

        "score_epoch_equals_the_rollout_and_seed_epoch": (
            '    try:\n'
            '        records = invocations()\n'
            '    except Exception as exc:\n'
            '        return False, evidence(target="chain_manifest", reason="manifest_malformed",\n'
            '                               error=str(exc))\n'
            '    seeds = [r for r in records if r.get("stage") == "seed"]\n'
            '    if not seeds:\n'
            '        return False, evidence(target="chain_manifest", reason="epoch_unbound",\n'
            '                               why="no_seed_invocation_to_bind_against")\n'
            '    seed_epoch = seeds[0].get("epoch_id")\n'
            '    origin = {}\n'
            '    for record in records:\n'
            '        for entry in (record.get("produced") or []):\n'
            '            if isinstance(entry, dict) and is_digest(entry.get("sha256")):\n'
            '                origin[entry["sha256"]] = record\n'
            '    unbound = []\n'
            '    scored = 0\n'
            '    for index, record in enumerate(records):\n'
            '        if record.get("stage") != "score":\n'
            '            continue\n'
            '        scored += 1\n'
            '        if record.get("epoch_id") != seed_epoch:\n'
            '            unbound.append({"invocation": index, "why": "score_epoch_differs_from_seed"})\n'
            '        sources = [origin.get((e or {}).get("sha256")) for e in (record.get("consumed") or [])\n'
            '                   if isinstance(e, dict)]\n'
            '        rollouts = [s for s in sources if s is not None and s.get("stage") == "rollout"]\n'
            '        if not rollouts:\n'
            '            unbound.append({"invocation": index, "why": "score_consumed_no_rollout_artifact"})\n'
            '        for source in rollouts:\n'
            '            if source.get("epoch_id") != record.get("epoch_id"):\n'
            '                unbound.append({"invocation": index,\n'
            '                                "why": "rollout_epoch_differs_from_score_epoch"})\n'
            '    ok = scored > 0 and not unbound\n'
            '    return ok, evidence(target="chain_manifest", seed_epoch=seed_epoch,\n'
            '                        score_invocations=scored, offending=unbound[:6],\n'
            '                        reason=None if ok else "epoch_unbound")'
        ),

        "strata_counts_partition_the_1200_private_episodes": (
            '    book = ledger()\n'
            '    if book is None:\n'
            '        return False, evidence(target="episode_ledger", reason="episode_ledger_unrecorded")\n'
            '    absent = [name for name in sorted(STRATUM_TOTALS) if name not in book]\n'
            '    if absent:\n'
            '        return False, evidence(target="episode_ledger", strata_absent=absent,\n'
            '                               reason="episode_ledger_incomplete")\n'
            '    wrong_total, out_of_range = [], []\n'
            '    summed = 0\n'
            '    for name, expected in sorted(STRATUM_TOTALS.items()):\n'
            '        row = book[name] if isinstance(book[name], dict) else {}\n'
            '        total, counted = row.get("total"), row.get("counted")\n'
            '        if total != expected:\n'
            '            wrong_total.append({"stratum": name, "declared": total, "published": expected})\n'
            '        if not isinstance(counted, int) or counted < 0 or counted > expected:\n'
            '            out_of_range.append({"stratum": name, "counted": counted})\n'
            '        summed += expected\n'
            '    partitions = summed == EPISODE_TOTAL\n'
            '    ok = partitions and not wrong_total and not out_of_range\n'
            '    return ok, evidence(target="episode_ledger", summed_totals=summed,\n'
            '                        published_total=EPISODE_TOTAL, wrong_totals=wrong_total,\n'
            '                        counts_out_of_range=out_of_range,\n'
            '                        reason=None if ok else "strata_do_not_partition_the_split")'
        ),

        "timed_window_is_host_measured_not_self_reported": (
            '    record = run_record()\n'
            '    elapsed = record.get("harness_measured_elapsed_s")\n'
            '    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool):\n'
            '        return False, evidence(target="run_record", reason="harness_elapsed_unrecorded")\n'
            '    elapsed = float(elapsed)\n'
            '    ceiling = BUDGET_SECONDS + BUDGET_GRACE_SECONDS\n'
            '    within = elapsed <= ceiling\n'
            '    try:\n'
            '        records = invocations()\n'
            '    except Exception as exc:\n'
            '        return False, evidence(target="run_record", reason="manifest_malformed",\n'
            '                               error=str(exc))\n'
            '    outside = []\n'
            '    for index, invocation in enumerate(records):\n'
            '        start, end = invocation.get("started_s"), invocation.get("ended_s")\n'
            '        if not all(isinstance(v, (int, float)) and not isinstance(v, bool)\n'
            '                   for v in (start, end)):\n'
            '            outside.append({"invocation": index, "why": "offsets_unrecorded"})\n'
            '        elif start < 0 or end > elapsed or start > end:\n'
            '            outside.append({"invocation": index, "window": [start, end]})\n'
            '    ok = within and not outside\n'
            '    return ok, evidence(target="run_record", harness_measured_elapsed_s=elapsed,\n'
            '                        ceiling_with_grace=ceiling, offending=outside[:6],\n'
            '                        reason=None if ok else\n'
            '                        ("budget_exceeded" if not within\n'
            '                         else "invocation_outside_the_timed_window"))'
        ),

        "twelve_ablation_rewards_lie_inside_a_007_band": (
            '    table = run_record().get("link_ablation_accuracy")\n'
            '    if not isinstance(table, dict) or not table:\n'
            '        return False, evidence(target="run_record", reason="link_ablation_unrecorded")\n'
            '    try:\n'
            '        measured = [float(table[name]) for name in BOUND_ARGUMENTS if name in table]\n'
            '    except Exception as exc:\n'
            '        return False, evidence(target="run_record", reason="link_ablation_unparseable",\n'
            '                               error=str(exc))\n'
            '    complete = len(measured) == len(BOUND_ARGUMENTS)\n'
            '    width = (max(measured) - min(measured)) if measured else None\n'
            '    ok = complete and width is not None and width <= ABLATION_BAND\n'
            '    return ok, evidence(target="run_record", ablations=len(measured), band_width=width,\n'
            '                        band_ceiling=ABLATION_BAND,\n'
            '                        reason=None if ok\n'
            '                        else "ablation_band_localizes_the_broken_link")'
        ),

        "twelve_interface_arguments_bound_inside_published_domains": (
            '    try:\n'
            '        bindings = section("argument_bindings")\n'
            '    except Exception as exc:\n'
            '        return False, evidence(target="argument_bindings", reason="manifest_malformed",\n'
            '                               error=str(exc))\n'
            '    domains = run_record().get("argument_domains")\n'
            '    if not isinstance(domains, dict) or not domains:\n'
            '        return False, evidence(target="argument_bindings",\n'
            '                               reason="argument_domains_unpublished")\n'
            '    unbound = [name for name in BOUND_ARGUMENTS if name not in bindings]\n'
            '    undeclared = sorted(set(map(str, bindings)) - set(BOUND_ARGUMENTS))\n'
            '    no_domain = [name for name in BOUND_ARGUMENTS if not isinstance(domains.get(name), list)]\n'
            '    outside = [name for name in BOUND_ARGUMENTS\n'
            '               if name in bindings and isinstance(domains.get(name), list)\n'
            '               and bindings[name] not in domains[name]]\n'
            '    sizes = [len(domains[name]) for name in BOUND_ARGUMENTS\n'
            '             if isinstance(domains.get(name), list)]\n'
            '    published = sizes == list(ARGUMENT_DOMAIN_SIZES)\n'
            '    ok = not (unbound or undeclared or no_domain or outside) and published\n'
            '    return ok, evidence(target="argument_bindings", unbound=unbound, undeclared=undeclared,\n'
            '                        domainless=no_domain, out_of_domain=outside,\n'
            '                        domain_sizes=sizes, combinations=product(sizes),\n'
            '                        reason=None if ok else "manifest_malformed")'
        ),

        "two_axis_composite_uses_the_published_weights_verbatim": (
            '    try:\n'
            '        declared = section("composite_declaration")\n'
            '    except Exception as exc:\n'
            '        return False, evidence(reason="manifest_malformed", error=str(exc))\n'
            '    try:\n'
            '        quality_weight = float(declared["quality_weight"])\n'
            '        speed_weight = float(declared["speed_weight"])\n'
            '        quality_fraction = float(declared["quality_fraction"])\n'
            '        speed_fraction = float(declared["speed_fraction"])\n'
            '        composite = float(declared["composite"])\n'
            '    except Exception as exc:\n'
            '        return False, evidence(reason="composite_declaration_unparseable", error=str(exc))\n'
            '    weights_published = (quality_weight == QUALITY_WEIGHT\n'
            '                         and speed_weight == SPEED_WEIGHT\n'
            '                         and abs(quality_weight + speed_weight - 1.0) <= 1e-9)\n'
            '    restated = quality_fraction * (quality_weight + speed_weight * speed_fraction)\n'
            '    reproduces = abs(restated - composite) <= 1e-9\n'
            '    ok = weights_published and reproduces\n'
            '    return ok, evidence(quality_weight=quality_weight, speed_weight=speed_weight,\n'
            '                        published=[QUALITY_WEIGHT, SPEED_WEIGHT],\n'
            '                        declared_composite=composite, recomputed=restated,\n'
            '                        reason=None if ok else "composite_reweighted_by_the_submission")'
        ),

    }
    # REFUSE rather than fall through. A generic obligation lookup used to stand here and quietly
    # supply a body for any item nobody had written one for; the body it supplied read a run
    # record key nothing ever wrote, so one bundle's outcome check consulted nothing at all and
    # another's invariant became a self-report. An unwritten body is an authoring gap and it
    # stops the generator.
    unwritten = sorted(it["id"] for it in g["items"]
                       if it["mode"] == "compiled" and it["id"] not in bodies)
    if unwritten:
        raise SystemExit("C2-S2: no compiled body is written for %s; refusing to emit a checker "
                         "that would consult nothing" % ", ".join(unwritten))
    # The outcome body must OPEN on require_measured, before it reads the submission or the run
    # record. Otherwise a missing chain_manifest.json raises first, the ConstantUnmeasured signal
    # never fires, and the verifier reports gradeable=true with r_det=0.0 -- which says the agent
    # scored zero when the truth is that no ramp constant has been measured yet.
    opening = first_statement(bodies[OUTCOME_ITEM])
    if not opening.startswith("c = require_measured("):
        raise SystemExit("C2-S2: %s must call require_measured first, not %r"
                         % (OUTCOME_ITEM, opening))

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
        # The tail arrives FINAL. It is family-shaped and has already resolved every placeholder
        # it owns, and it legitimately contains percent signs of its own (format strings inside
        # the emitted helpers), so formatting it a second time here raised TypeError on the first
        # slot that carried one. Only the prelude is substituted.
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


PRELUDE = '"""Compiled checks for %(slot_id)s. GENERATED by solution/recompute.py. DO NOT HAND-EDIT.\n\nEvery check below runs against delivered bytes and references no unmeasured quantity, except\nchain_end_state_accuracy_ramp, which raises ConstantUnmeasured while tests/constants.json carries\nnulls. The graded quantity is %(metric)s.\n"""\nimport json\nimport os\n\nHERE = os.path.dirname(os.path.abspath(__file__))\nSUBMISSION = %(path)r\nCHECKS = []\n\n\nclass ConstantUnmeasured(Exception):\n    """Raised when a check needs a ramp constant that Phase 0 deliberately left null."""\n\n\ndef check(id, weight, knockout, kind):\n    def deco(fn):\n        CHECKS.append({"id": id, "weight": weight, "knockout": knockout, "kind": kind, "fn": fn})\n        return fn\n    return deco\n\n\ndef constants():\n    with open(os.path.join(HERE, "constants.json")) as fh:\n        return json.load(fh)\n\n\ndef require_measured(*names):\n    c = constants()\n    missing = [n for n in names if c.get(n) is None]\n    if missing:\n        raise ConstantUnmeasured(\n            "%(slot_id)s: " + ", ".join(missing) + " are null. Phase 0 authors no measurement; "\n            "the Phase 2 measurement wave writes these and this check becomes gradeable then.")\n    return c\n\n\ndef evidence(**kw):\n    return {"evaluation_target": kw.pop("target", "submitted_artifact"), "values": kw}\n'


TAIL = '\n# --- the facts this slot is about ---------------------------------------------------------\n# The graded object is an ORCHESTRATION over eight pinned tools the agent may compose and may\n# not modify. Everything named below is a fact about that composition: which tool ran, in what\n# order, under which of the twelve bound arguments, against which sandbox epoch, and what each\n# invocation ate and emitted. None of it is a fact about a model.\nSECTIONS = ("argument_bindings", "chain_manifest", "composite_declaration", "export_record",\n            "stage_digests", "stage_seconds")\n\n# The eight pinned stage tools, sorted. Digest-locked and read-only: the lever is the order they\n# are composed in and the arguments bound across their seven link boundaries, never their bytes.\nSTAGES = ("export", "merge", "render", "rollout", "score", "seed", "sft", "update")\n\nINVOCATION_KEYS = ("bindings", "consumed", "epoch_id", "produced", "stage", "stage_digest")\n\n# The twelve declared interface arguments, in the order instruction.md publishes their domain\n# sizes. Each one is bound across a link boundary, and each mis-binding alone sinks the run\n# below T1 while every stage still exits 0.\nBOUND_ARGUMENTS = ("--mask-convention", "--generation-prompt", "--pad-side", "--epoch-binding",\n                   "--score-input", "--export-form", "--embed-resize",\n                   "--tokenizer-fingerprint", "--advantage-baseline", "--reference-dtype",\n                   "--merge-before-rollout", "--decoy-downweight")\nARGUMENT_DOMAIN_SIZES = (2, 3, 2, 3, 3, 2, 3, 3, 3, 3, 2, 3)\n\n# Fallback partial order, used only when the run record publishes none. Seven edges over eight\n# stages; at least forty distinct linearizations satisfy it and every one runs to a clean exit.\nDECLARED_EDGES = (("render", "sft"), ("sft", "merge"), ("merge", "rollout"), ("seed", "rollout"),\n                  ("rollout", "score"), ("score", "update"), ("update", "export"))\n\n# Fallback artifact-producer map, used only when the run record publishes none. merged_weights\n# has two declared producers because the loop re-produces it; nothing else does.\nDECLARED_ARTIFACT_PRODUCERS = {\n    "world_state": frozenset(("seed",)),\n    "render_spec": frozenset(("render",)),\n    "rendered_episodes": frozenset(("render",)),\n    "adapter": frozenset(("sft",)),\n    "merged_weights": frozenset(("merge", "update")),\n    "rollout_samples": frozenset(("rollout",)),\n    "episode_rewards": frozenset(("score",)),\n    "export_bundle": frozenset(("export",)),\n}\n\nT = 0.42                       # the family\'s highest target, at family_position 3\nT1 = 0.34                      # the lowest target; each single-link ablation must fall below it\nBASE_ACCURACY = 0.08           # measured vendored-instruct-base end-state accuracy\nABLATION_BAND = 0.07           # the width the twelve ablation accuracies must fit inside\nHAYSTACK_MARGIN = 0.10\nTRANSFER_MARGIN = 0.16\nSTRATUM_CAP = 0.60\nQUALITY_WEIGHT = 0.55\nSPEED_WEIGHT = 0.45\nBUDGET_SECONDS = 3600.0\nBUDGET_GRACE_SECONDS = 300.0\nSTRATUM_TOTALS = {"haystack": 200, "main": 700, "transfer": 300}\nEPISODE_TOTAL = 1200\nCOUNTING_BASIS = "backend_end_state_digest_equality_with_order_dependent_pairs"\nPRIVATE_EPISODE_TREE = "/private/episodes"\nADMISSIBILITY_REASON_CODES = (\n    "boundary_violation", "budget_exceeded", "chain_closure_broken", "empty_submission",\n    "epoch_unbound", "export_form_rejected", "export_unloadable", "grading_tree_written",\n    "heldout_mutated", "manifest_absent", "manifest_malformed", "stage_tree_written",\n    "unsanctioned_checkpoint")\n\n\ndef load_submission():\n    """The orchestration\'s account of itself: what it composed, how it bound the interface,\n    which tool bytes it invoked, what it exported, how it restated the published composite and\n    where its seconds went. Not a model and not a training log."""\n    with open(SUBMISSION) as handle:\n        doc = json.load(handle)\n    if not isinstance(doc, dict):\n        raise ValueError("the deliverable must be a JSON object")\n    return doc\n\n\ndef section(name):\n    doc = load_submission()\n    if name not in doc:\n        raise KeyError("the submission carries no %s section" % name)\n    return doc[name]\n\n\ndef invocations():\n    """The realized invocation sequence, in the order the manifest lists it.\n\n    List order IS the claim about execution order. Nothing here re-sorts by a timestamp the\n    submission supplies, because sorting would repair the very ordering the closure and the\n    linearization checks exist to measure.\n    """\n    chain = section("chain_manifest")\n    if not isinstance(chain, list) or not chain:\n        raise ValueError("chain_manifest must be a non-empty list of invocation records")\n    for record in chain:\n        if not isinstance(record, dict):\n            raise ValueError("every chain_manifest entry must be an invocation record")\n    return chain\n\n\ndef is_digest(value):\n    return (isinstance(value, str) and len(value) == 64\n            and all(c in "0123456789abcdef" for c in value))\n\n\ndef binding_at(record, argument):\n    """The value one invocation was called with for one declared argument.\n\n    Read per invocation rather than from the top-level bindings map: the whole point of the\n    render-to-rollout and rollout-to-update links is that one argument can carry two different\n    values at two invocations, with both stages exiting 0.\n    """\n    bindings = record.get("bindings")\n    return bindings.get(argument) if isinstance(bindings, dict) else None\n\n\ndef product(sizes):\n    total = 1\n    for size in sizes:\n        total *= size\n    return total\n\n\ndef declared_edges():\n    """The declared partial order, read from the run record; the constant is the fallback."""\n    published = run_record().get("declared_partial_order")\n    if published is None:\n        return DECLARED_EDGES\n    if not isinstance(published, list):\n        return None\n    edges = []\n    for edge in published:\n        if not isinstance(edge, (list, tuple)) or len(edge) != 2:\n            return None\n        edges.append((str(edge[0]), str(edge[1])))\n    return tuple(edges)\n\n\ndef artifact_producers():\n    """Which stage is declared to produce each named artifact.\n\n    Published in instruction.md and echoed in the run record. It is a per-artifact map rather\n    than a per-stage one because the GRPO cycle re-produces merged_weights at update and hands\n    it back to rollout: the chain is not acyclic and closure cannot be stated as "an acyclic\n    predecessor produced this". What closure DOES state is that the digest eaten is the most\n    recent one emitted under that name by a stage entitled to emit it.\n    """\n    published = run_record().get("artifact_producers")\n    if published is None:\n        return DECLARED_ARTIFACT_PRODUCERS\n    if not isinstance(published, dict) or not published:\n        return None\n    out = {}\n    for name, stages in published.items():\n        if not isinstance(stages, (list, tuple)):\n            return None\n        out[str(name)] = frozenset(str(stage) for stage in stages)\n    return out\n\n\ndef run_record():\n    """The record the graded run and the harness produced together.\n\n    The wall clock, the pinned tool digests, the declared partial order, the argument domains,\n    the episode ledger, the twelve single-link ablations, the mount manifest and the grader\'s\n    reason-code vocabulary all live here. Read from here rather than from a scan taken\n    afterwards, which cannot separate what happened inside the timed window from what happened\n    before it, and never from a number the submission reports about itself.\n    """\n    path = os.path.join(HERE, "..", "run_record.json")\n    if not os.path.exists(path):\n        return {}\n    with open(path) as handle:\n        return json.load(handle)\n\n\ndef ledger():\n    book = run_record().get("episode_ledger")\n    return book if isinstance(book, dict) and book else None\n\n\ndef stratum_accuracy(book, name):\n    row = book.get(name) if isinstance(book, dict) else None\n    if not isinstance(row, dict):\n        return None\n    counted = row.get("counted")\n    total = STRATUM_TOTALS.get(name)\n    if not isinstance(counted, int) or isinstance(counted, bool) or not total:\n        return None\n    return counted / float(total)\n\n\ndef harvest(node, words, numbers):\n    """Every string and every number anywhere in the submission, object keys included."""\n    if isinstance(node, dict):\n        for key, value in node.items():\n            words.add(str(key).lower())\n            harvest(value, words, numbers)\n    elif isinstance(node, list):\n        for value in node:\n            harvest(value, words, numbers)\n    elif isinstance(node, bool):\n        pass\n    elif isinstance(node, (int, float)):\n        numbers.add(float(node))\n    elif isinstance(node, str):\n        words.add(node.lower())\n    return words, numbers\n\n\ndef measure():\n    """acc = summed counted episodes over the summed 1200, from the verifier\'s own ledger.\n\n    The denominator is rebuilt by summing the three published stratum totals rather than read\n    from any total the record states beside them, and the numerator counts only episodes whose\n    backend end-state digest equalled gold\'s. No number the submission reports enters either.\n    """\n    book = ledger()\n    if book is None:\n        raise ConstantUnmeasured("the verifier has written no private episode ledger")\n    counted, total = 0, 0\n    for name, published in STRATUM_TOTALS.items():\n        row = book.get(name)\n        if not isinstance(row, dict) or not isinstance(row.get("counted"), int):\n            raise ConstantUnmeasured("stratum %s carries no counted episodes" % name)\n        counted += row["counted"]\n        total += published\n    if total != EPISODE_TOTAL:\n        raise ConstantUnmeasured("the strata do not partition the 1200 private episodes")\n    return counted / float(total)\n'


if __name__ == "__main__":
    raise SystemExit(main())
