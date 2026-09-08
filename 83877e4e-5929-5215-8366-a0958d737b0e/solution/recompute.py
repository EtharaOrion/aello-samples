"""Derivation for A5-01: every canonical artifact descends from solution/grounding.yaml.

Wired through seed/build/generator_common.py, which carries the five slot-agnostic functions
verified byte-for-byte against the three frozen generators. build_provenance stays here, per
family, for the reason recorded in that module.

The graded axis of this slot is BITS PER BYTE OF THE LAST ATTEMPT'S EXPORTED PROGRAM, measured by
the verifier's own executor on the worse of two private post-2026-06-01 PG-19 partitions, at fixed
data and fixed compute. Every compiled body below names something that axis has and that a
quality-graded, throughput-graded or time-to-bar slot does not: an artifact that is a PROGRAM under
a seven-symbol operator envelope rather than a checkpoint of weights, a 48 MiB serialized ceiling
that makes shape the only lever once data and compute are pinned, a bank denominated in analytic
accounted GPU-seconds rather than in wall clock, a 2048-byte carry-forward channel over a 204-key
evidence packet whose field ORDER is permuted every attempt, one keyed graded partition against
three corroborating decoys, a single unannounced rotation between genre-separated partitions, and
a final_selection bound to LAST rather than to best.

The generic spine that stood here graded the existence and shape of a deliverable, a self-reported
budget declaration and a replay digest. Not one of those six checks would have changed its verdict
if this slot had been a classifier, a forecaster or a tokenizer, which is precisely why they said
nothing about the task.
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
    """Assemble the compiled-check module for the bits-per-byte grading."""
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
SLOT_ID = 'A5-01'
TITLE = "# A5-01 -- what this task actually tests"


def load_grounding():
    with open(os.path.join(HERE, "grounding.yaml")) as fh:
        return yaml.safe_load(fh)


def build_provenance(g, identity):
    p = g["provenance"]
    layout = g["corpus_layout"]
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
        # SLOT-SPECIFIC. These are the disclosed shape of the task, not measurements: the graded
        # object, the selection rule, the envelope the program must live inside, the ceiling that
        # makes shape the lever, the currency the bank is denominated in and the width of the only
        # channel that crosses an attempt boundary.
        "graded_object": ("the LAST attempt's exported program, re-executed by the verifier's own "
                          "executor on partitions the agent never reads"),
        "graded_selection": "last attempt, not best attempt",
        "operator_envelope": list(layout["operator_envelope"]),
        "param_byte_ceiling": layout["param_byte_ceiling"],
        "episode_bank_accounted_s": layout["episode_bank_accounted_s"],
        "bank_currency": "analytic accounted GPU-seconds, not NVML busy-seconds",
        "carry_channel_bytes": layout["carry_notes_byte_cap"],
        "evidence_vocabulary_size": layout["evidence_vocabulary_size"],
        "graded_partition_genres": list(layout["graded_partition_genres"]),
        "rotation_events": layout["rotation_events"],
        "corpus_status": layout["corpus_status"],
        "timed_axis": False,
    }
    out.update(read_screening(HERE))
    return out


# The five facts that ARE this slot. If any one of them goes unasserted the item list has stopped
# describing an architecture search at fixed data and fixed compute and has started describing
# some other task that also happens to emit a file.
REQUIRED_ASSERTIONS = ("bank_debits_are_analytic_accounted_seconds_not_wall_clock",
                       "carry_forward_entries_reference_diagnostics_by_name",
                       "exactly_the_header_and_2048_notes_bytes_crossed_each_boundary",
                       "exported_operator_list_inside_the_bound_envelope",
                       "graded_export_is_the_last_attempt_not_the_best_one",
                       "quantization_scales_recomputed_at_the_export_own_width",
                       "ranking_rows_carry_the_partition_epoch_in_force",
                       "serialized_parameter_bytes_recounted_not_declared")
RAMP_CONSTANTS = ("dichotomizing_threshold", "floor", "knee", "reward_gate_pass_threshold")


def validate(grounding):
    """An architecture-design slot at fixed data and fixed compute is well formed only if the
    facts that make it one are asserted by some item.

    If the export is not held to an operator envelope and a serialized byte ceiling, the artifact
    is not a program under a shape budget and "fixed compute" has no teeth. If the bank is not
    held to an integer accounted-second debit, the budget is a stopwatch and the campaign is not
    replayable. If the carry-forward record is not held to the 2048-byte channel and to
    name-keyed references, the archetype's whole memory surface is graded by declaration. If the
    epoch is not carried, a stale table ranks two distributions as one. And if the graded export
    is not pinned to the LAST attempt, the slot has quietly become best-of-fifty, which is a
    different and much easier task.
    """
    problems = []
    items = grounding["items"]
    carried = {i["id"] for i in items}
    for name in REQUIRED_ASSERTIONS:
        if name not in carried:
            problems.append("no item asserts %s" % name)
    uncompiled = sorted(i["id"] for i in items if i["mode"] != "compiled")
    if uncompiled:
        problems.append("compilation_floor is 1.0 but these items are not compiled: "
                        + ", ".join(uncompiled))
    outcome = sorted(i["id"] for i in items if i["dimension"] == "outcome")
    if len(outcome) != 1:
        problems.append("expected exactly one outcome item, found %r" % outcome)
    for name in RAMP_CONSTANTS:
        if grounding["constants"].get(name) is not None:
            problems.append("%s is authored at Phase 0, which the convention forbids" % name)
    targets = {i["evaluation_target"] for i in items}
    stray = sorted(targets - set(grounding["evaluation_target_vocabulary"]))
    if stray:
        problems.append("items name evaluation targets the vocabulary does not: "
                        + ", ".join(stray))
    unused = sorted(set(grounding["evaluation_target_vocabulary"]) - targets)
    if unused:
        problems.append("the vocabulary names targets no item evaluates: " + ", ".join(unused))
    layout = grounding.get("corpus_layout") or {}
    envelope = list(layout.get("operator_envelope") or [])
    if len(envelope) != 7 or len(set(envelope)) != 7:
        problems.append("the bound operator envelope is seven distinct symbols, found %d"
                        % len(envelope))
    if int(layout.get("param_byte_ceiling") or 0) != 50331648:
        problems.append("the serialized parameter ceiling is 48 MiB, found %r"
                        % layout.get("param_byte_ceiling"))
    if int(layout.get("carry_notes_byte_cap") or 0) != 2048:
        problems.append("the carry-forward channel is 2048 bytes, found %r"
                        % layout.get("carry_notes_byte_cap"))
    if int(layout.get("in_force_partitions") or 0) != 2:
        problems.append("exactly two private siblings are in force at a time, found %r"
                        % layout.get("in_force_partitions"))
    if int(layout.get("rotation_events") or 0) != 1:
        problems.append("the graded partition rotates exactly once, found %r"
                        % layout.get("rotation_events"))
    graded = list(layout.get("graded_partitions") or [])
    decoys = list(layout.get("decoy_partitions") or [])
    genres = list(layout.get("graded_partition_genres") or [])
    if len(graded) != 4 or len(decoys) != 4:
        problems.append("four graded and four decoy partitions are pinned, found %d and %d"
                        % (len(graded), len(decoys)))
    if len(genres) != len(graded) or len(set(genres)) != len(genres):
        problems.append("each graded partition carries its own distinct genre, found %r" % genres)
    keys = int(layout.get("heldout_keys_per_packet") or 0)
    keyed = int(layout.get("graded_heldout_keys_per_packet") or 0)
    if keys != 4 or keyed != 1:
        problems.append("every packet carries four held-out keys of which exactly one is graded, "
                        "found %d and %d" % (keys, keyed))
    if int(layout.get("evidence_vocabulary_size") or 0) != 204:
        problems.append("the evidence vocabulary is 204 names, found %r"
                        % layout.get("evidence_vocabulary_size"))
    return problems


def facts_block(g):
    """The literal facts the compiled checks read, rendered from grounding.yaml.

    They live in the emitted module rather than in tests/constants.json because none of them is a
    MEASUREMENT: they are the DISCLOSED shape of this task -- a seven-symbol operator envelope, a
    48 MiB serialized ceiling, a 9,000 accounted-second bank at a pinned reference FLOP rate, a
    2048-byte carry channel over a 204-key vocabulary, four held-out keys of which one is graded,
    and two 2 MiB private siblings in force at a time. tests/constants.json carries only the ramp,
    which Phase 0 does not author and leaves null; keeping every non-null value out of that file
    is what stops a disclosed constant from being mistaken for a measured one.
    """
    layout = g["corpus_layout"]
    return (
        '\n# --- the facts this slot is about ------------------------------------------------\n'
        '# The five sections of the graded artifact. A missing section is a malformed\n'
        '# submission, not a weak one.\n'
        'SECTIONS = %r\n'
        '# The seven symbols the verifier executor implements. The deliverable is a PROGRAM, so\n'
        '# an operator outside this set cannot be run on the private partition at all.\n'
        'OPERATOR_ENVELOPE = frozenset(%r)\n'
        '# 48 MiB of SERIALIZED parameter bytes, recounted from the blob and never declared.\n'
        '# Data and compute are fixed, so shape is the only lever and these are its currency.\n'
        'PARAM_BYTE_CEILING = %r\n'
        '# One depleting bank shared across all fifty attempts, denominated in ANALYTIC\n'
        '# accounted GPU-seconds: ceil(tokens * flops_per_token / R_ref). Both inputs are\n'
        '# integers the harness observes, so the step count is fixed by the config and not by\n'
        '# the clock, which is the precondition for bit-identical weights.\n'
        'EPISODE_BANK_ACCOUNTED_S = %r\n'
        'ACCOUNTED_FLOPS_PER_SECOND = %r\n'
        '# A debit within this many seconds of the NVML busy reading is indistinguishable from a\n'
        '# stopwatch. NVML is the fraud cross-check on this slot, never the meter.\n'
        'NVML_SEPARATION = 0.5\n'
        'MAX_ATTEMPTS = %r\n'
        'FRESH_MEASUREMENTS_PER_ATTEMPT = %r\n'
        '# The evidence packet: a fixed, named vocabulary whose FIELD ORDER is permuted every\n'
        '# attempt. The vocabulary is what a carry-forward record may reference; the order is\n'
        '# what destroys a record that referenced positions instead.\n'
        'EVIDENCE_VOCABULARY_SIZE = %r\n'
        'HELDOUT_KEYS_PER_PACKET = %r\n'
        'GRADED_HELDOUT_KEYS = %r\n'
        '# Exactly what crosses an attempt boundary, and nothing else.\n'
        'CARRY_HEADER_LINES = %r\n'
        'CARRY_NOTES_BYTE_CAP = %r\n'
        '# Ordinal-shaped keys a positionally-keyed record betrays itself with.\n'
        'ORDINAL_KEYS = ("field_index", "packet_offset", "ordinal", "position")\n'
        '# The four residue classes attempt teardown is supposed to destroy. An empty residue\n'
        '# roster over an empty INSPECTION roster passes vacuously, so both are read.\n'
        'RESIDUE_CLASSES = %r\n'
        '# On this slot these two files ARE the measurement: the launcher writes the\n'
        '# accounted-second debit the bank is drawn against, the emitter writes the 204-key\n'
        '# packet the carry-forward record is built from.\n'
        'STAMPING_INSTRUMENTS = %r\n'
        'EGRESS_METHODS = frozenset(%r)\n'
        '# Two 2 MiB private siblings in force at a time, out of four genre-separated graded\n'
        '# partitions and four decoys drawn from the shipped training distribution.\n'
        'IN_FORCE_PARTITIONS = %r\n'
        'PARTITION_BYTES = %r\n'
        'GRADED_PARTITIONS = %r\n'
        'DECOY_PARTITIONS = %r\n'
        'GRADED_PARTITION_GENRES = %r\n'
        'SIBLING_GAP_TOLERANCE = %r\n'
        'PERMUTATION_REWARD_TOLERANCE = %r\n'
        '# Exactly one rotation, at an index instruction.md never discloses. The check asserts\n'
        '# the epoch is RECORDED, never that the agent predicted the instant.\n'
        'ROTATION_EVENTS = %r\n'
        'EVAL_WINDOW_TOKENS = %r\n'
        'SCALE_TOLERANCE = %r\n'
        '# The bits-per-byte identity is arithmetic over harness-recorded totals, so it is tight.\n'
        'BPB_IDENTITY_TOL = 1e-09\n'
        'LN2 = 0.6931471805599453\n'
        'TRAIN_SHARD = %r\n'
        'HARNESS_EPISODE = %r\n'
        % (tuple(layout["submission_sections"]), tuple(sorted(layout["operator_envelope"])),
           int(layout["param_byte_ceiling"]), int(layout["episode_bank_accounted_s"]),
           float(layout["accounted_flops_per_second"]), int(layout["max_attempts"]),
           int(layout["fresh_measurements_per_attempt"]),
           int(layout["evidence_vocabulary_size"]), int(layout["heldout_keys_per_packet"]),
           int(layout["graded_heldout_keys_per_packet"]), int(layout["carry_header_lines"]),
           int(layout["carry_notes_byte_cap"]), tuple(sorted(layout["residue_classes"])),
           tuple(layout["stamping_instruments"]), tuple(sorted(layout["egress_methods"])),
           int(layout["in_force_partitions"]), int(layout["partition_bytes"]),
           tuple(layout["graded_partitions"]), tuple(layout["decoy_partitions"]),
           tuple(layout["graded_partition_genres"]),
           float(layout["sibling_gap_tolerance_bpb"]),
           float(layout["permutation_reward_tolerance"]), int(layout["rotation_events"]),
           int(layout["eval_window_tokens"]), float(layout["scale_tolerance"]),
           layout["train_shard"], layout["harness_episode"]))


def main():
    g = load_grounding()
    # The family validator runs FIRST and refuses rather than emitting: a generator that writes a
    # carrier it knows is malformed hands the drift check a moving target.
    problems = validate(g)
    if problems:
        raise SystemExit("A5-01 grounding is not well-formed: " + "; ".join(problems))
    dl = g["deliverable"]
    layout = g["corpus_layout"]
    outcome_id = [i["id"] for i in g["items"] if i["dimension"] == "outcome"][0]
    metric = ('Bits per byte of the LAST attempt\'s exported program, measured by the verifier\'s '
              'own executor at the announced %d-token window over the two in-force private '
              'sibling PG-19 partitions and taken as the WORSE (max) of the two, byte-normalized '
              'against the pinned %d-byte partition length so a self-designed tokenizer is '
              'neither rewarded nor punished for vocabulary size. Last, not best: task.toml binds '
              'final_selection to last, so an earlier attempt that measured better contributes '
              'nothing. Data and compute are both fixed -- a shipped shard and one depleting bank '
              'of %d analytic accounted GPU-seconds -- so the only lever left is shape, and it is '
              'spent against a %d-byte serialized parameter ceiling inside a seven-symbol '
              'operator envelope'
              % (int(layout["eval_window_tokens"]), int(layout["partition_bytes"]),
                 int(layout["episode_bank_accounted_s"]), int(layout["param_byte_ceiling"])))
    subst = {"slot_id": SLOT_ID, "path": dl["path"], "metric": metric, "outcome_id": outcome_id}

    # BODIES ARE WRITTEN FOR THIS SLOT. Each names a fact of an architecture search at fixed data
    # and fixed compute whose LAST attempt is graded on an exported program: an operator envelope,
    # a recounted serialized byte ceiling, a scale recomputed at the export's own width, a
    # byte-normalized denominator over two in-force siblings, a 204-key packet with one graded key
    # against three corroborating decoys, a 2048-byte name-keyed carry channel, an integer
    # accounted-second bank, a rotation epoch, and a selection pinned to LAST. None of them would
    # return a different verdict on a classifier, a forecaster, a tokenizer or a time-to-bar run.
    bodies = {

        "export_program_and_parameter_blob_land": '''\
    if not os.path.exists(SUBMISSION):
        return False, evidence(target="exported_program", reason="no-deliverable",
                               detail="submission_missing", path=SUBMISSION)
    try:
        doc = load_submission()
    except Exception as exc:
        return False, evidence(target="exported_program", reason="submission_malformed",
                               error=str(exc))
    absent = [name for name in SECTIONS if name not in doc]
    if absent:
        return False, evidence(target="exported_program", reason="submission_malformed",
                               absent=absent, carried=sorted(doc))
    export = doc["export"]
    if not isinstance(export, dict):
        return False, evidence(target="exported_program", reason="export_section_malformed")
    manifest, blob = export.get("program_manifest"), export.get("param_blob")
    on_disk_manifest = bool(manifest) and os.path.exists(workspace_path(manifest))
    on_disk_blob = bool(blob) and os.path.exists(workspace_path(blob))
    facts = export_facts()
    operators = facts.get("operators")
    per_tensor = facts.get("per_tensor_bytes")
    # WEIGHTS ARE NOT A SUBMISSION and a graph with nothing to load is not one either. The graded
    # number comes from re-EXECUTING this pair on partitions the agent never reads, so either
    # half alone is the no-deliverable red line rather than a weak entry.
    ok = (on_disk_manifest and on_disk_blob and isinstance(operators, list) and bool(operators)
          and isinstance(per_tensor, dict) and bool(per_tensor))
    return ok, evidence(target="exported_program", program_manifest=manifest, param_blob=blob,
                        manifest_on_disk=on_disk_manifest, blob_on_disk=on_disk_blob,
                        operators_declared=len(operators) if isinstance(operators, list) else None,
                        tensors_declared=len(per_tensor) if isinstance(per_tensor, dict) else None,
                        reason=None if ok else "program_or_parameter_blob_missing")''',

        "graded_export_is_the_last_attempt_not_the_best_one": '''\
    rows = attempts()
    if not rows:
        return False, evidence(target="attempt_ledger", reason="episode_record_absent")
    indices = [int(row["attempt_index"]) for row in rows if row.get("attempt_index") is not None]
    if len(indices) != len(rows):
        return False, evidence(target="attempt_ledger", reason="attempt_index_unrecorded")
    rising = all(b > a for a, b in zip(indices, indices[1:]))
    last = rows[-1]
    scored = [row for row in rows if row.get("graded_bpb") is not None]
    best = min(scored, key=lambda row: float(row["graded_bpb"])) if scored else None
    try:
        chosen = load_submission()["final_selection"]
    except Exception as exc:
        return False, evidence(target="attempt_ledger", reason="submission_malformed",
                               error=str(exc))
    if not isinstance(chosen, dict):
        return False, evidence(target="attempt_ledger", reason="final_selection_undeclared")
    same_index = chosen.get("attempt_index") == last.get("attempt_index")
    same_digest = str(chosen.get("config_digest")) == str(last.get("config_digest"))
    exported_is_last = str(export_facts().get("config_digest")) == str(last.get("config_digest"))
    # LOWER bits per byte is better, so `best` is the MINIMUM. When it is not the last row, the
    # difference between the two is exactly what this slot declines to pay for.
    ok = rising and same_index and same_digest and exported_is_last
    return ok, evidence(target="attempt_ledger", attempts=len(rows),
                        last_attempt_index=last.get("attempt_index"),
                        last_config_digest=last.get("config_digest"),
                        last_bpb=last.get("graded_bpb"),
                        declared_attempt_index=chosen.get("attempt_index"),
                        best_attempt_index=best.get("attempt_index") if best else None,
                        best_bpb=float(best["graded_bpb"]) if best else None,
                        best_is_not_last=bool(best and best.get("attempt_index")
                                              != last.get("attempt_index")),
                        indices_strictly_increasing=rising,
                        export_digest_matches_last=exported_is_last,
                        reason=None if ok else "graded_export_is_not_the_last_attempt")''',

        "exported_operator_list_inside_the_bound_envelope": '''\
    ops = export_facts().get("operators")
    if ops is None:
        return False, evidence(target="exported_program", reason="export_facts_unrecorded")
    if not isinstance(ops, list) or not ops:
        # An empty operator list would clear a subset test vacuously. Nothing was exported.
        return False, evidence(target="exported_program", reason="no_operator_exported",
                               declared=ops)
    symbols = sorted({str(op) for op in ops})
    outside = sorted(set(symbols) - OPERATOR_ENVELOPE)
    ok = not outside
    return ok, evidence(target="exported_program", operators=symbols,
                        envelope=sorted(OPERATOR_ENVELOPE), outside=outside,
                        unused_envelope_symbols=sorted(OPERATOR_ENVELOPE - set(symbols)),
                        reason=None if ok else "envelope-op-refused")''',

        "serialized_parameter_bytes_recounted_not_declared": '''\
    facts = export_facts()
    recount = facts.get("recounted_param_bytes")
    if recount is None:
        return False, evidence(target="exported_program", reason="export_facts_unrecorded")
    recount = int(recount)
    per_tensor = facts.get("per_tensor_bytes")
    summed = (sum(int(v) for v in per_tensor.values())
              if isinstance(per_tensor, dict) and per_tensor else None)
    try:
        declared = load_submission()["export"].get("serialized_param_bytes")
    except Exception as exc:
        return False, evidence(target="exported_program", reason="submission_malformed",
                               error=str(exc))
    within = 0 < recount <= PARAM_BYTE_CEILING
    # The RECOUNT is the graded quantity and the declaration is never substituted for it: on a
    # shape-limited axis the cheapest way to buy quality is to under-report the bytes it cost.
    agrees = declared is not None and int(declared) == recount
    tallies = summed is not None and summed == recount
    ok = within and agrees and tallies
    return ok, evidence(target="exported_program", recounted_bytes=recount,
                        declared_bytes=declared, per_tensor_sum=summed,
                        tensors=len(per_tensor) if isinstance(per_tensor, dict) else None,
                        ceiling=PARAM_BYTE_CEILING, headroom=PARAM_BYTE_CEILING - recount,
                        reason=None if ok else ("param-bytes-exceeded" if not within
                                                else "recount_disagrees_with_declaration"))''',

        "quantization_scales_recomputed_at_the_export_own_width": '''\
    facts = export_facts()
    trained, measured = facts.get("declared_width_s"), facts.get("measured_at_width_s")
    if trained is None or measured is None:
        return False, evidence(target="exported_program", reason="export_facts_unrecorded")
    trained, measured = float(trained), float(measured)
    same_width = trained == measured
    drifted = []
    for label, said_key, recomputed_key in (
            ("scale", "declared_scales", "recomputed_scales"),
            ("schedule", "declared_schedule_constants", "recomputed_schedule_constants")):
        said, recomputed = facts.get(said_key), facts.get(recomputed_key)
        if not isinstance(said, dict) or not isinstance(recomputed, dict) or not said:
            drifted.append(label + ":unrecorded")
            continue
        for name in sorted(set(said) | set(recomputed)):
            if name not in said or name not in recomputed:
                drifted.append("%s:%s:absent" % (label, name))
            elif abs(float(said[name]) - float(recomputed[name])) > SCALE_TOLERANCE:
                drifted.append("%s:%s" % (label, name))
    # A scale measured on a short probe and carried forward unchanged clips a few percent of
    # channels at the convergence width. It raises no traceback, clears the envelope, the byte
    # ceiling and the determinism replay, and is visible only in bits per byte. STALE, not wrong.
    ok = same_width and not drifted
    return ok, evidence(target="exported_program", trained_at_width_s=trained,
                        constants_measured_at_width_s=measured,
                        width_ratio=(trained / measured) if measured else None,
                        drifted=drifted[:8], drifted_count=len(drifted),
                        tolerance=SCALE_TOLERANCE,
                        reason=None if ok else "stale-constant")''',

        "byte_denominator_prices_the_self_designed_vocabulary_out": '''\
    row = last_attempt()
    accounting = episode().get("bpb_accounting")
    if row is None or not isinstance(accounting, dict) or not accounting:
        return False, evidence(target="heldout_partitions", reason="episode_record_absent")
    in_force = row.get("in_force")
    if not isinstance(in_force, dict) or not in_force:
        return False, evidence(target="heldout_partitions", reason="no_partition_in_force")
    wrong_length, byte_identity, token_identity = [], [], []
    for name in sorted(in_force):
        part = in_force[name] or {}
        nats, byts = part.get("total_nats"), part.get("bytes")
        toks, value = part.get("tokens"), part.get("bpb")
        if None in (nats, byts, toks, value):
            return False, evidence(target="heldout_partitions", partition=name,
                                   reason="partition_table_incomplete")
        if int(byts) != PARTITION_BYTES:
            wrong_length.append(name)
        byte_identity.append(
            abs(float(nats) / LN2 / float(byts) - float(value)) <= BPB_IDENTITY_TOL)
        # Dividing by TOKENS instead of by bytes is the one substitution this axis exists to
        # refuse: it would pay for a coarser vocabulary rather than for a better architecture.
        # Where the two counts differ, the token form must NOT also reproduce the number.
        if int(toks) != int(byts):
            token_identity.append(
                abs(float(nats) / LN2 / float(toks) - float(value)) <= BPB_IDENTITY_TOL)
    ok = (accounting.get("window_tokens") == EVAL_WINDOW_TOKENS and not wrong_length
          and all(byte_identity) and not any(token_identity))
    return ok, evidence(target="heldout_partitions",
                        window_tokens=accounting.get("window_tokens"),
                        announced_window=EVAL_WINDOW_TOKENS, partitions=sorted(in_force),
                        pinned_partition_bytes=PARTITION_BYTES, wrong_length=wrong_length,
                        byte_identity_holds=all(byte_identity),
                        token_identity_also_holds=any(token_identity),
                        declared_vocab_entries=accounting.get("vocab_entries"),
                        reason=None if ok else "denominator_is_not_partition_bytes")''',

        "graded_bpb_is_the_worse_of_the_two_in_force_partitions": '''\
    row = last_attempt()
    if row is None:
        return False, evidence(target="heldout_partitions", reason="episode_record_absent")
    in_force = row.get("in_force")
    if not isinstance(in_force, dict) or not in_force:
        return False, evidence(target="heldout_partitions", reason="no_partition_in_force")
    values = {}
    for name in sorted(in_force):
        value = (in_force[name] or {}).get("bpb")
        if value is None:
            return False, evidence(target="heldout_partitions", partition=name,
                                   reason="partition_table_incomplete")
        values[name] = float(value)
    if len(values) != IN_FORCE_PARTITIONS:
        return False, evidence(target="heldout_partitions", reason="wrong_partition_count",
                               in_force=sorted(values), expected=IN_FORCE_PARTITIONS)
    worse = max(values.values())
    try:
        declared = load_submission()["export"].get("graded_bpb")
    except Exception as exc:
        return False, evidence(target="heldout_partitions", reason="submission_malformed",
                               error=str(exc))
    recorded = row.get("graded_bpb")
    # The WORSE of the two, never the mean and never the keyed one alone: that is what stops a
    # design tuned to one draw of post-2026 Gutenberg volumes from being paid for a distribution.
    consumes_worse = recorded is not None and float(recorded) == worse
    agrees = declared is not None and float(declared) == worse
    ok = consumes_worse and agrees
    return ok, evidence(target="heldout_partitions", per_partition=values, worse=worse,
                        better=min(values.values()),
                        mean=sum(values.values()) / len(values), declared=declared,
                        recorded_graded_bpb=recorded,
                        reason=None if ok else "graded_value_is_not_the_worse_partition")''',

        "sibling_partition_disagreement_within_five_hundredths": '''\
    row = last_attempt()
    if row is None:
        return False, evidence(target="heldout_partitions", reason="episode_record_absent")
    in_force = row.get("in_force") or {}
    values = dict((name, float(in_force[name]["bpb"])) for name in sorted(in_force)
                  if isinstance(in_force[name], dict) and in_force[name].get("bpb") is not None)
    if len(values) != IN_FORCE_PARTITIONS:
        return False, evidence(target="heldout_partitions", reason="wrong_partition_count",
                               in_force=sorted(values), expected=IN_FORCE_PARTITIONS)
    names = sorted(values)
    gap = abs(values[names[0]] - values[names[1]])
    ok = gap <= SIBLING_GAP_TOLERANCE
    return ok, evidence(target="heldout_partitions", per_partition=values, gap=gap,
                        tolerance=SIBLING_GAP_TOLERANCE,
                        genres=dict((n, (in_force[n] or {}).get("genre")) for n in names),
                        worse_sibling=max(values, key=lambda n: values[n]),
                        reason=None if ok else "shard-gap-exceeded")''',

        "one_keyed_graded_partition_against_three_decoy_keys": '''\
    packs, vocab = packets(), vocabulary()
    if not packs:
        return False, evidence(target="evidence_packet", reason="no_evidence_packet_emitted")
    if len(vocab) != EVIDENCE_VOCABULARY_SIZE or len(set(vocab)) != len(vocab):
        return False, evidence(target="evidence_packet", reason="vocabulary_not_pinned",
                               named=len(vocab), expected=EVIDENCE_VOCABULARY_SIZE)
    allowed = set(vocab)
    faults, spreads, margins = [], [], []
    for pack in packs:
        index = pack.get("attempt_index")
        keys = pack.get("keys")
        if not isinstance(keys, list) or set(map(str, keys)) - allowed:
            faults.append("attempt_%s:keys_outside_vocabulary" % index)
            continue
        held = pack.get("heldout_keys")
        if not isinstance(held, dict) or len(held) != HELDOUT_KEYS_PER_PACKET:
            faults.append("attempt_%s:heldout_key_count" % index)
            continue
        graded = sorted(k for k, v in held.items() if isinstance(v, dict) and v.get("graded"))
        if len(graded) != GRADED_HELDOUT_KEYS:
            faults.append("attempt_%s:graded_key_count_%d" % (index, len(graded)))
            continue
        decoys = sorted(float(v["value"]) for k, v in held.items()
                        if k not in graded and isinstance(v, dict) and v.get("value") is not None)
        keyed = held[graded[0]].get("value")
        if len(decoys) == HELDOUT_KEYS_PER_PACKET - GRADED_HELDOUT_KEYS and keyed is not None:
            # The three decoys come from the shipped training distribution, so they agree with
            # each other and disagree with the graded key. Those two numbers ARE the trap: a
            # corroborated majority of three against one dissenting keyed value.
            spreads.append(decoys[-1] - decoys[0])
            margins.append(abs(float(keyed) - decoys[len(decoys) // 2]))
    ok = not faults
    return ok, evidence(target="evidence_packet", packets=len(packs), vocabulary_size=len(vocab),
                        heldout_keys_per_packet=HELDOUT_KEYS_PER_PACKET,
                        graded_keys_per_packet=GRADED_HELDOUT_KEYS, faults=faults[:6],
                        widest_decoy_spread=max(spreads) if spreads else None,
                        widest_decoy_median_to_graded=max(margins) if margins else None,
                        reason=None if ok else "evidence_schema_violated")''',

        "carry_forward_entries_reference_diagnostics_by_name": '''\
    vocab = set(vocabulary())
    if not vocab:
        return False, evidence(target="carry_record", reason="no_evidence_packet_emitted")
    try:
        entries = list(load_submission()["carry_record"]["entries"])
    except Exception as exc:
        return False, evidence(target="carry_record", reason="submission_malformed",
                               error=str(exc))
    if not entries:
        return False, evidence(target="carry_record", reason="carry_record_empty")
    positional, unknown, named = [], [], 0
    for entry in entries:
        if not isinstance(entry, dict):
            positional.append("entry_is_not_an_object")
            continue
        for ordinal in ORDINAL_KEYS:
            if entry.get(ordinal) is not None:
                positional.append("%s=%r" % (ordinal, entry[ordinal]))
        reference = entry.get("diagnostic")
        if isinstance(reference, bool) or isinstance(reference, (int, float)):
            positional.append("diagnostic=%r" % (reference,))
        elif not isinstance(reference, str):
            positional.append("diagnostic=%r" % (reference,))
        elif reference not in vocab:
            unknown.append(reference)
        else:
            named += 1
    # Recorded on every run, not only the failing ones: this is how far a diagnostic's POSITION
    # moves between packets. A name-keyed record is indifferent to it; a positional one is noise
    # from the first permutation onward, with no error raised anywhere to notice.
    ok = (not positional) and (not unknown) and named == len(entries)
    return ok, evidence(target="carry_record", entries=len(entries), named_references=named,
                        positional_references=positional[:6], unknown_names=unknown[:6],
                        max_field_order_drift=permutation_drift(),
                        vocabulary_size=len(vocab),
                        reason=None if ok else "carry_record_is_position_keyed")''',

        "exactly_the_header_and_2048_notes_bytes_crossed_each_boundary": '''\
    rows = [row for row in attempts() if isinstance(row.get("carry"), dict) and row["carry"]]
    if not attempts():
        return False, evidence(target="carry_record", reason="episode_record_absent")
    if not rows:
        return False, evidence(target="carry_record", reason="carry_boundaries_unrecorded")
    faults, widest = [], 0
    for row in rows:
        carry, index = row["carry"], row.get("attempt_index")
        crossed, expected = carry.get("crossing_sha256"), carry.get("expected_sha256")
        if not crossed or not expected or crossed != expected:
            faults.append("attempt_%s:extra_bytes_crossed" % index)
        if carry.get("header_lines") != CARRY_HEADER_LINES:
            faults.append("attempt_%s:header_is_not_three_lines" % index)
        notes = carry.get("notes_bytes")
        if notes is None or int(notes) > CARRY_NOTES_BYTE_CAP:
            faults.append("attempt_%s:notes_over_cap" % index)
        else:
            widest = max(widest, int(notes))
        if carry.get("tail_decodes_utf8") is not True:
            faults.append("attempt_%s:truncated_mid_codepoint" % index)
        dropped, reported = carry.get("notes_dropped_bytes"), carry.get("reported_dropped_bytes")
        if dropped is None or reported is None or int(dropped) != int(reported):
            faults.append("attempt_%s:dropped_count_unreported" % index)
    ok = not faults
    return ok, evidence(target="carry_record", boundaries=len(rows), faults=faults[:6],
                        widest_notes_bytes=widest, cap=CARRY_NOTES_BYTE_CAP,
                        header_lines=CARRY_HEADER_LINES,
                        reason=None if ok else "summary-channel-bypass")''',

        "no_prior_attempt_bytes_present_when_an_attempt_opens": '''\
    rows = attempts()
    if not rows:
        return False, evidence(target="run_record", reason="episode_record_absent")
    inspected, found, mounts, boundaries = set(), [], [], 0
    for row in rows:
        residue = row.get("residue")
        if not isinstance(residue, dict):
            continue
        boundaries += 1
        inspected |= {str(name) for name in (residue.get("inspected") or [])}
        for item in (residue.get("found") or []):
            found.append("attempt_%s:%s" % (row.get("attempt_index"), item))
        for mount in (residue.get("persistent_mounts") or []):
            mounts.append("attempt_%s:%s" % (row.get("attempt_index"), mount))
    if not boundaries:
        return False, evidence(target="run_record", reason="residue_never_inspected")
    # An empty residue roster over an empty INSPECTION roster passes vacuously. Every class the
    # teardown is supposed to destroy has to have been looked for on at least one boundary.
    unchecked = sorted(set(RESIDUE_CLASSES) - inspected)
    ok = (not found) and (not mounts) and (not unchecked)
    return ok, evidence(target="run_record", boundaries_inspected=boundaries,
                        residue_found=found[:6], persistent_mounts=mounts[:4],
                        classes_never_inspected=unchecked, classes_expected=list(RESIDUE_CLASSES),
                        reason=None if ok else ("residue_never_inspected" if unchecked
                                                else "prior_attempt_state_survived"))''',

        "bank_debits_are_analytic_accounted_seconds_not_wall_clock": '''\
    rows = attempts()
    if not rows:
        return False, evidence(target="bank_ledger", reason="bank_ledger_absent")
    mismatched, clock_shaped, checked = [], [], 0
    for row in rows:
        index = row.get("attempt_index")
        debit, tokens = row.get("accounted_seconds_debited"), row.get("tokens")
        per_token = row.get("flops_per_token")
        if debit is None or tokens is None or per_token is None:
            mismatched.append("attempt_%s:unrecorded" % index)
            continue
        checked += 1
        recomputed = int(math.ceil(int(tokens) * float(per_token)
                                   / ACCOUNTED_FLOPS_PER_SECOND))
        if isinstance(debit, bool) or not isinstance(debit, int) or debit != recomputed:
            mismatched.append("attempt_%s:%r_not_%d" % (index, debit, recomputed))
        busy = row.get("nvml_busy_s")
        # NVML is the FRAUD CROSS-CHECK on this slot, never the meter. If every debit is within
        # half a second of the busy reading, the bank is a stopwatch and the campaign is not
        # replayable no matter what the ledger calls its column.
        if busy is not None and abs(float(busy) - float(debit)) < NVML_SEPARATION:
            clock_shaped.append("attempt_%s" % index)
    ok = checked > 0 and not mismatched and len(clock_shaped) < checked
    return ok, evidence(target="bank_ledger", rows_checked=checked, mismatched=mismatched[:6],
                        debits_indistinguishable_from_nvml=len(clock_shaped),
                        reference_flops_per_second=ACCOUNTED_FLOPS_PER_SECOND,
                        reason=None if ok else "debit_is_not_the_analytic_accounted_second")''',

        "bank_balance_non_increasing_and_never_below_zero": '''\
    rows = attempts()
    if not rows:
        return False, evidence(target="bank_ledger", reason="bank_ledger_absent")
    opening = (episode().get("metering") or {}).get("opening_balance_s")
    balances, faults, previous = [], [], opening
    for row in rows:
        index, after = row.get("attempt_index"), row.get("bank_balance_after_s")
        debit = row.get("accounted_seconds_debited")
        if after is None:
            faults.append("attempt_%s:balance_unrecorded" % index)
            continue
        after = int(after)
        balances.append(after)
        if after < 0:
            faults.append("attempt_%s:balance_negative" % index)
        if previous is not None:
            if after > int(previous):
                faults.append("attempt_%s:balance_rose" % index)
            if debit is not None and after != int(previous) - int(debit):
                faults.append("attempt_%s:balance_does_not_reconcile" % index)
        previous = after
    opens_right = opening is not None and int(opening) == EPISODE_BANK_ACCOUNTED_S
    ok = opens_right and not faults
    return ok, evidence(target="bank_ledger", opening_balance_s=opening,
                        pinned_opening_s=EPISODE_BANK_ACCOUNTED_S,
                        closing_balance_s=balances[-1] if balances else None,
                        spent_s=((int(opening) - balances[-1]) if opens_right and balances
                                 else None),
                        rows=len(rows), faults=faults[:6],
                        reason=None if ok else "bank_ledger_not_monotone")''',

        "bank_exhaustion_grades_the_export_on_disk_rather_than_zeroing": '''\
    rows = attempts()
    if not rows:
        return False, evidence(target="bank_ledger", reason="bank_ledger_absent")
    refusals, ran_at_zero = [], []
    for row in rows:
        reason = row.get("refused_reason")
        if reason is None:
            continue
        refusals.append("attempt_%s:%s" % (row.get("attempt_index"), reason))
        if str(reason) == "BANK_EXHAUSTED" and row.get("executed"):
            ran_at_zero.append("attempt_%s" % row.get("attempt_index"))
    zero_reasons = sorted({str(r) for r in (episode().get("zero_reasons") or [])})
    last = last_attempt() or {}
    graded_last = last.get("graded_bpb") is not None
    # The ONE budget overrun this slot declines to zero. Spending the bank early is priced in
    # bits per byte -- whatever the last attempt could train -- and never in a gate.
    ok = ("BANK_EXHAUSTED" not in zero_reasons) and not ran_at_zero and graded_last
    return ok, evidence(target="bank_ledger", refusals=refusals[:6],
                        bank_exhausted_refusals=len([r for r in refusals
                                                     if "BANK_EXHAUSTED" in r]),
                        zero_reasons=zero_reasons,
                        bank_exhaustion_in_zero_set="BANK_EXHAUSTED" in zero_reasons,
                        last_attempt_graded=graded_last, ran_at_zero_balance=ran_at_zero[:4],
                        reason=None if ok else "bank_exhaustion_was_gated_not_priced")''',

        "metering_opens_at_first_corpus_read_not_interpreter_start": '''\
    metering = episode().get("metering")
    if not isinstance(metering, dict) or not metering:
        return False, evidence(target="bank_ledger", reason="bank_ledger_absent")
    opened_on = metering.get("origin_event")
    before = metering.get("gpu_processes_before_origin")
    outside = metering.get("gpu_processes_outside_launcher")
    if before is None or outside is None:
        return False, evidence(target="bank_ledger", reason="bank_ledger_absent",
                               origin_event=opened_on)
    # Opening at interpreter start would put a statistics pass over the 2.5 GB shard outside the
    # bank; opening at the first optimizer step would put the whole data pipeline outside it.
    ok = (opened_on == "first_corpus_read" and int(before) == 0 and int(outside) == 0)
    return ok, evidence(target="bank_ledger", origin_event=opened_on,
                        origin_stamp_s=metering.get("origin_stamp_s"),
                        gpu_processes_before_origin=before,
                        gpu_processes_outside_launcher=outside,
                        reason=None if ok else ("unmetered-gpu-access" if int(outside)
                                                else "meter_did_not_open_at_first_corpus_read"))''',

        "repeat_configurations_cost_bank_but_buy_no_measurement": '''\
    rows = attempts()
    if not rows:
        return False, evidence(target="attempt_ledger", reason="episode_record_absent")
    seen, faults, fresh, over_cap = set(), [], 0, []
    for row in rows:
        index, digest = row.get("attempt_index"), row.get("config_digest")
        if digest is None:
            faults.append("attempt_%s:no_config_digest" % index)
            continue
        repeated = digest in seen
        seen.add(digest)
        bought = row.get("fresh_measurement")
        if bought:
            fresh += 1
        if isinstance(bought, int) and not isinstance(bought, bool):
            if bought > FRESH_MEASUREMENTS_PER_ATTEMPT:
                over_cap.append("attempt_%s" % index)
        if repeated:
            if not row.get("duplicate_config"):
                faults.append("attempt_%s:repeat_not_flagged" % index)
            if bought:
                faults.append("attempt_%s:repeat_bought_a_measurement" % index)
            # The debit already incurred is NOT refunded: the cost of having forgotten is real.
            if not row.get("accounted_seconds_debited"):
                faults.append("attempt_%s:repeat_was_refunded" % index)
        elif row.get("duplicate_config"):
            faults.append("attempt_%s:flagged_without_a_repeat" % index)
    ok = (not faults) and (not over_cap) and fresh <= MAX_ATTEMPTS
    return ok, evidence(target="attempt_ledger", attempts=len(rows),
                        distinct_configurations=len(seen), repeats=len(rows) - len(seen),
                        fresh_measurements=fresh, measurement_cap=MAX_ATTEMPTS,
                        per_attempt_cap_breached=over_cap[:4], faults=faults[:6],
                        reason=None if ok else "duplicate_configuration_accounting_wrong")''',

        "ranking_rows_carry_the_partition_epoch_in_force": '''\
    rows = attempts()
    if not rows:
        return False, evidence(target="attempt_ledger", reason="episode_record_absent")
    epochs, by_index = [], {}
    for row in rows:
        epoch, index = row.get("partition_epoch"), row.get("attempt_index")
        if epoch is None or index is None:
            return False, evidence(target="attempt_ledger", attempt=index,
                                   reason="partition_epoch_unrecorded")
        epochs.append(int(epoch))
        by_index[int(index)] = int(epoch)
    non_decreasing = all(b >= a for a, b in zip(epochs, epochs[1:]))
    rotations = sum(1 for a, b in zip(epochs, epochs[1:]) if b != a)
    final_epoch = epochs[-1]
    try:
        compared = load_submission()["final_selection"].get("compared_attempt_indices")
    except Exception as exc:
        return False, evidence(target="attempt_ledger", reason="submission_malformed",
                               error=str(exc))
    if not isinstance(compared, list) or not compared:
        return False, evidence(target="attempt_ledger", reason="comparison_set_undeclared",
                               final_epoch=final_epoch)
    unknown = [i for i in compared if int(i) not in by_index]
    stale = sorted({by_index[int(i)] for i in compared if int(i) in by_index} - {final_epoch})
    # The rotation index is never disclosed, only that one occurs, so the discipline that
    # survives it is RECORDING the epoch -- not predicting the instant.
    ok = non_decreasing and rotations == ROTATION_EVENTS and not stale and not unknown
    return ok, evidence(target="attempt_ledger", epoch_sequence=epochs[:12],
                        rotations_observed=rotations, rotations_expected=ROTATION_EVENTS,
                        epochs_non_decreasing=non_decreasing, final_epoch=final_epoch,
                        genres=list(GRADED_PARTITION_GENRES), compared_rows=len(compared),
                        stale_epochs_compared=stale, unknown_rows_compared=unknown[:4],
                        reason=None if ok else "ranking_mixed_rotation_epochs")''',

        "regrading_the_frozen_export_reproduces_bit_identically": '''\
    regrade = episode().get("regrade")
    if not isinstance(regrade, dict) or not regrade:
        return False, evidence(target="exported_program", reason="regrade_not_performed")
    first, second = regrade.get("pass_1_bpb"), regrade.get("pass_2_bpb")
    if first is None or second is None:
        return False, evidence(target="exported_program", reason="regrade_not_performed")
    # BIT equality, not a tolerance. Grading is forward-only over frozen bytes with the model in
    # eval and no sampling, so a stochastic operator in the exported graph is the only way this
    # can differ -- and it is caught rather than averaged away.
    left, right = struct.pack(">d", float(first)), struct.pack(">d", float(second))
    ok = left == right
    return ok, evidence(target="exported_program", pass_1_bpb=float(first),
                        pass_2_bpb=float(second), pass_1_bits=left.hex(),
                        pass_2_bits=right.hex(),
                        reason=None if ok else "export-nondeterministic")''',

        "reward_unmoved_under_a_second_alphabet_permutation": '''\
    trial = episode().get("second_permutation")
    if not isinstance(trial, dict) or not trial:
        return False, evidence(target="exported_program", reason="second_permutation_not_run")
    first, second = trial.get("reward_first"), trial.get("reward_second")
    if first is None or second is None:
        return False, evidence(target="exported_program", reason="second_permutation_not_run")
    moved = abs(float(first) - float(second))
    # Parameters not derived from the bytes THIS run read cannot survive a change of the private
    # bijection. Recorded honestly: frequency analysis can invert a permutation, so this raises
    # the cost of transfer rather than closing the route.
    ok = moved <= PERMUTATION_REWARD_TOLERANCE
    return ok, evidence(target="exported_program", reward_first=float(first),
                        reward_second=float(second), moved=moved,
                        tau=PERMUTATION_REWARD_TOLERANCE,
                        reason=None if ok else "precomputed-params")''',

        "training_inputs_are_the_pinned_shipped_shard_bytes": '''\
    inputs = episode().get("training_inputs")
    if not isinstance(inputs, dict) or not inputs:
        return False, evidence(target="run_record", reason="training_inputs_unrecorded")
    checked = inputs.get("records_checked")
    mismatched = [str(r) for r in (inputs.get("mismatched") or [])]
    unpinned = [str(r) for r in (inputs.get("unpinned") or [])]
    # Data is FIXED here, so the lever is shape and not corpus. The graded partitions are
    # Gutenberg volumes a public mirror will serve on request, and fetching them during the
    # open-egress solving phase would replace an architecture problem with a retrieval one.
    ok = checked is not None and int(checked) > 0 and not mismatched and not unpinned
    return ok, evidence(target="run_record", records_checked=checked,
                        mismatched=mismatched[:6], unpinned=unpinned[:6], shard=TRAIN_SHARD,
                        reason=None if ok else "external-supervision")''',

        "no_upload_shaped_request_left_the_solving_sandbox": '''\
    log = episode().get("egress_log")
    if not isinstance(log, dict) or not log:
        return False, evidence(target="run_record", reason="egress_log_absent")
    requests = [r for r in (log.get("requests") or []) if isinstance(r, dict)]
    attested = log.get("attested_empty") is True
    if not requests and not attested:
        return False, evidence(target="run_record", reason="egress_log_absent")
    methods = sorted({str(r.get("method")) for r in requests})
    disallowed = sorted(set(methods) - EGRESS_METHODS)
    with_body = [str(r.get("url")) for r in requests
                 if r.get("request_body_bytes") is not None
                 and int(r["request_body_bytes"]) > 0]
    # The carry channel is only narrow if it is the widest one available. A POST of the whole
    # ledger at attempt k, fetched back at attempt k+1, would restore an unbounded memory.
    ok = not disallowed and not with_body
    return ok, evidence(target="run_record", requests=len(requests), attested_empty=attested,
                        allowed_methods=sorted(EGRESS_METHODS), methods_seen=methods,
                        disallowed_methods=disallowed, requests_with_a_body=with_body[:4],
                        reason=None if ok else "summary-channel-bypass")''',

        "metering_and_evidence_instruments_unwritten": '''\
    digests, heldout = episode().get("tree_digests"), episode().get("heldout_digests")
    if not isinstance(digests, dict) or not digests:
        return False, evidence(target="run_record", reason="tree_digests_unrecorded")
    before, after = digests.get("before") or {}, digests.get("after") or {}
    if not before or not after:
        return False, evidence(target="run_record", reason="tree_digests_unrecorded")
    if not isinstance(heldout, dict) or not heldout:
        return False, evidence(target="run_record", reason="heldout_digests_unrecorded")
    changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    roster = set(before) & set(after)
    # An empty digest roster would clear the comparison vacuously. The launcher and the emitter
    # ARE the measurement on this slot, so they have to be inside the roster that was compared.
    absent_instruments = [name for name in STAMPING_INSTRUMENTS
                          if not any(str(p).endswith(name) for p in roster)]
    tampered = sorted(p for p in heldout
                      if (heldout[p] or {}).get("observed") != (heldout[p] or {}).get("pinned"))
    expected = len(GRADED_PARTITIONS) + len(DECOY_PARTITIONS)
    ok = (not changed and not absent_instruments and not tampered
          and len(heldout) == expected)
    return ok, evidence(target="run_record", paths_digested=len(roster), changed=changed[:6],
                        partitions_digested=len(heldout), partitions_expected=expected,
                        tampered=tampered[:6], instruments_absent_from_roster=absent_instruments,
                        instruments=list(STAMPING_INSTRUMENTS),
                        reason=None if ok else ("heldout-digest-mismatch" if tampered
                                                else "grading-tree-written"))''',

        "bits_per_byte_of_the_last_attempt_export_ramp": '''\
    # require_measured runs FIRST, before a single submitted byte is read. If a missing export or
    # an absent episode record raised here instead, the verifier would report gradeable=true with
    # r_det=0.0 and collapse "not yet measurable" into "the agent scored zero";
    # seed/build/freeze_batch.py refuses to freeze a bundle that does that.
    c = require_measured("floor", "knee", "dichotomizing_threshold")
    bpb = measure()
    # LOWER bits per byte is better, so the ramp DESCENDS: the floor is the poor end.
    span = c["floor"] - c["knee"]
    s = 0.0 if span <= 0 else max(0.0, min(1.0, (c["floor"] - bpb) / span))
    row = last_attempt() or {}
    in_force = row.get("in_force") or {}
    return bpb <= c["dichotomizing_threshold"], evidence(
        target="exported_program", bits_per_byte=bpb, s_outcome=s,
        graded_attempt_index=row.get("attempt_index"),
        per_partition=dict((n, float((in_force[n] or {}).get("bpb")))
                           for n in sorted(in_force)
                           if isinstance(in_force[n], dict)
                           and in_force[n].get("bpb") is not None),
        floor=c["floor"], knee=c["knee"],
        dichotomizing_threshold=c["dichotomizing_threshold"])''',
    }
    # REFUSE rather than fall through. A generic obligation lookup used to stand here and quietly
    # supply a body for any item nobody had written one for; the body it supplied read a run
    # record key nothing ever wrote, so an outcome check consulted nothing at all and still
    # reported a verdict. An unwritten body is an authoring gap and it stops the generator.
    unwritten = sorted(it["id"] for it in g["items"]
                       if it["mode"] == "compiled" and it["id"] not in bodies)
    if unwritten:
        raise SystemExit("A5-01: no compiled body is written for %s; refusing to emit a checker "
                         "that would consult nothing" % ", ".join(unwritten))
    orphaned = sorted(set(bodies) - {it["id"] for it in g["items"]})
    if orphaned:
        raise SystemExit("A5-01: bodies are written for %s, which grounding.yaml does not carry"
                         % ", ".join(orphaned))

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
        fh.write(build_tests(g, PRELUDE % subst, bodies) + facts_block(g) + TAIL)
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


PRELUDE = '''"""Compiled checks for %(slot_id)s. GENERATED by solution/recompute.py. DO NOT HAND-EDIT.

Every check below runs against delivered bytes and references no unmeasured quantity, except
%(outcome_id)s, which raises ConstantUnmeasured while tests/constants.json carries nulls.

The graded quantity is %(metric)s.
"""
import json
import math
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
SUBMISSION = %(path)r
CHECKS = []


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
    return {"evaluation_target": kw.pop("target", "exported_program"), "values": kw}
'''


TAIL = '''

def load_submission():
    """The graded account of a fifty-attempt architecture search at fixed data and fixed compute:
    the exported PROGRAM and its parameter blob, the per-attempt ledger with configuration
    digests and rotation epochs, the fixed-schema carry-forward record that crossed the 2048-byte
    channel, the accounted-second bank, and the final selection. Not a checkpoint and not a
    training script -- the graded object is a program the verifier re-executes itself."""
    with open(SUBMISSION) as handle:
        doc = json.load(handle)
    if not isinstance(doc, dict):
        raise ValueError("the deliverable must be a JSON object")
    return doc


def workspace_path(relative):
    """Resolve a path the submission declares, against the agent workspace beside tests/."""
    return os.path.normpath(os.path.join(HERE, "..", str(relative)))


def episode():
    """The harness's own record of the fifty-attempt episode, under the verifier-owned held-out
    tree.

    Read from there and never from the submission: the per-attempt bits per byte, the
    accounted-second debits, the crossing digests at every attempt boundary and the evidence
    packet field order are exactly the quantities the agent must not be able to author. Making a
    write into this tree a red line is only a rule if the number is read from inside it.
    """
    path = os.path.join(HERE, "heldout", "harness_episode.json")
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def run_record():
    """The record the graded run itself produced. Everything here is the agent's own account and
    is checked for honesty against the episode record, never substituted for it."""
    path = os.path.join(HERE, "..", "run_record.json")
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def attempts():
    """Every attempt the harness observed, in ATTEMPT-INDEX order.

    The sort is what makes the last-attempt rule below independent of how the ledger happened to
    be serialised: the graded export is the one submitted at the highest index, and a ledger
    written out of order must not be able to move which row that is.
    """
    rows = episode().get("attempts")
    if not isinstance(rows, list):
        return []
    kept = [row for row in rows
            if isinstance(row, dict) and row.get("attempt_index") is not None]
    return sorted(kept, key=lambda row: int(row["attempt_index"]))


def last_attempt():
    """The attempt the score is read off. LAST, not best.

    task.toml binds final_selection to last, so the entire value of the first forty-nine attempts
    reaches the graded float only through what the agent carried forward in 2048 bytes. An
    earlier attempt that measured better contributes nothing at all.
    """
    rows = attempts()
    return rows[-1] if rows else None


def export_facts():
    """What the harness observed about the exported program itself: its operator list, the
    recounted serialized parameter bytes, the per-tensor byte table, the width it was trained at
    and the width its quantization scales were measured at."""
    facts = episode().get("export")
    return facts if isinstance(facts, dict) else {}


def packets():
    """The evidence packets the emitter wrote, one per attempt, each a 204-key object whose FIELD
    ORDER follows the private per-attempt permutation schedule."""
    rows = episode().get("evidence_packets")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def vocabulary():
    """The pinned 204-name evidence vocabulary. Fixed and named for the whole episode -- it is
    the ORDER that moves, which is exactly why a record may reference names and never slots."""
    names = episode().get("evidence_vocabulary")
    return [str(n) for n in names] if isinstance(names, list) else []


def permutation_drift():
    """The furthest a diagnostic's POSITION moves between the first packet and any later one.

    Recorded as evidence on every run rather than only on failing ones. A name-keyed
    carry-forward record is indifferent to this number; a positionally-keyed one is turned into
    noise by it, silently, from the first permutation onward and with no traceback anywhere.
    """
    orders = [p.get("field_order") for p in packets() if isinstance(p.get("field_order"), list)]
    if len(orders) < 2:
        return None
    first = dict((str(name), i) for i, name in enumerate(orders[0]))
    worst = 0
    for order in orders[1:]:
        for i, name in enumerate(order):
            if str(name) in first:
                worst = max(worst, abs(i - first[str(name)]))
    return worst


def measure():
    """Bits per byte of the LAST attempt's exported program, on the WORSE of the two in-force
    private sibling partitions.

    Both halves of that sentence are the axis. Last, not best: an earlier attempt that measured
    better is not the export the verifier runs. Worse, not mean: taking the maximum over the two
    siblings is what stops a design tuned to one draw of post-2026 Gutenberg volumes from being
    paid for a distribution it did not fit. The value is the harness executor's own measurement
    over frozen bytes, byte-normalized, and never a number the submission reports.
    """
    row = last_attempt()
    if row is None:
        raise ConstantUnmeasured(
            "the harness recorded no attempt; there is no last export to measure")
    in_force = row.get("in_force")
    if not isinstance(in_force, dict):
        raise ConstantUnmeasured(
            "the last attempt records no in-force private partitions")
    values = [float(part["bpb"]) for part in in_force.values()
              if isinstance(part, dict) and part.get("bpb") is not None]
    if len(values) != IN_FORCE_PARTITIONS:
        raise ConstantUnmeasured(
            "the last attempt has no pair of in-force private partitions to take the worse of")
    return max(values)
'''


if __name__ == "__main__":
    raise SystemExit(main())
