"""Derivation for C1-S2: every canonical artifact descends from solution/grounding.yaml.

Wired through seed/build/generator_common.py, which carries the five slot-agnostic functions
verified byte-for-byte against the three frozen generators. build_provenance stays here, per
family, for the reason recorded in that module.

The graded axis of this slot is TIME TO A BAR: harness-measured seconds from the first read of
the shipped corpus to the harness stamp of the earliest emitted checkpoint at or below 1.02 bits
per byte on the private held-out shard. Every compiled body below names something that axis has
and a quality-graded or throughput-graded slot does not -- a clock origin at the first corpus
read, an emission cadence that sets the resolution of the score, an earliest-clearing scan rather
than a best-of scan, an ordered family of bars, and a per-domain token trace over three labelled
corpora that the declared recipe has to match.
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
    """Assemble the compiled-check module for the time-to-bar grading."""
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
SLOT_ID = 'C1-S2'
TITLE = "# C1-S2 -- what this task actually tests"


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
        "timed_axis": True,
        # Recorded here because these are the two facts the whole slot turns on and the two an
        # agent is most likely to assume away: the clock opens at the FIRST READ of the shipped
        # corpus, so every second of setup is graded, and the terminus is the EARLIEST emitted
        # checkpoint that clears the bar, so a better checkpoint after it moves nothing and a
        # checkpoint that never clears is worth nothing at all.
        "clock_origin": ("the harness stamp of the first read of the shipped corpus, before "
                         "tokenizer construction, memmap open and dataloader warm-up"),
        "clock_terminus": ("the harness stamp of the earliest emitted checkpoint measured at or "
                           "below %s bits per byte on %s"
                           % (layout["graded_bar_bpb"], layout["private_shard"])),
        "labelled_domains": list(layout["domains"]),
        "heldout_shares": dict(sorted(layout["heldout_shares"].items())),
        "corpus_status": layout["corpus_status"],
    }
    out.update(read_screening(HERE))
    return out


REQUIRED_ASSERTIONS = ("checkpoint_cadence_bounds_the_time_to_bar",
                       "clock_opens_at_first_corpus_read",
                       "declared_crossing_is_the_earliest_checkpoint_to_clear_the_bar",
                       "executed_domain_segments_match_declared_stage_order",
                       "harness_stamping_instruments_unwritten")
RAMP_CONSTANTS = ("dichotomizing_threshold", "floor", "knee", "reward_gate_pass_threshold")


def validate(grounding):
    """A time-to-bar slot is well formed only if four facts are asserted by some item.

    The clock origin, the earliest-clearing rule, the emission cadence and the executed domain
    order are not hygiene here, they are the axis. If the clock does not open at the first corpus
    read then setup is free and the graded span is not the one the objective names. If the
    crossing is taken as the best checkpoint rather than the earliest one, the quantity measured
    is loss-at-budget wearing a stopwatch. If the cadence is unbound, the resolution of the score
    is unbounded and a run can be charged for time it had already earned. And if the executed
    per-domain trace is never compared with the declared recipe, the ordering lever this corpus
    exists to expose is graded by declaration alone.
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
    if "run_record" not in targets:
        problems.append("a timed slot must grade against the harness run record")
    stray = sorted(targets - set(grounding["evaluation_target_vocabulary"]))
    if stray:
        problems.append("items name evaluation targets the vocabulary does not: "
                        + ", ".join(stray))
    layout = grounding.get("corpus_layout") or {}
    domains = list(layout.get("domains") or [])
    if len(domains) != 3 or len(set(domains)) != 3:
        problems.append("the shipped corpus is three labelled domains, found %d" % len(domains))
    shares = layout.get("heldout_shares") or {}
    if sorted(shares) != sorted(domains):
        problems.append("the disclosed held-out shares must cover exactly the three domains")
    elif abs(sum(float(v) for v in shares.values()) - 1.0) > 1e-9:
        problems.append("the disclosed held-out shares must sum to one")
    family = [float(v) for v in (layout.get("bar_family") or [])]
    if len(family) != 3 or family != sorted(family, reverse=True):
        problems.append("the ordered bar family must run easiest to hardest, found %r" % family)
    elif float(layout.get("graded_bar_bpb", -1.0)) not in family:
        problems.append("the graded bar must be a member of the ordered family")
    return problems


def facts_block(g):
    """The literal facts the compiled checks read, rendered from grounding.yaml.

    They live in the emitted module rather than in tests/constants.json because none of them is
    a MEASUREMENT: they are the disclosed shape of this task -- three labelled domains, a
    disclosed held-out mixture, an ordered family of bars, a token budget and a parameter cap.
    tests/constants.json carries only the ramp, which Phase 0 does not author and leaves null.
    """
    layout = g["corpus_layout"]
    domains = tuple(sorted(layout["domains"]))
    shares = dict((k, layout["heldout_shares"][k]) for k in domains)
    return (
        '\n# --- the facts this slot is about ------------------------------------------------\n'
        '# The five sections of the graded artifact. A missing section is a malformed\n'
        '# submission, not a weak one.\n'
        'SECTIONS = %r\n'
        '# The three licence-clean corpora, carrying per-document domain labels. Mixture\n'
        '# weighting is a lever here only because every document knows which corpus it is from.\n'
        'DOMAINS = %r\n'
        '# Disclosed in instruction.md, and deliberately NOT the shipped training mixture.\n'
        'HELDOUT_SHARES = %r\n'
        'SHARE_TOLERANCE = 0.005\n'
        '# Easiest to hardest. Crossing stamps along one trajectory are non-decreasing here.\n'
        'BAR_FAMILY = %r\n'
        'GRADED_BAR = %r\n'
        'CONTEXT = %r\n'
        'TOKEN_CAP = %r\n'
        'PARAM_CAP = %r\n'
        'BOUNDARY_TOLERANCE_TOKENS = %r\n'
        'SIBLING_TOLERANCE = %r\n'
        'SEQUENCE_LENGTHS = %r\n'
        'VOCAB_ENTRIES = %r\n'
        'IMPORT_ALLOWLIST = frozenset(("math", "safetensors", "torch"))\n'
        'FORBIDDEN_ATTRIBUTE_ROOTS = frozenset(("ctypes", "importlib", "os", "requests",\n'
        '                                       "socket", "subprocess", "sys", "urllib"))\n'
        '# A stamp comparison is a comparison of harness clocks, so it is exact to the\n'
        '# millisecond and no looser: the graded quantity IS the stamp.\n'
        'STAMP_TOLERANCE = 0.001\n'
        'LEDGER_GAP_TOLERANCE_S = 1.0\n'
        'CADENCE_SLACK = 1.05\n'
        'LOG_TOLERANCE = 1e-06\n'
        'TRAINING_PHASES = ("train",)\n'
        '# The two files that ARE the measurement on this slot: the emitter assigns the stamp\n'
        '# the score is read off, the shim writes the per-domain trace the recipe is checked\n'
        '# against. Editing either would forge the graded instant rather than earn it.\n'
        'STAMPING_INSTRUMENTS = ("aello_emit.py", "memmap_shim.py")\n'
        '# A bar that was never crossed sorts after every bar that was.\n'
        'UNCROSSED = float("inf")\n'
        % (tuple(layout["submission_sections"]), domains, shares,
           tuple(float(v) for v in layout["bar_family"]), float(layout["graded_bar_bpb"]),
           int(layout["context"]), int(layout["token_budget"]), int(layout["param_cap"]),
           int(layout["boundary_tolerance_tokens"]), float(layout["sibling_tolerance_bpb"]),
           tuple(int(n) for n in layout["sequence_lengths"]), int(layout["vocab_entries"])))


def main():
    g = load_grounding()
    # The family validator runs FIRST and refuses rather than emitting: a generator that writes a
    # carrier it knows is malformed hands the drift check a moving target.
    problems = validate(g)
    if problems:
        raise SystemExit("C1-S2 grounding is not well-formed: " + "; ".join(problems))
    dl = g["deliverable"]
    layout = g["corpus_layout"]
    outcome_id = [i["id"] for i in g["items"] if i["dimension"] == "outcome"][0]
    metric = ('Harness-measured wall-clock seconds, from the first read of the shipped corpus to '
              'the harness stamp of the EARLIEST emitted checkpoint scoring bits per byte at or '
              'below %s on %s, evaluated at context %d over non-overlapping windows excluding '
              'each window first position, in nats divided by ln 2 and by total shard bytes. '
              'Earliest, not best: a checkpoint that models better after the crossing moves '
              'nothing and a checkpoint that never crosses is worth nothing, so the emission '
              'cadence is itself the resolution of the score. The executed per-domain token '
              'trace from the harness memmap shim is recorded for grading against the declared '
              'recipe.json, not for the reward axis'
              % (layout["graded_bar_bpb"], layout["private_shard"], int(layout["context"])))
    subst = {"slot_id": SLOT_ID, "path": dl["path"], "metric": metric, "outcome_id": outcome_id}

    # BODIES ARE WRITTEN FOR THIS SLOT. Each one names a fact of a time-to-bar run over three
    # labelled corpora -- a clock origin at the first corpus read, an earliest-clearing scan, an
    # emission gap, an ordered family of bars, a per-domain token trace, a 2047-of-2048 window --
    # and not one of them would mean anything on a classifier, a forecaster or a tokenizer. The
    # generic spine that stood here graded the existence and shape of a deliverable and said
    # nothing about when the bar was crossed, which is the only thing this task is about.
    bodies = {

        "emitted_checkpoint_series_is_stamped": '''\
    if not os.path.exists(SUBMISSION):
        return False, evidence(target="checkpoint_series", reason="no-deliverable",
                               detail="submission_missing", path=SUBMISSION)
    try:
        doc = load_submission()
    except Exception as exc:
        return False, evidence(target="checkpoint_series", reason="submission_malformed",
                               error=str(exc))
    absent = [name for name in SECTIONS if name not in doc]
    if absent:
        return False, evidence(target="checkpoint_series", reason="submission_malformed",
                               absent=absent, carried=sorted(doc))
    series = doc["checkpoint_series"]
    if not isinstance(series, list) or not series:
        return False, evidence(target="checkpoint_series", reason="no-deliverable",
                               detail="no_checkpoint_emitted")
    unstamped, stamps = [], []
    for record in series:
        if not isinstance(record, dict):
            unstamped.append(None)
            continue
        if (record.get("checkpoint_id") is None or record.get("harness_stamp_s") is None
                or record.get("bpb_private") is None):
            unstamped.append(record.get("checkpoint_id"))
        else:
            stamps.append(float(record["harness_stamp_s"]))
    # Emission ORDER is stamp order. The score is read off a stamp, so a series that does not
    # rise has either replayed an old checkpoint or restamped one, and neither can be timed.
    ascending = all(b > a for a, b in zip(stamps, stamps[1:]))
    ok = (not unstamped) and ascending and len(stamps) == len(series)
    return ok, evidence(target="checkpoint_series", emitted=len(series),
                        unstamped=unstamped[:4], stamps_ascending=ascending,
                        first_stamp_s=stamps[0] if stamps else None,
                        last_stamp_s=stamps[-1] if stamps else None,
                        reason=None if ok else "unstamped_or_unordered_emission")''',

        "recipe_declares_order_mixture_and_token_boundaries": '''\
    try:
        recipe = load_submission()["recipe"]
        stages = list(recipe["stages"])
    except Exception as exc:
        return False, evidence(target="recipe_declaration", reason="submission_malformed",
                               error=str(exc))
    if not stages:
        return False, evidence(target="recipe_declaration", reason="no-deliverable",
                               detail="no_stage_declared")
    names, boundaries, malformed = [], [], []
    for stage in stages:
        if not isinstance(stage, dict):
            malformed.append("stage_is_not_an_object")
            continue
        label = str(stage.get("stage"))
        names.append(label)
        mixture = stage.get("mixture")
        if not isinstance(mixture, dict) or sorted(map(str, mixture)) != sorted(DOMAINS):
            malformed.append("mixture_not_over_the_three_domains_" + label)
        else:
            weights = [float(mixture[name]) for name in DOMAINS]
            if any(w < 0.0 for w in weights) or abs(sum(weights) - 1.0) > 1e-6:
                malformed.append("mixture_is_not_a_distribution_" + label)
        edge = stage.get("token_boundary")
        if not isinstance(edge, int) or isinstance(edge, bool) or edge <= 0:
            malformed.append("boundary_is_not_a_token_offset_" + label)
        else:
            boundaries.append(edge)
    rising = all(b > a for a, b in zip(boundaries, boundaries[1:]))
    distinct = len(set(names)) == len(names)
    budget = recipe.get("token_budget")
    # The last boundary IS the budget: a declaration whose stages stop short of it has left a
    # tail of consumed tokens outside every declared stage.
    closes = bool(boundaries) and budget == boundaries[-1] and 0 < budget <= TOKEN_CAP
    ok = (not malformed) and rising and distinct and closes
    return ok, evidence(target="recipe_declaration", stage_order=names,
                        token_boundaries=boundaries, declared_budget=budget,
                        boundaries_rising=rising, names_distinct=distinct,
                        malformed=malformed[:4],
                        reason=None if ok else "recipe_declaration_incomplete")''',

        "declared_crossing_is_the_earliest_checkpoint_to_clear_the_bar": '''\
    observed_id, observed_stamp = earliest_clearing(GRADED_BAR)
    if observed_stamp is None:
        return False, evidence(bar=GRADED_BAR, crossed=False,
                               checkpoints_stamped=len(harness_checkpoints()),
                               reason="no_checkpoint_reached_the_bar")
    try:
        rows = load_submission()["bar_crossings"]
        declared = rows[str(GRADED_BAR)]
    except Exception as exc:
        return False, evidence(bar=GRADED_BAR, reason="submission_malformed", error=str(exc))
    if not isinstance(declared, dict):
        return False, evidence(bar=GRADED_BAR, reason="crossing_undeclared",
                               earliest_clearing_id=observed_id,
                               earliest_clearing_stamp_s=observed_stamp)
    said = declared.get("harness_stamp_s")
    same_id = str(declared.get("checkpoint_id")) == str(observed_id)
    same_stamp = said is not None and abs(float(said) - observed_stamp) <= STAMP_TOLERANCE
    # Naming a LATER clearing checkpoint prices the run slower than it was; naming an EARLIER
    # one that never cleared claims a bar that was never crossed. Both are the same error about
    # the same axis: the graded instant is the first crossing, not the best checkpoint.
    ok = same_id and same_stamp
    return ok, evidence(bar=GRADED_BAR, earliest_clearing_id=observed_id,
                        earliest_clearing_stamp_s=observed_stamp,
                        declared_id=declared.get("checkpoint_id"), declared_stamp_s=said,
                        reason=None if ok else "declared_crossing_is_not_the_earliest_clearing")''',

        "clock_opens_at_first_corpus_read": '''\
    opened = timeline().get("first_corpus_read_stamp_s")
    if opened is None:
        return False, evidence(target="run_record", reason="clock_origin_unrecorded")
    crossing_id, crossing = earliest_clearing(GRADED_BAR)
    if crossing is None:
        return False, evidence(target="run_record", clock_origin_s=opened,
                               reason="no_checkpoint_reached_the_bar")
    harness_seconds = crossing - float(opened)
    try:
        clock = load_submission()["clock"]
        declared_origin = clock["first_corpus_read_stamp_s"]
        declared_seconds = float(clock["graded_seconds_to_bar"])
    except Exception as exc:
        return False, evidence(target="run_record", reason="submission_malformed",
                               error=str(exc))
    same_origin = (declared_origin is not None
                   and abs(float(declared_origin) - float(opened)) <= STAMP_TOLERANCE)
    same_span = abs(declared_seconds - harness_seconds) <= STAMP_TOLERANCE
    # The clock opens at the FIRST READ of the shipped corpus, so tokenizer construction, memmap
    # open, any statistics pass and dataloader warm-up are all inside the graded interval. An
    # agent that starts its own timer at the first optimizer step reports a smaller span, and the
    # difference reported below is exactly the setup it declined to pay for.
    ok = same_origin and same_span
    return ok, evidence(target="run_record", harness_origin_s=opened,
                        declared_origin_s=declared_origin,
                        crossing_checkpoint=crossing_id,
                        harness_seconds_to_bar=round(harness_seconds, 6),
                        declared_seconds_to_bar=declared_seconds,
                        understated_by_s=round(harness_seconds - declared_seconds, 6),
                        reason=None if ok else "clock_not_opened_at_first_corpus_read")''',

        "no_phase_escapes_the_graded_interval": '''\
    opened = timeline().get("first_corpus_read_stamp_s")
    crossing_id, crossing = earliest_clearing(GRADED_BAR)
    if opened is None or crossing is None:
        return False, evidence(target="run_record", reason="graded_interval_unbounded")
    try:
        ledger = load_submission()["clock"]["phases"]
    except Exception as exc:
        return False, evidence(target="run_record", reason="submission_malformed",
                               error=str(exc))
    if not isinstance(ledger, list) or not ledger:
        return False, evidence(target="run_record", reason="no_phase_ledger")
    opened, spans, before_open, named = float(opened), [], [], []
    for phase in ledger:
        if not isinstance(phase, dict):
            continue
        start, end = phase.get("start_s"), phase.get("end_s")
        named.append(str(phase.get("phase")))
        if start is None or end is None:
            continue
        start, end = float(start), float(end)
        if start < opened - STAMP_TOLERANCE:
            before_open.append(phase.get("phase"))
        if start < crossing:
            spans.append((start, min(end, crossing)))
    covered, cursor = 0.0, opened
    for start, end in sorted(spans):
        if end <= cursor:
            continue
        covered += end - max(start, cursor)
        cursor = max(cursor, end)
    interval = crossing - opened
    gap = round(interval - covered, 6)
    setup = sorted(set(name for name in named if name not in TRAINING_PHASES))
    # A ledger made only of training phases has hidden the corpus read that opened the clock.
    # That span is the one this task charges for and a throughput-graded task would not.
    ok = (not before_open) and gap <= LEDGER_GAP_TOLERANCE_S and bool(setup)
    return ok, evidence(target="run_record", graded_interval_s=round(interval, 6),
                        ledger_covers_s=round(covered, 6), unaccounted_gap_s=gap,
                        phases_before_clock_open=before_open[:4], setup_phases=setup[:6],
                        reason=None if ok else "setup_seconds_outside_the_graded_interval")''',

        "checkpoint_cadence_bounds_the_time_to_bar": '''\
    opened = timeline().get("first_corpus_read_stamp_s")
    records = harness_checkpoints()
    if opened is None or not records:
        return False, evidence(target="checkpoint_series", reason="no_stamped_emission")
    try:
        declared = float(load_submission()["clock"]["checkpoint_interval_s"])
    except Exception as exc:
        return False, evidence(target="checkpoint_series", reason="submission_malformed",
                               error=str(exc))
    crossing_id, crossing = earliest_clearing(GRADED_BAR)
    horizon = crossing if crossing is not None else float(records[-1]["stamp_s"])
    marks = [float(opened)] + [float(r["stamp_s"]) for r in records
                               if float(r["stamp_s"]) <= horizon]
    gaps = [b - a for a, b in zip(marks, marks[1:])]
    widest = max(gaps) if gaps else horizon - float(opened)
    # The harness can only stamp a checkpoint that was emitted, so it cannot stamp a crossing
    # that happened between two emissions. The widest gap up to the crossing is therefore the
    # GRANULARITY of the graded quantity: a run that reaches the bar but emits every 600 s is
    # charged for up to 600 s it had already earned. Emitting often is a lever on the axis.
    honoured = declared > 0.0 and widest <= declared * CADENCE_SLACK
    return honoured, evidence(target="checkpoint_series", declared_interval_s=declared,
                              widest_emission_gap_s=round(widest, 6),
                              score_granularity_s=round(widest, 6),
                              emissions_before_the_bar=len(gaps), crossed=crossing is not None,
                              reason=None if honoured else "emission_cadence_not_honoured")''',

        "bits_per_byte_excludes_each_window_first_position": '''\
    accounting = timeline().get("bpb_window_accounting")
    if not isinstance(accounting, dict):
        return False, evidence(target="heldout_shards", reason="window_accounting_unrecorded")
    try:
        context = int(accounting["context"])
        scored = int(accounting["positions_scored_per_window"])
        windows = int(accounting["windows"])
        nats = float(accounting["nats_total"])
        shard_bytes = int(accounting["shard_bytes"])
        shard_tokens = int(accounting["shard_tokens"])
        reported = float(accounting["bits_per_byte"])
    except Exception as exc:
        return False, evidence(target="heldout_shards", reason="window_accounting_malformed",
                               error=str(exc))
    # Each window first position has no in-window context to condition on, so it is not scored:
    # 2047 positions per 2048-token window, never 2048.
    excludes_first = context == CONTEXT and scored == CONTEXT - 1
    non_overlapping = windows > 0 and windows * context <= shard_tokens < (windows + 1) * context
    bits = nats / math.log(2.0)
    scored_total = windows * scored
    tolerance = 1e-6 * max(1.0, abs(bits))
    over_bytes = shard_bytes > 0 and abs(reported * shard_bytes - bits) <= tolerance
    over_scored = scored_total > 0 and abs(reported * scored_total - bits) <= tolerance
    # Dividing by scored positions instead of shard bytes moves the number the BAR is set on
    # without any modelling having changed, so the byte identity has to hold and the position
    # identity must not -- except where the two totals coincide and the distinction is undefined.
    indistinguishable = shard_bytes == scored_total
    ok = (excludes_first and non_overlapping and over_bytes
          and (indistinguishable or not over_scored))
    return ok, evidence(target="heldout_shards", context=context,
                        positions_scored_per_window=scored, windows=windows,
                        shard_bytes=shard_bytes, shard_tokens=shard_tokens,
                        identity_over_shard_bytes=over_bytes,
                        identity_over_scored_positions=over_scored,
                        reason=None if ok else "bits_per_byte_not_nats_over_shard_bytes")''',

        "ordered_bar_family_crossed_in_non_decreasing_time": '''\
    if not harness_checkpoints():
        return False, evidence(target="checkpoint_series", reason="no_stamped_emission")
    try:
        rows = load_submission()["bar_crossings"]
    except Exception as exc:
        return False, evidence(target="checkpoint_series", reason="submission_malformed",
                               error=str(exc))
    if not isinstance(rows, dict):
        return False, evidence(target="checkpoint_series", reason="bar_crossings_malformed")
    stamps, disagree = {}, []
    for bar in BAR_FAMILY:
        crossing_id, stamp = earliest_clearing(bar)
        stamps[str(bar)] = stamp
        row = rows.get(str(bar))
        said = row.get("harness_stamp_s") if isinstance(row, dict) else None
        if (stamp is None) != (said is None):
            disagree.append(str(bar))
        elif stamp is not None and abs(float(said) - stamp) > STAMP_TOLERANCE:
            disagree.append(str(bar))
    # BAR_FAMILY runs easiest to hardest and an uncrossed bar sorts after every crossed one, so
    # an uncrossed 1.09 beside a crossed 1.02 is an INVERSION rather than a missing row. Bits
    # per byte falls monotonically along one training trajectory; stamps that invert did not
    # come from one, which is what the ordered family is there to detect.
    order = [UNCROSSED if stamps[str(bar)] is None else stamps[str(bar)] for bar in BAR_FAMILY]
    monotone = all(a <= b for a, b in zip(order, order[1:]))
    ok = monotone and not disagree
    return ok, evidence(target="checkpoint_series", family=list(BAR_FAMILY),
                        crossing_stamps=stamps, monotone=monotone,
                        bars_the_submission_disagrees_on=disagree,
                        reason=None if ok else "target-order-inverted")''',

        "executed_domain_segments_match_declared_stage_order": '''\
    trace = timeline().get("domain_trace")
    if not isinstance(trace, list) or not trace:
        return False, evidence(target="domain_token_trace", reason="no_domain_trace")
    try:
        stages = list(load_submission()["recipe"]["stages"])
    except Exception as exc:
        return False, evidence(target="domain_token_trace", reason="submission_malformed",
                               error=str(exc))
    declared_majority, declared_edges = [], []
    for stage in stages:
        mixture = stage.get("mixture") if isinstance(stage, dict) else None
        if not isinstance(mixture, dict):
            return False, evidence(target="domain_token_trace",
                                   reason="submission_malformed",
                                   detail="stage_declares_no_mixture")
        # The majority domain of a stage is its argmax mixture weight. Per-document domain
        # labels are what let the shim resolve the executed stream into the same alphabet.
        declared_majority.append(max(DOMAINS, key=lambda d: float(mixture.get(d, 0.0))))
        declared_edges.append(stage.get("token_boundary"))
    segments = [seg for seg in trace if isinstance(seg, dict)]
    executed_majority = [str(seg.get("domain")) for seg in segments]
    executed_edges = [seg.get("end_token") for seg in segments]
    order_matches = executed_majority == declared_majority
    displaced = []
    for name, said, saw in zip(declared_majority, declared_edges, executed_edges):
        if said is None or saw is None:
            displaced.append(name)
        elif abs(int(saw) - int(said)) > BOUNDARY_TOLERANCE_TOKENS:
            displaced.append(name)
    # A smoothly annealed mixture declares discrete boundaries it never executes: the order can
    # match while every boundary is displaced, and the run zeroes even though the model is good.
    ok = order_matches and not displaced
    return ok, evidence(target="domain_token_trace", declared_stage_order=declared_majority,
                        executed_segment_order=executed_majority,
                        declared_boundaries=declared_edges, executed_boundaries=executed_edges,
                        boundary_tolerance_tokens=BOUNDARY_TOLERANCE_TOKENS,
                        displaced_stages=displaced,
                        reason=None if ok else "recipe-divergence")''',

        "stage_boundaries_counted_in_tokens_not_sequences": '''\
    served = timeline().get("per_domain_served_tokens")
    if not isinstance(served, dict) or not served:
        return False, evidence(target="domain_token_trace", reason="no_served_token_counters")
    try:
        recipe = load_submission()["recipe"]
        final_edge = int(list(recipe["stages"])[-1]["token_boundary"])
        lengths = [int(n) for n in (recipe.get("sequence_lengths") or SEQUENCE_LENGTHS)]
    except Exception as exc:
        return False, evidence(target="domain_token_trace", reason="submission_malformed",
                               error=str(exc))
    total = sum(int(v) for v in served.values())
    slack = max(1, int(BOUNDARY_TOLERANCE_TOKENS))
    in_tokens = abs(final_edge - total) <= slack
    # The shipped starter accounts the budget in SEQUENCES. A boundary copied from it is the
    # right number in the wrong currency: multiply it by any packing length and the served total
    # reappears, which is the signature this looks for and names.
    packing = [n for n in lengths if abs(final_edge * n - total) <= slack]
    ok = in_tokens and not packing
    return ok, evidence(target="domain_token_trace", final_declared_boundary=final_edge,
                        served_tokens_total=total, boundary_unit_is_tokens=in_tokens,
                        packing_factors_that_reconcile=packing,
                        sequence_lengths_declared=lengths,
                        reason=None if ok else "boundaries_counted_in_sequences_not_tokens")''',

        "per_domain_served_tokens_within_the_budget": '''\
    served = timeline().get("per_domain_served_tokens")
    if not isinstance(served, dict):
        return False, evidence(target="domain_token_trace", reason="no_served_token_counters")
    missing = [name for name in DOMAINS if name not in served]
    stray = sorted(set(map(str, served)) - set(DOMAINS))
    negative = sorted(name for name in served if float(served[name]) < 0.0)
    total = sum(float(served[name]) for name in served)
    # The agent own accounting is recorded and NEVER substituted: the counters the harness
    # memmap shim wrote are the only tokens that count against the cap.
    try:
        self_reported = load_submission()["recipe"].get("declared_served_tokens")
    except Exception:
        self_reported = None
    if missing or stray:
        why = "domain_labels_do_not_match_the_shipped_corpora"
    elif negative or total <= 0.0:
        why = "no_tokens_served"
    elif total > TOKEN_CAP:
        why = "token-budget-exceeded"
    else:
        why = None
    return why is None, evidence(target="domain_token_trace",
                                 per_domain=dict((n, served.get(n)) for n in DOMAINS),
                                 served_total=total, token_cap=TOKEN_CAP,
                                 agent_declared_and_ignored=self_reported,
                                 missing_domains=missing, unlabelled_domains=stray, reason=why)''',

        "heldout_shards_carry_the_disclosed_shifted_mixture": '''\
    line = timeline()
    shares = line.get("heldout_domain_shares")
    digests = line.get("shard_sha256")
    if not isinstance(shares, dict) or not isinstance(digests, dict):
        return False, evidence(target="heldout_shards", reason="heldout_manifest_unrecorded")
    off = sorted(name for name in HELDOUT_SHARES
                 if abs(float(shares.get(name, -1.0)) - HELDOUT_SHARES[name]) > SHARE_TOLERANCE)
    private, sibling = digests.get("private"), digests.get("sibling")
    pinned = bool(private) and bool(sibling) and private != sibling
    # The shipped corpora and the graded shards are the SAME three labelled domains at different
    # shares, and that shift is the whole point: an agent that steers on a validation split
    # carrying the training mixture is improving a distribution the bar is not set on. The
    # distance below is recorded rather than gated, because training at the graded shares is a
    # legitimate choice and so is training away from them.
    trained = declared_training_mixture()
    drift = None if trained is None else round(
        max(abs(trained.get(name, 0.0) - HELDOUT_SHARES[name]) for name in HELDOUT_SHARES), 6)
    ok = (not off) and pinned
    return ok, evidence(target="heldout_shards", disclosed=dict(HELDOUT_SHARES),
                        measured_shares=dict((n, shares.get(n)) for n in HELDOUT_SHARES),
                        shares_off_disclosure=off, both_shards_digested=pinned,
                        declared_training_mixture=trained, largest_share_shift=drift,
                        reason=None if ok else "heldout-tampered")''',

        "sibling_shard_agrees_within_four_hundredths_bpb": '''\
    crossing_id, crossing = earliest_clearing(GRADED_BAR)
    if crossing_id is None:
        return False, evidence(target="heldout_shards", reason="no_checkpoint_reached_the_bar")
    row = None
    for record in harness_checkpoints():
        if str(record.get("checkpoint_id")) == str(crossing_id):
            row = record
            break
    if row is None or row.get("bpb_sibling") is None or row.get("bpb_private") is None:
        return False, evidence(target="heldout_shards", crossing_checkpoint=crossing_id,
                               reason="sibling_bpb_unrecorded")
    private, sibling = float(row["bpb_private"]), float(row["bpb_sibling"])
    gap = abs(private - sibling)
    # Both shards come from ONE private document-id hash partition at the SAME shifted mixture,
    # so a wider gap says the bar was cleared on one draw of documents rather than on the
    # distribution the bar is set on, and the crossing stamp is timing the wrong event.
    ok = gap <= SIBLING_TOLERANCE
    return ok, evidence(target="heldout_shards", crossing_checkpoint=crossing_id,
                        bpb_private=private, bpb_sibling=sibling, gap=round(gap, 6),
                        tolerance=SIBLING_TOLERANCE,
                        reason=None if ok else "sibling-divergence")''',

        "model_py_imports_confined_to_the_allowlist": '''\
    scan = timeline().get("model_py_import_scan")
    if not isinstance(scan, dict):
        return False, evidence(target="submitted_source", reason="no_import_scan")
    imported = sorted(set(map(str, scan.get("imported_modules") or [])))
    roots = sorted(set(map(str, scan.get("attribute_roots") or [])))
    outside = [name for name in imported if name.split(".")[0] not in IMPORT_ALLOWLIST]
    reaching = [name for name in roots if name in FORBIDDEN_ATTRIBUTE_ROOTS]
    # The lane grants the solver OPEN EGRESS, so on a time-to-bar axis the fastest route to the
    # bar is a downloaded checkpoint reaching it in a time no training run could. The envelope,
    # not the network, is what keeps the axis about the recipe.
    ok = bool(imported) and not outside and not reaching
    return ok, evidence(target="submitted_source", imported=imported,
                        allowlist=sorted(IMPORT_ALLOWLIST), outside_allowlist=outside,
                        attribute_paths_reaching=reaching,
                        reason=None if ok else "import-envelope-violation")''',

        "step_zero_digests_reproduce_under_the_declared_seed": '''\
    recomputed = timeline().get("step0_digests_recomputed")
    if not isinstance(recomputed, dict) or not recomputed:
        return False, evidence(target="submitted_source", reason="no_recomputed_step0_digests")
    try:
        spec = load_submission()["init_spec"]
        declared = dict(spec["step0_digests"])
        seed = spec["master_seed"]
        entries = int(spec["vocab_entries"])
    except Exception as exc:
        return False, evidence(target="submitted_source", reason="submission_malformed",
                               error=str(exc))
    moved = sorted(name for name in set(recomputed) | set(declared)
                   if str(recomputed.get(name)) != str(declared.get(name)))
    # The shipped and graded shards are written in a private PERMUTED byte-level BPE. An
    # embedding table of any other row count was not initialised against the vocabulary the
    # graded bytes are in, whatever seed it declares beside itself.
    right_vocab = entries == VOCAB_ENTRIES
    ok = (not moved) and right_vocab and seed is not None
    return ok, evidence(target="submitted_source", tensors_digested=len(recomputed),
                        digests_that_moved=moved[:4], declared_seed=seed,
                        declared_vocab_entries=entries, expected_vocab_entries=VOCAB_ENTRIES,
                        reason=None if ok else "init-not-reproducible")''',

        "reward_log_claims_no_better_than_the_harness_saw": '''\
    claims = run_record().get("reward_log_records")
    if not isinstance(claims, list) or not claims:
        return False, evidence(target="run_record", reason="no_reward_log")
    observed = [float(record["bpb_private"]) for record in harness_checkpoints()
                if record.get("bpb_private") is not None]
    if not observed:
        return False, evidence(target="run_record", reason="no_harness_observation")
    best = min(observed)
    fabricated = []
    for row in claims:
        if not isinstance(row, dict) or row.get("bpb_private") is None:
            continue
        # Lower bits per byte is better, so a claim BELOW the best the harness ever saw is a
        # claim about a checkpoint that was never emitted -- and on a time-to-bar axis that is
        # precisely the claim that pays, because it moves the crossing earlier.
        if float(row["bpb_private"]) < best - LOG_TOLERANCE:
            fabricated.append({"claimed": float(row["bpb_private"]), "best_observed": best})
    ok = not fabricated
    return ok, evidence(target="run_record", records=len(claims),
                        best_harness_observation=best,
                        claims_better_than_observed=fabricated[:4],
                        reason=None if ok else "log-fabrication")''',

        "parameter_count_recomputed_from_the_shipped_weights": '''\
    counted = timeline().get("param_count_from_weights")
    if counted is None:
        return False, evidence(target="submitted_source", reason="no_recomputed_param_count")
    counted = int(counted)
    try:
        series = [r for r in load_submission()["checkpoint_series"] if isinstance(r, dict)]
    except Exception as exc:
        return False, evidence(target="submitted_source", reason="submission_malformed",
                               error=str(exc))
    crossing_id, _crossing = earliest_clearing(GRADED_BAR)
    declared, beside = None, None
    for record in series:
        if crossing_id is not None and str(record.get("checkpoint_id")) == str(crossing_id):
            declared, beside = record.get("param_count"), record.get("checkpoint_id")
            break
    if declared is None and series:
        declared, beside = series[-1].get("param_count"), series[-1].get("checkpoint_id")
    # weights.safetensors is RECOUNTED and the number beside it is only checked for honesty. On
    # a time-to-bar axis a larger model is the fastest way to reach the bar in fewer steps, so
    # the cap is what keeps the axis about the recipe rather than about scale.
    within = 0 < counted <= PARAM_CAP
    agrees = declared is not None and int(declared) == counted
    if not within:
        why = "param-budget-exceeded"
    elif not agrees:
        why = "declared_param_count_disagrees_with_the_recount"
    else:
        why = None
    return why is None, evidence(target="submitted_source", recounted=counted,
                                 declared=declared, declared_beside=beside, cap=PARAM_CAP,
                                 reason=why)''',

        "harness_stamping_instruments_unwritten": '''\
    record = run_record()
    digests = record.get("verifier_tree_digests")
    if not isinstance(digests, dict) or "before" not in digests or "after" not in digests:
        return False, evidence(target="run_record", reason="no_verifier_tree_digest_record")
    before, after = digests["before"] or {}, digests["after"] or {}
    moved = sorted(name for name in set(before) | set(after)
                   if before.get(name) != after.get(name))
    touched = sorted(record.get("guarded_paths_touched", []))
    # An empty digest map would pass this vacuously, so the two instruments the graded number
    # actually comes out of have to be present in the roster that was digested. Everything else
    # under the pinned tree is checked for movement; these two are checked for existence first.
    undigested = [name for name in STAMPING_INSTRUMENTS
                  if not any(str(path).endswith(name) for path in before)]
    if moved or touched:
        why = "grading-tree-written"
    elif undigested:
        why = "stamping_instrument_not_digested"
    else:
        why = None
    return why is None, evidence(target="run_record", files_digested=len(before),
                                 digests_that_moved=moved[:4],
                                 guarded_paths_touched=touched[:4],
                                 instruments_undigested=undigested, reason=why)''',

        "seconds_to_the_bar_ramp": '''\
    # require_measured FIRST, before any read of the submission or of the harness timeline.
    # Reading the artifact first lets a missing-file error mask the unmeasured signal: the
    # verifier then reports gradeable=true with r_det=0.0, which collapses "not yet measurable"
    # into "the agent scored zero". They are different claims and seed/build/freeze_batch.py
    # refuses to freeze a bundle that conflates them.
    c = require_measured("floor", "knee", "dichotomizing_threshold")
    crossing_id, crossing = earliest_clearing(GRADED_BAR)
    if crossing is None:
        # A checkpoint that never crosses is worth nothing, and that is a MEASURED zero on the
        # axis rather than an arbitrarily large time: there is no crossing instant to report.
        return False, evidence(target="checkpoint_series", bar=GRADED_BAR, crossed=False,
                               seconds_to_bar=None, s_outcome=0.0,
                               reason="no_checkpoint_reached_the_bar")
    t = measure()
    # FEWER seconds is better, so the ramp DESCENDS: floor is the slow end.
    span = c["floor"] - c["knee"]
    s = 0.0 if span <= 0 else max(0.0, min(1.0, (c["floor"] - t) / span))
    return t <= c["dichotomizing_threshold"], evidence(target="checkpoint_series",
                                                       crossing_checkpoint=crossing_id,
                                                       crossing_stamp_s=crossing,
                                                       seconds_to_bar=t, s_outcome=s)''',
    }
    # REFUSE rather than fall through. A generic obligation lookup used to stand here and quietly
    # supply a body for any item nobody had written one for; the body it supplied read a run
    # record key nothing ever wrote, so an outcome check consulted nothing at all and still
    # reported. An unwritten body is an authoring gap and it stops the generator.
    unwritten = sorted(it["id"] for it in g["items"]
                       if it["mode"] == "compiled" and it["id"] not in bodies)
    if unwritten:
        raise SystemExit("C1-S2: no compiled body is written for %s; refusing to emit a checker "
                         "that would consult nothing" % ", ".join(unwritten))
    orphaned = sorted(set(bodies) - {it["id"] for it in g["items"]})
    if orphaned:
        raise SystemExit("C1-S2: bodies are written for %s, which grounding.yaml does not carry"
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
    return {"evaluation_target": kw.pop("target", "submitted_artifact"), "values": kw}
'''


TAIL = '''

def load_submission():
    """The graded description of a timed run: every emitted checkpoint with its harness stamp
    and held-out bits per byte, the earliest clearing checkpoint for each bar of the ordered
    family, the declared recipe, the clock ledger and the init spec. Not a model and not a loss
    curve -- the graded object here is the account of WHEN a bar was first crossed."""
    with open(SUBMISSION) as handle:
        doc = json.load(handle)
    if not isinstance(doc, dict):
        raise ValueError("the deliverable must be a JSON object")
    return doc


def timeline():
    """The harness stamped record of the graded run, under the verifier-owned held-out tree.

    Read from there and never from the submission or the run record: the stamp the score is read
    off, the per-domain token counters and the held-out bits per byte are the quantities the
    agent must not be able to author. CK-TREE makes writing into this tree a red line rather
    than a way to move the number, which is only a rule if the number is read from inside it.
    """
    path = os.path.join(HERE, "heldout", "harness_timeline.json")
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def run_record():
    """The record the graded run itself produced: its phase ledger, its own reward log and its
    own budget declaration. Everything here is the agent account of the run and is checked for
    honesty against the timeline, never substituted for it."""
    path = os.path.join(HERE, "..", "run_record.json")
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def harness_checkpoints():
    """Every emitted checkpoint the harness stamped, in STAMP order.

    Emission order and stamp order are the same order for a run that emitted forward, and the
    sort is what makes the earliest-clearing scan below independent of how the series was
    serialised.
    """
    rows = timeline().get("checkpoints")
    if not isinstance(rows, list):
        return []
    kept = [row for row in rows if isinstance(row, dict) and row.get("stamp_s") is not None]
    return sorted(kept, key=lambda row: float(row["stamp_s"]))


def earliest_clearing(bar):
    """The FIRST emitted checkpoint whose held-out bits per byte is at or below `bar`.

    Earliest, not best. This is the whole axis: a later checkpoint that models better does not
    move the graded instant, a checkpoint that never crosses contributes nothing at all, and the
    gap between two emissions is time the run cannot be credited for even if it had already
    earned it. Returns (None, None) when nothing crossed, which is a measured zero on the axis
    rather than an unmeasurable one.
    """
    for record in harness_checkpoints():
        value = record.get("bpb_private")
        if value is not None and float(value) <= float(bar):
            return record.get("checkpoint_id"), float(record["stamp_s"])
    return None, None


def declared_training_mixture():
    """Token-weighted aggregate of the declared per-stage mixture over the three domains.

    Recorded beside the disclosed held-out shares so the distance between what a run trained on
    and what it is graded on is visible. It is evidence, not a gate: training at the graded
    shares is a legitimate choice and so is training away from them.
    """
    try:
        stages = list(load_submission()["recipe"]["stages"])
    except Exception:
        return None
    total, cursor = 0, 0
    weighted = dict((name, 0.0) for name in DOMAINS)
    for stage in stages:
        if not isinstance(stage, dict):
            return None
        edge, mixture = stage.get("token_boundary"), stage.get("mixture")
        if edge is None or not isinstance(mixture, dict):
            return None
        width = int(edge) - cursor
        cursor = int(edge)
        if width <= 0:
            return None
        total += width
        for name in DOMAINS:
            weighted[name] += width * float(mixture.get(name, 0.0))
    if total <= 0:
        return None
    return dict((name, round(weighted[name] / total, 6)) for name in DOMAINS)


def measure():
    """Harness-measured seconds from the first read of the shipped corpus to the earliest
    emitted checkpoint at or below the graded bar.

    Both ends are harness stamps. A self-reported duration is a declaration, and this quantity
    is the one the harness measures itself; the clock opens before tokenizer construction and
    memmap open, so setup is inside the span rather than beside it.
    """
    opened = timeline().get("first_corpus_read_stamp_s")
    if opened is None:
        raise ConstantUnmeasured(
            "the harness recorded no first-corpus-read stamp; the clock has no origin")
    crossing_id, crossing = earliest_clearing(GRADED_BAR)
    if crossing is None:
        raise ConstantUnmeasured(
            "no emitted checkpoint reached the bar; there is no crossing instant to time")
    return crossing - float(opened)
'''


if __name__ == "__main__":
    raise SystemExit(main())
