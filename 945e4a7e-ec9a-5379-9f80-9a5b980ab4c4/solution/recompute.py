"""Derivation for C2-S1: every canonical artifact descends from solution/grounding.yaml.

Wired through seed/build/generator_common.py, which carries the five slot-agnostic functions
verified byte-for-byte against the three frozen generators. build_provenance stays here, per
family, for the reason recorded in that module.

The compiled bodies below are written for THIS slot. Each one names a fact of a post-training
run over an opaque ninety-six-shard pool graded on a minimum over four strata -- a twelve-shard
capacity, a probe ledger that has to be paid into, a train-time render that must equal the decode
render byte for byte, a prompt mask, a packed-sequence attention reset, a preference reference
pass in eval mode, a 6000-byte carrier with one entry per completed attempt, a final-attempt
export, a read-only pool digested either side of the run. None of them would mean anything on a
different task, which is the test they were written to pass.
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
    """Assemble the compiled-check module for accuracy grading."""
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
SLOT_ID = 'C2-S1'
TITLE = "# C2-S1 -- what this task actually tests"


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
    }
    out.update(read_screening(HERE))
    return out


def validate(grounding):
    """Refuse to emit a carrier this slot's own shape does not support.

    Every clause below is a fact of a min-over-four-strata post-training task, not a generic
    well-formedness rule: ninety-six shards admitting twelve, four named strata of five hundred,
    a carrier cap, a budget, and an evaluation-target vocabulary the items actually stay inside.
    """
    problems = []
    if not grounding.get("measurement_tier"):
        problems.append("no measurement tier recorded; the wave cannot schedule this slot")
    layout = grounding.get("corpus_layout") or {}
    shards = int(layout.get("pool_shards", 0))
    capacity = int(layout.get("selection_capacity", 0))
    if shards <= 0 or capacity <= 0 or capacity >= shards:
        problems.append("the pool must admit a capacity strictly smaller than itself, "
                        "found %d of %d" % (capacity, shards))
    strata = sorted((layout.get("strata") or {}))
    if len(strata) != 4:
        problems.append("the graded minimum is over four strata, found %d" % len(strata))
    if int(layout.get("stratum_items", 0)) <= 0:
        problems.append("each stratum needs a positive item count to divide by")
    if int(layout.get("carrier_bytes", 0)) <= 0 or int(layout.get("max_attempts", 0)) <= 0:
        problems.append("the carrier cap and the attempt cap are what make this AR10; "
                        "both must be positive")
    if float(layout.get("budget_seconds", 0.0)) <= 0.0:
        problems.append("the timed window needs a positive budget")
    targets = set(i["evaluation_target"] for i in grounding["items"])
    stray = sorted(targets - set(grounding["evaluation_target_vocabulary"]))
    if stray:
        problems.append("items name evaluation targets the vocabulary does not: "
                        + ", ".join(stray))
    outcome = [i for i in grounding["items"] if i["dimension"] == "outcome"]
    if len(outcome) != 1:
        problems.append("exactly one outcome item carries the ramp, found %d" % len(outcome))
    return problems


def facts_block(g):
    """The literal facts the compiled checks read, rendered from grounding.yaml.

    They live in the emitted module rather than in tests/constants.json because none of them is
    a MEASUREMENT: they are the disclosed shape of this task -- ninety-six opaque shards, a
    twelve-shard capacity, four named strata of five hundred, a 6000-byte carrier, a 1200 s
    window with 120 s of grace, an arithmetic floor and a sibling tolerance. Every one of them
    is published in instruction.md. tests/constants.json carries only the ramp, which Phase 0
    does not author and leaves null.
    """
    layout = g["corpus_layout"]
    strata = tuple(sorted(layout["strata"]))
    return (
        '\n# --- the facts this slot is about ------------------------------------------------\n'
        '# The five sections of the graded artifact. A missing section is a malformed\n'
        '# submission, not a weak one.\n'
        'SECTIONS = %r\n'
        '# Ninety-six opaque shards of four thousand rows; twelve of them fit the window.\n'
        'POOL_SHARDS = %r\n'
        'SHARD_ROWS = %r\n'
        'SELECTION_CAPACITY = %r\n'
        'STAGE_ROLES = %r\n'
        '# Recorded so a checker can REFUTE them. Filenames are salted hashes, byte counts are\n'
        '# equalized by within-source row padding and mtimes are normalized, so a selection\n'
        '# that reproduces a ranking over these has found a leak, not a method.\n'
        'SURFACE_KEYS = %r\n'
        '# Four strata of five hundred, entering the graded minimum on equal terms.\n'
        'STRATA = %r\n'
        'STRATUM_ITEMS = %r\n'
        'SIBLING_ITEMS = %r\n'
        'RETENTION_ITEMS = %r\n'
        'RETENTION_FLOOR = %r\n'
        'SIBLING_TOLERANCE = %r\n'
        '# The timed single invocation, opened at the first read under the pool mount.\n'
        'BUDGET_S = %r\n'
        'GRACE_S = %r\n'
        '# The only channel between attempts. Capacity is fixed while the entry count grows.\n'
        'CARRIER_BYTES = %r\n'
        'MAX_ATTEMPTS = %r\n'
        'CARRIER_PATHS = ("rolling_summary.txt",)\n'
        'REWARD_TOLERANCE = 1e-06\n'
        '# A stamp comparison is a comparison of harness clocks, exact to the millisecond.\n'
        'STAMP_TOLERANCE = 0.001\n'
        'TENSOR_SUFFIXES = %r\n'
        '# The four booleans that separate "the reference pass was in eval mode" from\n'
        '# "everything was in eval mode", which is a different and useless run.\n'
        'REFERENCE_FLAGS = ("policy_module_training", "reference_dropout_active",\n'
        '                   "reference_effective_dropout_p", "reference_module_training")\n'
        '# On this slot the vendored Apache-2.0 constraint verifier IS the measurement: every\n'
        '# per-stratum count is the number of prompts it passed on every constraint.\n'
        'GRADING_INSTRUMENTS = %r\n'
        % (tuple(layout["submission_sections"]), int(layout["pool_shards"]),
           int(layout["shard_rows"]), int(layout["selection_capacity"]),
           tuple(layout["stage_roles"]), tuple(layout["surface_keys"]), strata,
           int(layout["stratum_items"]), int(layout["sibling_items"]),
           int(layout["retention_items"]), float(layout["retention_floor"]),
           float(layout["sibling_tolerance"]), float(layout["budget_seconds"]),
           float(layout["grace_seconds"]), int(layout["carrier_bytes"]),
           int(layout["max_attempts"]), tuple(layout["tensor_suffixes"]),
           tuple(layout["grading_instruments"])))


PRELUDE = '''"""Compiled checks for %(slot_id)s. GENERATED by solution/recompute.py. DO NOT HAND-EDIT.

Every check below runs against delivered bytes and references no unmeasured quantity, except
%(outcome_id)s, which raises ConstantUnmeasured while tests/constants.json carries nulls.

The graded quantity is %(metric)s
"""
import json
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
    """The graded account of a post-training run over an opaque pool: which shards were chosen
    and in which stage role, which preference lineage was optimised, what the rolling carrier
    held when the final attempt began, which export was written and inside what window.

    Not a model and not a training log. Every number in here is checked against a quantity the
    harness measured for itself, and none of them is substituted for one.
    """
    with open(SUBMISSION) as handle:
        doc = json.load(handle)
    if not isinstance(doc, dict):
        raise ValueError("the deliverable must be a JSON object")
    return doc


def heldout(name):
    return os.path.join(HERE, "heldout", name)


def _read_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as handle:
            return json.load(handle)
    except Exception:
        return {}


def harness():
    """What the harness measured for itself over the timed invocation.

    The render digests, the mask and packing counters, the preference-stage flags, the attempt
    ledger and its rewards, the workspace rebuild ledger, the export digest, the timed window,
    the four stratum counts and the retention count all live here, under the verifier-owned
    held-out tree. They are read from there and never from the submission: these are exactly the
    quantities a run must not be able to author, and writing into this tree is a red line rather
    than a way to move the number.
    """
    return _read_json(heldout("harness_record.json"))


def pool_manifest():
    """Per-shard digests either side of the run, plus the equalized surface features.

    The surface columns are recorded to be REFUTED. Filenames are salted hashes, byte counts are
    equalized by row-count padding drawn from within the same source, and mtimes are normalized,
    so a twelve-shard selection that reproduces a ranking over any of them has found a leak that
    converts a paid measurement into a free lookup.
    """
    return _read_json(heldout("pool_manifest.json"))


def probe_ledger():
    """Every probe the run spent an attempt on, one row per measurement.

    Shard worth is not inspectable and is not derivable from the manifest, so this ledger is the
    only record of what a selection actually cost. About six probes fit an attempt and the pool
    needs on the order of a hundred and fifty of them.
    """
    return _read_json(heldout("probe_ledger.json"))


def run_record():
    """The run's own account of itself. Checked for honesty against the harness, never
    substituted for it."""
    return _read_json(os.path.join(HERE, "..", "run_record.json"))


def pool_shard_ids():
    """The shard ids the read-only mount actually carries."""
    manifest = pool_manifest()
    before = manifest.get("digests_before")
    if isinstance(before, dict) and before:
        return sorted(before)
    listed = manifest.get("shard_ids")
    return sorted(map(str, listed)) if isinstance(listed, list) else []


def selected_ids():
    """The shard ids the submission committed to, in declaration order."""
    rows = load_submission()["shard_selection"]["selected"]
    return [str(row["shard_id"]) for row in rows]


def stratum_counts(key):
    """The four integer per-stratum counts, or None when the roster is not the four strata.

    Integers out of five hundred, never a rate the run reports: q is an exact rational over a
    count the vendored constraint verifier produced, which is what makes the replay spread
    exactly zero rather than a tolerance band.
    """
    block = harness().get(key)
    if not isinstance(block, dict):
        return None
    counts = block.get("counts")
    if not isinstance(counts, dict) or sorted(map(str, counts)) != sorted(STRATA):
        return None
    out = {}
    for name in STRATA:
        value = counts[name]
        if value is None:
            return None
        value = int(value)
        if value < 0 or value > STRATUM_ITEMS:
            return None
        out[name] = value
    return out


def q_of(counts):
    """The graded scalar: the MINIMUM over the four strata, never their mean."""
    return min(counts[name] / float(STRATUM_ITEMS) for name in STRATA)


def measure():
    """Worst-stratum strict all-constraints-satisfied accuracy of the final attempt's export.

    Strict and all-correct per instance: a prompt with three of its four constraints satisfied
    contributes nothing at all. That is why a minimum over four strata of such a rate is so much
    harder than a mean over items, and why the haystack stratum -- a long real persona preamble
    with the constraint clause placed late -- can hold the whole score down on its own.
    """
    counts = stratum_counts("strata")
    if counts is None:
        raise ConstantUnmeasured(
            "the harness recorded no four-stratum count roster; there is no minimum to take")
    return q_of(counts)
'''


def main():
    g = load_grounding()
    # The family validator runs FIRST and refuses rather than emitting: a generator that writes a
    # carrier it knows is malformed hands the drift check a moving target.
    problems = validate(g)
    if problems:
        raise SystemExit("C2-S1 grounding is not well-formed: " + "; ".join(problems))
    dl = g["deliverable"]
    layout = g["corpus_layout"]
    outcome_id = [i["id"] for i in g["items"] if i["dimension"] == "outcome"][0]
    metric = ('the MINIMUM over the four private %d-item strata (%s) of the fraction of prompts '
              'on which EVERY verifiable constraint passes under the vendored Apache-2.0 '
              'constraint verifier, decoded with the published fixed harness on the export the '
              'FINAL attempt wrote inside a %.0f s single-invocation window on one H100. The '
              'pool the export was trained from is %d opaque shards of %d rows mounted read-only, '
              'of which at most %d fit the window, and the only channel between the %d attempts '
              'is a %d-byte rolling summary carrying one entry per completed attempt with its '
              'reward. The minimum is the point: a stratum that rises cannot pay for a stratum '
              'that falls, and the score is the worst of the four.'
              % (int(layout["stratum_items"]), ", ".join(sorted(layout["strata"])),
                 float(layout["budget_seconds"]), int(layout["pool_shards"]),
                 int(layout["shard_rows"]), int(layout["selection_capacity"]),
                 int(layout["max_attempts"]), int(layout["carrier_bytes"])))
    subst = {"slot_id": SLOT_ID, "path": dl["path"], "metric": metric, "outcome_id": outcome_id}

    # BODIES ARE WRITTEN FOR THIS SLOT. There is no generic fallback below: an item without a
    # body makes the generator REFUSE. A previous revision fell through to an obligation lookup
    # keyed by the item id, which made one bundle's outcome check consult nothing at all and
    # another's invariant a self-report of the thing it was supposed to measure.
    bodies = {

        "shard_selection_names_at_most_twelve_of_the_ninety_six_opaque_shards": '''\
    try:
        doc = load_submission()
    except Exception as exc:
        return False, evidence(reason="submission_malformed", error=str(exc))
    absent = [name for name in SECTIONS if name not in doc]
    if absent:
        return False, evidence(reason="submission_malformed", absent=absent,
                               carried=sorted(doc))
    known = set(pool_shard_ids())
    if not known:
        return False, evidence(reason="pool_manifest_unavailable",
                               detail="no shard roster to draw the selection from")
    try:
        rows = list(doc["shard_selection"]["selected"])
    except Exception as exc:
        return False, evidence(reason="submission_malformed", error=str(exc))
    ids, unknown, roleless = [], [], []
    for row in rows:
        if not isinstance(row, dict) or row.get("shard_id") is None:
            roleless.append(None)
            continue
        shard = str(row["shard_id"])
        ids.append(shard)
        if shard not in known:
            unknown.append(shard)
        if str(row.get("stage_role")) not in STAGE_ROLES:
            roleless.append(shard)
    distinct = len(set(ids)) == len(ids)
    # A thirteenth shard is not a greedier recipe. Twelve is the token capacity the 1200 s
    # window can render, so the selection is a refusal of eighty-four alternatives.
    ok = (bool(ids) and len(ids) <= SELECTION_CAPACITY and distinct
          and not unknown and not roleless)
    return ok, evidence(selected=len(ids), capacity=SELECTION_CAPACITY,
                        pool_shards=len(known), refused=len(known) - len(set(ids)),
                        not_in_the_manifest=unknown[:4], without_a_stage_role=roleless[:4],
                        distinct=distinct,
                        reason=None if ok else "selection_out_of_domain")''',

        "every_selected_shard_was_bought_by_a_recorded_probe": '''\
    rows = probe_ledger().get("probes")
    if not isinstance(rows, list) or not rows:
        return False, evidence(target="probe_ledger", reason="no_probe_recorded")
    probed = set()
    for row in rows:
        if isinstance(row, dict) and row.get("shard_id") is not None:
            probed.add(str(row["shard_id"]))
    try:
        wanted = selected_ids()
    except Exception as exc:
        return False, evidence(target="probe_ledger", reason="submission_malformed",
                               error=str(exc))
    # The pool is OPAQUE and about two thirds of it is real adversarial distractor mass, so a
    # shard nobody probed was chosen by something other than measurement.
    unbought = sorted(set(wanted) - probed)
    ok = bool(wanted) and not unbought
    return ok, evidence(target="probe_ledger", probes_spent=len(rows),
                        shards_probed=len(probed), selected=len(wanted),
                        never_probed=unbought[:6],
                        reason=None if ok else "selected_a_shard_no_probe_measured")''',

        "selection_does_not_reproduce_a_surface_ranking_of_the_pool": '''\
    surfaces = pool_manifest().get("surface_features")
    if not isinstance(surfaces, dict) or not surfaces:
        return False, evidence(target="pool_manifest", reason="surface_features_unrecorded")
    try:
        selection = load_submission()["shard_selection"]
        chosen = set(str(row["shard_id"]) for row in selection["selected"])
        basis = str(selection.get("ranking_basis", ""))
    except Exception as exc:
        return False, evidence(target="pool_manifest", reason="submission_malformed",
                               error=str(exc))
    if not chosen:
        return False, evidence(target="pool_manifest", reason="no_selection_declared")
    matched = []
    for key in SURFACE_KEYS:
        column = surfaces.get(key)
        if not isinstance(column, dict) or len(column) < len(chosen):
            continue
        order = sorted(column, key=lambda shard: (column[shard], shard))
        for direction, ranked in (("ascending", order), ("descending", order[::-1])):
            if set(ranked[:len(chosen)]) == chosen:
                matched.append("%s_%s" % (key, direction))
    named = [key for key in SURFACE_KEYS if key in basis.lower()]
    # These columns are equalized on purpose. Reproducing one is either a free-signal leak that
    # dissolves the discovery cost, or a pool that was built wrong; both block rather than pay.
    ok = not matched and not named
    return ok, evidence(target="pool_manifest", surface_keys=sorted(surfaces),
                        ranking_basis=basis[:80], reproduced_a_ranking=matched,
                        basis_names_a_surface_key=named,
                        reason=None if ok else "selection_recoverable_from_a_surface_feature")''',

        "train_render_matches_the_declared_decode_template_byte_for_byte": '''\
    render = harness().get("render")
    if not isinstance(render, dict):
        return False, evidence(target="run_record", reason="render_digests_unrecorded")
    spans = []
    for span in ("template", "system_turn", "generation_prompt"):
        trained = render.get("train_%s_sha256" % span)
        decoded = render.get("decode_%s_sha256" % span)
        if trained is None or decoded is None:
            spans.append(span + "_unrecorded")
        elif str(trained) != str(decoded):
            spans.append(span)
    # A template differing only in the system-turn and generation-prompt bytes reaches a LOWER
    # training loss and raises no error, so the loss curve cannot see it and only the worst
    # stratum records it. The spans are compared individually for exactly that reason.
    ok = not spans
    return ok, evidence(target="run_record",
                        train_template_sha256=str(render.get("train_template_sha256"))[:16],
                        decode_template_sha256=str(render.get("decode_template_sha256"))[:16],
                        first_span_that_differs=spans[0] if spans else None,
                        spans_that_differ=spans,
                        reason=None if ok else "train_render_is_not_the_decode_template")''',

        "prompt_tokens_carry_no_gradient_in_the_sft_loss": '''\
    ledger = harness().get("sft_mask")
    if not isinstance(ledger, dict):
        return False, evidence(target="run_record", reason="mask_ledger_unrecorded")
    try:
        prompt = int(ledger["prompt_positions"])
        completion = int(ledger["completion_positions"])
        scored = int(ledger["loss_positions"])
    except Exception as exc:
        return False, evidence(target="run_record", reason="mask_ledger_malformed",
                               error=str(exc))
    # These persona prompts carry their machine-verifiable constraint clause inside the PROMPT,
    # so an unmasked loss teaches the model to reproduce constraint text rather than satisfy it,
    # and it lowers the loss faster because prompt tokens are the easy ones.
    ok = completion > 0 and prompt > 0 and scored == completion
    return ok, evidence(target="run_record", prompt_positions=prompt,
                        completion_positions=completion, loss_positions=scored,
                        prompt_positions_in_the_loss=max(0, scored - completion),
                        reason=None if ok else "prompt_tokens_left_in_the_sft_loss")''',

        "packed_sequences_reset_attention_at_every_document_boundary": '''\
    ledger = harness().get("packing")
    if not isinstance(ledger, dict):
        return False, evidence(target="run_record", reason="packing_ledger_unrecorded")
    try:
        enabled = bool(ledger["enabled"])
        packed = int(ledger["packed_sequences"])
        boundaries = int(ledger["document_boundaries"])
        resets = int(ledger["attention_resets"])
        leaked = int(ledger["cross_document_attention_positions"])
    except Exception as exc:
        return False, evidence(target="run_record", reason="packing_ledger_malformed",
                               error=str(exc))
    # Packing is how twelve shards of four thousand rows fit the window at all. Without a reset
    # at each boundary a four-constraint row conditions on the persona preamble packed in front
    # of it, which is the confusion the haystack stratum is built to find.
    ok = enabled and packed > 0 and boundaries > 0 and resets == boundaries and leaked == 0
    return ok, evidence(target="run_record", packing_enabled=enabled,
                        packed_sequences=packed, document_boundaries=boundaries,
                        attention_resets=resets,
                        cross_document_attention_positions=leaked,
                        reason=None if ok else "packed_rows_attend_across_a_boundary")''',

        "preference_reference_forward_pass_ran_with_dropout_off": '''\
    ledger = harness().get("preference")
    if not isinstance(ledger, dict):
        return False, evidence(target="run_record", reason="preference_ledger_unrecorded")
    absent = [flag for flag in REFERENCE_FLAGS if flag not in ledger]
    if absent:
        return False, evidence(target="run_record", reason="preference_ledger_malformed",
                               absent=absent)
    rate = ledger["reference_effective_dropout_p"]
    # The implicit reward is a DIFFERENCE of two log-likelihoods, so dropout left active on the
    # reference side puts noise inside the signal rather than inside the gradient: the loss
    # falls normally, the adapter merges, the export loads, and only the minimum records it.
    # Requiring the policy side to still be training is what stops a wholly-eval run passing.
    ok = (ledger["reference_module_training"] is False
          and ledger["reference_dropout_active"] is False
          and rate is not None and float(rate) == 0.0
          and ledger["policy_module_training"] is True)
    return ok, evidence(target="run_record",
                        reference_module_training=ledger["reference_module_training"],
                        reference_dropout_active=ledger["reference_dropout_active"],
                        reference_effective_dropout_p=rate,
                        policy_module_training=ledger["policy_module_training"],
                        reason=None if ok else "reference_forward_pass_left_in_train_mode")''',

        "preference_pairs_are_verifier_scored_on_both_sides": '''\
    ledger = harness().get("preference")
    if not isinstance(ledger, dict):
        return False, evidence(target="run_record", reason="preference_ledger_unrecorded")
    try:
        total = int(ledger["pairs_total"])
        scored = int(ledger["pairs_scored_by_the_constraint_verifier"])
        wrong = int(ledger["pairs_whose_chosen_side_failed_the_verifier"])
    except Exception as exc:
        return False, evidence(target="run_record", reason="preference_ledger_malformed",
                               error=str(exc))
    declared = None
    try:
        declared = str(load_submission()["preference"]["lineage"])
    except Exception:
        declared = None
    # The shipped dataset cards and the published mixture guidance both select the volume
    # majority, and they corroborate each other, and they are still wrong here. The vendored
    # constraint verifier is the only instrument in the bundle that can rank a pair.
    ok = total > 0 and scored == total and wrong == 0
    return ok, evidence(target="run_record", pairs_total=total,
                        pairs_scored_by_the_constraint_verifier=scored,
                        pairs_whose_chosen_side_failed_the_verifier=wrong,
                        declared_lineage=declared,
                        reason=None if ok else "preference_pairs_not_verifier_ranked")''',

        "carrier_holds_one_entry_per_completed_attempt_under_six_thousand_bytes": '''\
    record = harness()
    ledger, carrier = record.get("attempts"), record.get("carrier")
    if not isinstance(ledger, list) or not ledger or not isinstance(carrier, dict):
        return False, evidence(target="attempt_carrier", reason="attempt_ledger_unrecorded")
    rewards = {}
    for row in ledger:
        if isinstance(row, dict) and row.get("attempt") is not None:
            rewards[int(row["attempt"])] = row.get("reward")
    completed = sorted(rewards)
    try:
        block = load_submission()["carrier"]
        entries = list(block["entries"])
        declared_bytes = int(block["bytes"])
    except Exception as exc:
        return False, evidence(target="attempt_carrier", reason="submission_malformed",
                               error=str(exc))
    restored = carrier.get("restored_bytes")
    seen, disagreeing = [], []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("attempt") is None:
            disagreeing.append(None)
            continue
        index = int(entry["attempt"])
        seen.append(index)
        recorded, claimed = rewards.get(index), entry.get("reward")
        if (recorded is None or claimed is None
                or abs(float(claimed) - float(recorded)) > REWARD_TOLERANCE):
            disagreeing.append(index)
    expected = list(range(1, len(completed) + 1))
    # Capacity is FIXED while the entry count grows: about 6000 characters of room at attempt
    # one, about 120 at attempt fifty. A carrier missing attempt seventeen has not compressed a
    # finding, it has lost one, and attempt fifty has nothing left to rediscover it with.
    covers = sorted(seen) == expected and completed == expected
    within = (restored is not None and int(restored) <= CARRIER_BYTES
              and declared_bytes == int(restored))
    ok = covers and within and not disagreeing
    return ok, evidence(target="attempt_carrier", completed_attempts=len(completed),
                        carrier_entries=len(seen), restored_bytes=restored,
                        declared_bytes=declared_bytes, cap_bytes=CARRIER_BYTES,
                        chars_per_entry=(CARRIER_BYTES // len(completed)) if completed else None,
                        attempts_missing_from_the_carrier=sorted(set(expected) - set(seen))[:6],
                        rewards_that_disagree=disagreeing[:6],
                        reason=None if ok else "carrier_dropped_an_attempt_or_overran_its_cap")''',

        "nothing_but_the_carrier_survived_the_workspace_rebuild": '''\
    rebuild = harness().get("workspace_rebuild")
    if not isinstance(rebuild, dict):
        return False, evidence(target="run_record", reason="rebuild_ledger_unrecorded")
    present = rebuild.get("present_at_final_attempt_start")
    pinned = rebuild.get("pinned_image_manifest")
    if not isinstance(present, list) or not isinstance(pinned, list) or not pinned:
        return False, evidence(target="run_record", reason="rebuild_ledger_malformed")
    allowed = set(map(str, pinned)) | set(CARRIER_PATHS)
    survivors = [str(path) for path in present if str(path) not in allowed]
    for channel in ("environment_variables_carried", "packages_installed_since_attempt_one",
                    "cached_model_artefacts", "pool_paths_written"):
        survivors += ["%s:%s" % (channel, item) for item in (rebuild.get(channel) or [])]
    # The workspace is rebuilt from the pinned image digest at every attempt boundary precisely
    # so that a finding either lives in 6000 characters or does not live. One survivor turns the
    # lossy channel into a lossless one and dissolves the archetype.
    ok = not survivors
    return ok, evidence(target="run_record", present=len(present),
                        pinned_image_entries=len(pinned),
                        first_survivor=survivors[0] if survivors else None,
                        survivors=survivors[:6],
                        reason=None if ok else "a_second_channel_crossed_the_attempt_boundary")''',

        "graded_export_is_the_final_attempts_and_not_the_best_attempts": '''\
    record = harness()
    ledger, export = record.get("attempts"), record.get("export")
    if not isinstance(ledger, list) or not isinstance(export, dict):
        return False, evidence(target="timed_export", reason="attempt_ledger_unrecorded")
    rows = [row for row in ledger if isinstance(row, dict) and row.get("attempt") is not None]
    if not rows:
        return False, evidence(target="timed_export", reason="attempt_ledger_unrecorded")
    last = max(rows, key=lambda row: int(row["attempt"]))
    best = max(rows, key=lambda row: (float(row.get("reward") or 0.0), int(row["attempt"])))
    graded = export.get("graded_sha256")
    try:
        declared_attempt = int(load_submission()["export"]["attempt"])
    except Exception as exc:
        return False, evidence(target="timed_export", reason="submission_malformed",
                               error=str(exc))
    # final_selection is bound to `last`. A run peaking at attempt thirty and regressing at
    # attempt fifty is scored at fifty; naming the peak digest is a claim about an export the
    # grader will not run. Both attempts are reported so the regression is legible.
    ok = (graded is not None and last.get("export_sha256") is not None
          and str(graded) == str(last["export_sha256"])
          and declared_attempt == int(last["attempt"]))
    return ok, evidence(target="timed_export", last_attempt=int(last["attempt"]),
                        best_attempt=int(best["attempt"]),
                        last_attempt_reward=last.get("reward"),
                        best_attempt_reward=best.get("reward"),
                        declared_attempt=declared_attempt,
                        graded_export_sha256=str(graded)[:16], final_selection="last",
                        reason=None if ok else "graded_export_is_not_the_final_attempts")''',

        "export_is_new_bytes_written_inside_the_timed_window": '''\
    record = harness()
    export = record.get("export")
    manifest = record.get("pre_run_workspace_manifest")
    window = record.get("timed_window")
    sanctioned = record.get("sanctioned_base_digests")
    if (not isinstance(export, dict) or not isinstance(manifest, dict)
            or not isinstance(window, dict) or not isinstance(sanctioned, list)):
        return False, evidence(target="timed_export", reason="pre_run_manifest_unrecorded")
    digest = str(export.get("graded_sha256"))
    base = set(map(str, sanctioned))
    preexisting = sorted(path for path, seen in manifest.items() if str(seen) == digest)
    smuggled = sorted(path for path, seen in manifest.items()
                      if str(path).endswith(TENSOR_SUFFIXES) and str(seen) not in base)
    written = export.get("written_at_s")
    opened, closed = window.get("first_pool_read_stamp_s"), window.get("closed_at_s")
    inside = (written is not None and opened is not None and closed is not None
              and float(opened) <= float(written) <= float(closed))
    # Qwen2.5-1.5B-Instruct is a public checkpoint that already scores well here, so the wrong
    # route is not a weak recipe but a downloaded post-trained checkpoint presented as an export.
    # That is a mutually exclusive wrong deliverable, not a judged opinion.
    ok = bool(base) and not preexisting and not smuggled and inside
    return ok, evidence(target="timed_export", export_sha256=digest[:16],
                        already_in_the_pre_run_manifest=preexisting[:4],
                        unsanctioned_tensor_files=smuggled[:4],
                        written_at_s=written, window_opened_s=opened, window_closed_s=closed,
                        written_inside_the_window=inside,
                        reason=None if ok else "export_outside_timed_window")''',

        "timed_window_opens_at_the_first_pool_read": '''\
    window = harness().get("timed_window")
    if not isinstance(window, dict):
        return False, evidence(target="run_record", reason="timed_window_unrecorded")
    opened, closed = window.get("first_pool_read_stamp_s"), window.get("closed_at_s")
    if opened is None or closed is None:
        return False, evidence(target="run_record", reason="timed_window_unrecorded")
    measured = float(closed) - float(opened)
    try:
        block = load_submission()["budget"]
        declared_seconds = float(block["timed_seconds"])
        declared_origin = float(block["first_pool_read_stamp_s"])
    except Exception as exc:
        return False, evidence(target="run_record", reason="submission_malformed",
                               error=str(exc))
    # Shard render, tokenizer construction, on-policy generation, the preference pass and the
    # merge are all inside this span. A run that starts its own clock at the first optimizer
    # step understates the window by exactly the rendering of twelve shards it declined to price.
    agrees = (abs(declared_seconds - measured) <= STAMP_TOLERANCE
              and abs(declared_origin - float(opened)) <= STAMP_TOLERANCE)
    origin_event = str(window.get("origin_event"))
    within = measured <= BUDGET_S + GRACE_S
    ok = agrees and within and origin_event == "first_pool_read"
    return ok, evidence(target="run_record", first_pool_read_stamp_s=opened,
                        origin_event=origin_event, harness_seconds=round(measured, 6),
                        declared_seconds=declared_seconds, ceiling_s=BUDGET_S,
                        grace_s=GRACE_S, declaration_agrees=agrees,
                        reason=None if ok else ("budget_exceeded" if not within
                                                else "timed_window_misdeclared"))''',

        "every_pool_shard_is_unmutated_after_the_timed_invocation": '''\
    manifest = pool_manifest()
    before, after = manifest.get("digests_before"), manifest.get("digests_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return False, evidence(target="pool_manifest", reason="pool_digests_unrecorded")
    moved = sorted(shard for shard in before if str(before[shard]) != str(after.get(shard)))
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    # The read-only pool is the one durable surface an attempt shares with the attempt after it,
    # so a rewritten shard is simultaneously the pool_mutated red line and the memory shortcut
    # this archetype exists to close: a run that can annotate a shard stopped paying for probes.
    complete = len(before) == POOL_SHARDS and len(after) == POOL_SHARDS
    ok = complete and not moved and not added and not removed
    return ok, evidence(target="pool_manifest", shards_before=len(before),
                        shards_after=len(after), expected=POOL_SHARDS,
                        first_shard_whose_digest_moved=moved[0] if moved else None,
                        digests_moved=moved[:4], added=added[:4], removed=removed[:4],
                        reason=None if ok else "pool_mutated")''',

        "arithmetic_retention_holds_on_the_three_hundred_item_split": '''\
    retention = harness().get("retention")
    if not isinstance(retention, dict):
        return False, evidence(target="private_strata", reason="retention_count_unrecorded")
    try:
        correct = int(retention["exact_match_correct"])
        items = int(retention["items"])
    except Exception as exc:
        return False, evidence(target="private_strata", reason="retention_count_malformed",
                               error=str(exc))
    if items != RETENTION_ITEMS or correct < 0 or correct > items:
        return False, evidence(target="private_strata", items=items,
                               expected_items=RETENTION_ITEMS,
                               reason="retention_split_is_not_the_graded_split")
    rate = correct / float(items)
    # A policy that learns to emit the constrained surface form for everything lifts all four
    # strata at once and loses the arithmetic the base model already had. The floor is what
    # makes that trade exactly 0.0 rather than a discounted score.
    ok = rate >= RETENTION_FLOOR
    return ok, evidence(target="private_strata", exact_match_correct=correct, items=items,
                        exact_match=round(rate, 6), floor=RETENTION_FLOOR,
                        reason=None if ok else "retention_gate_violated")''',

        "sibling_worst_stratum_agrees_within_eight_hundredths": '''\
    private = stratum_counts("strata")
    sibling = stratum_counts("sibling_strata")
    if private is None or sibling is None:
        return False, evidence(target="private_strata", reason="stratum_counts_unrecorded")
    q_private, q_sibling = q_of(private), q_of(sibling)
    gap = abs(q_private - q_sibling)
    # Both splits are held-out persona prompts recomposed the same way from the same constraint
    # vocabulary, so a wider gap says fifty attempts of reward feedback fitted the graded draw
    # rather than the recomposition the score is set on. The tolerance is derived, not copied.
    ok = gap <= SIBLING_TOLERANCE
    return ok, evidence(target="private_strata", q_private=round(q_private, 6),
                        q_sibling=round(q_sibling, 6), gap=round(gap, 6),
                        tolerance=SIBLING_TOLERANCE,
                        reason=None if ok else "sibling_divergence")''',

        "graded_scalar_is_the_minimum_of_the_four_strata_never_their_mean": '''\
    counts = stratum_counts("strata")
    if counts is None:
        return False, evidence(target="private_strata", reason="stratum_counts_unrecorded")
    rates = dict((name, counts[name] / float(STRATUM_ITEMS)) for name in STRATA)
    lowest, highest = min(rates.values()), max(rates.values())
    mean = sum(rates.values()) / float(len(STRATA))
    reported = harness().get("strata", {}).get("q")
    if reported is None:
        return False, evidence(target="private_strata", per_stratum=rates,
                               reason="graded_scalar_unreported")
    is_minimum = abs(float(reported) - lowest) <= 1e-09
    flat = (highest - lowest) <= 1e-09
    is_mean = abs(float(reported) - mean) <= 1e-09
    # The four enter on equal terms, so the haystack stratum cannot be averaged away by three
    # easier ones. Where the counts are not all equal the reported scalar must be the minimum
    # AND must not coincide with the mean; where they are equal the two are indistinguishable.
    ok = is_minimum and (flat or not is_mean)
    worst = sorted(rates, key=lambda name: (rates[name], name))[0]
    return ok, evidence(target="private_strata", per_stratum=rates,
                        counts_out_of=STRATUM_ITEMS, q_reported=float(reported),
                        minimum=round(lowest, 6), mean=round(mean, 6),
                        worst_stratum=worst, strata_are_flat=flat,
                        reason=None if ok else "graded_scalar_is_not_the_minimum")''',

        "grading_instruments_unwritten_either_side_of_the_run": '''\
    record = harness()
    before = record.get("instrument_digests_before")
    after = record.get("instrument_digests_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return False, evidence(target="grading_instruments",
                               reason="instrument_digests_unrecorded")
    if not before:
        return False, evidence(target="grading_instruments", reason="instrument_roster_empty")
    moved = sorted(path for path in before if str(before[path]) != str(after.get(path)))
    unrostered = [name for name in GRADING_INSTRUMENTS
                  if not any(str(path).endswith(name) for path in before)]
    touched = sorted(map(str, record.get("guarded_paths_touched") or []))
    # The vendored Apache-2.0 constraint verifier IS the measurement here: every k_s is the
    # count of prompts it passed on every constraint. Editing an instruction checker authors the
    # worst stratum rather than earning it. An empty roster would pass vacuously, so it fails.
    ok = not moved and not unrostered and not touched
    return ok, evidence(target="grading_instruments", digested=len(before),
                        digests_moved=moved[:4], instruments_not_digested=unrostered,
                        guarded_paths_touched=touched[:4],
                        reason=None if ok else "grading_tree_written")''',

        "worst_stratum_ramp": '''\
    # require_measured FIRST, before a single submitted byte is read. Reading the export or the
    # stratum roster first lets a missing-file error raise ahead of the unmeasured signal, and
    # the verifier then reports gradeable=true with r_det=0.0 -- "the agent scored zero" -- when
    # the true claim is "this slot is not yet measurable". Those are different claims.
    c = require_measured("floor", "knee", "dichotomizing_threshold")
    q = measure()
    span = c["knee"] - c["floor"]
    s = 0.0 if span <= 0 else max(0.0, min(1.0, (q - c["floor"]) / span))
    return q >= c["dichotomizing_threshold"], evidence(target="private_strata",
                                                       worst_stratum_accuracy=q,
                                                       s_outcome=s)''',
    }

    # NO FALLBACK. An item without a body is a refusal, not a generic obligation lookup.
    for it in g["items"]:
        if it["mode"] == "compiled" and it["id"] not in bodies:
            raise SystemExit("C2-S1: no body written for compiled item %r; the generator "
                             "refuses rather than emitting a check that consults nothing"
                             % it["id"])

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


if __name__ == "__main__":
    raise SystemExit(main())
