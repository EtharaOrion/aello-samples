"""Derivation for C3-S2: every canonical artifact descends from solution/grounding.yaml.

The graded object on this slot is a CORPUS and the RECIPE MANIFEST that emitted it. The
trainer, tokenizer, optimizer, schedule, seed and architecture are frozen and hash-checked, so
no compiled check here inspects a model, a checkpoint or a parameter tensor; every one of them
reads the submitted manifest, the emitted record set, or the harness's own run record.

Wired through seed/build/generator_common.py, which carries the five slot-agnostic functions
verified byte-for-byte against the three frozen generators. build_provenance stays here, per
family, for the reason recorded in that module.
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
    """Assemble the compiled-check module for scalar grading."""
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
SLOT_ID = 'C3-S2'
TITLE = "# C3-S2 -- what this task actually tests"

# The outcome item, named once so the ordering guard below and the body table cannot drift apart.
OUTCOME_ITEM = "worse_shard_bits_per_byte_ramp"


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
    """C3-S2 refuses to emit unless the curation items it names are actually checkable.

    The default family validator asserted only that SOME items existed and that no ramp constant
    had been authored. That is true of every slot in the batch and it is why a generic item list
    could sit here unnoticed. These assertions are about THIS grounding: the evaluation targets a
    curation slot uses, a unique id per operator-order claim, and an outcome dimension that
    actually carries the graded bits-per-byte quantity.
    """
    problems = []
    if not grounding["items"]:
        problems.append("no rubric items are carried")
    if grounding["constants"]["floor"] is not None:
        problems.append("floor is authored at Phase 0, which the convention forbids")
    vocabulary = set(grounding["evaluation_target_vocabulary"])
    ids = [i["id"] for i in grounding["items"]]
    if len(set(ids)) != len(ids):
        problems.append("item ids are not unique")
    for item in grounding["items"]:
        if item["evaluation_target"] not in vocabulary:
            problems.append("%s names evaluation target %r, outside the declared vocabulary"
                            % (item["id"], item["evaluation_target"]))
    if OUTCOME_ITEM not in ids:
        problems.append("the graded bits-per-byte item %r is absent" % OUTCOME_ITEM)
    if not any(i["dimension"] == "operator_order" for i in grounding["items"]):
        problems.append("no item constrains operator order, which is the design space this "
                        "slot grades")
    return problems


def _body(text):
    """A check body, inserted verbatim beneath `def test_<id>():`, so it keeps its own indent.

    Written as a block rather than as an escaped one-line literal: these bodies are the part of
    the bundle a reader has to be able to audit, and an escaped literal is unreadable exactly
    where reading matters most.
    """
    return text.strip("\n").rstrip()


PRELUDE = '''"""Compiled checks for %(slot_id)s. GENERATED by solution/recompute.py. DO NOT HAND-EDIT.

The graded object is a curated CORPUS and the RECIPE MANIFEST that emitted it; the trainer is
frozen and hash-pinned, so nothing below inspects a model. Every check reads delivered bytes --
the submitted manifest and span table, or the harness's own record of what the frozen trainer
consumed and what the verifier measured -- and references no unmeasured quantity, except
%(outcome)s, which calls require_measured before it reads anything at all.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SUBMISSION = %(path)r
CHECKS = []

# DISCLOSED TASK PARAMETERS, not measurements. Every one is stated in instruction.md and none is
# a ramp constant: the token BAND is announced, the value of N drawn inside it never is.
TOKEN_BAND = (560000000, 600000000)
BUILD_WALL_SECONDS = 1200
TRAIN_WALL_SECONDS = 900
NGRAM_WIDTH = 13
SPAN_SAMPLE_FRACTION = 0.02

# The closed reason set from seed/contract.yaml. A failing check names one of these or names
# nothing at all: CHK-B9 admits exactly one float and at most one reason code on the
# agent-visible surface, so an open-ended per-stage diagnostic is itself a leak.
ZERO_REASONS = (
    "no-submission", "eval-leak", "recipe-corpus-divergence", "span-containment-violation",
    "token-budget-exceeded", "build-wallclock-exceeded", "train-wallclock-exceeded",
    "heldout-touched", "grading-tree-write", "frozen-trainer-modified",
    "submission-size-exceeded", "nondeterministic-submission",
)

# Operator families, matched on the declared op name. Recognition is by substring because the
# agent names its own operators; the ORDER between families is what is graded, never the names.
NORMALISE_OPS = ("normalis", "normaliz", "encoding_repair", "mojibake", "unicode")
DECONTAM_OPS = ("decontam", "fingerprint", "leak_filter", "ngram_filter")
DEDUP_OPS = ("dedup", "duplicate", "minhash", "near_dup", "shingle")
TRUNCATE_OPS = ("truncat", "token_cap", "budget", "prefix", "emit_n")


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
    """The graded artifact: the recipe, and the digest of the corpus that recipe emitted."""
    with open(SUBMISSION) as handle:
        doc = json.load(handle)
    if not isinstance(doc, dict):
        raise ValueError("the deliverable must be a JSON object carrying a recipe manifest")
    return doc


def describe(obj):
    return "keys=" + ",".join(sorted(obj)[:8])


def manifest():
    """The ordered operator list.

    ORDER IS ARRAY POSITION. A mapping keyed by operator name cannot express an order and is
    rejected here rather than sorted into one, because the verifier RE-EXECUTES this list: a
    manifest whose order had to be guessed is a manifest that re-executes to a different corpus.
    """
    man = load_submission().get("recipe_manifest")
    if not isinstance(man, list) or not man:
        raise ValueError("recipe_manifest is not a non-empty ordered operator list")
    return man


def operator_order():
    """The declared op names in executed order, lowercased for family matching."""
    return [str(entry.get("op", "")).lower() if isinstance(entry, dict) else ""
            for entry in manifest()]


def first_index(order, family):
    for position, name in enumerate(order):
        if any(needle in name for needle in family):
            return position
    return None


def all_indices(order, family):
    return [position for position, name in enumerate(order)
            if any(needle in name for needle in family)]


def hex64(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def run_record():
    path = os.path.join(HERE, "..", "run_record.json")
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def measure():
    """b = max(b_P1, b_P2): the WORSE of the two private shards, in bits per byte.

    Both shards are required. Taking the max of whatever happens to be present would silently
    turn a one-shard report into a graded value, which is the aggregation this slot exists to
    forbid -- distribution-matching to one held-out source must not be a partial win.
    """
    per = run_record().get("shard_bits_per_byte") or {}
    values = [per.get("P1"), per.get("P2")]
    if any(v is None for v in values):
        raise ConstantUnmeasured(
            "the frozen evaluator has not reported both private shards; b is the worse of two "
            "and is undefined while either is missing")
    return max(float(v) for v in values)
'''


def slot_bodies():
    """One body per compiled item. Each names a curation fact: an operator's position in the
    declared order, a 13-gram containment count, a span triple, a worse-of-two aggregation.

    None of them would mean anything on another slot, and that is the point. A body that graded
    'the artifact exists and has some list in it' would be true of every slot in the batch and
    would measure nothing about curating a corpus.
    """
    return {

"recipe_manifest_and_corpus_digest_present": _body("""
    if not os.path.exists(SUBMISSION):
        return False, evidence(reason="no-submission", path=SUBMISSION)
    try:
        doc = load_submission()
    except Exception as exc:
        return False, evidence(reason="no-submission", error=str(exc))
    # Both halves or neither. A corpus digest with no recipe leaves CHK-B1 nothing to
    # re-execute; a recipe with no digest leaves the re-execution nothing to compare against.
    required = ("recipe_manifest", "corpus_sha256", "ids", "outputs", "spans")
    missing = [key for key in required if key not in doc]
    digest_wellformed = hex64(doc.get("corpus_sha256"))
    ok = (not missing) and digest_wellformed
    return ok, evidence(present=describe(doc), missing=missing,
                        corpus_digest_wellformed=digest_wellformed,
                        reason=None if ok else "manifest-or-digest-missing")
"""),

"manifest_operators_carry_parameters": _body("""
    try:
        man = manifest()
    except Exception as exc:
        return False, evidence(target="recipe_manifest", error=str(exc),
                               reason="manifest-not-an-ordered-operator-list")
    malformed = [position for position, entry in enumerate(man)
                 if not isinstance(entry, dict)
                 or not isinstance(entry.get("op"), str)
                 or not entry["op"].strip()
                 or not isinstance(entry.get("params"), dict)]
    return (not malformed), evidence(target="recipe_manifest", operators=len(man),
                                     malformed_positions=malformed[:8],
                                     reason=None if not malformed
                                     else "manifest-not-an-ordered-operator-list")
"""),

"manifest_reexecutes_to_emitted_digest": _body("""
    try:
        submitted = load_submission().get("corpus_sha256")
    except Exception as exc:
        return False, evidence(target="recipe_manifest", error=str(exc), reason="no-submission")
    rerun = run_record().get("manifest_reexecution") or {}
    produced = rerun.get("corpus_sha256")
    if produced is None:
        return False, evidence(target="recipe_manifest", reason="recipe-corpus-divergence",
                               detail="the verifier recorded no manifest re-execution")
    fresh = bool(rerun.get("fresh_checkout"))
    ok = fresh and hex64(produced) and produced == submitted
    return ok, evidence(target="recipe_manifest", submitted=submitted, reexecuted=produced,
                        fresh_checkout=fresh,
                        reason=None if ok else "recipe-corpus-divergence")
"""),

"decontamination_after_normalisation": _body("""
    try:
        order = operator_order()
    except Exception as exc:
        return False, evidence(target="recipe_manifest", error=str(exc),
                               reason="manifest-not-an-ordered-operator-list")
    normalisation = first_index(order, NORMALISE_OPS)
    decontamination = first_index(order, DECONTAM_OPS)
    if decontamination is None:
        return False, evidence(target="recipe_manifest", operators=order,
                               reason="decontamination-before-normalisation",
                               detail="no decontamination operator is declared at all")
    # CTL-B-PLACEMENT is exactly the inverted order. It exits 0, reports zero fingerprint hits
    # under its own pre-normalisation count, and leaves normalised held-out 13-grams in the
    # corpus for the containment check to zero. Nothing in the intermediate state separates the
    # two pipelines, so the order is checked here rather than inferred from a clean log.
    ok = normalisation is not None and normalisation < decontamination
    return ok, evidence(target="recipe_manifest", normalisation_index=normalisation,
                        decontamination_index=decontamination, operators=order,
                        reason=None if ok else "decontamination-before-normalisation")
"""),

"dedup_before_token_truncation": _body("""
    try:
        order = operator_order()
    except Exception as exc:
        return False, evidence(target="recipe_manifest", error=str(exc),
                               reason="manifest-not-an-ordered-operator-list")
    dedup = all_indices(order, DEDUP_OPS)
    cut = first_index(order, TRUNCATE_OPS)
    if not dedup:
        return False, evidence(target="recipe_manifest", operators=order,
                               reason="dedup-after-truncation",
                               detail="no deduplication operator is declared at all")
    # Dedup after truncation is the order-collapse trap: the survivor set was already fixed by
    # where the single pass stops, so the operator runs, logs a duplicate count, and changes
    # nothing the frozen trainer ever sees.
    late = [position for position in dedup if cut is not None and position > cut]
    return (not late), evidence(target="recipe_manifest", dedup_indices=dedup,
                                truncation_index=cut, after_truncation=late,
                                reason=None if not late else "dedup-after-truncation")
"""),

"heldout_13gram_containment_zero": _body("""
    scan = run_record().get("fingerprint_scan") or {}
    if not scan:
        return False, evidence(target="emitted_corpus", reason="eval-leak",
                               detail="the verifier recorded no containment scan")
    # The agent's own count does not discharge this. A pre-normalisation scan reports zero on
    # exactly the corpus that leaks, so the scan is required to be the verifier's, at width 13,
    # over the FULL emitted corpus rather than a sample of it.
    hits = scan.get("normalised_13gram_hits")
    width = scan.get("ngram_width")
    by_verifier = scan.get("scanned_by") == "verifier"
    full = bool(scan.get("covered_full_corpus"))
    ok = hits == 0 and width == NGRAM_WIDTH and by_verifier and full
    return ok, evidence(target="emitted_corpus", hits=hits, ngram_width=width,
                        scanned_by=scan.get("scanned_by"), covered_full_corpus=full,
                        reason=None if ok else "eval-leak")
"""),

"emitted_bytes_are_raw_document_spans": _body("""
    try:
        doc = load_submission()
    except Exception as exc:
        return False, evidence(target="emitted_corpus", error=str(exc), reason="no-submission")
    ids, spans = doc.get("ids"), doc.get("spans")
    if not isinstance(ids, list) or not isinstance(spans, list) or len(ids) != len(spans):
        return False, evidence(target="emitted_corpus", reason="span-containment-violation",
                               detail="spans do not align with ids position for position")
    malformed = [position for position, span in enumerate(spans)
                 if not isinstance(span, dict)
                 or not isinstance(span.get("source_id"), str)
                 or not isinstance(span.get("offset"), int)
                 or not isinstance(span.get("length"), int)
                 or span["offset"] < 0 or span["length"] <= 0]
    proof = run_record().get("span_coverage") or {}
    covered = proof.get("records_covered")
    mismatches = proof.get("sample_mismatches")
    fraction = proof.get("sample_fraction")
    ok = (not malformed and covered == len(ids) and mismatches == 0
          and fraction is not None and float(fraction) >= SPAN_SAMPLE_FRACTION)
    return ok, evidence(target="emitted_corpus", records=len(ids),
                        malformed_spans=malformed[:8], records_covered=covered,
                        sample_mismatches=mismatches, sample_fraction=fraction,
                        reason=None if ok else "span-containment-violation")
"""),

"consumed_prefix_within_drawn_budget": _body("""
    accounting = run_record().get("token_accounting") or {}
    counted = accounting.get("consumed_tokens")
    drawn = accounting.get("drawn_budget_tokens")
    if counted is None or drawn is None:
        return False, evidence(target="emitted_corpus", reason="token-budget-exceeded",
                               detail="the verifier recorded no token accounting")
    try:
        claimed = load_submission().get("consumed_tokens")
    except Exception:
        claimed = None
    # The verifier's count under the FROZEN tokenizer is the count. The submitted figure has to
    # agree with it rather than stand in for it, or the budget is graded on a self-report.
    in_band = TOKEN_BAND[0] <= drawn <= TOKEN_BAND[1]
    ok = in_band and counted <= drawn and claimed == counted
    return ok, evidence(target="emitted_corpus", verifier_counted=counted, drawn_budget=drawn,
                        agent_claimed=claimed, band=list(TOKEN_BAND), drawn_in_band=in_band,
                        reason=None if ok else "token-budget-exceeded")
"""),

"emission_order_is_consumption_order": _body("""
    try:
        ids = load_submission().get("ids")
    except Exception as exc:
        return False, evidence(target="emitted_corpus", error=str(exc), reason="no-submission")
    consumed = run_record().get("consumption_order")
    if not isinstance(ids, list) or not isinstance(consumed, list):
        return False, evidence(target="emitted_corpus", reason="emission-order-not-consumed",
                               detail="the harness recorded no single-pass consumption order")
    divergence = None
    for position in range(min(len(ids), len(consumed))):
        if ids[position] != consumed[position]:
            divergence = position
            break
    if divergence is None and len(ids) != len(consumed):
        divergence = min(len(ids), len(consumed))
    ok = divergence is None
    return ok, evidence(target="emitted_corpus", emitted=len(ids), consumed=len(consumed),
                        first_divergence=divergence,
                        reason=None if ok else "emission-order-not-consumed")
"""),

"executed_operator_sequence_declared": _body("""
    declaration = run_record().get("curation_declaration")
    if declaration is None:
        return False, evidence(target="run_record",
                               reason="declared-order-diverges-from-manifest",
                               detail="the agent declared no curation accounting")
    try:
        declaration = (json.loads(declaration) if isinstance(declaration, str)
                       else dict(declaration))
    except Exception as exc:
        return False, evidence(target="run_record", error=str(exc),
                               reason="declared-order-diverges-from-manifest")
    required = ("operator_order", "emission_curriculum", "build_seconds", "train_seconds")
    missing = [key for key in required if key not in declaration]
    try:
        submitted = operator_order()
    except Exception:
        submitted = None
    declared = declaration.get("operator_order")
    if isinstance(declared, list):
        declared = [str(name).lower() for name in declared]
    ok = (not missing) and submitted is not None and declared == submitted
    return ok, evidence(target="run_record", missing=missing, declared=declared,
                        submitted=submitted,
                        reason=None if ok else "declared-order-diverges-from-manifest")
"""),

"curation_wallclock_within_dual_walls": _body("""
    walls = run_record().get("wallclock") or {}
    build, train = walls.get("build_seconds"), walls.get("train_seconds")
    if build is None or train is None:
        return False, evidence(target="run_record", reason="build-wallclock-exceeded",
                               detail="the host measured no build or train region")
    # Two ceilings, two reasons. They are exact-zero cliffs and never slopes, which is what
    # keeps the measured wall-clock noise out of the graded bits-per-byte quantity entirely.
    if float(build) > BUILD_WALL_SECONDS:
        return False, evidence(target="run_record", build_seconds=build,
                               build_wall=BUILD_WALL_SECONDS,
                               reason="build-wallclock-exceeded")
    if float(train) > TRAIN_WALL_SECONDS:
        return False, evidence(target="run_record", train_seconds=train,
                               train_wall=TRAIN_WALL_SECONDS,
                               reason="train-wallclock-exceeded")
    return True, evidence(target="run_record", build_seconds=build, train_seconds=train,
                          build_wall=BUILD_WALL_SECONDS, train_wall=TRAIN_WALL_SECONDS)
"""),

"frozen_trainer_and_heldout_digests_unchanged": _body("""
    record = run_record()
    digests = record.get("frozen_tree_digests") or {}
    before, after = digests.get("before"), digests.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return False, evidence(target="frozen_grading_tree", reason="grading-tree-write",
                               detail="no before/after digest map covers the frozen trees")
    # PATH SETS AS WELL AS DIGESTS. A deleted held-out file changes nothing in the intersection
    # and would read clean if only shared paths were compared, so a disappearance is a change.
    vanished = sorted(set(before) - set(after))
    appeared = sorted(set(after) - set(before))
    changed = sorted(p for p in set(before) & set(after) if before[p] != after[p])
    touched = sorted(record.get("guarded_paths_touched", []))
    trainer = [p for p in changed + vanished + appeared if "trainer" in p]
    ok = not (vanished or appeared or changed or touched)
    reason = None if ok else ("frozen-trainer-modified" if trainer else "grading-tree-write")
    return ok, evidence(target="frozen_grading_tree", changed=changed[:8],
                        vanished=vanished[:8], appeared=appeared[:8], touched=touched[:8],
                        reason=reason)
"""),

"worse_shard_is_the_graded_value": _body("""
    per = run_record().get("shard_bits_per_byte") or {}
    encyclopedic, scientific = per.get("P1"), per.get("P2")
    if encyclopedic is None or scientific is None:
        return False, evidence(target="run_record", encyclopedic=encyclopedic,
                               scientific=scientific,
                               reason="shard-aggregation-not-worse-of-two",
                               detail="b is the worse of two shards and one was not reported")
    encyclopedic, scientific = float(encyclopedic), float(scientific)
    finite = all(v == v and v not in (float("inf"), float("-inf"))
                 for v in (encyclopedic, scientific))
    worse = max(encyclopedic, scientific)
    graded = run_record().get("graded_bits_per_byte")
    ok = finite and graded is not None and float(graded) == worse
    return ok, evidence(target="run_record", encyclopedic=encyclopedic, scientific=scientific,
                        worse_of_two=worse, graded=graded,
                        reason=None if ok else "shard-aggregation-not-worse-of-two")
"""),

OUTCOME_ITEM: _body("""
    # require_measured FIRST, before the corpus, the manifest or the run record is touched.
    # Reading a submitted byte first lets a missing-artifact error be caught AHEAD of the
    # unmeasured signal; the verifier then reports gradeable true with r_det 0.0, collapsing
    # "the slot is not yet measurable" into "the agent scored zero". Those are two claims.
    c = require_measured("floor", "knee", "dichotomizing_threshold")
    b = measure()
    # Bits per byte DESCENDS, so the ramp is read in the descending direction: the floor is the
    # LARGER bits-per-byte value and the knee the smaller one. Subtracting in the ascending
    # order here would score a good corpus at zero and an uncurated one at full reward.
    span = c["floor"] - c["knee"]
    s = 0.0 if span <= 0 else max(0.0, min(1.0, (c["floor"] - b) / span))
    return s >= c["dichotomizing_threshold"], evidence(target="emitted_corpus",
                                                       worse_shard_bpb=b, s_outcome=s)
"""),

    }


def main():
    g = load_grounding()
    # The family validator runs FIRST and refuses rather than emitting: a generator that writes a
    # carrier it knows is malformed hands the drift check a moving target.
    problems = validate(g)
    if problems:
        raise SystemExit("C3-S2 grounding is not well-formed: " + "; ".join(problems))

    bodies = slot_bodies()

    # NO FALLBACK. The retired generator filled any body it lacked with a generic obligation
    # lookup against run_record()["obligations"][<ID>], which meant a compiled item could ship
    # consulting a key nothing ever writes -- an outcome check that read nothing at all. A
    # missing body is a refusal here, so the gap is visible at generation instead of at grading.
    for item in g["items"]:
        if item["mode"] == "compiled" and item["id"] not in bodies:
            raise SystemExit("C3-S2: no body is written for compiled item %r; refusing rather "
                             "than falling through to a generic obligation lookup" % item["id"])
    orphan = sorted(set(bodies) - {i["id"] for i in g["items"]})
    if orphan:
        raise SystemExit("C3-S2: bodies are written for items the grounding does not carry: "
                         + ", ".join(orphan))

    # THE OUTCOME BODY MUST ASK FOR ITS CONSTANTS BEFORE IT READS A SUBMITTED BYTE.
    # seed/build/freeze_batch.py refuses to freeze a bundle that gets this wrong, so the
    # generator refuses one line earlier, where the fix is a body edit rather than a re-freeze.
    statements = [line.strip() for line in bodies[OUTCOME_ITEM].splitlines()
                  if line.strip() and not line.strip().startswith("#")]
    if not statements or not statements[0].startswith("c = require_measured("):
        raise SystemExit("C3-S2: %s must call require_measured as its first statement; it "
                         "begins %r" % (OUTCOME_ITEM, statements[0] if statements else ""))

    dl = g["deliverable"]
    subst = {"slot_id": SLOT_ID, "path": dl["path"], "outcome": OUTCOME_ITEM}

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
        # Only the PRELUDE is substituted. The tail is already resolved and legitimately carries
        # percent signs of its own inside the helpers it emits, so formatting it a second time
        # here raised TypeError on the first slot that carried one.
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
