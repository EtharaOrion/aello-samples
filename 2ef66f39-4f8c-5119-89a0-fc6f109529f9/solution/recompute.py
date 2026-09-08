"""Derivation for C1-S1: every canonical artifact descends from solution/grounding.yaml.

The five slot-agnostic functions are vendored here rather than imported from seed/build, for
the reason recorded in generator_common.py: a staged tree can resolve seed/stage/<slug>/../../
build and a DELIVERED tree at dataset/<uuid>/ cannot, so reaching outside would make the
delivered bundle non-self-contained -- it would import in the stage and fail everywhere it
actually ships.
"""
import json
import os
import sys

import yaml

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)

ITEM_KEYS = ('id', 'dimension', 'weight', 'evaluation_target', 'criterion', 'judgment',
             'knockout', 'mode')
ITEM_SCHEMA = frozenset(ITEM_KEYS + ('outcome_class',))
OUTCOME_CLASSES = frozenset(("EFFECT", "INVARIANT", "ABSENCE", "VALUE", "ORDERING", "DIVERGENCE"))
RAMP = ("floor", "knee", "dichotomizing_threshold", "reward_gate_pass_threshold")
REQUIRE_FIRST = 'c = require_measured("floor", "knee", "dichotomizing_threshold")'
SLOT_ID = 'C1-S1'
TITLE = "# C1-S1 -- what this task actually tests"

# The four assertions this slot cannot be graded without. Bits-per-byte on its own is a
# one-bar task that half the batch could carry; what makes THIS task is that the bar is a
# conjunction taken on one checkpoint, that the second half of it is a difference between two
# shards at different distances past a cutoff, and that the training stream is bounded by two
# facts about publication dates rather than about tokens.
REQUIRED_ASSERTIONS = ("crossing_checkpoint_clears_both_bars_in_the_same_row",
                       "forward_slope_is_h3_minus_h1_not_a_within_shard_trend",
                       "no_served_token_dated_at_or_after_the_cutoff",
                       "recency_quota_share_read_from_served_tokens_not_declared_weights")

# Every run-record key some body below reads. validate() refuses unless corpus_layout documents
# each one: a check reading a key the layout never names is a check nobody can implement the
# harness for, and that is how an obligation ends up consulting nothing.
RUN_RECORD_KEYS_READ = ("corpus_max_pub_epoch_day", "docmeta_sha256_after",
                        "docmeta_sha256_before", "docmeta_sha256_pinned",
                        "first_corpus_read_stamp_s", "first_optimizer_step_stamp_s",
                        "global_batch_tokens", "guarded_paths_touched",
                        "harness_checkpoint_stamps", "optimizer_steps",
                        "phase_median_pub_epoch_day", "shim_ledger")
SHIM_LEDGER_KEYS_READ = ("max_served_pub_epoch_day", "post_cutoff_served_tokens",
                         "recency_window_served_tokens", "served_tokens_by_era",
                         "served_tokens_total")


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


def load_grounding():
    with open(os.path.join(HERE, "grounding.yaml")) as fh:
        return yaml.safe_load(fh)


def build_tests(g, prelude, bodies):
    """Assemble the compiled-check module. kind carries the item's own dimension."""
    chunks = [prelude]
    for raw in sorted(g["items"], key=lambda i: i["id"]):
        if raw["mode"] != "compiled":
            continue
        chunks.append('\n\n@check(id="%s", weight=%d, knockout=%s, kind="%s")\ndef test_%s():\n%s\n'
                      % (raw["id"], raw["weight"], raw["knockout"], raw["dimension"], raw["id"],
                         bodies[raw["id"]].strip("\n")))
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
        "corpus_status": g["corpus_layout"]["status"],
    }
    out.update(read_screening(HERE))
    return out


def validate(grounding):
    """Refuse on a malformed derivation source rather than emitting a checker from one.

    The family validator that stood here asserted one thing -- that some item evaluated the
    run record -- which was true of an item set whose eighteen obligation rows named a contract
    identifier and graded nothing. These are the invariants the emitted checker depends on.
    """
    problems = []
    items = grounding.get("items") or []
    if not items:
        problems.append("no items to compile")
    seen = set()
    for item in items:
        iid = item.get("id", "<unnamed>")
        if iid in seen:
            problems.append("duplicate item id %r" % iid)
        seen.add(iid)
        if set(item) != ITEM_SCHEMA:
            problems.append("item %r does not carry the closed schema: %s"
                            % (iid, sorted(set(item) ^ ITEM_SCHEMA)))
            continue
        if item["outcome_class"] not in OUTCOME_CLASSES:
            problems.append("item %r has outcome_class %r" % (iid, item["outcome_class"]))
        if item["evaluation_target"] not in grounding["evaluation_target_vocabulary"]:
            problems.append("item %r evaluates %r, which the vocabulary does not admit"
                            % (iid, item["evaluation_target"]))
        if not isinstance(item["weight"], int) or item["weight"] <= 0:
            problems.append("item %r carries weight %r" % (iid, item["weight"]))
        if item["mode"] != "compiled":
            problems.append("compilation_floor is 1.0 and item %r is %r" % (iid, item["mode"]))

    for name in REQUIRED_ASSERTIONS:
        if name not in seen:
            problems.append("no item asserts %s" % name)

    outcome = [i["id"] for i in items if i.get("dimension") == "outcome"]
    if len(outcome) != 1:
        problems.append("expected exactly one outcome item, found %r" % outcome)

    # A timed axis is graded on harness measurement, never on a self-reported duration.
    if "run_record" not in {i["evaluation_target"] for i in items}:
        problems.append("a timed slot must grade against the harness run record")

    for name in RAMP:
        if grounding["constants"].get(name) is not None:
            problems.append("constant %r is authored at Phase 0; it is measured, not authored"
                            % name)

    layout = grounding.get("corpus_layout") or {}
    if not layout:
        problems.append("no corpus_layout: the corpus has not landed and the paths the checks "
                        "read must still be written down")
    documented = set(layout.get("run_record_keys") or {})
    missing = sorted(set(RUN_RECORD_KEYS_READ) - documented)
    if missing:
        problems.append("bodies read run-record keys corpus_layout does not document: %s"
                        % ", ".join(missing))
    documented = set(layout.get("shim_ledger_keys") or {})
    missing = sorted(set(SHIM_LEDGER_KEYS_READ) - documented)
    if missing:
        problems.append("bodies read shim-ledger keys corpus_layout does not document: %s"
                        % ", ".join(missing))

    # The two bars are a CONJUNCTION on one checkpoint, so both must be real numbers and the
    # slope bar must be the smaller quantity; a slope bar at or above the bpb bar would make the
    # conjunction vacuous and turn this slot into the one-bar task it is not.
    bpb, slope = layout.get("bpb_bar_h3"), layout.get("slope_bar")
    if not isinstance(bpb, float) or not isinstance(slope, float):
        problems.append("the two bars must both be floats, not %r and %r" % (bpb, slope))
    elif not 0.0 < slope < bpb:
        problems.append("slope_bar %r does not sit strictly inside (0, bpb_bar_h3=%r)"
                        % (slope, bpb))

    quota = layout.get("recency_quota_share")
    if not isinstance(quota, float) or not 0.0 < quota < 1.0:
        problems.append("recency_quota_share is %r, which is not a share in (0,1)" % quota)
    if not isinstance(layout.get("recency_window_days"), int):
        problems.append("recency_window_days is %r" % layout.get("recency_window_days"))

    # The horizons must be ordered, contiguous and start AT the cutoff: the slope is a
    # difference between the nearest and the furthest of them, so an overlap or a gap would
    # make bpb(H3) - bpb(H1) a difference over an undefined span.
    offsets = layout.get("horizon_offsets_days") or {}
    windows = [offsets.get(name) for name in ("h1", "h2", "h3")]
    if any(not (isinstance(w, list) and len(w) == 2) for w in windows):
        problems.append("horizon_offsets_days must carry h1, h2 and h3 as [start, end] pairs")
    else:
        flat = [windows[0][0], windows[0][1], windows[1][0], windows[1][1],
                windows[2][0], windows[2][1]]
        if windows[0][0] != 0:
            problems.append("h1 must open at the cutoff, not at day %r" % windows[0][0])
        if any(w[0] >= w[1] for w in windows) or flat != sorted(flat):
            problems.append("the three horizons are not ordered half-open windows: %r"
                            % (flat,))
        # Half-open and contiguous: each window ends exactly where the next one opens, so no
        # published day falls between two horizons and none falls in both.
        if windows[0][1] != windows[1][0] or windows[1][1] != windows[2][0]:
            problems.append("the horizons are not contiguous: %r" % (flat,))

    sections = tuple(layout.get("submission_sections") or ())
    if sections != ("era_calendar", "schedule", "token_ledger", "checkpoint_ledger", "tokenizer"):
        problems.append("submission_sections is %r; the bodies read the five named sections"
                        % (sections,))
    return problems


def main():
    g = load_grounding()
    # The validator runs FIRST and refuses rather than emitting: a generator that writes a
    # carrier it knows is malformed hands the drift check a moving target.
    problems = validate(g)
    if problems:
        raise SystemExit("C1-S1 grounding is not well-formed: " + "; ".join(problems))

    layout = g["corpus_layout"]
    outcome_id = [i["id"] for i in g["items"] if i["dimension"] == "outcome"][0]
    metric = ("harness-measured wall-clock seconds from the first read of the shipped corpus to "
              "the earliest emitted checkpoint that JOINTLY clears bits-per-byte at or below "
              "1.12 on the far-horizon shard h3.txt and a forward degradation slope bpb(H3) "
              "minus bpb(H1) at or below 0.06. Both terms are computed host-side from the "
              "submission's own tokenizer over the frozen raw held-out bytes, so the metric is "
              "invariant to whatever vocabulary the run builds")
    subst = {
        "slot_id": SLOT_ID,
        "path": g["deliverable"]["path"],
        "metric": metric,
        "outcome_id": outcome_id,
        "bpb_bar": layout["bpb_bar_h3"],
        "slope_bar": layout["slope_bar"],
        "common_target": layout["common_target_bpb"],
        "recency_days": layout["recency_window_days"],
        "quota": layout["recency_quota_share"],
        "token_cap": layout["token_cap"],
        "reconcile_tol": layout["reconciliation_tolerance_fraction"],
        "era_tol": layout["era_share_tolerance"],
        "sections": tuple(layout["submission_sections"]),
        "horizon_report": relative_to_tests(layout["horizon_report"]),
        "rows_key": layout["horizon_report_rows_key"],
        "run_record": layout["run_record"],
        "docmeta": layout["docmeta"],
    }

    # EVERY compiled item carries a written body. The generic obligation lookup that stood here
    # read run_record()["obligations"][<CK-ID>], a key nothing in the harness writes, so an item
    # that fell through to it consulted nothing at all and still reported a clean False. Refuse
    # instead of falling through.
    bodies = dict(BODIES)
    for it in g["items"]:
        if it["mode"] == "compiled" and it["id"] not in bodies:
            raise SystemExit("C1-S1: no body is written for compiled item %r; the generator "
                             "refuses rather than emitting a check that consults nothing"
                             % it["id"])
    stray = sorted(set(bodies) - {i["id"] for i in g["items"]})
    if stray:
        raise SystemExit("C1-S1: bodies are written for items the grounding does not carry: %s"
                         % ", ".join(stray))

    # The outcome check must SAY the slot is unmeasurable before it reads anything at all. If it
    # reads first, a missing horizon report reports as an ordinary failure, the verifier then
    # says gradeable=true with r_det=0.0, and "not yet measurable" has been collapsed into "the
    # agent scored zero" -- the one distinction the null-constant convention exists to keep.
    statements = [ln.strip() for ln in bodies[outcome_id].splitlines()
                  if ln.strip() and not ln.strip().startswith("#")]
    if not statements or statements[0] != REQUIRE_FIRST:
        raise SystemExit("C1-S1: %r must call require_measured as its first statement, before it "
                         "reads any submitted or measured byte" % outcome_id)

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
        # The tail arrives FINAL. It is family-shaped, has already resolved every placeholder it
        # owns, and legitimately contains percent formatting of its own, so formatting it a
        # second time here raised TypeError on the first slot that carried one. Only the prelude
        # is substituted.
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


def relative_to_tests(path):
    """corpus_layout writes bundle-relative paths; the emitted module lives in tests/."""
    prefix = "tests/"
    return path[len(prefix):] if path.startswith(prefix) else os.path.join("..", path)


PRELUDE = '''"""Compiled checks for %(slot_id)s. GENERATED by solution/recompute.py. DO NOT HAND-EDIT.

The graded quantity is %(metric)s.

Bits-per-byte never reaches these checks from the submission. It arrives only through
%(horizon_report)s, which the host-side evaluator writes after the graded run, one row per
emitted checkpoint. Wall-clock stamps arrive only through the harness run record. Nothing here
reads a duration the agent timed for itself or a quality number the agent reported, and every
check but %(outcome_id)s runs against delivered bytes with no unmeasured constant.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SUBMISSION = %(path)r
CHECKS = []

# --- the frozen frame of this slot ---------------------------------------------------------
# The bar is a CONJUNCTION taken on one checkpoint: far-horizon bits-per-byte AND the forward
# degradation slope between the nearest and the furthest shard past the cutoff.
BPB_BAR_H3 = %(bpb_bar)r
SLOPE_BAR = %(slope_bar)r
COMMON_TARGET_BPB = %(common_target)r
# The training stream is bounded by two facts about publication dates rather than about tokens.
RECENCY_WINDOW_DAYS = %(recency_days)r
RECENCY_QUOTA_SHARE = %(quota)r
TOKEN_CAP = %(token_cap)r
RECONCILE_TOLERANCE = %(reconcile_tol)r
ERA_SHARE_TOLERANCE = %(era_tol)r
SECTIONS = %(sections)r
HORIZON_REPORT = %(horizon_report)r
ROWS_KEY = %(rows_key)r
RUN_RECORD = %(run_record)r
DOCMETA = %(docmeta)r


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

# --- reading the three carriers this slot is graded from -----------------------------------

def load_submission():
    """The run's own account of its calendar, its schedule, its served tokens, its checkpoints
    and its tokenizer currency. It carries no bits-per-byte and no wall-clock of its own that
    anything here believes: every number it declares is checked against a harness carrier."""
    with open(SUBMISSION) as handle:
        doc = json.load(handle)
    if not isinstance(doc, dict):
        raise ValueError("the deliverable must be a JSON object")
    return doc


def sub():
    try:
        return load_submission()
    except Exception:
        return None


def section(name):
    doc = sub()
    if doc is None:
        return None
    value = doc.get(name)
    return value if isinstance(value, dict) else None


def section_value(name, key):
    node = section(name)
    return None if node is None else node.get(key)


def declared_phases():
    """The ordered phases of schedule.json as the submission mirrors them."""
    node = section("schedule")
    if node is None:
        return []
    phases = node.get("phases")
    if not isinstance(phases, list):
        return []
    return [p for p in phases if isinstance(p, dict)]


def as_number(value):
    """A number or None. True is not 1 here: a boolean where a day count belongs is malformed
    input, and coercing it would let a schedule declare its anneal window as True."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def number(node, key):
    if not isinstance(node, dict):
        return None
    return as_number(node.get(key))


def run_record():
    """What the harness metered about the graded run: the clock origin, the harness-assigned
    checkpoint stamps, the per-phase median consumed publication day, the docmeta digests and
    the shim's token attribution counters. Read from the record the run itself produced, never
    from a scan taken afterwards, which cannot separate a write made inside the graded region
    from one made before it."""
    path = os.path.join(HERE, RUN_RECORD)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as handle:
            doc = json.load(handle)
    except Exception:
        return {}
    return doc if isinstance(doc, dict) else {}


def shim_ledger():
    """The harness-owned counters that attribute every served token to a publication date
    through docmeta.bin. The agent's own accounting is never consulted."""
    led = run_record().get("shim_ledger")
    return led if isinstance(led, dict) else {}


def horizon_rows():
    """One row per emitted checkpoint, each carrying its harness stamp and its bits-per-byte on
    each of the four shards under the submission's own tokenizer.

    Returns None when the report has not landed, so a check reports horizon_report_absent by
    name rather than raising or, worse, passing over an empty list.
    """
    path = os.path.join(HERE, HORIZON_REPORT)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            doc = json.load(handle)
    except Exception:
        return None
    rows = doc.get(ROWS_KEY) if isinstance(doc, dict) else None
    if not isinstance(rows, list):
        return None
    return [r for r in rows if isinstance(r, dict) and number(r, "stamp_s") is not None]


def clears_both(row):
    """The conjunction, evaluated on ONE row.

    Both terms come from the same checkpoint under the same tokenizer. A row that clears the
    bits-per-byte bar with a slope of 0.09 clears nothing: the ramp does not open until one
    checkpoint satisfies both, which is why emitting often is a lever and why a checkpoint that
    clears a single bar is worth what a checkpoint that clears neither is worth.
    """
    far, near = number(row, "bpb_h3"), number(row, "bpb_h1")
    if far is None or near is None:
        return False
    return far <= BPB_BAR_H3 and (far - near) <= SLOPE_BAR


def earliest_joint_row(rows):
    joint = [r for r in rows if clears_both(r)]
    return min(joint, key=lambda r: number(r, "stamp_s")) if joint else None


def earliest_clearing_stamp(rows, field):
    """When the series first reached the common target on one named horizon."""
    reached = [number(r, "stamp_s") for r in rows
               if number(r, field) is not None and number(r, field) <= COMMON_TARGET_BPB]
    return min(reached) if reached else None
'''


# ==========================================================================================
# BODIES. Every compiled item carries one, written for THIS slot. Each reads a cutoff, a
# publication-date channel, two horizons, one checkpoint or the clock that runs between them,
# and not one of them would mean anything on a classifier, a forecaster or an IID trainer.
# ==========================================================================================

BODIES = {

    "submission_carries_calendar_schedule_ledger_and_crossing": '''
    if not os.path.exists(SUBMISSION):
        return False, evidence(reason="submission_missing", path=SUBMISSION)
    doc = sub()
    if doc is None:
        return False, evidence(reason="submission_malformed", path=SUBMISSION)
    absent = [name for name in SECTIONS if not isinstance(doc.get(name), dict)]
    return (not absent), evidence(path=SUBMISSION, carried=sorted(doc), absent=absent,
                                  reason=None if not absent else "submission_malformed")
''',

    "crossing_checkpoint_clears_both_bars_in_the_same_row": '''
    rows = horizon_rows()
    if rows is None:
        return False, evidence(target="horizon_report", reason="horizon_report_absent",
                               path=HORIZON_REPORT)
    named = section_value("checkpoint_ledger", "joint_crossing")
    if not named:
        return False, evidence(target="horizon_report", reason="no_crossing_named")
    row = next((r for r in rows if r.get("checkpoint_id") == named), None)
    if row is None:
        return False, evidence(target="horizon_report", reason="crossing_not_evaluated",
                               crossing=named,
                               evaluated=[r.get("checkpoint_id") for r in rows])
    far, near = number(row, "bpb_h3"), number(row, "bpb_h1")
    if far is None or near is None:
        return False, evidence(target="horizon_report", reason="crossing_row_incomplete",
                               crossing=named, bpb_h3=far, bpb_h1=near)
    slope = far - near
    cleared_bpb, cleared_slope = far <= BPB_BAR_H3, slope <= SLOPE_BAR
    both = cleared_bpb and cleared_slope
    return both, evidence(target="horizon_report", crossing=named, bpb_h3=far, slope=slope,
                          bpb_bar=BPB_BAR_H3, slope_bar=SLOPE_BAR, cleared_bpb=cleared_bpb,
                          cleared_slope=cleared_slope,
                          reason=None if both else "bars-cleared-separately")
''',

    "forward_slope_is_h3_minus_h1_not_a_within_shard_trend": '''
    rows = horizon_rows()
    if rows is None:
        return False, evidence(target="horizon_report", reason="horizon_report_absent",
                               path=HORIZON_REPORT)
    named = section_value("checkpoint_ledger", "joint_crossing")
    row = next((r for r in rows if r.get("checkpoint_id") == named), None)
    if row is None:
        return False, evidence(target="horizon_report", reason="crossing_not_evaluated",
                               crossing=named)
    far, near = number(row, "bpb_h3"), number(row, "bpb_h1")
    declared = as_number(section_value("checkpoint_ledger", "declared_forward_slope"))
    if far is None or near is None or declared is None:
        return False, evidence(target="horizon_report", reason="slope-not-across-horizons",
                               detail="the row carries no h1/h3 pair, or the ledger declares "
                                      "no forward slope",
                               bpb_h1=near, bpb_h3=far, declared_forward_slope=declared)
    recomputed = far - near
    ok = abs(recomputed - declared) <= 1e-06
    return ok, evidence(target="horizon_report", crossing=named, bpb_h1=near, bpb_h3=far,
                        recomputed_slope=recomputed, declared_forward_slope=declared,
                        reason=None if ok else "slope-not-across-horizons")
''',

    "declared_crossing_is_the_earliest_row_that_clears_both": '''
    rows = horizon_rows()
    if rows is None:
        return False, evidence(reason="horizon_report_absent", path=HORIZON_REPORT)
    named = section_value("checkpoint_ledger", "joint_crossing")
    earliest = earliest_joint_row(rows)
    if earliest is None:
        return False, evidence(reason="no-joint-crossing", named=named, checkpoints=len(rows),
                               detail="no row clears both bars, so no checkpoint stopped the "
                                      "clock")
    ok = named == earliest.get("checkpoint_id")
    return ok, evidence(named_crossing=named, earliest_joint=earliest.get("checkpoint_id"),
                        earliest_joint_stamp_s=number(earliest, "stamp_s"),
                        joint_rows=[r.get("checkpoint_id") for r in rows if clears_both(r)],
                        reason=None if ok else "crossing-not-earliest")
''',

    "setup_seconds_charged_from_first_corpus_read": '''
    rec = run_record()
    if not rec:
        return False, evidence(target="run_record", reason="run_record_absent", path=RUN_RECORD)
    read = number(rec, "first_corpus_read_stamp_s")
    step = number(rec, "first_optimizer_step_stamp_s")
    if read is None or step is None:
        return False, evidence(target="run_record", reason="setup-outside-the-clock",
                               detail="the harness recorded no corpus-read or first-step stamp",
                               first_corpus_read_stamp_s=read,
                               first_optimizer_step_stamp_s=step)
    declared = as_number(section_value("checkpoint_ledger", "setup_seconds"))
    if declared is None:
        return False, evidence(target="run_record", reason="setup-outside-the-clock",
                               detail="checkpoint_ledger declares no setup_seconds")
    measured = step - read
    ok = read < step and abs(measured - declared) <= 1e-06
    return ok, evidence(target="run_record", first_corpus_read_stamp_s=read,
                        first_optimizer_step_stamp_s=step, measured_setup_seconds=measured,
                        declared_setup_seconds=declared,
                        reason=None if ok else "setup-outside-the-clock")
''',

    "recency_quota_share_read_from_served_tokens_not_declared_weights": '''
    led = shim_ledger()
    if not led:
        return False, evidence(target="shim_ledger", reason="shim_ledger_absent",
                               path=RUN_RECORD)
    total = number(led, "served_tokens_total")
    recent = number(led, "recency_window_served_tokens")
    if not total or recent is None:
        return False, evidence(target="shim_ledger", reason="quota_counters_unrecorded",
                               served_tokens_total=total,
                               recency_window_served_tokens=recent)
    share = recent / total
    ok = share <= RECENCY_QUOTA_SHARE
    return ok, evidence(target="shim_ledger", recency_share=share,
                        ceiling=RECENCY_QUOTA_SHARE, window_days=RECENCY_WINDOW_DAYS,
                        recency_window_served_tokens=recent, served_tokens_total=total,
                        reason=None if ok else "recency-quota-exceeded")
''',

    "no_served_token_dated_at_or_after_the_cutoff": '''
    led = shim_ledger()
    if not led:
        return False, evidence(target="shim_ledger", reason="shim_ledger_absent",
                               path=RUN_RECORD)
    post = number(led, "post_cutoff_served_tokens")
    newest = number(led, "max_served_pub_epoch_day")
    cutoff = as_number(section_value("era_calendar", "cutoff_epoch_day"))
    if post is None or newest is None or cutoff is None:
        return False, evidence(target="shim_ledger", reason="cutoff_attribution_unrecorded",
                               post_cutoff_served_tokens=post,
                               max_served_pub_epoch_day=newest, cutoff_epoch_day=cutoff)
    ok = post == 0 and newest < cutoff
    return ok, evidence(target="shim_ledger", post_cutoff_served_tokens=post,
                        max_served_pub_epoch_day=newest, cutoff_epoch_day=cutoff,
                        docmeta=DOCMETA,
                        reason=None if ok else "future-token-consumed")
''',

    "corpus_end_measured_from_docmeta_not_assumed_equal_to_the_cutoff": '''
    calendar = section("era_calendar")
    if calendar is None:
        return False, evidence(reason="submission_malformed",
                               detail="the submission carries no era_calendar")
    rec = run_record()
    if not rec:
        return False, evidence(reason="run_record_absent", path=RUN_RECORD)
    observed = number(rec, "corpus_max_pub_epoch_day")
    declared_end = as_number(calendar.get("corpus_last_pub_epoch_day"))
    cutoff = as_number(calendar.get("cutoff_epoch_day"))
    window = as_number(calendar.get("recency_window_start_epoch_day"))
    band = as_number(calendar.get("blind_band_days"))
    if None in (observed, declared_end, cutoff, window, band):
        return False, evidence(reason="boundary-assumed",
                               detail="the calendar does not carry four measured days",
                               corpus_max_pub_epoch_day=observed,
                               corpus_last_pub_epoch_day=declared_end, cutoff_epoch_day=cutoff,
                               recency_window_start_epoch_day=window, blind_band_days=band)
    ok = (declared_end == observed and window == cutoff - RECENCY_WINDOW_DAYS
          and band == cutoff - declared_end and band >= 0)
    return ok, evidence(observed_corpus_end=observed, declared_corpus_end=declared_end,
                        cutoff_epoch_day=cutoff, declared_window_start=window,
                        expected_window_start=cutoff - RECENCY_WINDOW_DAYS,
                        blind_band_days=band, reason=None if ok else "boundary-assumed")
''',

    "recency_anneal_executes_last_by_median_publication_day": '''
    rec = run_record()
    if not rec:
        return False, evidence(target="run_record", reason="run_record_absent", path=RUN_RECORD)
    executed = rec.get("phase_median_pub_epoch_day")
    phases = declared_phases()
    if not isinstance(executed, list) or not executed or not phases:
        return False, evidence(target="run_record", reason="schedule-order-diverged",
                               detail="the shim recorded no per-phase median publication day, "
                                      "or the schedule declares no phases",
                               declared_phases=len(phases))
    rows = [e for e in executed if isinstance(e, dict)]
    declared_names = [p.get("name") for p in phases]
    executed_names = [e.get("name") for e in rows]
    medians = [as_number(e.get("median_pub_epoch_day")) for e in rows]
    anneal = [i for i, p in enumerate(phases) if p.get("is_recency_anneal") is True]
    monotone = all(a <= b for a, b in zip(medians, medians[1:])
                   if a is not None and b is not None)
    ok = (declared_names == executed_names and monotone and None not in medians
          and anneal == [len(phases) - 1])
    return ok, evidence(target="run_record", declared_order=declared_names,
                        executed_order=executed_names, executed_medians=medians,
                        anneal_positions=anneal, phase_count=len(phases),
                        reason=None if ok else "schedule-order-diverged")
''',

    "every_phase_keeps_a_floor_of_the_oldest_era": '''
    floor = as_number(section_value("schedule", "oldest_era_floor_share"))
    rows = section_value("token_ledger", "per_phase")
    if floor is None or not isinstance(rows, list) or not rows:
        return False, evidence(reason="oldest-era-floor-breached",
                               detail="no per-phase token ledger, or no declared floor share",
                               oldest_era_floor_share=floor)
    if floor <= 0:
        return False, evidence(reason="oldest-era-floor-breached",
                               oldest_era_floor_share=floor,
                               detail="a floor of zero is not a floor; the anneal may then run "
                                      "with no oldest-era tokens at all")
    breached = []
    for row in rows:
        if not isinstance(row, dict):
            breached.append("<malformed>")
            continue
        served = as_number(row.get("served_tokens"))
        oldest = as_number(row.get("oldest_era_served_tokens"))
        if not served or oldest is None or oldest <= 0 or (oldest / served) < floor:
            breached.append(row.get("name"))
    return (not breached), evidence(oldest_era_floor_share=floor, phases=len(rows),
                                    breached_phases=breached,
                                    reason=None if not breached
                                    else "oldest-era-floor-breached")
''',

    "declared_era_weights_reconcile_with_served_token_shares": '''
    led = shim_ledger()
    phases = declared_phases()
    by_era = led.get("served_tokens_by_era")
    total = number(led, "served_tokens_total")
    if not phases or not isinstance(by_era, dict) or not total:
        return False, evidence(target="shim_ledger", reason="mixture-not-realised",
                               detail="no declared phases, or the shim recorded no per-era "
                                      "served tokens",
                               declared_phases=len(phases), served_tokens_total=total)
    planned_total, declared, unnormalised = 0.0, {}, []
    for phase in phases:
        planned = as_number(phase.get("planned_tokens"))
        weights = phase.get("era_weights")
        if planned is None or planned <= 0 or not isinstance(weights, dict) or not weights:
            return False, evidence(target="shim_ledger", reason="mixture-not-realised",
                                   detail="a phase declares no planned tokens or no era weights",
                                   phase=phase.get("name"))
        if abs(sum(as_number(v) or 0.0 for v in weights.values()) - 1.0) > 1e-06:
            unnormalised.append(phase.get("name"))
        planned_total += planned
        for era, weight in weights.items():
            declared[era] = declared.get(era, 0.0) + planned * (as_number(weight) or 0.0)
    if unnormalised:
        return False, evidence(target="shim_ledger", reason="mixture-not-realised",
                               unnormalised_phases=unnormalised)
    gaps = {}
    for era in sorted(set(declared) | set(by_era)):
        want = declared.get(era, 0.0) / planned_total
        got = (as_number(by_era.get(era)) or 0.0) / total
        if abs(want - got) > ERA_SHARE_TOLERANCE:
            gaps[str(era)] = round(got - want, 6)
    return (not gaps), evidence(target="shim_ledger", diverging_eras=gaps,
                                tolerance=ERA_SHARE_TOLERANCE, eras=len(declared),
                                reason=None if not gaps else "mixture-not-realised")
''',

    "token_budget_priced_at_the_newest_readable_era_fertility": '''
    calendar = section("era_calendar")
    tokenizer = section("tokenizer")
    total = as_number(section_value("token_ledger", "served_tokens_total"))
    if calendar is None or tokenizer is None or total is None:
        return False, evidence(reason="fertility-not-measured-forward",
                               detail="no tokenizer section, no era_calendar, or no served "
                                      "token total to price")
    corpus_end = as_number(calendar.get("corpus_last_pub_epoch_day"))
    slice_start = as_number(tokenizer.get("fertility_slice_start_epoch_day"))
    fertility = as_number(tokenizer.get("realized_bytes_per_token"))
    trained = as_number(tokenizer.get("trained_bytes_total"))
    if None in (corpus_end, slice_start, fertility, trained):
        return False, evidence(reason="fertility-not-measured-forward",
                               detail="the tokenizer section does not price the token budget",
                               corpus_last_pub_epoch_day=corpus_end,
                               fertility_slice_start_epoch_day=slice_start,
                               realized_bytes_per_token=fertility,
                               trained_bytes_total=trained)
    expected_start = corpus_end - RECENCY_WINDOW_DAYS
    product = total * fertility
    ok = (slice_start == expected_start and fertility > 0 and trained > 0
          and abs(product - trained) <= 1e-06 * max(1.0, abs(trained)))
    return ok, evidence(fertility_slice_start_epoch_day=slice_start,
                        expected_slice_start=expected_start,
                        realized_bytes_per_token=fertility, served_tokens_total=total,
                        priced_bytes=product, trained_bytes_total=trained,
                        reason=None if ok else "fertility-not-measured-forward")
''',

    "every_trained_token_passed_through_the_attributing_shim": '''
    rec = run_record()
    led = shim_ledger()
    steps = number(rec, "optimizer_steps")
    batch = number(rec, "global_batch_tokens")
    served = number(led, "served_tokens_total")
    if steps is None or batch is None or served is None:
        return False, evidence(target="run_record", reason="unaccounted-tokens",
                               detail="the run record or the shim ledger is missing a counter",
                               optimizer_steps=steps, global_batch_tokens=batch,
                               served_tokens_total=served)
    if served <= 0:
        return False, evidence(target="run_record", reason="unaccounted-tokens",
                               served_tokens_total=served,
                               detail="the shim served nothing, so nothing it served is dated")
    if served > TOKEN_CAP:
        return False, evidence(target="run_record", reason="token-budget-exceeded",
                               served_tokens_total=served, token_cap=TOKEN_CAP)
    divergence = abs(steps * batch - served) / served
    ok = divergence <= RECONCILE_TOLERANCE
    return ok, evidence(target="run_record", stepped_tokens=steps * batch,
                        served_tokens_total=served, divergence_fraction=divergence,
                        tolerance=RECONCILE_TOLERANCE, token_cap=TOKEN_CAP,
                        reason=None if ok else "unaccounted-tokens")
''',

    "docmeta_epoch_day_channel_unaltered_by_the_graded_run": '''
    rec = run_record()
    if not rec:
        return False, evidence(target="run_record", reason="run_record_absent", path=RUN_RECORD)
    before = rec.get("docmeta_sha256_before")
    after = rec.get("docmeta_sha256_after")
    pinned = rec.get("docmeta_sha256_pinned")
    touched = rec.get("guarded_paths_touched")
    if not (before and after and pinned) or not isinstance(touched, list):
        return False, evidence(target="run_record", reason="grading-tree-written",
                               detail="the run record carries no docmeta digest triple, or no "
                                      "guarded-path list",
                               docmeta=DOCMETA, digests_recorded=bool(before and after and pinned),
                               guarded_paths_recorded=isinstance(touched, list))
    stable = before == after
    pinned_ok = after == pinned
    ok = stable and pinned_ok and not touched
    return ok, evidence(target="run_record", docmeta=DOCMETA, digest_stable=stable,
                        matches_pinned=pinned_ok, guarded_paths_touched=sorted(map(str, touched)),
                        reason=None if ok else "grading-tree-written")
''',

    "checkpoint_stamps_are_harness_assigned_not_submission_authored": '''
    rec = run_record()
    stamps = rec.get("harness_checkpoint_stamps")
    ledger = section_value("checkpoint_ledger", "checkpoints")
    named = section_value("checkpoint_ledger", "joint_crossing")
    declared_seconds = as_number(section_value("checkpoint_ledger", "seconds_to_joint_bar"))
    origin = number(rec, "first_corpus_read_stamp_s")
    if not isinstance(stamps, dict) or not isinstance(ledger, list) or origin is None:
        return False, evidence(target="run_record", reason="stamp-divergence",
                               detail="the harness stamped no checkpoints, the ledger is not a "
                                      "list, or the clock has no recorded origin",
                               harness_stamped=isinstance(stamps, dict),
                               first_corpus_read_stamp_s=origin)
    diverging = []
    for entry in ledger:
        if not isinstance(entry, dict):
            diverging.append("<malformed>")
            continue
        cid = entry.get("checkpoint_id")
        declared = as_number(entry.get("stamp_s"))
        assigned = as_number(stamps.get(cid))
        if declared is None or assigned is None or declared != assigned:
            diverging.append(cid)
    crossing_stamp = as_number(stamps.get(named))
    seconds_ok = (crossing_stamp is not None and declared_seconds is not None
                  and abs((crossing_stamp - origin) - declared_seconds) <= 1e-06)
    ok = (not diverging) and seconds_ok
    return ok, evidence(target="run_record", diverging_checkpoints=diverging,
                        crossing=named, crossing_stamp_s=crossing_stamp,
                        first_corpus_read_stamp_s=origin,
                        declared_seconds_to_joint_bar=declared_seconds,
                        recomputed_seconds=None if crossing_stamp is None
                        else crossing_stamp - origin,
                        reason=None if ok else "stamp-divergence")
''',

    "further_horizon_never_clears_before_a_nearer_one": '''
    rows = horizon_rows()
    if rows is None:
        return False, evidence(target="horizon_report", reason="horizon_report_absent",
                               path=HORIZON_REPORT)
    stamps = [earliest_clearing_stamp(rows, field)
              for field in ("bpb_h1", "bpb_h2", "bpb_h3")]
    pairs = [(a, b) for a, b in zip(stamps, stamps[1:]) if a is not None and b is not None]
    ok = all(a <= b for a, b in pairs)
    return ok, evidence(target="horizon_report", earliest_clearing_stamps=stamps,
                        common_target=COMMON_TARGET_BPB, ordered_pairs=len(pairs),
                        reason=None if ok else "horizon-order-inverted")
''',

    "seconds_from_corpus_read_to_the_conjunctive_bar_ramp": '''
    c = require_measured("floor", "knee", "dichotomizing_threshold")
    rows = horizon_rows()
    if rows is None:
        return False, evidence(target="run_record", reason="horizon_report_absent",
                               path=HORIZON_REPORT)
    origin = number(run_record(), "first_corpus_read_stamp_s")
    if origin is None:
        return False, evidence(target="run_record", reason="run_record_absent", path=RUN_RECORD)
    earliest = earliest_joint_row(rows)
    if earliest is None:
        return False, evidence(target="run_record", reason="no-joint-crossing",
                               checkpoints=len(rows),
                               detail="no checkpoint cleared both bars, so the clock never "
                                      "stopped and there is no t to map")
    t = number(earliest, "stamp_s") - origin
    # Fewer seconds is better, so the ramp DESCENDS: floor is the slow end, knee the fast one.
    span = c["floor"] - c["knee"]
    s = 0.0 if span <= 0 else max(0.0, min(1.0, (c["floor"] - t) / span))
    return t <= c["dichotomizing_threshold"], evidence(target="run_record",
                                                       crossing=earliest.get("checkpoint_id"),
                                                       seconds_to_joint_bar=t, s_outcome=s)
''',
}


if __name__ == "__main__":
    raise SystemExit(main())
