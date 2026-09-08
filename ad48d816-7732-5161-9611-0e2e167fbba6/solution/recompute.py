"""Derivation for AELLO-C8-S13: every canonical artifact descends from solution/grounding.yaml.

Wired through seed/build/generator_common.py, which carries the five slot-agnostic functions
verified byte-for-byte against the three frozen generators. build_provenance stays here, per
family, for the reason recorded in that module.

WHAT IS DIFFERENT HERE. This generator carries a compiled BODY for every compiled item and
REFUSES to emit if one is missing. The family default was to fall through to a generic
obligation lookup -- read run_record()["obligations"][ID] and believe whatever it says -- which
on this slot silently turned the CK-BINFAM invariant into a self-report and, on a sibling
bundle, turned the outcome check into a check that consulted nothing at all. A missing body is
now a build failure rather than a check that always agrees with the agent.
"""
import json
import os
import sys

# Set BEFORE the first third-party import: a generator that leaves __pycache__ behind inside a
# bundle changes the tree the freeze hashes, and the freeze re-runs this file after hashing it.
sys.dont_write_bytecode = True

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
    """Assemble the compiled-check module for the corrupted-calibration grading."""
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
SLOT_ID = 'AELLO-C8-S13'
TITLE = "# AELLO-C8-S13 -- what this task actually tests"


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
        "graded_binning": {k: g["constants"].get(k) for k in
                           ("graded_bin_edges_file", "graded_bin_edges_sha256",
                            "graded_bin_count")},
        "graded_layout": {k: g["corpus_layout"][k] for k in
                          ("cells", "rows_per_cell", "corrupted_rows", "clean_graded_rows",
                           "classes")},
    }
    out.update(read_screening(HERE))
    return out


# --------------------------------------------------------------------------------------------
# COMPILED BODIES. One per compiled item, keyed by item id. Every body is real Python that runs
# against delivered bytes; every one that can meet an absent submission or an absent run record
# returns a named reason instead of raising, because the only exception the verifier catches is
# ConstantUnmeasured and anything else would be reported as an unreadable run rather than as a
# failed check. The two bodies that CANNOT return a score open with require_measured.
# --------------------------------------------------------------------------------------------

BODIES = {

"posterior_file_lands": '''\
    if not os.path.exists(SUBMISSION):
        return False, evidence(reason="submission_missing", path=SUBMISSION)
    try:
        arr = load_submission()
    except Exception as exc:
        return False, evidence(reason="posterior_unloadable", error=str(exc))
    return True, evidence(path=SUBMISSION, shape=describe(arr))''',

"stacked_block_layout": '''\
    import numpy as np
    try:
        arr = load_submission()
    except Exception as exc:
        return False, evidence(reason="posterior_unloadable", error=str(exc))
    if arr.ndim != 2:
        return False, evidence(reason="not_two_dimensional", shape=list(arr.shape))
    rows, cols = int(arr.shape[0]), int(arr.shape[1])
    if cols != N_CLASSES:
        return False, evidence(reason="wrong_class_count", columns=cols, want=N_CLASSES)
    if rows == N_CORRUPTED:
        # The interesting failure: a perfectly good answer to the stated objective that removes
        # the only block the clean-accuracy tolerance gate can read.
        return False, evidence(reason="clean_block_missing", rows=rows, want=N_ROWS)
    if rows != N_ROWS:
        return False, evidence(reason="wrong_row_count", rows=rows, want=N_ROWS)
    try:
        spans = cell_spans()
    except Exception as exc:
        return False, evidence(reason="corpus_layout_mismatch", error=str(exc))
    return True, evidence(rows=rows, columns=cols, cells=len(spans),
                          corrupted_block=[0, N_CORRUPTED],
                          clean_block=[N_CORRUPTED, N_ROWS])''',

"rows_are_posteriors": '''\
    import numpy as np
    try:
        arr = posterior()
    except Exception as exc:
        return False, evidence(reason="submission_malformed", error=str(exc))
    negative = int((arr < 0.0).sum())
    if negative:
        return False, evidence(reason="negative_mass", negative_entries=negative,
                               most_negative=float(arr.min()),
                               largest_confidence=float(arr.max(axis=1).max()))
    off = np.abs(arr.sum(axis=1) - 1.0)
    bad = int((off > ROW_SUM_TOLERANCE).sum())
    return (bad == 0), evidence(rows=int(arr.shape[0]), tolerance=ROW_SUM_TOLERANCE,
                                max_row_mass_error=float(off.max()),
                                rows_out_of_tolerance=bad,
                                reason=None if bad == 0 else "row_mass_out_of_tolerance")''',

"uniform_posterior_rejected": '''\
    import numpy as np
    try:
        arr = posterior()
    except Exception as exc:
        return False, evidence(reason="submission_unusable", error=str(exc))
    flat = 1.0 / N_CLASSES
    every_row_uniform = bool(np.allclose(arr, flat, rtol=0.0, atol=1e-09))
    corrupted_conf = arr[:N_CORRUPTED].max(axis=1)
    distinct = int(np.unique(corrupted_conf).size)
    clean_classes = int(np.unique(arr[N_CORRUPTED:].argmax(axis=1)).size)
    ok = (not every_row_uniform) and distinct >= 2 and clean_classes >= 2
    return ok, evidence(uniform_value=flat, every_row_uniform=every_row_uniform,
                        distinct_corrupted_confidences=distinct,
                        clean_classes_predicted=clean_classes,
                        reason=None if ok else "uniform_posterior")''',

"pinned_binning_intact": '''\
    import hashlib
    import numpy as np
    c = constants_json()
    path = os.path.join(HERE, os.path.basename(c["graded_bin_edges_file"]))
    if not os.path.exists(path):
        return False, evidence(target="submitted_source", reason="pinned_binning_missing",
                               path=path)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if digest != c["graded_bin_edges_sha256"]:
        return False, evidence(target="submitted_source", reason="pinned_binning_moved",
                               digest=digest, pinned=c["graded_bin_edges_sha256"])
    edges = np.load(path, allow_pickle=False)
    want = np.linspace(0.0, 1.0, c["graded_bin_count"] + 1)
    if edges.shape != want.shape or not np.allclose(edges, want, rtol=0.0, atol=1e-12):
        return False, evidence(target="submitted_source",
                               reason="pinned_binning_not_equal_width",
                               edges=[float(e) for e in np.asarray(edges).ravel()])
    rewritten = [p for p in sorted(run_record().get("guarded_paths_touched", []))
                 if "graded_bin_edges" in p]
    return (not rewritten), evidence(target="submitted_source", digest=digest,
                                     bins=c["graded_bin_count"], touched=rewritten,
                                     reason=None if not rewritten else "pinned_binning_rewritten")''',

"graded_fold_boundary": '''\
    rec = run_record()
    if not rec:
        # Fails closed. A post-hoc scan cannot separate a write made during the graded run from
        # one made before it, so an absent record is an unanswered question, not a clean answer.
        return False, evidence(target="submitted_source", reason="run_record_absent")
    touched = sorted(rec.get("guarded_paths_touched", []))
    named = [p for p in touched
             if p.startswith("tests/") or "/tests/" in p
             or "corrupted_graded_labels" in p or "clean_graded_labels" in p
             or "/logs/verifier" in p]
    return (not touched), evidence(target="submitted_source", touched=touched,
                                   guarded_named=named,
                                   reason=None if not touched else "wrote_into_guarded_tree")''',

"heldout_label_isolation": '''\
    import hashlib
    import numpy as np
    root = os.path.join(HERE, "..", "agent_src")
    if not os.path.isdir(root):
        return False, evidence(target="submitted_source", reason="agent_source_unavailable",
                               root=os.path.normpath(root))
    try:
        corrupted, clean = corrupted_truth(), clean_truth()
    except Exception as exc:
        return False, evidence(target="submitted_source", reason="private_labels_unreadable",
                               error=str(exc))
    # The 200-entry cell vector is the length that matters: the 19000-entry vector is 95 tilings
    # of it, so holding 200 integers is holding every graded corrupted label.
    wanted_values = ((corrupted, "corrupted_19000"), (clean, "clean_10000"),
                     (corrupted[:PER_CELL], "cell_200"))
    wanted_digests = {}
    for name in ("corrupted_graded_labels.npz", "clean_graded_labels.npz"):
        wanted_digests[hashlib.sha256(open(heldout(name), "rb").read()).hexdigest()] = name
    hits, scanned = [], 0
    for dirpath, _dirs, filenames in os.walk(root):
        for filename in sorted(filenames):
            full = os.path.join(dirpath, filename)
            rel = os.path.relpath(full, root)
            try:
                blob = open(full, "rb").read()
            except Exception:
                continue
            scanned += 1
            digest = hashlib.sha256(blob).hexdigest()
            if digest in wanted_digests:
                hits.append({"path": rel, "match": wanted_digests[digest],
                             "how": "byte_identical"})
                continue
            if not filename.endswith((".npy", ".npz")):
                continue
            try:
                loaded = np.load(full, allow_pickle=False)
            except Exception:
                continue
            arrays = [loaded] if hasattr(loaded, "dtype") else [loaded[k] for k in loaded.files]
            for array in arrays:
                array = np.asarray(array)
                if array.dtype.kind not in "iu":
                    continue
                held = array.ravel().astype("int64")
                for want, label in wanted_values:
                    if held.size == want.size and bool(np.array_equal(held, want)):
                        hits.append({"path": rel, "match": label, "how": "value_identical"})
    return (not hits), evidence(target="submitted_source", files_scanned=scanned, hits=hits,
                                reason=None if not hits else "private_labels_held")''',

"calibration_fit_split_declared": '''\
    rec = run_record()
    declaration = rec.get("agent_budget_declaration")
    if declaration is None:
        return False, evidence(target="run_record", reason="no_agent_declaration")
    try:
        declaration = (json.loads(declaration) if isinstance(declaration, str)
                       else dict(declaration))
    except Exception as exc:
        return False, evidence(target="run_record", reason="declaration_unparseable",
                               error=str(exc))
    passes = declaration.get("graded_forward_passes")
    if isinstance(passes, bool) or not isinstance(passes, int) or passes < 1:
        return False, evidence(target="run_record", reason="graded_forward_passes_undeclared",
                               declared=passes)
    split = declaration.get("calibration_fit_split")
    named = [split] if isinstance(split, str) else [str(s) for s in (split or [])]
    if not named:
        return False, evidence(target="run_record", reason="calibration_fit_split_undeclared")
    on_graded = sorted(set(n for n in named if n in GRADED_SPLITS))
    if on_graded:
        return False, evidence(target="run_record", reason="fit_on_graded_split",
                               named=on_graded, fittable=list(FITTABLE_SPLITS))
    unknown = sorted(set(n for n in named if n not in FITTABLE_SPLITS))
    if unknown:
        return False, evidence(target="run_record", reason="unknown_fit_split", named=unknown,
                               fittable=list(FITTABLE_SPLITS))
    return True, evidence(target="run_record", calibration_fit_split=sorted(set(named)),
                          graded_forward_passes=passes)''',

"posterior_replay_identity": '''\
    first = artifact_digest()
    if first is None:
        return False, evidence(target="submitted_source", reason="submission_missing",
                               path=SUBMISSION)
    replay = replay_digest()
    if replay is None:
        return False, evidence(target="submitted_source", reason="replay_unavailable")
    return first == replay, evidence(target="submitted_source", first=first, replay=replay,
                                     reason=None if first == replay else "replay_diverged")''',

"ck_binfam": '''\
    import numpy as np
    try:
        submission = confidence_correct(corrupted_block(), corrupted_truth())
    except Exception as exc:
        return False, evidence(reason="submission_unusable", error=str(exc))
    record = run_record().get("controls", {}).get("CTL-NOOP", {})
    if not isinstance(record, dict) or record.get("confidences") is None \\
            or record.get("correct") is None:
        return False, evidence(reason="control_record_absent", control="CTL-NOOP",
                               needs="the per-sample (confidence, correctness) record of "
                                     "CTL-NOOP's graded inference pass")
    control = (np.asarray(record["confidences"], dtype="float64"),
               np.asarray(record["correct"], dtype="float64"))
    if control[0].size != N_CORRUPTED or control[1].size != N_CORRUPTED:
        return False, evidence(reason="control_record_wrong_length", got=int(control[0].size),
                               want=N_CORRUPTED)
    span = (min(float(control[0].min()), float(submission[0].min())),
            max(float(control[0].max()), float(submission[0].max())))
    try:
        family = binning_family(control[0], span)
    except Exception as exc:
        return False, evidence(reason="family_unbuildable", error=str(exc))
    spans, signs, per_member = cell_spans(), set(), {}
    for name, edges in family:
        noop = macro_over_cells(control[0], control[1], edges, spans)
        subm = macro_over_cells(submission[0], submission[1], edges, spans)
        per_member[name] = {"ctl_noop": noop, "submission": subm}
        signs.add(0 if noop == subm else (1 if noop > subm else -1))
    ok = len(signs) == 1 and 0 not in signs
    reason = None
    if not ok:
        reason = "ordering_ties_under_some_binning" if 0 in signs else "ordering_flips_with_binning"
    return ok, evidence(members=len(family), sign_set=sorted(signs), per_member=per_member,
                        reason=reason)''',

"ck_binpin": '''\
    import numpy as np
    try:
        block, truth = corrupted_block(), corrupted_truth()
        edges = graded_bin_edges()
    except Exception as exc:
        return False, evidence(reason="submission_unusable", error=str(exc))
    confidence, correct = confidence_correct(block, truth)
    pinned = ece(confidence, correct, edges)
    # The finding is a single measured fact: over 19000 rows, mean confidence equals accuracy to
    # within a millionth in every bin of the PINNED partition. Binomial noise alone puts an
    # honest solve's floor near 1/sqrt(19000), about 0.007, so 1e-06 sits four orders of
    # magnitude below anything a classifier reaches without having read these labels.
    fitted = pinned <= 1e-06
    # Corroboration, recorded and not required: a value lookup stays calibrated under any
    # partition that separates its values, so refining the pinned bins ten ways moves it very
    # little, while its confidence takes far fewer levels than a softmax maximum can.
    refined = ece(confidence, correct, refined_edges(edges, REFINEMENT))
    return (not fitted), evidence(
        pinned_ece=pinned, refined_ece=refined, refinement=REFINEMENT,
        refined_bins=(int(edges.size) - 1) * REFINEMENT, rows=N_CORRUPTED,
        honest_noise_floor=float(N_CORRUPTED) ** -0.5,
        distinct_confidence_levels=int(np.unique(np.round(confidence, 9)).size),
        reason="calibrated_to_float_precision_on_the_pinned_partition" if fitted else None)''',

"per_cell_ece_recorded": '''\
    import math
    try:
        rows = per_cell_ece()
        pooled = pooled_ece()
    except Exception as exc:
        return False, evidence(reason="per_cell_ece_uncomputable", error=str(exc))
    values = [r[2] for r in rows]
    ok = len(rows) == N_CELLS and all(math.isfinite(v) for v in values)
    worst = max(rows, key=lambda r: r[2])
    macro = sum(values) / len(values)
    return ok, evidence(cells=len(rows), macro_ece=macro, pooled_ece=pooled,
                        macro_minus_pooled=macro - pooled,
                        worst_cell={"corruption": worst[0], "severity": worst[1],
                                    "ece": worst[2]},
                        per_cell=dict(("%s@%d" % (r[0], r[1]), r[2]) for r in rows),
                        reason=None if ok else "per_cell_ece_incomplete")''',

"clean_accuracy_tolerance": '''\
    c = require_measured("clean_accuracy_floor")
    try:
        block, truth = clean_block(), clean_truth()
    except Exception as exc:
        return False, evidence(reason="clean_block_unusable", error=str(exc))
    accuracy = float((block.argmax(axis=1) == truth).mean())
    floor = c["clean_accuracy_floor"]
    ok = accuracy >= floor
    return ok, evidence(clean_accuracy=accuracy, clean_accuracy_floor=floor, rows=N_CLEAN,
                        reason=None if ok else "clean_accuracy_below_tolerance")''',

"corruption_macro_ece_ramp": '''\
    c = require_measured("floor", "knee", "dichotomizing_threshold")
    try:
        rows = per_cell_ece()
    except Exception as exc:
        return False, evidence(reason="metric_uncomputable", error=str(exc))
    macro = sum(r[2] for r in rows) / len(rows)
    value = 1.0 - macro
    span = c["knee"] - c["floor"]
    s = 0.0 if span <= 0 else max(0.0, min(1.0, (value - c["floor"]) / span))
    worst = max(rows, key=lambda r: r[2])
    return value >= c["dichotomizing_threshold"], evidence(
        metric=value, macro_ece=macro, cells=len(rows), s_outcome=s,
        worst_cell={"corruption": worst[0], "severity": worst[1], "ece": worst[2]})''',

}


PRELUDE = '''\
"""Compiled checks for %(slot_id)s. GENERATED by solution/recompute.py. DO NOT HAND-EDIT.

Every check below runs against delivered bytes. Two of them -- clean_accuracy_tolerance and
corruption_macro_ece_ramp -- open with require_measured and raise ConstantUnmeasured while
tests/constants.json carries nulls, because a bar that has not been measured is not a bar the
agent failed to clear.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SUBMISSION = %(path)r
CHECKS = []

%(layout)s

class ConstantUnmeasured(Exception):
    """Raised when a check needs a constant that Phase 0 deliberately left null."""


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
    """The graded posterior. A floating-point array is required, not merely a numeric one: this
    slot grades a distribution, so an integer label vector is a different object rather than a
    degraded one."""
    import numpy as np
    arr = np.load(SUBMISSION, allow_pickle=False)
    if arr.dtype == object:
        raise ValueError("object arrays are not a valid deliverable")
    if arr.dtype.kind != "f":
        raise ValueError("a posterior must be floating point, got dtype %s" % arr.dtype)
    return arr


def describe(obj):
    return "%s%s" % (obj.dtype, obj.shape)


def posterior():
    """The submission as float64, shape-checked against the pinned stacking order."""
    import numpy as np
    arr = np.asarray(load_submission(), dtype="float64")
    if arr.ndim != 2:
        raise ValueError("a stacked posterior must be two dimensional, got %dD" % arr.ndim)
    if int(arr.shape[1]) != N_CLASSES:
        raise ValueError("expected %d fine-label columns, got %d" % (N_CLASSES, arr.shape[1]))
    if int(arr.shape[0]) != N_ROWS:
        raise ValueError("expected %d rows (%d corrupted then %d clean), got %d"
                         % (N_ROWS, N_CORRUPTED, N_CLEAN, arr.shape[0]))
    if not np.isfinite(arr).all():
        raise ValueError("the posterior carries non-finite entries")
    return arr


def corrupted_block():
    return posterior()[:N_CORRUPTED]


def clean_block():
    return posterior()[N_CORRUPTED:]


def cell_spans():
    """The 95 (corruption, severity) cells as (name, severity, lo, hi) row spans.

    Derived arithmetically from the authored layout and, when the shipped corpus is readable,
    cross-checked cell for cell against the corruption_id and severity arrays the corpus itself
    carries. A corpus that disagrees raises rather than being silently overridden: the row order
    is the only thing binding a submitted row to a corruption.
    """
    import numpy as np
    spans = []
    for corruption in range(N_CORRUPTIONS):
        for severity in range(1, N_SEVERITIES + 1):
            lo = corruption * N_SEVERITIES * PER_CELL + (severity - 1) * PER_CELL
            spans.append((CORRUPTION_NAMES[corruption], severity, lo, lo + PER_CELL))
    path = os.path.join(HERE, "..", CORRUPTED_GRADED)
    if os.path.exists(path):
        shipped = np.load(path, allow_pickle=False)
        names = [str(n) for n in shipped["corruption_names"]]
        if names != list(CORRUPTION_NAMES):
            raise ValueError("the shipped corruption names are not the authored ones")
        cid, sev = shipped["corruption_id"], shipped["severity"]
        if cid.size != N_CORRUPTED or sev.size != N_CORRUPTED:
            raise ValueError("the shipped grid is %d rows, not %d" % (cid.size, N_CORRUPTED))
        for name, severity, lo, hi in spans:
            want = CORRUPTION_NAMES.index(name)
            if not (int(cid[lo:hi].min()) == int(cid[lo:hi].max()) == want
                    and int(sev[lo:hi].min()) == int(sev[lo:hi].max()) == severity):
                raise ValueError("rows %d:%d are not %s at severity %d in the shipped corpus"
                                 % (lo, hi, name, severity))
    return spans


def heldout(name):
    return os.path.join(HERE, "heldout", name)


def corrupted_truth():
    """The private corrupted labels, checked to be the 95 tilings the corpus was built as."""
    import numpy as np
    y = np.asarray(np.load(heldout("corrupted_graded_labels.npz"))["y"]).astype("int64")
    if y.size != N_CORRUPTED:
        raise ValueError("the private corrupted label vector is %d long, not %d"
                         % (y.size, N_CORRUPTED))
    if not np.array_equal(y, np.tile(y[:PER_CELL], N_CELLS)):
        raise ValueError("the private corrupted labels are not %d tilings of one %d-entry cell "
                         "vector; this bundle was authored against a corpus in which one fixed "
                         "draw of images is corrupted into every cell" % (N_CELLS, PER_CELL))
    return y


def clean_truth():
    import numpy as np
    y = np.asarray(np.load(heldout("clean_graded_labels.npz"))["y"]).astype("int64")
    if y.size != N_CLEAN:
        raise ValueError("the private clean label vector is %d long, not %d" % (y.size, N_CLEAN))
    return y


def constants_json():
    with open(os.path.join(HERE, "constants.json")) as handle:
        return json.load(handle)


def graded_bin_edges():
    """The PINNED graded partition, verified against the digest the bundle carries and never
    read from any submitted file."""
    import hashlib
    import numpy as np
    c = constants_json()
    path = os.path.join(HERE, os.path.basename(c["graded_bin_edges_file"]))
    blob = open(path, "rb").read()
    if hashlib.sha256(blob).hexdigest() != c["graded_bin_edges_sha256"]:
        raise ValueError("the graded bin edges do not match their pinned digest")
    edges = np.load(path, allow_pickle=False)
    if edges.size != c["graded_bin_count"] + 1:
        raise ValueError("the pinned partition does not carry %d bins" % c["graded_bin_count"])
    return edges


def ece(confidence, correct, edges):
    """Expected calibration error of one block under one partition.

    The first bin closes on both sides so a confidence sitting exactly on the lower edge is
    counted rather than dropped; every later bin is left-open, which is the usual convention and
    is what makes the bins a partition rather than an overlapping cover.
    """
    total = int(confidence.size)
    if total == 0:
        return 0.0
    error = 0.0
    for k in range(int(edges.size) - 1):
        lo, hi = edges[k], edges[k + 1]
        if k == 0:
            inside = (confidence >= lo) & (confidence <= hi)
        else:
            inside = (confidence > lo) & (confidence <= hi)
        weight = int(inside.sum())
        if weight:
            error += (weight / total) * abs(float(correct[inside].mean())
                                            - float(confidence[inside].mean()))
    return error


def refined_edges(edges, factor):
    """Each pinned bin split `factor` ways. The refinement is a strict refinement of the pinned
    partition, so a posterior calibrated at both resolutions is calibrated, while one calibrated
    only at the coarse resolution has been fitted to the coarse edges themselves."""
    import numpy as np
    pieces = [np.linspace(edges[k], edges[k + 1], factor + 1)
              for k in range(int(edges.size) - 1)]
    return np.unique(np.concatenate(pieces))


def confidence_correct(block, truth):
    """The per-sample (confidence, correctness) record every binning in the family reads."""
    return block.max(axis=1), (block.argmax(axis=1) == truth).astype("float64")


def macro_over_cells(confidence, correct, edges, spans):
    values = [ece(confidence[lo:hi], correct[lo:hi], edges) for _n, _s, lo, hi in spans]
    return sum(values) / len(values)


def binning_family(reference, span):
    """The 21 members CK-BINFAM fixes: ten bin counts crossed with equal-width and equal-mass
    edges, plus the pinned graded partition.

    Equal-width members span the OBSERVED confidence range rather than [0, 1]. That is what keeps
    the 15-bin equal-width member off the pinned [0, 1] fifteen-bin graded partition, which the
    contract requires of the twenty alternates: none of them is the graded scheme. Equal-mass
    edges are the quantiles of the CONTROL's confidence vector, computed once and applied
    unchanged to both candidates, so neither is measured under edges fitted to its own outputs.
    The outermost two edges are widened to cover both candidates, so no row of either falls
    outside every bin and is dropped from its own mean.
    """
    import numpy as np
    reference = np.asarray(reference, dtype="float64")
    lo, hi = float(reference.min()), float(reference.max())
    if not hi > lo:
        raise ValueError("the control's confidence vector is constant; no family can be built")
    outer_lo, outer_hi = min(lo, float(span[0])), max(hi, float(span[1]))
    members = []
    for k in BIN_COUNTS:
        width = np.linspace(lo, hi, k + 1)
        mass = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, k + 1)))
        if mass.size < 2:
            raise ValueError("equal-mass edges collapsed at %d bins" % k)
        for kind, edges in (("equal_width_%d" % k, width), ("equal_mass_%d" % k, mass)):
            edges = np.array(edges, dtype="float64", copy=True)
            edges[0], edges[-1] = outer_lo, outer_hi
            members.append((kind, edges))
    members.append(("graded_pinned", graded_bin_edges()))
    return members


def per_cell_ece():
    """One expected calibration error per (corruption, severity) cell, under the pinned edges."""
    confidence, correct = confidence_correct(corrupted_block(), corrupted_truth())
    edges = graded_bin_edges()
    return [(name, severity, ece(confidence[lo:hi], correct[lo:hi], edges))
            for name, severity, lo, hi in cell_spans()]


def pooled_ece():
    """The 19000-row figure, recorded beside the macro average so the gap is visible."""
    confidence, correct = confidence_correct(corrupted_block(), corrupted_truth())
    return ece(confidence, correct, graded_bin_edges())


def measure():
    """1 minus the macro-averaged corrupted calibration error: higher is better."""
    rows = per_cell_ece()
    return 1.0 - sum(r[2] for r in rows) / len(rows)


def run_record():
    path = os.path.join(HERE, "..", "run_record.json")
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def artifact_digest():
    import hashlib
    if not os.path.exists(SUBMISSION):
        return None
    return hashlib.sha256(open(SUBMISSION, "rb").read()).hexdigest()


def replay_digest():
    return run_record().get("replay_artifact_sha256")
'''


def render_layout(g):
    """The pinned layout constants the emitted module reads, derived from grounding alone.

    Every number the checks use is written once, here, from solution/grounding.yaml. A constant
    typed a second time into a check body is a constant that can drift from the corpus.
    """
    layout, submission = g["corpus_layout"], g["submission_layout"]
    lines = [
        "N_CLASSES = %d" % layout["classes"],
        "N_CORRUPTIONS = %d" % layout["corruptions"],
        "N_SEVERITIES = %d" % layout["severities"],
        "N_CELLS = %d" % layout["cells"],
        "PER_CELL = %d" % layout["rows_per_cell"],
        "N_CORRUPTED = %d" % layout["corrupted_rows"],
        "N_CLEAN = %d" % layout["clean_graded_rows"],
        "N_ROWS = %d" % submission["rows"],
        "ROW_SUM_TOLERANCE = %r" % float(submission["row_sum_tolerance"]),
        "CORRUPTION_NAMES = %r" % (tuple(layout["corruption_names"]),),
        "CORRUPTED_GRADED = %r" % layout["corrupted_graded"],
        "BIN_COUNTS = %r" % (tuple(g["binning_family"]["bin_counts"]),),
        "REFINEMENT = %d" % g["binning_family"]["graded_refinement"],
        "FITTABLE_SPLITS = %r" % (tuple(g["declaration_vocabulary"]["fittable_splits"]),),
        "GRADED_SPLITS = %r" % (tuple(g["declaration_vocabulary"]["graded_splits"]),),
    ]
    return "\n".join(lines) + "\n"


def validate(grounding):
    """Calibration is the one family whose INSTRUMENT is authored: the binning is pinned.

    Without a pinned partition there is nothing for an alternate-binning family to be compared
    against, and CTL-GRADEDBINS has no concrete object to be rejected for fitting to.
    """
    problems = []
    constants = grounding["constants"]
    pinned = [k for k in ("graded_bin_edges_file", "graded_bin_edges_sha256", "graded_bin_count")
              if constants.get(k)]
    # A PARTIAL pin is an inconsistency and refuses. A wholly absent pin is a thin design record,
    # which is a gap to record and not a reason to block authoring: refusing there would be a
    # threshold no slot without a design record could ever meet.
    if pinned and len(pinned) != 3:
        problems.append("the graded binning is only partially pinned: have %s" % sorted(pinned))

    # The layout arithmetic. These numbers reach the checker module and bind a submitted row to a
    # corruption; an inconsistent set would produce a checker that mislabels cells rather than
    # one that fails.
    layout, submission = grounding["corpus_layout"], grounding["submission_layout"]
    if layout["cells"] != layout["corruptions"] * layout["severities"]:
        problems.append("cells is not corruptions x severities")
    if layout["corrupted_rows"] != layout["cells"] * layout["rows_per_cell"]:
        problems.append("corrupted_rows is not cells x rows_per_cell")
    if len(layout["corruption_names"]) != layout["corruptions"]:
        problems.append("corruption_names carries %d names for %d corruptions"
                        % (len(layout["corruption_names"]), layout["corruptions"]))
    if sorted(layout["corruption_names"]) != list(layout["corruption_names"]):
        problems.append("corruption_names is not in the alphabetical order corruption_id indexes")
    if submission["rows"] != layout["corrupted_rows"] + layout["clean_graded_rows"]:
        problems.append("submission rows is not corrupted_rows + clean_graded_rows")
    if submission["columns"] != layout["classes"]:
        problems.append("submission columns is not the class count")
    if submission["corrupted_block"] != [0, layout["corrupted_rows"]]:
        problems.append("the corrupted block is not the leading rows")
    if submission["clean_block"] != [layout["corrupted_rows"], submission["rows"]]:
        problems.append("the clean block is not the trailing rows")

    family = grounding["binning_family"]
    if family["members"] != 2 * len(family["bin_counts"]) + 1:
        problems.append("the binning family declares %d members for %d bin counts"
                        % (family["members"], len(family["bin_counts"])))
    return problems


def compile_bodies(grounding):
    """One compiled body per compiled item, or a refusal.

    THE FALLBACK IS GONE ON PURPOSE. The family default filled a missing body with a lookup into
    run_record()["obligations"][ID] -- a check that reads what the graded run says about itself
    and agrees. On this slot that would have turned CK-BINFAM, an invariant over recomputed
    calibration errors, into a self-report, and on a sibling bundle the same fallback left the
    outcome check consulting nothing at all. Refusing to emit is the only response that cannot
    be mistaken for a passing check.
    """
    compiled = [it["id"] for it in grounding["items"] if it["mode"] == "compiled"]
    missing = sorted(i for i in compiled if i not in BODIES)
    if missing:
        raise SystemExit("%s: no compiled body for %s. A compiled item without a body would fall "
                         "through to a generic obligation lookup, which consults the graded run's "
                         "own account of itself; this generator refuses instead."
                         % (SLOT_ID, ", ".join(missing)))
    orphans = sorted(set(BODIES) - set(compiled))
    if orphans:
        raise SystemExit("%s: BODIES carries %s with no matching compiled item; a body that is "
                         "emitted for no item is a check nobody runs."
                         % (SLOT_ID, ", ".join(orphans)))

    # The ramp item's body must OPEN with require_measured. A body that read the submission first
    # would let a missing file raise its own error, the verifier would report gradeable true with
    # r_det 0.0, and a declared absence would have been recorded as a measured zero.
    ramp = grounding["ramp_item"]
    opening = 'c = require_measured("floor", "knee", "dichotomizing_threshold")'
    if BODIES[ramp].strip().splitlines()[0].strip() != opening:
        raise SystemExit("%s: the body for %s must open with %s" % (SLOT_ID, ramp, opening))
    return dict(BODIES)


def main():
    g = load_grounding()
    # The family validator runs FIRST and refuses rather than emitting: a generator that writes a
    # carrier it knows is malformed hands the drift check a moving target.
    problems = validate(g)
    if problems:
        raise SystemExit("AELLO-C8-S13 grounding is not well-formed: " + "; ".join(problems))
    bodies = compile_bodies(g)
    dl = g["deliverable"]
    subst = {"slot_id": SLOT_ID, "path": dl["path"], "layout": render_layout(g)}

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
        # it owns, and it legitimately contains percent signs of its own (format strings inside the
        # emitted helpers), so formatting it a second time here raised TypeError on the first slot
        # that carried one. Only the prelude is substituted.
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
