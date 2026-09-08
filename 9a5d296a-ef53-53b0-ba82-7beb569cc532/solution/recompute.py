"""Derivation for AELLO-C8-S10: every canonical artifact descends from solution/grounding.yaml.

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
    """Assemble the compiled-check module for accuracy grading."""
    chunks = [prelude]
    for raw in sorted(g["items"], key=lambda i: i["id"]):
        if raw["mode"] != "compiled":
            continue
        chunks.append(
            '\n@check(id="%s", weight=%d, knockout=%s, kind="process")\ndef test_%s():\n%s\n'
            % (raw["id"], raw["weight"], raw["knockout"], raw["id"], bodies[raw["id"]]))
    return "".join(chunks)


def _annex(g):
    """Compose the contract-record annex from the STRUCTURED keys this grounding carries.

    An earlier draft reached for a pre-rendered g["truth_annex"] blob that this slot never
    authored; it authored truth_annex_preamble beside a structured rejected_routes list instead.
    Composing from the structure keeps the annex derived rather than transcribed, so a route added
    to grounding.yaml cannot go missing from TRUTH.md.
    """
    out = [g["truth_annex_preamble"], "",
           "Ordered path, with the checker identifier each step satisfies:", ""]
    for n, item in enumerate(sorted(g["items"], key=lambda i: i["id"]), 1):
        out.append("%d. `%s` (%s, weight %d%s) -- %s"
                   % (n, item["id"], item["mode"], item["weight"],
                      ", knockout" if item["knockout"] else "", item["criterion"]))
    out += ["", "Rejected routes, each bound to the control that measures it wrong:", ""]
    for r in g["rejected_routes"]:
        out.append("- %s -- control `%s`, measured %.5f against reference %.5f"
                   % (r["route"], r["control"], r["measured"], r["against_reference"]))
    return "\n".join(out) + "\n"


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
              "annex is a recorded deviation rather than an omission.", "", _annex(g), ""]
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
SLOT_ID = 'AELLO-C8-S10'
TITLE = "# AELLO-C8-S10 -- what this task actually tests"


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
    """Top-1 agreement needs the graded fold's row count fixed before the run."""
    problems = []
    if not grounding.get("measurement_tier"):
        problems.append("no measurement tier recorded; the wave cannot schedule this slot")
    return problems



def main():
    g = load_grounding()
    # The family validator runs FIRST and refuses rather than emitting: a generator that writes a
    # carrier it knows is malformed hands the drift check a moving target.
    problems = validate(g)
    if problems:
        raise SystemExit("AELLO-C8-S10 grounding is not well-formed: " + "; ".join(problems))
    dl = g["deliverable"]
    metric = 'Accuracy over the private test nodes'
    subst = {"slot_id": SLOT_ID, "path": dl["path"], "kind": dl["kind"],
             "shape": dl["shape"], "metric": metric}

    bodies = dict({'deliverable_present': '    if not os.path.exists(SUBMISSION):\n        return False, evidence(reason="submission_missing", path=SUBMISSION)\n    try:\n        load_submission()\n    except Exception as exc:\n        return False, evidence(reason="submission_malformed", error=str(exc))\n    return True, evidence(path=SUBMISSION)', 'deliverable_shape': '    try:\n        obj = load_submission()\n    except Exception as exc:\n        return False, evidence(reason="submission_malformed", error=str(exc))\n    ok, why = shape_ok(obj)\n    return ok, evidence(reason=None if ok else why, shape=describe(obj))', 'no_graded_writes': '    touched = sorted(guarded_paths_touched())\n    return (not touched), evidence(target="submitted_source", touched=touched,\n                                   reason=None if not touched else "wrote_into_guarded_tree")', 'budget_declared': '    rec = run_record()\n    decl = rec.get("agent_budget_declaration")\n    if decl is None:\n        return False, evidence(target="run_record", reason="no_agent_declaration")\n    try:\n        json.loads(decl) if isinstance(decl, str) else dict(decl)\n    except Exception as exc:\n        return False, evidence(target="run_record", reason="declaration_unparseable",\n                               error=str(exc))\n    return True, evidence(target="run_record", declared=True)', 'deterministic_replay': '    a = artifact_digest()\n    b = replay_digest()\n    if b is None:\n        return False, evidence(target="submitted_source", reason="replay_unavailable")\n    return a == b, evidence(target="submitted_source", first=a, replay=b,\n                            reason=None if a == b else "replay_diverged")', 'graded_metric': '    c = require_measured("floor", "knee", "dichotomizing_threshold")\n    value = measure()\n    span = c["knee"] - c["floor"]\n    s = 0.0 if span <= 0 else max(0.0, min(1.0, (value - c["floor"]) / span))\n    return value >= c["dichotomizing_threshold"], evidence(metric=value, s_outcome=s)'})
    # graded_accuracy is THIS slot's outcome item, and it must consult the ramp. Without an
    # explicit body it fell through to the generic obligation lookup, which consults nothing --
    # so the verifier reported gradeable=True with r_det=0.0 while floor and knee were null,
    # collapsing "not yet measurable" into "the agent scored zero". Those are different claims and
    # the whole null-constant convention rests on keeping them apart.
    bodies["graded_accuracy"] = (
        '    c = require_measured("floor", "knee", "dichotomizing_threshold")\n'
        '    value = measure()\n'
        '    span = c["knee"] - c["floor"]\n'
        '    s = 0.0 if span <= 0 else max(0.0, min(1.0, (value - c["floor"]) / span))\n'
        '    return value >= c["dichotomizing_threshold"], evidence(metric=value, s_outcome=s)')

    for it in g["items"]:
        if it["mode"] == "compiled" and it["id"] not in bodies:
            bodies[it["id"]] = '    rec = run_record().get("obligations", {}).get(%(oid)r)\n    if rec is None:\n        return False, evidence(reason="obligation_unrecorded", obligation=%(oid)r)\n    return bool(rec.get("satisfied")), evidence(obligation=%(oid)r,\n                                                detail=rec.get("detail"),\n                                                reason=None if rec.get("satisfied")\n                                                else "obligation_unsatisfied")' % {"oid": it["id"].upper().replace("_", "-")}

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
        fh.write(build_tests(g, '"""Compiled checks for %(slot_id)s. GENERATED by solution/recompute.py. DO NOT HAND-EDIT.\n\nEvery check below runs against delivered bytes and references no unmeasured quantity, except\ngraded_metric, which raises ConstantUnmeasured while tests/constants.json carries nulls.\n"""\nimport json\nimport os\n\nHERE = os.path.dirname(os.path.abspath(__file__))\nSUBMISSION = %(path)r\nCHECKS = []\n\n\nclass ConstantUnmeasured(Exception):\n    """Raised when a check needs a ramp constant that Phase 0 deliberately left null."""\n\n\ndef check(id, weight, knockout, kind):\n    def deco(fn):\n        CHECKS.append({"id": id, "weight": weight, "knockout": knockout, "kind": kind, "fn": fn})\n        return fn\n    return deco\n\n\ndef constants():\n    with open(os.path.join(HERE, "constants.json")) as fh:\n        return json.load(fh)\n\n\ndef require_measured(*names):\n    c = constants()\n    missing = [n for n in names if c.get(n) is None]\n    if missing:\n        raise ConstantUnmeasured(\n            "%(slot_id)s: " + ", ".join(missing) + " are null. Phase 0 authors no measurement; "\n            "the Phase 2 measurement wave writes these and this check becomes gradeable then.")\n    return c\n\n\ndef evidence(**kw):\n    return {"evaluation_target": kw.pop("target", "submitted_artifact"), "values": kw}\n' % subst, bodies) + '\ndef load_submission():\n    """A dense array deliverable: integer label array, one row per graded instance"""\n    import numpy as np\n    arr = np.load(SUBMISSION, allow_pickle=False)\n    if arr.dtype == object:\n        raise ValueError("object arrays are not a valid deliverable")\n    return arr\n\n\ndef describe(obj):\n    return "%s%s" % (obj.dtype, obj.shape)\n\n\ndef shape_ok(obj):\n    if getattr(obj, "ndim", 0) < 1:\n        return False, "not_an_array"\n    if obj.size == 0:\n        return False, "empty_array"\n    import numpy as np\n    if not np.isfinite(np.asarray(obj, dtype="float64")).all():\n        return False, "non_finite_entries"\n    return True, None\n\ndef graded_pair():\n    """Predicted labels beside the private truth, both verifier-side."""\n    import numpy as np\n    predicted = load_submission()\n    truth = np.load(heldout("graded_labels.npz"))["y"]\n    predicted = np.asarray(predicted).ravel()[: len(truth)]\n    return predicted.astype("int64"), np.asarray(truth).astype("int64")\n\ndef measure():\n    """Top-1 agreement over the graded fold, computed verifier-side."""\n    import numpy as np\n    pred, truth = graded_pair()\n    return float(np.mean(pred == truth))\n\n\nGUARDED = ("tests/", "environment/data/graded", "/logs/verifier/")\n\n\ndef constants_json():\n    with open(os.path.join(HERE, "constants.json")) as handle:\n        return json.load(handle)\n\n\ndef heldout(name):\n    return os.path.join(HERE, "heldout", name)\n\n\ndef run_record():\n    path = os.path.join(HERE, "..", "run_record.json")\n    if not os.path.exists(path):\n        return {}\n    with open(path) as handle:\n        return json.load(handle)\n\n\ndef guarded_paths_touched():\n    """Paths under the checker tree or graded fold modified during the graded run.\n\n    Read from the harness record the run itself produced, never from a scan performed after the\n    fact: a post-hoc scan cannot separate a write made during the graded run from one made before.\n    """\n    return run_record().get("guarded_paths_touched", [])\n\n\ndef artifact_digest():\n    import hashlib\n    if not os.path.exists(SUBMISSION):\n        return None\n    return hashlib.sha256(open(SUBMISSION, "rb").read()).hexdigest()\n\n\ndef replay_digest():\n    return run_record().get("replay_artifact_sha256")\n')
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
