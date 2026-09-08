"""Derivation for C3-S1: every canonical artifact descends from solution/grounding.yaml.

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
SLOT_ID = 'C3-S1'
TITLE = "# C3-S1 -- what this task actually tests"


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
        # Recorded here because it is the fact the whole slot turns on and the one an agent is
        # most likely to assume: the divisor is raw corpus bytes, not tokens and not normalised
        # bytes, so the tokenizer never appears in the denominator it is trying to move.
        "graded_denominator": ("total raw UTF-8 byte count of the held-out text as it arrived, "
                               "before any normalisation the pipeline applies"),
    }
    out.update(read_screening(HERE))
    return out


REQUIRED_ASSERTIONS = ("byte_alphabet_fully_covered", "byte_denominator_is_raw_not_normalised",
                       "roundtrip_decodes_to_input_bytes")
RAMP_CONSTANTS = ("dichotomizing_threshold", "floor", "knee", "reward_gate_pass_threshold")


def validate(grounding):
    """A bits-per-byte slot is well formed only if three facts are asserted by some item.

    Losslessness, byte coverage and the raw-byte denominator are not hygiene items here. If
    encode and decode do not compose to identity the numerator is measured over a stream that is
    not the corpus; if a byte value is unrepresentable the denominator still charges for it; and
    if the denominator is taken after normalisation, b moves without any modelling changing. A
    rubric that asserts none of the three cannot tell a better tokenizer from a shorter corpus.
    """
    problems = []
    carried = {i["id"] for i in grounding["items"]}
    for name in REQUIRED_ASSERTIONS:
        if name not in carried:
            problems.append("no item asserts %s" % name)
    uncompiled = sorted(i["id"] for i in grounding["items"] if i["mode"] != "compiled")
    if uncompiled:
        problems.append("compilation_floor is 1.0 but these items are not compiled: "
                        + ", ".join(uncompiled))
    for name in RAMP_CONSTANTS:
        if grounding["constants"][name] is not None:
            problems.append("%s is authored at Phase 0, which the convention forbids" % name)
    return problems


def main():
    g = load_grounding()
    # The family validator runs FIRST and refuses rather than emitting: a generator that writes a
    # carrier it knows is malformed hands the drift check a moving target.
    problems = validate(g)
    if problems:
        raise SystemExit("C3-S1 grounding is not well-formed: " + "; ".join(problems))
    dl = g["deliverable"]
    metric = ('Held-out bits per byte, b = max(b_P1, b_P2) over two disjoint private shards. Each '
              'b is the frozen evaluator total negative log-likelihood in bits over the encoded '
              'held-out stream divided by the total raw UTF-8 byte count of the same held-out '
              'text. Lower b is better. Because the denominator counts raw corpus bytes and never '
              'tokens, a tokenizer that shrinks the token count without predicting better moves '
              'b by nothing at all')
    subst = {"slot_id": SLOT_ID, "path": dl["path"], "kind": dl["kind"],
             "shape": dl["shape"], "metric": metric}

    # BODIES ARE WRITTEN FOR THIS SLOT. Each one names a tokenizer fact -- a byte-fallback entry,
    # a round-trip fixture, a raw UTF-8 length, a declared stage -- and not one of them would mean
    # anything on a forecaster or a classifier. The generic spine that stood here graded the
    # existence and shape of a deliverable and said nothing about losslessness or about the
    # denominator, which are the only two things this task is about.
    bodies = {
        "vocabulary_artifact_emitted": (
            '    if not os.path.exists(SUBMISSION):\n'
            '        return False, evidence(reason="submission_missing", path=SUBMISSION)\n'
            '    try:\n'
            '        doc = load_submission()\n'
            '    except Exception as exc:\n'
            '        return False, evidence(reason="submission_malformed", error=str(exc))\n'
            '    absent = [name for name in SECTIONS if name not in doc]\n'
            '    return (not absent), evidence(path=SUBMISSION, carried=sorted(doc),\n'
            '                                  absent=absent,\n'
            '                                  reason=None if not absent else "submission_malformed")'),

        "vocabulary_entry_count_read_from_entries": (
            '    try:\n'
            '        vocabulary = load_submission()["vocabulary"]\n'
            '        entries = list(vocabulary["entries"])\n'
            '    except Exception as exc:\n'
            '        return False, evidence(target="emitted_vocabulary",\n'
            '                               reason="submission_malformed", error=str(exc))\n'
            '    counted = len(set(entries))\n'
            '    declared = vocabulary.get("declared_entry_count")\n'
            '    within = counted <= VOCABULARY_CEILING\n'
            '    agrees = declared == counted\n'
            '    return (within and agrees), evidence(target="emitted_vocabulary",\n'
            '                                        counted_from_entries=counted,\n'
            '                                        declared_beside_them=declared,\n'
            '                                        ceiling=VOCABULARY_CEILING,\n'
            '                                        reason=None if within and agrees\n'
            '                                        else "vocab-budget-exceeded")'),

        "byte_alphabet_fully_covered": (
            '    try:\n'
            '        entries = set(load_submission()["vocabulary"]["entries"])\n'
            '    except Exception as exc:\n'
            '        return False, evidence(target="emitted_vocabulary",\n'
            '                               reason="submission_malformed", error=str(exc))\n'
            '    uncovered = sorted(BYTE_FALLBACK_ENTRIES - entries)\n'
            '    return (not uncovered), evidence(target="emitted_vocabulary",\n'
            '                                     unrepresentable_byte_values=len(uncovered),\n'
            '                                     first_uncovered=uncovered[:4],\n'
            '                                     reason=None if not uncovered\n'
            '                                     else "byte-fallback-incomplete")'),

        "roundtrip_decodes_to_input_bytes": (
            '    try:\n'
            '        rows = roundtrip_rows()\n'
            '    except Exception as exc:\n'
            '        return False, evidence(reason="submission_malformed", error=str(exc))\n'
            '    if not rows:\n'
            '        return False, evidence(reason="no_roundtrip_evidence")\n'
            '    lossy = []\n'
            '    for row in rows:\n'
            '        if utf8(row.get("decoded_utf8", "")) != utf8(row.get("raw_utf8", "")):\n'
            '            lossy.append(row.get("fixture_class"))\n'
            '    return (not lossy), evidence(fixtures_checked=len(rows),\n'
            '                                 not_byte_identity=lossy,\n'
            '                                 reason=None if not lossy\n'
            '                                 else "roundtrip-not-identity")'),

        "roundtrip_covers_adversarial_classes": (
            '    try:\n'
            '        rows = roundtrip_rows()\n'
            '    except Exception as exc:\n'
            '        return False, evidence(reason="submission_malformed", error=str(exc))\n'
            '    witnessed = set()\n'
            '    for row in rows:\n'
            '        if row.get("raw_utf8"):\n'
            '            witnessed.add(row.get("fixture_class"))\n'
            '    uncovered = [name for name in ADVERSARIAL_CLASSES if name not in witnessed]\n'
            '    return (not uncovered), evidence(\n'
            '        covered=[n for n in ADVERSARIAL_CLASSES if n in witnessed],\n'
            '        uncovered=uncovered,\n'
            '        reason=None if not uncovered else "adversarial_class_unwitnessed")'),

        "byte_denominator_is_raw_not_normalised": (
            '    try:\n'
            '        rows = roundtrip_rows()\n'
            '        declared_total = load_submission()["bpb_accounting"]["raw_utf8_byte_total"]\n'
            '    except Exception as exc:\n'
            '        return False, evidence(reason="submission_malformed", error=str(exc))\n'
            '    mismeasured, recomputed_total = [], 0\n'
            '    for row in rows:\n'
            '        length = len(utf8(row.get("raw_utf8", "")))\n'
            '        recomputed_total += length\n'
            '        if row.get("raw_utf8_bytes") != length:\n'
            '            mismeasured.append(row.get("fixture_class"))\n'
            '    honest = (not mismeasured) and declared_total == recomputed_total\n'
            '    return honest, evidence(recomputed_total=recomputed_total,\n'
            '                            declared_total=declared_total,\n'
            '                            mismeasured=mismeasured,\n'
            '                            reason=None if honest\n'
            '                            else "denominator_not_raw_utf8_bytes")'),

        "bits_per_byte_divides_by_bytes_not_tokens": (
            '    try:\n'
            '        accounting = load_submission()["bpb_accounting"]\n'
            '        bits = float(accounting["nll_bits_total"])\n'
            '        per_byte = float(accounting["declared_bits_per_byte"])\n'
            '        byte_total = int(accounting["raw_utf8_byte_total"])\n'
            '        token_total = int(accounting["token_total"])\n'
            '    except Exception as exc:\n'
            '        return False, evidence(reason="submission_malformed", error=str(exc))\n'
            '    tolerance = 1e-6 * max(1.0, abs(bits))\n'
            '    over_bytes = abs(per_byte * byte_total - bits) <= tolerance\n'
            '    over_tokens = abs(per_byte * token_total - bits) <= tolerance\n'
            '    # A byte-level tokenizer makes the two totals equal and the distinction\n'
            '    # undefined, so the byte identity is allowed to stand alone in that one case.\n'
            '    indistinguishable = byte_total == token_total\n'
            '    basis = accounting.get("denominator_basis")\n'
            '    divides_by_bytes = (basis == "raw_utf8_bytes" and over_bytes\n'
            '                        and (indistinguishable or not over_tokens))\n'
            '    return divides_by_bytes, evidence(denominator_basis=basis,\n'
            '                                      byte_total=byte_total,\n'
            '                                      token_total=token_total,\n'
            '                                      identity_over_bytes=over_bytes,\n'
            '                                      identity_over_tokens=over_tokens,\n'
            '                                      reason=None if divides_by_bytes\n'
            '                                      else "denominator_is_not_the_byte_total")'),

        "stage_manifest_order_matches_execution": (
            '    try:\n'
            '        stages = list(load_submission()["stage_manifest"])\n'
            '        declared = [str(stage["stage"]) for stage in stages]\n'
            '    except Exception as exc:\n'
            '        return False, evidence(target="stage_manifest",\n'
            '                               reason="submission_malformed", error=str(exc))\n'
            '    executed = [str(name) for name in run_record().get("stage_execution_order", [])]\n'
            '    undigested = [stage.get("stage") for stage in stages\n'
            '                  if not stage.get("entry_point") or not stage.get("sha256")]\n'
            '    repeats = len(set(declared)) != len(declared)\n'
            '    faithful = bool(declared) and declared == executed and not undigested\\\n'
            '        and not repeats\n'
            '    return faithful, evidence(target="stage_manifest", declared=declared,\n'
            '                              executed=executed, undigested=undigested,\n'
            '                              repeated_stage_name=repeats,\n'
            '                              reason=None if faithful\n'
            '                              else "manifest-order-divergence")'),

        "manifest_replay_reproduces_vocabulary_digest": (
            '    replayed = run_record().get("manifest_replay_vocabulary_sha256")\n'
            '    if replayed is None:\n'
            '        return False, evidence(target="run_record", reason="replay_unavailable")\n'
            '    try:\n'
            '        built = load_submission()["vocabulary"]["artifact_sha256"]\n'
            '    except Exception as exc:\n'
            '        return False, evidence(target="run_record",\n'
            '                               reason="submission_malformed", error=str(exc))\n'
            '    rebuilt = str(replayed) == str(built)\n'
            '    return rebuilt, evidence(target="run_record", built=built, replayed=replayed,\n'
            '                             reason=None if rebuilt\n'
            '                             else "manifest-order-divergence")'),

        "frozen_trainer_digest_unchanged": (
            '    record = run_record()\n'
            '    digests = record.get("frozen_trainer_digests")\n'
            '    if not isinstance(digests, dict) or "before" not in digests \\\n'
            '            or "after" not in digests:\n'
            '        return False, evidence(target="run_record", reason="no_digest_record")\n'
            '    before, after = digests["before"] or {}, digests["after"] or {}\n'
            '    moved = sorted(k for k in set(before) | set(after)\n'
            '                   if before.get(k) != after.get(k))\n'
            '    touched = sorted(record.get("guarded_paths_touched", []))\n'
            '    intact = (not moved) and (not touched)\n'
            '    if moved:\n'
            '        why = "frozen-trainer-modified"\n'
            '    elif touched:\n'
            '        why = "grading-tree-write"\n'
            '    else:\n'
            '        why = None\n'
            '    return intact, evidence(target="run_record", files_digested=len(before),\n'
            '                            moved=moved, touched=touched, reason=why)'),

        "encode_and_train_seconds_declared": (
            '    declaration = run_record().get("agent_budget_declaration")\n'
            '    if declaration is None:\n'
            '        return False, evidence(target="run_record", reason="no_agent_declaration")\n'
            '    try:\n'
            '        named = (json.loads(declaration) if isinstance(declaration, str)\n'
            '                 else dict(declaration))\n'
            '    except Exception as exc:\n'
            '        return False, evidence(target="run_record",\n'
            '                               reason="declaration_unparseable", error=str(exc))\n'
            '    absent = [key for key in BUDGET_KEYS if key not in named]\n'
            '    return (not absent), evidence(target="run_record", declared=sorted(named),\n'
            '                                  absent=absent,\n'
            '                                  reason=None if not absent\n'
            '                                  else "declaration_incomplete")'),

        "graded_region_within_wall": (
            '    measured = run_record().get("graded_region_wall_seconds")\n'
            '    if measured is None:\n'
            '        return False, evidence(target="run_record", reason="wall_unrecorded")\n'
            '    try:\n'
            '        measured = float(measured)\n'
            '    except Exception as exc:\n'
            '        return False, evidence(target="run_record", reason="wall_unparseable",\n'
            '                               error=str(exc))\n'
            '    inside = measured <= WALL_SECONDS\n'
            '    return inside, evidence(target="run_record", measured_seconds=measured,\n'
            '                            wall_seconds=WALL_SECONDS,\n'
            '                            reason=None if inside else "wallclock-exceeded")'),

        "no_per_shard_breakdown_in_submission": (
            '    try:\n'
            '        doc = load_submission()\n'
            '    except Exception as exc:\n'
            '        return False, evidence(reason="submission_malformed", error=str(exc))\n'
            '    words, numbers = harvest(doc, set(), set())\n'
            '    named = sorted(w for w in words\n'
            '                   if any(token in w for token in PRIVATE_SURFACE_TOKENS))\n'
            '    pinned = sorted(str(name) for name, value in constants().items()\n'
            '                    if isinstance(value, (int, float))\n'
            '                    and not isinstance(value, bool) and float(value) in numbers)\n'
            '    sealed = (not named) and (not pinned)\n'
            '    return sealed, evidence(private_names=named[:6], pinned_constants=pinned,\n'
            '                            reason=None if sealed\n'
            '                            else "private_shard_surface_leaked")'),

        "held_out_bits_per_byte_ramp": (
            '    # require_measured FIRST, before any read of the submission. Reading the\n'
            '    # artifact first lets a missing-file error mask the unmeasured signal: the\n'
            '    # verifier then reports gradeable=true with r_det=0.0, which collapses "not yet\n'
            '    # measurable" into "the agent scored zero". They are different claims.\n'
            '    c = require_measured("floor", "knee", "dichotomizing_threshold")\n'
            '    b = measure()\n'
            '    # Lower bits per byte is better, so the ramp DESCENDS: floor is the worse end.\n'
            '    span = c["floor"] - c["knee"]\n'
            '    s = 0.0 if span <= 0 else max(0.0, min(1.0, (c["floor"] - b) / span))\n'
            '    return b <= c["dichotomizing_threshold"], evidence(bits_per_byte=b,\n'
            '                                                      s_outcome=s)'),
    }
    # REFUSE rather than fall through. A generic obligation lookup used to stand here and quietly
    # supply a body for any item nobody had written one for; the body it supplied read a run
    # record key nothing ever wrote, so an outcome check consulted nothing at all and still
    # reported. An unwritten body is an authoring gap and it stops the generator.
    unwritten = sorted(it["id"] for it in g["items"]
                       if it["mode"] == "compiled" and it["id"] not in bodies)
    if unwritten:
        raise SystemExit("C3-S1: no compiled body is written for %s; refusing to emit a checker "
                         "that would consult nothing" % ", ".join(unwritten))

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


PRELUDE = '''"""Compiled checks for %(slot_id)s. GENERATED by solution/recompute.py. DO NOT HAND-EDIT.

Every check below runs against delivered bytes and references no unmeasured quantity, except
held_out_bits_per_byte_ramp, which raises ConstantUnmeasured while tests/constants.json carries
nulls. The graded quantity is %(metric)s.
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
# --- the facts this slot is about ---------------------------------------------------------
SECTIONS = ("bpb_accounting", "roundtrip_fixtures", "stage_manifest", "vocabulary")
# The five disclosed classes. Each one breaks a different assumption a tokenizer makes about
# text: ligature folding, an inserted kashida, mark reordering, a codepoint UTF-8 cannot carry
# on its own, and a sequence NFKC does not leave alone.
ADVERSARIAL_CLASSES = ("arabic_presentation_forms", "tatweel", "tamil_combining",
                       "lone_surrogate", "nfkc_unstable")
BYTE_FALLBACK_ENTRIES = frozenset("<0x%02X>" % value for value in range(256))
VOCABULARY_CEILING = 65536
WALL_SECONDS = 900.0
BUDGET_KEYS = ("encode_seconds", "script_sampling", "train_seconds", "vocab_size")
PRIVATE_SURFACE_TOKENS = ("b_p1", "b_p2", "shard_p1", "shard_p2", "heldout")


def load_submission():
    """The tokenizer description: vocabulary, ordered stage manifest, round-trip evidence and
    the byte accounting. Not a model and not a token stream -- the graded object here is the
    recipe that turns bytes into tokens."""
    with open(SUBMISSION) as handle:
        doc = json.load(handle)
    if not isinstance(doc, dict):
        raise ValueError("the deliverable must be a JSON object")
    return doc


def utf8(text):
    """Raw UTF-8 bytes with surrogatepass.

    The denominator counts these bytes and nothing else. surrogatepass is not a convenience: the
    lone-surrogate fixture exists precisely because a codepoint UTF-8 cannot carry alone is where
    a pipeline silently substitutes U+FFFD, and a strict encoder would raise here and hide the
    substitution behind an exception instead of measuring it.
    """
    return str(text).encode("utf-8", "surrogatepass")


def roundtrip_rows():
    rows = load_submission()["roundtrip_fixtures"]
    if not isinstance(rows, list):
        raise ValueError("roundtrip_fixtures must be a list of fixture records")
    return [row for row in rows if isinstance(row, dict)]


def harvest(node, words, numbers):
    """Every string and every number anywhere in the submission, object keys included."""
    if isinstance(node, dict):
        for key, value in node.items():
            words.add(str(key).lower())
            harvest(value, words, numbers)
    elif isinstance(node, list):
        for value in node:
            harvest(value, words, numbers)
    elif isinstance(node, bool):
        pass
    elif isinstance(node, (int, float)):
        numbers.add(float(node))
    elif isinstance(node, str):
        words.add(node.lower())
    return words, numbers


def evaluator_report():
    """The frozen evaluator's own per-shard result, under the verifier-owned held-out tree.

    Read from there and never from the submission or the run record: b is the one quantity the
    agent must not be able to author, and the worse-of-two rule is only a rule if both shard
    values come from the same evaluator the agent never runs.
    """
    path = os.path.join(HERE, "heldout", "evaluator_report.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def measure():
    """b = max(b_P1, b_P2): the worse of the two disjoint private shards, lower being better."""
    report = evaluator_report()
    if not report:
        raise ConstantUnmeasured("the frozen evaluator has written no bits-per-byte report")
    shards = [report.get("b_p1"), report.get("b_p2")]
    if any(value is None for value in shards):
        raise ConstantUnmeasured("the evaluator report carries only one shard; b is max of two")
    return max(float(value) for value in shards)


def run_record():
    """The record the graded run itself produced.

    Stage order, the replay digest, the frozen-trainer digests and the wall are all read from
    here rather than from a scan taken afterwards, which cannot separate what happened inside
    the graded region from what happened before it.
    """
    path = os.path.join(HERE, "..", "run_record.json")
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        return json.load(handle)
'''


if __name__ == "__main__":
    raise SystemExit(main())
