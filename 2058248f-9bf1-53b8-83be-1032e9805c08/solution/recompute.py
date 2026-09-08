"""Derivation for A5-02: every canonical artifact descends from solution/grounding.yaml.

WHAT THIS GENERATOR EMITS AND WHY IT IS NOT A TEMPLATE. A5-02 grades threshold-free macro average
precision over 19 labels on 12-band Sentinel-2 patches, twice, and multiplies the two normalized
values. Nothing about that reduces to "did a file appear and does it parse". The checks below
count label columns, count distinct values per label column, measure the spread of a twelve-entry
band-scale vector, count quantization scales against output channels, recount serialized bytes off
the shipped blob, rebuild a three-term analytic latency estimate from graph integers, read a
batch-64 live-tensor ledger, and compare per-record supervision digests against the shipped
shard's own label bytes. Each of those is an assertion about THIS task and would be meaningless on
any other one, which is the point: a checker that would pass unchanged on a forecaster or a
tokenizer is not checking this slot.

Bodies are authored flush-left in BODIES and indented once at emission. An item with no body stops
the generator; it never falls through to a generic obligation lookup, because the lookup that used
to stand here supplied a body that read a run-record key nothing ever wrote, so an outcome check
consulted nothing at all and still reported a verdict.
"""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import os  # noqa: E402

import yaml  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)
SLOT_ID = "A5-02"
TITLE = "# A5-02 -- what this task actually tests"
ITEM_KEYS = ("id", "dimension", "weight", "evaluation_target", "criterion", "judgment",
             "knockout", "mode")

# The graded quantity, written once and substituted into the emitted module's docstring.
GRADED_QUANTITY = (
    "the product of two normalized threshold-free macro average precisions over the 19 labels, one "
    "on the private in-distribution held-out split and one on the private held-out-country split, "
    "both measured by the verifier's own executor over the frozen export. Each patch carries "
    "several labels at once, so the graded artifact is a score matrix and never an argmax, and "
    "because average precision integrates over a whole ranking, a submission that has already "
    "applied a decision threshold has discarded the quantity being measured")


def load_grounding():
    with open(os.path.join(HERE, "grounding.yaml")) as handle:
        return yaml.safe_load(handle)


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
    return {"screening_measured_at": None, "screening_expires_at": None, "screening_result": None}


def indent_body(source):
    """Bodies are authored flush-left for readability and indented once at emission."""
    out = []
    for line in source.strip("\n").split("\n"):
        out.append("    " + line if line.strip() else "")
    return "\n".join(out)


def build_checks(grounding, prelude, bodies):
    """Assemble tests/test_output.py in the grounding's own item order."""
    parts = [prelude]
    for item in grounding["items"]:
        if item["mode"] != "compiled":
            continue
        parts.append("\n@check(id=%r, weight=%d, knockout=%s, target=%r)\ndef check_%s():\n%s\n"
                     % (item["id"], item["weight"], item["knockout"], item["evaluation_target"],
                        item["id"], indent_body(bodies[item["id"]])))
    return "".join(parts)


def build_truth(grounding, title):
    """Render solution/TRUTH.md from the frozen literals of solution/grounding.yaml."""
    narrative = grounding["truth_narrative"]
    lines = [title, "", "GENERATED SECTION. DO NOT HAND-EDIT.", "", narrative["opening"], "",
             "## The single most important insight", "", narrative["insight"], "",
             "## The ideal solve, step by step", ""]
    for index, step in enumerate(narrative["steps"], 1):
        lines += ["%d. **%s** %s" % (index, step["heading"], step["paragraph"]), ""]
    lines += ["## Traps that catch agents that are not thinking carefully", ""]
    lines += ["- " + trap for trap in narrative["traps"]]
    lines += ["", "---", "", "## Contract record (annex; not part of the narrative body)", "",
              "Retained under FORGE.md item 10e and Phase 2 item 7, which require this file to "
              "carry the ordered path through instruction.md with each satisfied checker "
              "identifier, and each rejected route bound to a measured known-wrong control. "
              "standards/truth-md-authoring-v1.md section 3 admits no fifth section, so this "
              "annex is a recorded deviation rather than an omission.", "",
              grounding["truth_annex"], ""]
    return "\n".join(lines) + "\n"


def build_rubrics(grounding):
    """The 9g rubric carrier. The item schema is CLOSED at eight keys, so item 10f's outcome
    classification rides as a greppable prefix of the judgment rather than as a ninth key."""
    rows = []
    for raw in grounding["items"]:
        row = dict((key, raw[key]) for key in ITEM_KEYS)
        row["judgment"] = "%s: %s" % (raw["outcome_class"], raw["judgment"])
        rows.append(row)
    compiled = sum(row["weight"] for row in rows if row["mode"] == "compiled")
    total = sum(row["weight"] for row in rows)
    return {"$schema": "forge.rubric/v1",
            "banner": "GENERATED SECTION. DO NOT HAND-EDIT.",
            "generator": "solution/recompute.py",
            "compilation_floor": grounding["compilation_floor"],
            "compiled_weight_share": round(compiled / total, 6) if total else 0.0,
            "evaluation_target_vocabulary": sorted(grounding["evaluation_target_vocabulary"]),
            "items": sorted(rows, key=lambda row: row["id"])}


def build_provenance(grounding, identity):
    p = grounding["provenance"]
    out = {
        "banner": "GENERATED SECTION. DO NOT HAND-EDIT.",
        "generator": "solution/recompute.py from solution/grounding.yaml",
        "schema_version": "1.0",
        "slot_id": grounding["slot_id"],
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
        "measurement_tier": grounding["measurement_tier"],
        "gradeable": grounding["gradeable"],
        "knee_anchor_status": grounding["knee_anchor_status"],
        # Recorded here because they are the two facts the whole slot turns on and the two an
        # agent is most likely to assume away: the metric reads a RANKING per label, so a
        # thresholded decision matrix is a different object; and the macro mean is unweighted
        # over the labels PRESENT in the split, so the rarest label is worth exactly as much as
        # the most common one and a label with no positive is not scoreable at all.
        "graded_object": ("a float score matrix of shape (n_graded_patches, 19); average "
                          "precision is threshold-free and reads the ordering, so no decision "
                          "threshold is applied anywhere in grading"),
        "macro_denominator": ("the labels present in the split, weighted uniformly; a per-label "
                              "average precision is undefined for a label with no positive "
                              "instance and is not charged as a zero"),
        "corpus_layout": grounding["corpus_layout"],
    }
    out.update(read_screening(HERE))
    return out


# --------------------------------------------------------------------------------------
# The family validator. It runs BEFORE anything is emitted, because a generator that writes a
# carrier it knows is malformed hands the drift check a moving target.
# --------------------------------------------------------------------------------------

LOAD_BEARING_ITEMS = (
    "macro_mean_divides_by_labels_present_in_the_split",
    "per_band_statistics_not_one_global_scale",
    "quantization_scales_are_per_output_channel",
    "scores_are_continuous_not_thresholded",
    "supervision_digests_equal_the_shipped_shard_labels",
)
NULL_AT_PHASE_ZERO = ("dichotomizing_threshold", "floor", "host_flops_per_second",
                      "host_hbm_bytes_per_second", "host_kernel_launch_seconds", "knee",
                      "reward_gate_pass_threshold", "shift_floor", "shift_knee")
OUTCOME_ITEM = "macro_ap_product_ramp"


def validate(grounding, bodies):
    """Five facts make this slot's rubric well formed, and none of them is hygiene.

    Threshold freedom, per-band normalization, per-channel quantization scales, the macro
    denominator and label provenance are the assertions the task is built on. Drop the first and
    a thresholded decision matrix passes as a ranking. Drop the second or the third and the two
    silent failures that cost the most macro AP raise nothing anywhere. Drop the fourth and a
    label that cannot be scored is charged as a zero. Drop the fifth and a distilled student is
    indistinguishable from a model that read the shard.
    """
    problems = []
    carried = set(item["id"] for item in grounding["items"])
    for name in LOAD_BEARING_ITEMS:
        if name not in carried:
            problems.append("no item asserts %s" % name)
    uncompiled = sorted(i["id"] for i in grounding["items"] if i["mode"] != "compiled")
    if uncompiled:
        problems.append("compilation_floor is %s but these items are not compiled: %s"
                        % (grounding["compilation_floor"], ", ".join(uncompiled)))
    targets = set(item["evaluation_target"] for item in grounding["items"])
    vocabulary = set(grounding["evaluation_target_vocabulary"])
    if targets - vocabulary:
        problems.append("items name evaluation targets outside the vocabulary: %s"
                        % ", ".join(sorted(targets - vocabulary)))
    if "run_record" not in targets:
        problems.append("a slot with a budget, a watchdog and a campaign log must grade against "
                        "the harness run record")
    for name in NULL_AT_PHASE_ZERO:
        if grounding["constants"].get(name) is not None:
            problems.append("%s is authored at Phase 0, which the convention forbids" % name)
    unwritten = sorted(i["id"] for i in grounding["items"]
                       if i["mode"] == "compiled" and i["id"] not in bodies)
    if unwritten:
        problems.append("no compiled body is written for %s; a checker that would consult "
                        "nothing is worse than a missing one" % ", ".join(unwritten))
    # The outcome body must ask for its constants BEFORE it reads a submitted byte. If the
    # artifact is read first, a missing-file error masks the unmeasured signal and the verifier
    # reports gradeable=true with r_det=0.0, which collapses "not yet measurable" into "the agent
    # scored zero". They are different claims and this refuses rather than trusting the comment.
    outcome = bodies.get(OUTCOME_ITEM, "")
    first = next((line.strip() for line in outcome.strip("\n").split("\n")
                  if line.strip() and not line.strip().startswith("#")), "")
    if 'require_measured("floor", "knee", "dichotomizing_threshold")' not in first:
        problems.append("%s does not call require_measured for the ramp constants as its first "
                        "statement; it begins with %r" % (OUTCOME_ITEM, first))
    return problems


# --------------------------------------------------------------------------------------
# BODIES. Authored flush-left, one per compiled item, each running against delivered bytes.
# --------------------------------------------------------------------------------------

BODIES = {}

BODIES["score_columns_span_the_nineteen_labels"] = '''
if not os.path.exists(SCORE_MATRIX):
    return False, evidence(reason="NO_SUBMISSION", path=SCORE_MATRIX)
try:
    scores = load_score_matrix()
except MALFORMED as exc:
    return False, evidence(reason="submission_malformed", error=str(exc))
if scores.ndim != 2:
    # A one-dimensional array is one chosen label per patch. The 19 labels do not partition a
    # patch -- several hold at once -- so an argmax is not a weak answer, it is a different one.
    return False, evidence(reason="not_a_score_matrix", ndim=int(scores.ndim),
                           shape=[int(n) for n in scores.shape])
rows, columns = int(scores.shape[0]), int(scores.shape[1])
if columns == BAND_COUNT:
    return False, evidence(reason="columns_are_bands_not_labels", columns=columns,
                           bands=BAND_COUNT, labels=LABEL_COUNT)
if columns != LABEL_COUNT:
    return False, evidence(reason="label_axis_wrong_width", columns=columns, labels=LABEL_COUNT)
if rows < 1:
    return False, evidence(reason="no_graded_rows")
return True, evidence(rows=rows, columns=columns, dtype=str(scores.dtype))
'''

BODIES["scores_are_continuous_not_thresholded"] = '''
numpy = array_backend()
try:
    scores = load_score_matrix()
except MALFORMED as exc:
    return False, evidence(reason="submission_malformed", error=str(exc))
if scores.dtype.kind != "f":
    return False, evidence(reason="scores_are_not_floating_point", dtype=str(scores.dtype))
if not numpy.isfinite(scores).all():
    return False, evidence(reason="non_finite_scores")
if scores.ndim != 2 or scores.shape[1] != LABEL_COUNT:
    return False, evidence(reason="label_axis_wrong_width",
                           shape=[int(n) for n in scores.shape])
already_decided = []
for column in range(LABEL_COUNT):
    if len(numpy.unique(scores[:, column])) <= 2:
        already_decided.append(column)
ranked = LABEL_COUNT - len(already_decided)
passed = ranked >= MIN_RANKED_COLUMNS
# Average precision sweeps the whole ranking. A column collapsed to two values has been
# reduced to one operating point, and calibrating that point is explicitly not the task.
return passed, evidence(ranked_columns=ranked, thresholded_columns=already_decided,
                        minimum_ranked=MIN_RANKED_COLUMNS,
                        reason=None if passed else "scores_already_thresholded")
'''

BODIES["every_label_column_is_ranked_not_constant"] = '''
try:
    scores = load_score_matrix()
except MALFORMED as exc:
    return False, evidence(reason="submission_malformed", error=str(exc))
if scores.ndim != 2 or scores.shape[1] != LABEL_COUNT:
    return False, evidence(reason="label_axis_wrong_width",
                           shape=[int(n) for n in scores.shape])
flat = []
for column in range(LABEL_COUNT):
    band = scores[:, column]
    if float(band.max()) == float(band.min()):
        flat.append(column)
# A constant column has no ordering, so its average precision falls to that label's positive
# rate. The macro mean is unweighted, so one collapsed label costs a full nineteenth.
return (not flat), evidence(constant_label_columns=flat, labels=LABEL_COUNT,
                            reason=None if not flat else "constant_label_column")
'''

BODIES["macro_mean_divides_by_labels_present_in_the_split"] = '''
try:
    block = manifest_section("macro_average")
except MALFORMED as exc:
    return False, evidence(target="export_manifest", reason="manifest_malformed", error=str(exc))
basis = block.get("denominator_basis")
if basis != "labels_present_in_split":
    return False, evidence(target="export_manifest", reason="macro_denominator_counts_all_labels",
                           denominator_basis=basis, labels=LABEL_COUNT)
weighting = block.get("per_label_weighting")
if weighting != "uniform":
    # A mean weighted by per-label support is a micro average wearing the word macro.
    return False, evidence(target="export_manifest", reason="macro_mean_is_support_weighted",
                           per_label_weighting=weighting)
scored = block.get("labels_scored")
if not isinstance(scored, int) or not 1 <= scored <= LABEL_COUNT:
    return False, evidence(target="export_manifest", reason="labels_scored_out_of_range",
                           labels_scored=scored, labels=LABEL_COUNT)
return True, evidence(target="export_manifest", labels_scored=scored, labels=LABEL_COUNT)
'''

BODIES["per_band_statistics_not_one_global_scale"] = '''
try:
    block = manifest_section("band_normalization")
    centres = [float(v) for v in block["mean"]]
    scales = [float(v) for v in block["scale"]]
except MALFORMED as exc:
    return False, evidence(target="export_manifest", reason="manifest_malformed", error=str(exc))
if len(centres) != BAND_COUNT or len(scales) != BAND_COUNT:
    return False, evidence(target="export_manifest", reason="normalization_not_per_band",
                           centres=len(centres), scales=len(scales), bands=BAND_COUNT)
if min(scales) <= 0.0:
    return False, evidence(target="export_manifest", reason="non_positive_band_scale",
                           smallest=min(scales))
spread = max(scales) / min(scales)
passed = spread >= BAND_SPREAD_MINIMUM
# Twelve copies of one number is a global statistic wearing a per-band shape. The Sentinel-2
# bands differ by more than an order of magnitude, so a genuine fit spreads by at least that.
return passed, evidence(target="export_manifest", scale_spread=spread,
                        required_spread=BAND_SPREAD_MINIMUM, bands=BAND_COUNT,
                        reason=None if passed else "band_scales_collapsed_to_global")
'''

BODIES["quantization_scales_are_per_output_channel"] = '''
try:
    block = manifest_section("quantization")
    tensors = list(block["tensors"])
except MALFORMED as exc:
    return False, evidence(target="export_manifest", reason="manifest_malformed", error=str(exc))
if block.get("granularity") != "per_channel":
    return False, evidence(target="export_manifest", reason="per_tensor_scale_declared",
                           granularity=block.get("granularity"))
if not tensors:
    return False, evidence(target="export_manifest", reason="no_quantized_tensors_listed")
collapsed = []
for entry in tensors:
    channels = int(entry.get("output_channels", 0))
    scales = entry.get("scales") or []
    if channels > 1 and len(scales) != channels:
        collapsed.append({"tensor": entry.get("name"), "output_channels": channels,
                          "scales": len(scales)})
passed = not collapsed
# One scale per tensor is set by the largest channel and rounds the smallest to zero. Those
# carry the rarest of the 19 labels, and each label that collapses costs a full nineteenth.
return passed, evidence(target="export_manifest", tensors=len(tensors), collapsed=collapsed,
                        reason=None if passed else "quantization_scale_shared_across_channels")
'''

BODIES["serialized_parameter_bytes_recounted_from_the_blob"] = '''
try:
    block = manifest_section("parameter_blob")
    declared = int(block["serialized_bytes"])
except MALFORMED as exc:
    return False, evidence(target="parameter_blob", reason="manifest_malformed", error=str(exc))
blob = resolve_submitted_path(block.get("path"))
if blob is None or not os.path.exists(blob):
    return False, evidence(target="parameter_blob", reason="parameter_blob_missing",
                           path=block.get("path"))
counted = os.path.getsize(blob)
if counted != declared:
    return False, evidence(target="parameter_blob", reason="declared_bytes_disagree_with_blob",
                           declared=declared, counted=counted)
passed = counted <= PARAM_BYTE_CEILING
# The count is taken off the blob. A truthful declaration of an oversized export fails, and so
# does a compliant declaration standing beside a larger blob.
return passed, evidence(target="parameter_blob", serialized_bytes=counted,
                        ceiling=PARAM_BYTE_CEILING,
                        reason=None if passed else "PARAM_BYTES_EXCEEDED")
'''

BODIES["analytic_latency_recomputes_from_graph_terms"] = '''
try:
    block = manifest_section("analytic_latency")
    declared = float(block["estimate_ms"])
except MALFORMED as exc:
    return False, evidence(target="export_manifest", reason="manifest_malformed", error=str(exc))
try:
    rebuilt = analytic_latency_ms(block)
except MALFORMED as exc:
    return False, evidence(target="export_manifest", reason="latency_terms_incomplete",
                           error=str(exc))
if abs(rebuilt - declared) > LATENCY_RECOMPUTE_TOLERANCE_MS:
    return False, evidence(target="export_manifest", reason="latency_estimate_not_recomputable",
                           declared=declared, recomputed=rebuilt)
passed = rebuilt <= LATENCY_CEILING_MS
# Three integers off the registered graph and three pinned host constants. No timing loop, so
# nothing here can be tuned against the agent's own stopwatch.
return passed, evidence(target="export_manifest", estimate_ms=rebuilt,
                        ceiling_ms=LATENCY_CEILING_MS,
                        flops_batch1=block.get("flops_batch1"),
                        kernel_count=block.get("kernel_count"),
                        bytes_moved=block.get("bytes_moved"),
                        reason=None if passed else "LATENCY_BUDGET_EXCEEDED")
'''

BODIES["sized_against_the_pinned_host_constants"] = '''
pinned = require_measured("host_flops_per_second", "host_kernel_launch_seconds",
                          "host_hbm_bytes_per_second")
try:
    host = manifest_section("analytic_latency")["host_constants"]
except MALFORMED as exc:
    return False, evidence(target="export_manifest", reason="manifest_malformed", error=str(exc))
stale = []
for field, constant in HOST_CONSTANT_FIELDS:
    if float(host.get(field, 0.0)) != float(pinned[constant]):
        stale.append({"field": field, "declared": host.get(field), "pinned": pinned[constant]})
passed = not stale
# The three rotate within a disclosed band between attempts, so a kernel count sized against
# last attempt's launch cost overshoots without a line of the export changing.
return passed, evidence(target="export_manifest", stale_constants=stale,
                        reason=None if passed else "stale_host_constants")
'''

BODIES["activation_high_water_taken_at_batch_64"] = '''
try:
    block = manifest_section("activation")
    batch = int(block["batch"])
    peak = int(block["peak_live_bytes"])
    ledger = dict(block["live_bytes_by_stage"])
except MALFORMED as exc:
    return False, evidence(target="export_manifest", reason="manifest_malformed", error=str(exc))
if batch != GRADED_ACTIVATION_BATCH:
    return False, evidence(target="export_manifest", reason="activation_measured_at_wrong_batch",
                           batch=batch, graded_batch=GRADED_ACTIVATION_BATCH)
if block.get("recomputation") not in (None, False, "none"):
    # Gradient checkpointing during training measures a different schedule entirely; the graded
    # figure is a plain forward pass and can be far larger.
    return False, evidence(target="export_manifest",
                           reason="checkpointing_hides_the_high_water",
                           recomputation=block.get("recomputation"))
if not ledger:
    return False, evidence(target="export_manifest", reason="no_stage_ledger")
high = max(int(v) for v in ledger.values())
if high != peak:
    return False, evidence(target="export_manifest", reason="peak_disagrees_with_stage_ledger",
                           declared=peak, ledger_high=high, stages=len(ledger))
passed = peak <= ACTIVATION_CEILING_BYTES
return passed, evidence(target="export_manifest", peak_live_bytes=peak, stages=len(ledger),
                        ceiling=ACTIVATION_CEILING_BYTES,
                        reason=None if passed else "ACTIVATION_MEMORY_EXCEEDED")
'''

BODIES["accounted_flop_debit_within_the_fixed_budget"] = '''
accounting = run_record().get("training_accounting")
if not isinstance(accounting, dict):
    return False, evidence(target="run_record", reason="no_flop_accounting")
try:
    steps = int(accounting["steps"])
    batch = int(accounting["batch_size"])
    per_sample = float(accounting["flops_per_sample"])
    debited = float(accounting["accounted_flops"])
except MALFORMED as exc:
    return False, evidence(target="run_record", reason="flop_accounting_incomplete",
                           error=str(exc))
rebuilt = steps * batch * per_sample
if abs(rebuilt - debited) > 1e-6 * max(1.0, abs(debited)):
    return False, evidence(target="run_record", reason="debit_disagrees_with_step_counters",
                           declared=debited, recomputed=rebuilt, steps=steps, batch_size=batch)
passed = debited <= FLOP_BUDGET
# Work rather than seconds: the step count is fixed by construction, which is the precondition
# for weights that reproduce bit for bit.
return passed, evidence(target="run_record", accounted_flops=debited, budget=FLOP_BUDGET,
                        steps=steps, reason=None if passed else "FLOP_BUDGET_EXCEEDED")
'''

BODIES["training_stopped_on_the_flop_budget_not_the_watchdog"] = '''
record = run_record()
stop = record.get("termination")
exported_at = record.get("export_complete_at_step")
steps = (record.get("training_accounting") or {}).get("steps")
if stop is None or exported_at is None or steps is None:
    return False, evidence(target="run_record", reason="termination_unrecorded",
                           termination=stop, export_complete_at_step=exported_at, steps=steps)
if stop == "watchdog":
    return False, evidence(target="run_record", reason="WATCHDOG_TIMEOUT",
                           watchdog_wall_seconds=record.get("watchdog_wall_seconds"))
if int(exported_at) > int(steps):
    return False, evidence(target="run_record", reason="export_after_last_debited_step",
                           export_complete_at_step=int(exported_at), steps=int(steps))
passed = stop == "flop_budget_reached"
return passed, evidence(target="run_record", termination=stop,
                        export_complete_at_step=int(exported_at), steps=int(steps),
                        reason=None if passed else "unrecognised_termination")
'''

BODIES["supervision_digests_equal_the_shipped_shard_labels"] = '''
record = run_record()
traced = record.get("training_supervision_trace")
pinned = record.get("shard_label_record_digests")
if not isinstance(traced, list) or not isinstance(pinned, list):
    return False, evidence(target="train_supervision_trace",
                           reason="supervision_trace_absent")
if len(traced) != len(pinned):
    return False, evidence(target="train_supervision_trace", reason="EXTERNAL_SUPERVISION",
                           traced_records=len(traced), shard_records=len(pinned))
first_divergent = None
divergent = 0
for index in range(len(pinned)):
    if traced[index] != pinned[index]:
        divergent += 1
        if first_divergent is None:
            first_divergent = index
passed = divergent == 0
# A distilled student trains honestly in budget, exports small, clears every gate and survives
# the band permutation. The per-record digest is the only place the teacher shows.
return passed, evidence(target="train_supervision_trace", records=len(pinned),
                        divergent_records=divergent, first_divergent_record=first_divergent,
                        reason=None if passed else "EXTERNAL_SUPERVISION")
'''

BODIES["reward_moves_under_second_permutation_by_at_most_tau"] = '''
pair = run_record().get("permutation_regrade")
if not isinstance(pair, dict):
    return False, evidence(target="run_record", reason="permutation_regrade_absent")
try:
    under_a = float(pair["r_det_permutation_a"])
    under_b = float(pair["r_det_permutation_b"])
except MALFORMED as exc:
    return False, evidence(target="run_record", reason="permutation_regrade_incomplete",
                           error=str(exc))
moved = abs(under_b - under_a)
passed = moved <= TRANSFER_TAU
# Both permutations hit the shipped shard and both graded splits identically, so a model that
# read pixels is unaffected and one keyed to a fixed band order cannot be.
return passed, evidence(target="run_record", r_det_permutation_a=under_a,
                        r_det_permutation_b=under_b, moved=moved, tau=TRANSFER_TAU,
                        reason=None if passed else "PRECOMPUTED_PARAMS")
'''

BODIES["sibling_shard_gap_within_bound"] = '''
shards = run_record().get("in_distribution_shards")
if not isinstance(shards, dict) or len(shards) < 2:
    return False, evidence(target="run_record", reason="sibling_shard_unmeasured",
                           shards=sorted(shards) if isinstance(shards, dict) else None)
measured = {}
for name in sorted(shards):
    try:
        measured[name] = float(shards[name])
    except MALFORMED:
        return False, evidence(target="run_record", reason="shard_value_unreadable", shard=name)
gap = max(measured.values()) - min(measured.values())
passed = gap <= SHARD_GAP_TAU
# The sibling rotates among four pinned partitions, so a fit to the partition that gave
# feedback diverges on the one that did not.
return passed, evidence(target="run_record", shards=measured, gap=gap, tau=SHARD_GAP_TAU,
                        reason=None if passed else "SHARD_GAP_EXCEEDED")
'''

BODIES["reward_is_the_product_of_both_normalized_axes"] = '''
lines = reward_log()
if not lines:
    return False, evidence(target="run_record", reason="reward_log_empty")
last = lines[-1]
try:
    r_in = float(last["r_in"])
    r_shift = float(last["r_shift"])
    claimed = float(last["r_det"])
except MALFORMED as exc:
    return False, evidence(target="run_record", reason="reward_log_incomplete", error=str(exc))
for name, fraction in (("r_in", r_in), ("r_shift", r_shift)):
    if not 0.0 <= fraction <= 1.0:
        return False, evidence(target="run_record", reason="normalized_fraction_out_of_range",
                               field=name, value=fraction)
expected = product_reward(r_in, r_shift)
if abs(claimed - expected) > 1e-9:
    return False, evidence(target="run_record", reason="reward_is_not_the_product",
                           claimed=claimed, product=expected, r_in=r_in, r_shift=r_shift)
# The two axes multiply. An axis parked at its baseline pins the reward near 0.05 however far
# the other one is pushed, which is what makes the country split unskippable.
return True, evidence(target="run_record", r_in=r_in, r_shift=r_shift, r_det=claimed,
                      floor_when_one_axis_is_zero=product_reward(1.0, 0.0))
'''

BODIES["every_attempt_appears_in_the_reward_log"] = '''
declared = budget_declaration()
if declared is None:
    return False, evidence(target="run_record", reason="no_agent_declaration")
used = declared.get("attempts_used")
if not isinstance(used, int) or used < 1:
    return False, evidence(target="run_record", reason="attempts_used_not_declared",
                           attempts_used=used)
lines = reward_log()
seen = []
incomplete = []
for line in lines:
    index = line.get("attempt")
    if index is None:
        incomplete.append({"attempt": None, "missing_fields": ["attempt"]})
        continue
    seen.append(int(index))
    absent = [field for field in REWARD_LOG_FIELDS if field not in line]
    if absent:
        incomplete.append({"attempt": int(index), "missing_fields": absent})
expected = list(range(1, used + 1))
missing = [n for n in expected if seen.count(n) != 1]
unexpected = sorted(set(seen) - set(expected))
passed = not missing and not unexpected and not incomplete
# The attempts most worth leaving out are the ones a deployment gate zeroed, so the check is
# on the index sequence rather than on whether the log reads plausibly.
return passed, evidence(target="run_record", attempts_declared=used, logged=len(lines),
                        missing=missing, unexpected=unexpected, incomplete=incomplete,
                        reason=None if passed else "LOG_DIVERGENCE")
'''

BODIES["reason_codes_come_from_the_closed_zero_set"] = '''
invented = []
for line in reward_log():
    code = line.get("reason")
    if code in (None, "", "ok"):
        continue
    if code not in ZERO_REASONS:
        invented.append(code)
passed = not invented
# The refusal is loud and carries its own code precisely so a constraint violation cannot be
# reframed afterwards as a run that merely scored badly.
return passed, evidence(target="run_record", invented_reason_codes=sorted(set(invented)),
                        closed_set_size=len(ZERO_REASONS),
                        reason=None if passed else "reason_code_outside_closed_set")
'''

BODIES["export_operator_list_inside_the_bound_envelope"] = '''
try:
    block = manifest_section("operator_envelope")
    operators = list(block["graph_operators"])
except MALFORMED as exc:
    return False, evidence(target="export_manifest", reason="manifest_malformed", error=str(exc))
if block.get("source") != "recovered_from_export_graph":
    return False, evidence(target="export_manifest", reason="operator_list_hand_written",
                           source=block.get("source"))
outside = sorted(set(op for op in operators if op not in ENVELOPE_OPERATORS))
if outside:
    return False, evidence(target="export_manifest", reason="ENVELOPE_OP_REFUSED",
                           outside_envelope=outside, envelope_size=len(ENVELOPE_OPERATORS))
try:
    priced = set(manifest_section("analytic_latency")["per_operator_flops"])
except MALFORMED as exc:
    return False, evidence(target="export_manifest", reason="manifest_malformed", error=str(exc))
unpriced = sorted(set(operators) - priced)
passed = not unpriced
# Both remaining gates are analytic. An operator the cost model has no term for cannot be
# priced, so admitting one would silently remove the latency and activation gates together.
return passed, evidence(target="export_manifest", operators=len(set(operators)),
                        unpriced=unpriced,
                        reason=None if passed else "operator_has_no_analytic_term")
'''

BODIES["regrading_the_frozen_export_twice_agrees_bitwise"] = '''
record = run_record()
passes = record.get("regrade_passes")
if not isinstance(passes, list) or len(passes) < 2:
    return False, evidence(target="run_record", reason="second_grading_pass_absent",
                           passes=len(passes) if isinstance(passes, list) else None)
first, second = passes[0], passes[1]
divergent = []
for axis in ("map_in", "map_shift"):
    if repr(first.get(axis)) != repr(second.get(axis)):
        divergent.append({"axis": axis, "first": first.get(axis), "second": second.get(axis)})
if divergent:
    return False, evidence(target="run_record", reason="EXPORT_NONDETERMINISTIC",
                           divergent_axes=divergent)
order = record.get("average_precision_tie_break")
passed = order == TIE_BREAK
# Equal scores are broken by record index, so a tie never depends on sort stability and the
# second pass cannot land on a different last bit for a reason the export did not cause.
return passed, evidence(target="run_record", tie_break=order, required_tie_break=TIE_BREAK,
                        reason=None if passed else "tie_break_not_pinned")
'''

BODIES["checker_tree_and_sweep_archive_unchanged"] = '''
record = run_record()
ledger = record.get("digest_ledger")
if not isinstance(ledger, dict):
    return False, evidence(target="run_record", reason="digest_ledger_absent")
before = ledger.get("before") or {}
after = ledger.get("after") or {}
changed = sorted(path for path in set(before) | set(after)
                 if before.get(path) != after.get(path))
if changed:
    archive = [path for path in changed if path.startswith(ARCHIVE_ROOT)]
    return False, evidence(target="run_record",
                           reason="ARCHIVE_DIGEST_MISMATCH" if archive
                           else "GRADING_TREE_WRITE",
                           changed=changed, archive_paths=archive)
touched = sorted(record.get("guarded_paths_touched") or [])
passed = not touched
# The archive is pinned as tightly as the checkers because it is evidence the agent reasons
# over: 53 runs at one envelope rank the families opposite to the 7 at this one.
return passed, evidence(target="run_record", ledger_entries=len(before), touched=touched,
                        reason=None if passed else "GRADING_TREE_WRITE")
'''

BODIES["budget_declaration_names_the_family_it_chose"] = '''
declared = budget_declaration()
if declared is None:
    return False, evidence(target="run_record", reason="no_agent_declaration")
absent = [field for field in DECLARATION_FIELDS if field not in declared]
passed = not absent
# The expensive failure here is spending a campaign shrinking a family whose batch-1 cost is
# kernel count rather than parameters. An agent that never switched has to say so.
return passed, evidence(target="run_record", missing_fields=absent,
                        declared_fields=sorted(declared),
                        reason=None if passed else "declaration_incomplete")
'''

BODIES["macro_ap_product_ramp"] = '''
c = require_measured("floor", "knee", "dichotomizing_threshold")
band = require_measured("shift_floor", "shift_knee")
map_in, map_shift = verifier_measured_macro_ap()
r_in = normalized_fraction(map_in, c["floor"], c["knee"])
r_shift = normalized_fraction(map_shift, band["shift_floor"], band["shift_knee"])
value = product_reward(r_in, r_shift)
return value >= c["dichotomizing_threshold"], evidence(map_in=map_in, map_shift=map_shift,
                                                       r_in=r_in, r_shift=r_shift,
                                                       s_outcome=value)
'''


PRELUDE = '''"""Compiled checks for %(slot_id)s. GENERATED by solution/recompute.py. DO NOT HAND-EDIT.

Every check runs against delivered bytes and names no unmeasured quantity, except
sized_against_the_pinned_host_constants and macro_ap_product_ramp, which raise ConstantUnmeasured
while tests/constants.json carries nulls. The graded quantity is %(metric)s.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCORE_MATRIX = %(scores)r
EXPORT_MANIFEST = %(manifest)r
CHECKS = []


class ConstantUnmeasured(Exception):
    """Raised when a check cannot reach something it needs.

    Two cases, one claim. A ramp constant Phase 0 deliberately left null, and a pinned
    executor dependency this environment does not carry, both mean the submission was
    never examined. Reporting a failure would assert something about bytes the check
    did not read, and that is the one distinction the null-constant convention exists
    to keep.
    """


def check(id, weight, knockout, target):
    def register(fn):
        CHECKS.append({"id": id, "weight": weight, "knockout": knockout, "kind": "process",
                       "target": target, "fn": fn})
        return fn
    return register


def constants():
    with open(os.path.join(HERE, "constants.json")) as handle:
        return json.load(handle)


def require_measured(*names):
    known = constants()
    missing = [name for name in names if known.get(name) is None]
    if missing:
        raise ConstantUnmeasured(
            "%(slot_id)s: " + ", ".join(missing) + " are null. Phase 0 authors no measurement; "
            "the Phase 2 measurement wave writes these and this check becomes gradeable then.")
    return known


def evidence(**fields):
    return {"evaluation_target": fields.pop("target", "score_matrix"), "values": fields}
'''


TAIL = '''
# --- the facts this slot is about -----------------------------------------------------------
# 12 spectral bands in, 19 labels out. The two counts are never interchangeable and a matrix
# that is 12 wide is scoring the input channels rather than the nomenclature.
BAND_COUNT = 12
LABEL_COUNT = 19
# Average precision reads a ranking. A column with two distinct values has already been
# thresholded, and a handful of genuinely sparse labels may look that way, so the check allows
# four before it calls the matrix a decision matrix.
MIN_RANKED_COLUMNS = 15
# The three deployment constraints, as integers rather than as prose.
PARAM_BYTE_CEILING = 4194304
LATENCY_CEILING_MS = 1.5
ACTIVATION_CEILING_BYTES = 268435456
GRADED_ACTIVATION_BATCH = 64
LATENCY_RECOMPUTE_TOLERANCE_MS = 1e-6
# Training is bounded by accounted work, not by a clock.
FLOP_BUDGET = 2.304e17
# The two divergence bounds, both 0.03 and both measured against a rotation the agent cannot see.
TRANSFER_TAU = 0.03
SHARD_GAP_TAU = 0.03
# Sentinel-2 band dynamic ranges differ by more than an order of magnitude, so a real per-band
# scale vector spreads by at least this much and twelve copies of one number does not.
BAND_SPREAD_MINIMUM = 10.0
TIE_BREAK = "score_desc_then_record_index_asc"
ARCHIVE_ROOT = "environment/experiments"
HOST_CONSTANT_FIELDS = (("flops_per_second", "host_flops_per_second"),
                        ("kernel_launch_seconds", "host_kernel_launch_seconds"),
                        ("hbm_bytes_per_second", "host_hbm_bytes_per_second"))
DECLARATION_FIELDS = ("accounted_flops_spent", "attempts_used", "family_chosen",
                      "family_switch_attempt")
REWARD_LOG_FIELDS = ("activation_peak_bytes", "latency_estimate_ms", "map_in", "map_shift",
                     "param_bytes", "reason")
ENVELOPE_OPERATORS = frozenset((
    "add", "avgpool2d", "batchnorm2d", "clip", "concat", "conv2d", "depthwise_conv2d",
    "dequantize", "gemm", "globalavgpool2d", "hardswish", "mul", "quantize", "relu", "reshape",
    "sigmoid", "transpose"))
ZERO_REASONS = frozenset((
    "ACTIVATION_MEMORY_EXCEEDED", "ARCHIVE_DIGEST_MISMATCH", "ENVELOPE_OP_REFUSED",
    "EXPORT_NONDETERMINISTIC", "EXTERNAL_SUPERVISION", "FLOP_BUDGET_EXCEEDED",
    "GRADING_TREE_WRITE", "HELDOUT_DIGEST_MISMATCH", "LATENCY_BUDGET_EXCEEDED",
    "LOG_DIVERGENCE", "NO_SUBMISSION", "PARAM_BYTES_EXCEEDED", "PRECOMPUTED_PARAMS",
    "SHARD_GAP_EXCEEDED", "WATCHDOG_TIMEOUT"))


# Exceptions that mean the submission is malformed, kept narrow on purpose: a check that
# swallowed every exception would report ConstantUnmeasured as a failed submission, which is
# the collapse the null-constant convention exists to prevent.
MALFORMED = (AttributeError, IndexError, KeyError, OSError, TypeError, ValueError)


def array_backend():
    """numpy, or a refusal that is not a verdict.

    The graded executor pins numpy 1.26.4. A checker running somewhere that cannot import it has
    not read the score matrix at all, so it says the slot is not measurable here rather than
    saying the submission failed.
    """
    try:
        import numpy
    except ImportError as exc:
        raise ConstantUnmeasured("this executor carries no numpy, so the score matrix was not "
                                 "read: %s" % exc)
    return numpy


def load_score_matrix():
    """The graded object: one continuous score per label per patch.

    Not an argmax, because several of the 19 labels hold on the same patch, and not a decision,
    because average precision integrates over the ordering rather than reading one operating
    point. Rows are not required to sum to anything; the 19 columns are 19 independent rankings.
    """
    matrix = array_backend().load(SCORE_MATRIX, allow_pickle=False)
    if matrix.dtype == object:
        raise ValueError("an object array carries no scores")
    return matrix


def load_export_manifest():
    """The export describing itself: band statistics, quantization granularity, parameter bytes,
    the analytic latency terms, the batch-64 activation ledger, the operator list and the macro
    denominator. None of that fits inside an npy array, so it ships beside it."""
    with open(EXPORT_MANIFEST) as handle:
        doc = json.load(handle)
    if not isinstance(doc, dict):
        raise ValueError("the export manifest must be a JSON object")
    return doc


def manifest_section(name):
    doc = load_export_manifest()
    if name not in doc:
        raise KeyError("the export manifest carries no %s section" % name)
    return doc[name]


def resolve_submitted_path(declared):
    """A path inside the submission directory, resolved relative to the score matrix."""
    if not declared:
        return None
    if os.path.isabs(declared):
        return declared
    return os.path.join(os.path.dirname(SCORE_MATRIX), declared)


def analytic_latency_ms(block):
    """flops / rate + kernels * launch + bytes / bandwidth, in milliseconds.

    A rational function of three integers recovered by static analysis of the registered graph
    and three host constants. There is no timing loop anywhere in it, which is why the gate has
    no noise term and why an export tuned on a private stopwatch does not move it.
    """
    host = block["host_constants"]
    seconds = (float(block["flops_batch1"]) / float(host["flops_per_second"])
               + int(block["kernel_count"]) * float(host["kernel_launch_seconds"])
               + float(block["bytes_moved"]) / float(host["hbm_bytes_per_second"]))
    return seconds * 1000.0


def normalized_fraction(value, low, high):
    """One axis of the reward, clamped to its own band. The two axes have separate bands
    because the in-distribution split and the held-out-country split are different
    distributions; normalizing both on one band would assert they are the same."""
    span = high - low
    if span <= 0:
        return 0.0
    return max(0.0, min(1.0, (value - low) / span))


def product_reward(r_in, r_shift):
    """The two normalized axes MULTIPLY. Saturating both returns the literal 1.0; otherwise the
    floor is 0.05 and an axis at its baseline holds the whole reward there."""
    if r_in == 1.0 and r_shift == 1.0:
        return 1.0
    return 0.05 + 0.95 * r_in * r_shift


def run_record():
    path = os.path.join(HERE, "..", "run_record.json")
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def reward_log():
    """The campaign log, one line per attempt, read from the record the run itself produced."""
    lines = run_record().get("reward_log")
    return [line for line in lines if isinstance(line, dict)] if isinstance(lines, list) else []


def budget_declaration():
    """The agent's own budget accounting. The declaration is graded and the harness's own
    measurement is not, so no field the harness writes can stand in for it."""
    declared = run_record().get("agent_budget_declaration")
    if isinstance(declared, str):
        try:
            declared = json.loads(declared)
        except ValueError:
            return None
    return declared if isinstance(declared, dict) else None


def verifier_measured_macro_ap():
    """The two threshold-free macro average precisions, taken from the verifier's own executor
    over the frozen export bytes and never from the agent's reward log."""
    graded = run_record().get("verifier_measured_macro_ap")
    if not isinstance(graded, dict):
        raise ConstantUnmeasured(
            "the verifier recorded no macro average precision over this export")
    for axis in ("held_out_country", "in_distribution"):
        if graded.get(axis) is None:
            raise ConstantUnmeasured("the %s axis was not measured over this export" % axis)
    return float(graded["in_distribution"]), float(graded["held_out_country"])
'''


def main():
    grounding = load_grounding()
    problems = validate(grounding, BODIES)
    if problems:
        raise SystemExit("%s grounding is not well-formed: %s" % (SLOT_ID, "; ".join(problems)))

    deliverable = grounding["deliverable"]
    prelude = PRELUDE % {"slot_id": SLOT_ID, "metric": GRADED_QUANTITY,
                         "scores": deliverable["path"],
                         "manifest": deliverable["companion"]["path"]}

    tests_dir = os.path.join(BUNDLE, "tests")
    os.makedirs(tests_dir, exist_ok=True)

    # solution/TRUTH.md and solution/rubrics.json are PRIVATE_CARRIERS: seed/identity.py plants a
    # canary block into them at freeze, and canary normalisation is what keeps planting from
    # moving identity. A generator that rewrote them plainly would DROP the planted tripwire, and
    # the next content_hash would differ from the frozen one. So the planted block is carried
    # across the regeneration rather than recreated.
    truth_path = os.path.join(HERE, "TRUTH.md")
    truth = build_truth(grounding, TITLE).rstrip("\n") + "\n"
    planted = None
    if os.path.exists(truth_path):
        import re
        found = re.search(r"<!-- AELLO-CANARY-BLOCK.*?-->\n?", open(truth_path).read(), re.S)
        planted = found.group(0) if found else None
    with open(truth_path, "w") as handle:
        handle.write(truth + ("\n" + planted if planted else ""))

    private_rubrics = os.path.join(HERE, "rubrics.json")
    carrier = build_rubrics(grounding)
    if os.path.exists(private_rubrics):
        try:
            previous = json.load(open(private_rubrics))
            if "canary" in previous:
                carrier["canary"] = previous["canary"]
        except ValueError:
            pass
    with open(private_rubrics, "w") as handle:
        json.dump(carrier, handle, indent=1, sort_keys=True)
        handle.write("\n")
    with open(os.path.join(tests_dir, "rubrics.json"), "w") as handle:
        json.dump(build_rubrics(grounding), handle, indent=1, sort_keys=True)
        handle.write("\n")
    with open(os.path.join(tests_dir, "test_output.py"), "w") as handle:
        # The tail arrives FINAL and unformatted. It carries percent signs of its own inside the
        # emitted helpers, so running it through a second substitution raises TypeError. Only the
        # prelude is substituted.
        handle.write(build_checks(grounding, prelude, BODIES) + TAIL)
    with open(os.path.join(tests_dir, "constants.json"), "w") as handle:
        json.dump(grounding["constants"], handle, indent=1, sort_keys=True)
        handle.write("\n")
    # ORDER MATTERS. `with open(path, "w")` truncates BEFORE its body's arguments are evaluated,
    # so calling read_identity() inside the with-statement reads the file this line just emptied
    # and writes nulls over the frozen identity. Read first, then open.
    provenance = build_provenance(grounding, read_identity(HERE))
    with open(os.path.join(HERE, "provenance.yaml"), "w") as handle:
        # MATCH seed/build/screen_bind.py and the frozen generators exactly: sort_keys=True,
        # default_flow_style=False, width=100. The screen writes this carrier after the freeze,
        # so a generator using a different dump convention reorders the keys on the next
        # regeneration and reports as drift even though no value changed.
        yaml.safe_dump(provenance, handle, sort_keys=True, default_flow_style=False, width=100,
                       allow_unicode=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
