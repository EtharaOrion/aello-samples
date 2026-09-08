"""Regenerate every canonical artifact of AELLO-C8-S4 from solution/grounding.yaml.

WHAT THIS FILE OWNS, and nothing else owns:

    solution/TRUTH.md          the narrative, rendered from frozen literals
    solution/rubrics.json      the closed eight-key compiled item schema
    solution/provenance.yaml   the step-8i carrier (hash-excluded; identity read back, not authored)
    tests/rubrics.json         the judged council surface under the aello_eval v2 shape
    tests/rubrics.jsonl        the trajectory grader's line format, id and rubric only
    tests/test_output.py       the compiled checks, as executable relations
    tests/constants.json       the published constants, four of them deliberately null

WHAT IT DOES NOT OWN. tests/verifier.py, tests/budget.py, tests/isolation.py, tests/ladder.py and
tests/compose.py are hand-authored modules with their own reasoning, and regenerating them from a
string literal would move that reasoning into this file's quoting rather than into the modules. They
are checked for importability by --check and are never rewritten here.

DETERMINISM IS THE CONTRACT. No clock, no network, no locale, no random source, no set iteration
without a sort. Running this file twice produces byte-identical bytes; seed/build/freeze_batch.py
re-runs it after planting the canary and REFUSES the freeze if the tree hash moves.

CANARY PRESERVATION. seed/identity.py plants a tripwire block into five private carriers, two of
which this file rewrites: solution/TRUTH.md (an HTML comment block) and solution/rubrics.json (a
wrapper-level "canary" key). Both are read off the existing file BEFORE the new content is written
and re-attached to it. A generator that regenerated them without the block would delete the tripwire
and move the bundle hash, and the freeze would refuse.

READ BEFORE OPEN-FOR-WRITE. Every output below is computed in full before any output file is opened
for writing. The failure this avoids is concrete and has shipped in this project once already:

    with open(p, "w") as fh: yaml.safe_dump(build(read_identity(p)), fh)

truncates p, and read_identity then reads the file it just emptied and writes nulls over the frozen
identity. open() for write is the LAST thing that happens here, after every read.

RECOMPUTATION, not transcription. Every derived figure grounding.yaml publishes -- the five ablation
bounds, the per-row metric weights, the rare-class recall requirements, the leaf-budget arithmetic,
the starter moments -- is recomputed here from the raw 32-point ladder and the raw distributions and
compared against what grounding publishes. A disagreement fails the run. The point is that the
published numbers are checkable against their evidence rather than asserted beside it.
"""
import hashlib
import json
import math
import os
import statistics
import sys

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)
TESTS = os.path.join(BUNDLE, "tests")

# FORGE.md item 9g fixes the compiled item schema at exactly these eight keys. outcome_class is a
# ninth field in grounding and rides into the published item as a prefix of the judgment relation,
# because adding a ninth key is the defect touchstone ts-01 caught.
ITEM_KEYS = ("id", "dimension", "weight", "evaluation_target", "criterion", "judgment",
             "evidence", "mode")

# rubric-authoring-v2 section 11 field list for the judged surface, plus residue_reason, which
# CRUCIBLE's G-RUB-RESIDUE requires on every judged item. The conflict between a closed schema and
# a mandatory extra key is reported here rather than resolved by dropping one of them.
COUNCIL_KEYS = ("id", "criterion", "dimension", "mode", "grader", "weight", "knockout",
                "evaluation_target", "evidence", "judgment", "residue_reason")

LEVERS = ("L1", "L2", "L3", "L4", "L5")


# ---------------------------------------------------------------------------------------------
# INPUT
# ---------------------------------------------------------------------------------------------

def load_grounding():
    """Read the derivation source. The planted tripwire is not rubric content and is dropped."""
    import yaml
    with open(os.path.join(HERE, "grounding.yaml"), encoding="utf-8") as fh:
        g = yaml.safe_load(fh)
    g.pop("canary", None)
    return g


def ladder_by_cfg(g):
    return {row["cfg"]: row for row in g["ablation_ladder"]}


# ---------------------------------------------------------------------------------------------
# RECOMPUTATION OF EVERY PUBLISHED DERIVED FIGURE
# ---------------------------------------------------------------------------------------------

def recompute_ablation(g):
    """Re-derive the five contract ablation measures from the raw 32-point ladder.

    Definitions are seed/contract.yaml's, not this bundle's:
      standalone      (score with lever i alone - floor) / (reference - floor)
      leave_one_out   (reference - score with lever i removed) / (reference - floor)
      reference share (reference - floor) / (best - floor)
      effective       inverse Herfindahl over each lever's share of total POSITIVE marginal worth
      breadth         fewest levers whose best combination reaches 90 percent of the floor-to-best span

    The denominator is floor-to-REFERENCE throughout. It is never floor-to-knee: a knee-anchored
    denominator is chosen by the slot author, which makes every fraction it produces either vacuous
    or self-referential, and this slot's knee is null in any case.
    """
    rows = ladder_by_cfg(g)
    floor = rows["00000"]["bacc"]
    reference = rows["11111"]["bacc"]
    best = max(r["bacc"] for r in g["ablation_ladder"])
    best_cfg = min(sorted(r["cfg"] for r in g["ablation_ladder"] if r["bacc"] == best))
    den = reference - floor

    standalone, loo = {}, {}
    for i, name in enumerate(LEVERS):
        alone = ["0"] * 5
        alone[i] = "1"
        without = ["1"] * 5
        without[i] = "0"
        standalone[name] = round((rows["".join(alone)]["bacc"] - floor) / den, 5)
        loo[name] = round((reference - rows["".join(without)]["bacc"]) / den, 5)

    positive = [v for v in loo.values() if v > 0]
    total = sum(positive)
    effective = round(1.0 / sum((v / total) ** 2 for v in positive), 5) if total else 0.0

    span = best - floor
    best_at_k, breadth = {}, None
    for k in range(1, 6):
        top = max(r["bacc"] for r in g["ablation_ladder"] if r["cfg"].count("1") == k)
        best_at_k[k] = round(top, 5)
        if breadth is None and (top - floor) / span >= 0.90:
            breadth = k

    return {
        "ladder_floor": round(floor, 5),
        "ladder_reference": round(reference, 5),
        "ladder_best": round(best, 5),
        "best_cfg": best_cfg,
        "denominator": round(den, 5),
        "standalone_fractions": standalone,
        "max_standalone": round(max(standalone.values()), 5),
        "max_standalone_lever": max(standalone, key=lambda k: standalone[k]),
        "leave_one_out_fractions": loo,
        "max_leave_one_out": round(max(loo.values()), 5),
        "reference_share_of_best": round(den / span, 5),
        "effective_levers": effective,
        "levers_to_90pct_of_best": breadth,
        "best_at_k": best_at_k,
    }


def recompute_bound_verdicts(g, ab):
    """Which of the five contract bounds this slot passes, computed rather than declared."""
    b = g["ablation_bounds"]
    return {
        "max_single_ablation_verdict":
            "pass" if ab["max_standalone"] <= b["max_single_ablation_fraction_contract"] else "fail",
        "max_marginal_ablation_verdict":
            "pass" if ab["max_leave_one_out"] <= b["max_marginal_ablation_fraction_contract"] else "fail",
        "reference_share_verdict":
            "pass" if ab["reference_share_of_best"] >= b["min_reference_share_contract"] else "fail",
        "effective_levers_verdict":
            "pass" if ab["effective_levers"] >= b["min_effective_levers_contract"] else "fail",
        "levers_to_90pct_verdict":
            "pass" if ab["levers_to_90pct_of_best"] >= b["min_levers_to_90pct_contract"] else "fail",
    }


def recompute_leverage(g, ab):
    """Per-row metric weight, the prevalence ratio, and the rare-class recall the anchors demand.

    Balanced accuracy is the mean of k per-class recalls, so a row of class c carries weight
    1/(k*count[c]). Everything else in this block is arithmetic on that one identity.
    """
    counts = g["graded_fold"]["class_counts"]
    k = len(counts)
    weight = {"class_%d" % i: round(1.0 / (k * n), 10) for i, n in enumerate(counts)}
    rare, common = min(counts), max(counts)
    out = {
        "per_row_weight": weight,
        "leverage_rare_over_common": round((1.0 / rare) / (1.0 / common), 4),
        "prevalence_ratio": round(common / rare, 4),
        "chance": 1.0 / k,
        "six_of_seven_ceiling": (k - 1) / k,
        "smallest_representable_step": round(1.0 / (k * common), 11),
        "largest_single_row_step": round(1.0 / (k * rare), 10),
    }
    rare_idx = counts.index(rare)
    for label, anchor, dp in (("reference", ab["ladder_reference"], 5),
                              ("best", ab["ladder_best"], 6)):
        # with the other k-1 recalls at 1.0, the rarest class must still carry k*anchor-(k-1)
        need = k * anchor - (k - 1)
        out["class_%d_recall_to_reach_%s" % (rare_idx, label)] = round(need, dp)
        out["class_%d_rows_to_reach_%s" % (rare_idx, label)] = math.ceil(need * rare)
    return out


def recompute_budget(g):
    """The leaf spend of each depth arm, from the ladder's own budget rule.

    boosting_rounds = leaf_budget_per_class_series // num_leaves, seven one-vs-rest trees per round.
    A tree of L leaves consumed L-1 splits, so the split count trails the leaf count by the tree
    count and the two are not interchangeable.
    """
    per_series = g["measurement_provenance"]["leaf_budget_per_class_series"]
    cap = g["split_budget"]["cap"]
    trees_per_round = 7
    out = {}
    for leaves in (31, 63):
        rounds = per_series // leaves
        trees = rounds * trees_per_round
        out["leaves_%d" % leaves] = {
            "num_leaves": leaves,
            "boosting_rounds": rounds,
            "trees_per_round": trees_per_round,
            "total_trees": trees,
            "total_leaves": trees * leaves,
            "total_splits": trees * (leaves - 1),
            "utilisation": round(trees * leaves / cap, 5),
        }
    out["cap_from_series"] = per_series * trees_per_round
    return out


def recompute_distributions(g):
    """Moments of the two measured starter distributions, and the planted defect's worth."""
    def moments(prefix, values):
        return {
            prefix + "_mean": round(statistics.fmean(values), 5),
            prefix + "_sd": round(statistics.stdev(values), 5),
            prefix + "_max": round(max(values), 5),
            prefix + "_n": len(values),
        }
    out = {}
    out.update(moments("starter", g["starter_distribution"]))
    out.update(moments("repaired_starter", g["repaired_starter_distribution"]))
    out["planted_defect_worth"] = round(
        statistics.fmean(g["repaired_starter_distribution"])
        - statistics.fmean(g["starter_distribution"]), 5)
    return out


def disagreements(g):
    """Every published figure that its own recomputation does not reproduce.

    Returns a sorted list of (dotted path, published, recomputed). Empty is the only passing
    result; this function is what makes the numbers in grounding.yaml evidence rather than claims.
    """
    ab = recompute_ablation(g)
    out = []

    def cmp(path, published, computed):
        if published != computed:
            out.append([path, published, computed])

    anchors = g["ladder_anchors"]
    for key in ("ladder_floor", "ladder_reference", "ladder_best", "denominator"):
        cmp("ladder_anchors.%s" % key, anchors[key], ab[key])
    cmp("ladder_anchors.best_cfg", anchors["best_cfg"], ab["best_cfg"])

    bounds = g["ablation_bounds"]
    for key in ("standalone_fractions", "max_standalone", "max_standalone_lever",
                "leave_one_out_fractions", "max_leave_one_out", "reference_share_of_best",
                "effective_levers", "levers_to_90pct_of_best", "best_at_k"):
        cmp("ablation_bounds.%s" % key, bounds[key], ab[key])
    for key, value in sorted(recompute_bound_verdicts(g, ab).items()):
        cmp("ablation_bounds.%s" % key, bounds[key], value)
    cmp("ablation_bounds.bounds_failed", bounds["bounds_failed"],
        sum(1 for v in recompute_bound_verdicts(g, ab).values() if v == "fail"))
    cmp("ablation_bounds.bounds_passed", bounds["bounds_passed"],
        sum(1 for v in recompute_bound_verdicts(g, ab).values() if v == "pass"))

    lev = recompute_leverage(g, ab)
    ml = g["metric_leverage"]
    cmp("metric_leverage.per_row_weight", ml["per_row_weight"], lev["per_row_weight"])
    for key in ("leverage_rare_over_common", "chance", "six_of_seven_ceiling",
                "smallest_representable_step", "largest_single_row_step",
                "class_3_recall_to_reach_reference", "class_3_rows_to_reach_reference",
                "class_3_recall_to_reach_best", "class_3_rows_to_reach_best"):
        cmp("metric_leverage.%s" % key, ml[key], lev[key])
    cmp("graded_fold.prevalence_ratio", g["graded_fold"]["prevalence_ratio"],
        lev["prevalence_ratio"])
    cmp("graded_fold.fold_sum", g["graded_fold"]["fold_sum"],
        g["graded_fold"]["train_rows"] + g["graded_fold"]["val_public_rows"]
        + g["graded_fold"]["rows"])
    cmp("graded_fold.rows", g["graded_fold"]["rows"], sum(g["graded_fold"]["class_counts"]))

    bud = recompute_budget(g)
    for arm in ("leaves_31", "leaves_63"):
        cmp("split_budget.ladder_consumption.%s" % arm,
            g["split_budget"]["ladder_consumption"][arm], bud[arm])
    cmp("split_budget.cap", g["split_budget"]["cap"], bud["cap_from_series"])

    dist = recompute_distributions(g)
    for key in ("starter_mean", "starter_sd", "starter_max", "starter_n",
                "repaired_starter_mean", "repaired_starter_sd", "repaired_starter_max",
                "repaired_starter_n"):
        cmp(key, g[key], dist[key])
    cmp("planted_defect.worth", g["planted_defect"]["worth"], dist["planted_defect_worth"])
    cmp("planted_defect.verified_to_carry", g["planted_defect"]["verified_to_carry"],
        dist["repaired_starter_mean"])
    cmp("planted_defect.against_reference", g["planted_defect"]["against_reference"],
        ab["ladder_reference"])
    cmp("reference_distribution", g["reference_distribution"], [ab["ladder_reference"]])

    weights = sum(i["weight"] for i in g["items"])
    compiled = sum(i["weight"] for i in g["items"] if i["mode"] == "compiled")
    cmp("compilation_floor", g["compilation_floor"], round(compiled / weights, 6))
    return sorted(out)


# ---------------------------------------------------------------------------------------------
# OUTPUT: solution/rubrics.json
# ---------------------------------------------------------------------------------------------

def build_compiled_rubrics(g):
    items = []
    for raw in g["items"]:
        item = {}
        for key in ITEM_KEYS:
            if key == "evidence":
                item[key] = "%s; recorded by tests/test_output.py check %s" % (
                    raw["evaluation_target"], raw["id"])
            elif key == "judgment":
                item[key] = "%s: %s" % (raw["outcome_class"], raw["judgment"].strip())
            else:
                item[key] = raw[key]
        items.append(item)
    total = sum(i["weight"] for i in g["items"])
    compiled = sum(i["weight"] for i in g["items"] if i["mode"] == "compiled")
    return {
        "$schema": "forge.rubric/v1",
        "banner": "GENERATED SECTION. DO NOT HAND-EDIT.",
        "generator": "solution/recompute.py",
        "slot_id": g["slot_id"],
        "compilation_floor": g["compilation_floor"],
        "compiled_weight_share": round(compiled / total, 6) if total else 0.0,
        "knockout_ids": sorted(i["id"] for i in g["items"] if i["knockout"]),
        "total_weight": total,
        "evaluation_target_vocabulary": sorted(g["evaluation_target_vocabulary"]),
        "items": sorted(items, key=lambda i: i["id"]),
    }


# ---------------------------------------------------------------------------------------------
# OUTPUT: tests/rubrics.json and tests/rubrics.jsonl
# ---------------------------------------------------------------------------------------------

def build_council(g):
    rubrics = [{k: r[k] for k in COUNCIL_KEYS} for r in g["council_rubrics"]]
    gate = g["reward_gate"]
    return {
        "schema": "aello_eval.rubric_set/v2_compiled_and_judged",
        "bundle_id": g["slot_id"],
        "floor": g["scoring"]["rubric_floor"],
        "compilation_floor": g["compilation_floor"],
        "rubrics": sorted(rubrics, key=lambda r: r["id"]),
        "reward_gate": {
            "applies_to": gate["applies_to"],
            "ceiling": gate["ceiling"],
            "form": gate["form"],
            "composition": gate["composition"],
            "coverage_rule": gate["coverage_rule"],
            "pass_threshold": gate["pass_threshold"],
            "enumerated_ids": sorted(r["id"] for r in rubrics),
        },
        "note": ("Judged trajectory rubrics for %s. The compiled checks live in "
                 "tests/test_output.py and are registered in solution/rubrics.json under FORGE.md "
                 "item 9g; per rubric-authoring-v2 section 9 they do not appear here. Every item "
                 "below is graded after the deterministic reward, composes through "
                 "tests/compose.py, and can only lower it." % g["slot_id"]),
    }


def build_council_lines(council):
    return "".join(json.dumps({"id": r["id"], "rubric": " ".join(r["criterion"].split())},
                              sort_keys=True) + "\n" for r in council["rubrics"])


# ---------------------------------------------------------------------------------------------
# OUTPUT: tests/constants.json
# ---------------------------------------------------------------------------------------------

def build_constants(g):
    """Published constants. Four are null and stay null.

    The ramp constants -- floor, knee, dichotomizing_threshold, reward_gate_pass_threshold -- are
    measured at Phase 1 on the grading host against the private fold. They ship beside the measured
    LADDER anchors, which are a different quantity measured on a different question, and the two
    are named apart so that no reader has to infer which one a bare word floor meant.
    """
    c = g["constants"]
    anchors = g["ladder_anchors"]
    return {
        "slot_id": g["slot_id"],
        "banner": "GENERATED SECTION. DO NOT HAND-EDIT.",
        "generator": "solution/recompute.py from solution/grounding.yaml",
        "floor": c["floor"],
        "knee": c["knee"],
        "dichotomizing_threshold": c["dichotomizing_threshold"],
        "reward_gate_pass_threshold": c["reward_gate_pass_threshold"],
        "measured_values_reason": c["measured_values_reason"],
        "gradeable": g["gradeable"]["value"],
        "gradeable_basis": g["gradeable"]["basis"],
        "reward_gate_form": g["reward_gate"]["form"],
        "reward_gate_ceiling": g["reward_gate"]["ceiling"],
        "graded_rows": g["graded_fold"]["rows"],
        "graded_classes": g["graded_fold"]["classes"],
        "graded_class_counts": g["graded_fold"]["class_counts"],
        "val_public_rows": g["graded_fold"]["val_public_rows"],
        "label_domain": [0, g["graded_fold"]["classes"] - 1],
        "submission_path": g["deliverable"]["path"],
        "run_record_path": g["run_record_contract"]["path"],
        "split_budget_cap": g["split_budget"]["cap"],
        "split_budget_quantity": g["split_budget"]["quantity"],
        "ladder_floor": anchors["ladder_floor"],
        "ladder_reference": anchors["ladder_reference"],
        "ladder_best": anchors["ladder_best"],
        "ladder_anchors_are_not_the_ramp": True,
        "chance": round(1.0 / g["graded_fold"]["classes"], 10),
        "starter_max": g["starter_max"],
    }


# ---------------------------------------------------------------------------------------------
# OUTPUT: tests/test_output.py
# ---------------------------------------------------------------------------------------------

PRELUDE = '''"""Compiled checks for AELLO-C8-S4. GENERATED by solution/recompute.py. DO NOT HAND-EDIT.

This module ships inside the verifier tree and carries NO criterion prose and no reference text --
only the executable relation each item implies. Every check runs against delivered bytes with numpy
and the standard library, on CPU, with no network and no vendored scoring engine.

Six of the seven checks return a verdict with no constant at all. The seventh, the outcome check,
needs a ramp this slot has not measured; it computes and records the seven per-class recalls, then
raises ConstantUnmeasured. The verifier turns that into r_det null -- an assertion that the SLOT is
not yet measurable, which is a different claim from an assertion that the agent scored zero.
"""
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    # budget.py, isolation.py and ladder.py are siblings in this tree. The verifier already puts
    # HERE on the path; this line makes the module importable on its own as well, which is what
    # any smoke test that does not go through the verifier will do.
    sys.path.insert(0, HERE)
CHECKS = []


class ConstantUnmeasured(Exception):
    """A check needs a ramp constant that Phase 0 deliberately left null."""


def check(id, weight, knockout, dimension, target):
    def register(fn):
        CHECKS.append({"id": id, "weight": weight, "knockout": knockout,
                       "dimension": dimension, "target": target, "fn": fn})
        return fn
    return register


def constants():
    with open(os.path.join(HERE, "constants.json"), encoding="utf-8") as fh:
        return json.load(fh)


def require_measured(*names):
    c = constants()
    missing = [n for n in names if c.get(n) is None]
    if missing:
        raise ConstantUnmeasured(
            "AELLO-C8-S4: %s null at Phase 0. %s"
            % (", ".join(missing), c.get("measured_values_reason", "")))
    return c


def evidence(target, **values):
    return {"evaluation_target": target, "values": values}


def submission_path():
    return constants()["submission_path"]


def run_record():
    """The agent-authored record beside the submission. Absent is a fact, not an exception."""
    path = constants()["run_record_path"]
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def load_labels():
    """The submitted array, loaded with pickling refused.

    allow_pickle=False is not decoration. A .npy file may carry a pickled Python object, and
    loading one executes it inside the verifier container; refusing it is the only reason this
    function exists rather than a bare np.load at each call site.
    """
    import numpy as np
    return np.load(submission_path(), allow_pickle=False)


def per_class_recall():
    """The seven recalls, verifier-side, in class order. The private labels are read through
    tests/isolation.py so that exactly one path literal in this tree reaches them."""
    import numpy as np
    import isolation
    truth = isolation.graded_labels()
    pred = np.asarray(load_labels())
    k = constants()["graded_classes"]
    return [round(float((pred[truth == c] == c).mean()), 10) if (truth == c).any() else None
            for c in range(k)]


def balanced_accuracy():
    present = [r for r in per_class_recall() if r is not None]
    return float(sum(present) / len(present))
'''

BODIES = {
    "label_vector_lands": '''    path = submission_path()
    if not os.path.exists(path):
        return False, evidence("submitted_artifact", reason="submission_missing", path=path)
    try:
        arr = load_labels()
    except Exception as exc:
        return False, evidence("submitted_artifact", reason="submission_unloadable",
                               error="%s: %s" % (type(exc).__name__, exc))
    if arr.dtype == object:
        return False, evidence("submitted_artifact", reason="submission_unloadable",
                               dtype="object")
    return True, evidence("submitted_artifact", path=path, dtype=str(arr.dtype),
                          shape=list(arr.shape),
                          sha256=hashlib.sha256(open(path, "rb").read()).hexdigest())''',

    "graded_row_alignment": '''    c = constants()
    try:
        arr = load_labels()
    except Exception as exc:
        return False, evidence("submitted_artifact", reason="submission_unloadable",
                               error="%s: %s" % (type(exc).__name__, exc))
    expected, public = c["graded_rows"], c["val_public_rows"]
    if arr.ndim != 1:
        return False, evidence("submitted_artifact", reason="not_one_dimensional",
                               shape=list(arr.shape), expected_rows=expected)
    n = int(arr.shape[0])
    if n == public:
        return False, evidence("submitted_artifact", reason="wrong_fold_length", rows=n,
                               expected_rows=expected, val_public_rows=public)
    if n != expected:
        return False, evidence("submitted_artifact", reason="row_count_mismatch", rows=n,
                               expected_rows=expected)
    return True, evidence("submitted_artifact", rows=n, expected_rows=expected)''',

    "cover_type_domain": '''    c = constants()
    try:
        arr = load_labels()
    except Exception as exc:
        return False, evidence("submitted_artifact", reason="submission_unloadable",
                               error="%s: %s" % (type(exc).__name__, exc))
    low, high = c["label_domain"]
    if arr.dtype.kind not in ("i", "u"):
        return False, evidence("submitted_artifact", reason="label_domain_violation",
                               dtype=str(arr.dtype), detail="non_integer_dtype")
    if arr.size == 0:
        return False, evidence("submitted_artifact", reason="label_domain_violation",
                               detail="empty_array")
    lo, hi = int(arr.min()), int(arr.max())
    if lo < low or hi > high:
        return False, evidence("submitted_artifact", reason="label_domain_violation",
                               observed=[lo, hi], allowed=[low, high])
    import numpy as np
    present = sorted(int(v) for v in np.unique(arr))
    return True, evidence("submitted_artifact", observed=[lo, hi], allowed=[low, high],
                          distinct_labels=present, classes_never_predicted=sorted(
                              set(range(low, high + 1)) - set(present)))''',

    "split_budget_declared": '''    import budget
    rec = run_record()
    if rec is None:
        return False, evidence("run_record", reason="run_record_missing",
                               path=constants()["run_record_path"])
    verdict = budget.audit(rec.get("split_budget_declaration"), constants()["split_budget_cap"])
    return verdict["ok"], evidence("run_record", **verdict)''',

    "seed_replay_identity": '''    rec = run_record()
    path = submission_path()
    if not os.path.exists(path):
        return False, evidence("submitted_source", reason="submission_missing")
    delivered = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if not rec:
        return False, evidence("submitted_source", reason="replay_unavailable",
                               delivered=delivered)
    claimed = rec.get("replay_sha256")
    if not claimed:
        return False, evidence("submitted_source", reason="replay_unavailable",
                               delivered=delivered, seed=rec.get("seed"))
    ok = str(claimed) == delivered
    return ok, evidence("submitted_source", delivered=delivered, replay=str(claimed),
                        seed=rec.get("seed"),
                        reason=None if ok else "replay_diverged")''',

    "private_fold_boundary": '''    import isolation
    rec = run_record()
    if rec is None:
        return False, evidence("submitted_source", reason="run_record_missing")
    declared = [str(p) for p in (rec.get("guarded_paths_touched") or [])]
    crossed = sorted(p for p in declared if isolation.is_guarded(p))
    tree = isolation.audit_tree()
    ok = not crossed and tree["ok"]
    return ok, evidence("submitted_source", declared_touched=sorted(declared),
                        red_lines_crossed=crossed, tree_audit=tree,
                        reason=None if ok else "red_line_crossed")''',

    "balanced_accuracy_ramp": '''    # ORDER MATTERS. require_measured comes FIRST. If the ramp constants are null this slot is
    # not yet measurable, and reading the submission before saying so lets an unrelated failure --
    # a missing artifact raising FileNotFoundError -- mask the unmeasured signal, so the verifier
    # reports gradeable=true with r_det=0.0. That collapses "not yet measurable" into "the agent
    # scored zero", which is the one distinction the null-constant convention exists to keep.
    c = require_measured("floor", "knee", "dichotomizing_threshold")
    recalls = per_class_recall()
    measured = balanced_accuracy()
    span = c["knee"] - c["floor"]
    s = 0.0 if span <= 0 else max(0.0, min(1.0, (measured - c["floor"]) / span))
    return measured >= c["dichotomizing_threshold"], evidence(
        "submitted_artifact", balanced_accuracy=measured, per_class_recall=recalls,
        s_outcome=s)''',
}


def build_test_module(g):
    parts = [PRELUDE]
    for raw in sorted(g["items"], key=lambda i: i["id"]):
        if raw["mode"] != "compiled":
            continue
        parts.append('\n\n@check(id="%s", weight=%d, knockout=%s, dimension="%s", target="%s")\n'
                     'def check_%s():\n%s\n'
                     % (raw["id"], raw["weight"], raw["knockout"], raw["dimension"],
                        raw["evaluation_target"], raw["id"], BODIES[raw["id"]]))
    parts.append('\n\nKNOCKOUT_IDS = %r\nTOTAL_WEIGHT = %d\n'
                 % (sorted(i["id"] for i in g["items"] if i["knockout"]),
                    sum(i["weight"] for i in g["items"])))
    return "".join(parts)


# ---------------------------------------------------------------------------------------------
# OUTPUT: solution/TRUTH.md
# ---------------------------------------------------------------------------------------------

def flow(text):
    """Collapse a folded YAML scalar to one line. Rendering must not depend on where the source
    file happened to wrap."""
    return " ".join(str(text).split())


def build_annex(g):
    """FORGE.md item 10e: the ordered path through instruction.md with the checker identifier each
    step satisfies, and every rejected route bound to a control that measures it wrong."""
    lines = ["Ordered path through instruction.md, with the checker identifier each step satisfies:",
             ""]
    for n, item in enumerate(sorted(g["items"], key=lambda i: i["id"]), 1):
        lines.append("%d. `%s` (%s, weight %d%s) -- %s"
                     % (n, item["id"], item["mode"], item["weight"],
                        ", knockout" if item["knockout"] else "", flow(item["criterion"])))
    lines += ["", "Judged surface, graded downstream and composing through the gate in "
                  "tests/compose.py:", ""]
    for n, r in enumerate(sorted(g["council_rubrics"], key=lambda r: r["id"]), 1):
        lines.append("%d. `%s` (judged, weight %d) -- %s"
                     % (n, r["id"], r["weight"], flow(r["criterion"])))
    lines += ["", "Rejected routes, each bound to the control that measures it wrong:", ""]
    for key in sorted(g["shortcut_controls"]):
        sc = g["shortcut_controls"][key]
        lines.append("- %s -- controlled by: %s" % (flow(sc["risk"]), flow(sc["control"])))
    lines += ["", "Deliverable manifest:", "",
              "- `%s` (%s) -- %s" % (g["deliverable"]["path"], g["deliverable"]["kind"],
                                     flow(g["deliverable"]["shape"])),
              "- `%s` (json) -- %s" % (g["run_record_contract"]["path"],
                                       ", ".join(g["run_record_contract"]["required_keys"])),
              "", "Control obligations, with the score each one actually carries:", ""]
    for name in sorted(g["controls"]):
        if name == "measurement_note":
            continue
        ctl = g["controls"][name]
        measured = ctl.get("measured")
        if measured is None:
            score = "UNMEASURED -- %s" % flow(ctl["measured_reason"])
        elif len(measured) == 1:
            score = "measured %.5f (n=1)" % measured[0]
        else:
            score = ("measured mean %.5f over n=%d, max %.5f"
                     % (statistics.fmean(measured), len(measured), max(measured)))
        lines.append("- `%s` targets %s; %s." % (name, flow(ctl["targets"]), score))
    lines += ["", "Contract ablation bounds, recomputed from the 32-point ladder by "
                  "tests/ladder.py:", ""]
    b = g["ablation_bounds"]
    for label, got, want, verdict in (
            ("max standalone fraction", b["max_standalone"],
             b["max_single_ablation_fraction_contract"], b["max_single_ablation_verdict"]),
            ("max leave-one-out fraction", b["max_leave_one_out"],
             b["max_marginal_ablation_fraction_contract"], b["max_marginal_ablation_verdict"]),
            ("reference share of best", b["reference_share_of_best"],
             b["min_reference_share_contract"], b["reference_share_verdict"]),
            ("effective levers", b["effective_levers"],
             b["min_effective_levers_contract"], b["effective_levers_verdict"]),
            ("levers to 90 percent of best", b["levers_to_90pct_of_best"],
             b["min_levers_to_90pct_contract"], b["levers_to_90pct_verdict"])):
        lines.append("- %s: measured %s against contract %s -- **%s**"
                     % (label, got, want, verdict.upper()))
    return "\n".join(lines)


def build_truth(g):
    """Render solution/TRUTH.md from frozen literals.

    Body: the four elements standards/truth-md-authoring-v1.md section 3 requires, in order -- an
    unheaded opening, the single most important insight, the ideal solve as numbered intent-named
    steps, and the traps bullets, which are the only bullets in the body.

    Annex: the material FORGE.md item 10e requires and section 3 has no section for. Kept below the
    body and marked as a recorded deviation rather than dropped, because discarding evidence to fit
    a section list is the wrong way to resolve a collision between two standards.
    """
    n = g["truth_narrative"]
    out = ["# AELLO-C8-S4 -- what this task actually tests", "",
           "GENERATED SECTION. DO NOT HAND-EDIT.", "", flow(n["opening"]), "",
           "## The single most important insight", "", flow(n["insight"]), "",
           "## The ideal solve, step by step", ""]
    for i, step in enumerate(n["steps"], 1):
        out += ["%d. **%s.** %s" % (i, flow(step["heading"]), flow(step["paragraph"])), ""]
    out += ["## Traps that catch agents that are not thinking carefully", ""]
    out += ["- " + flow(t) for t in n["traps"]]
    out += ["", "---", "",
            "## Contract record (annex; not part of the narrative body)", "",
            "Retained under FORGE.md item 10e and Phase 2 item 7, which require this file to carry "
            "the ordered path through instruction.md with each satisfied checker identifier, and "
            "each rejected route bound to a measured known-wrong control, and to reconcile with "
            "the checker set and the deliverable manifest by identifier set equality in both "
            "directions. standards/truth-md-authoring-v1.md section 3 admits no section beyond "
            "the four above, so this annex is a recorded deviation rather than an omission, and "
            "the section 4 word count is measured over the narrative body alone.", "",
            build_annex(g), ""]
    return "\n".join(out).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------------------------
# OUTPUT: solution/provenance.yaml  (hash-excluded carrier; identity is an INPUT)
# ---------------------------------------------------------------------------------------------

def read_carrier():
    """Read the existing carrier before anything is written.

    The identity block and the screening result are both MEASURED OVER the frozen tree by
    instruments outside this file -- seed/build/freeze_batch.py and seed/build/screen_bind.py.
    solution/provenance.yaml is excluded from the canonical hash precisely so that a value measured
    over the frozen bytes can be bound into them without moving them, which is what keeps the
    construction acyclic. Reading them back here is the only way regeneration can preserve them.
    """
    import yaml
    path = os.path.join(HERE, "provenance.yaml")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_provenance(g, existing):
    """The step-8i carrier: CLOSED SCHEMA, zero free-form-prose scalars.

    audit/g_con.py classifies a scalar as free-form prose when it carries at least eight words and
    ends a sentence. seed/build/provenance_split.py relocates any such scalar into
    provenance_notes.md -- which IS inside the canonical hash -- so a sentence written here would
    silently rewrite a hash-covered file and move the bundle's uuid after the freeze. Every reason
    this bundle owes therefore lives in solution/grounding.yaml, in solution/provenance_notes.md,
    or in a comment like this one. Never as a scalar in the returned mapping.
    """
    p = g["provenance"]
    carrier = {
        "banner": "GENERATED SECTION. DO NOT HAND-EDIT.",
        "generator": "solution/recompute.py from solution/grounding.yaml",
        "schema_version": "1.0",
        "slot_id": g["slot_id"],
        "identity": existing.get("identity", {
            "canonical_content_hash": None,
            "uuid": None,
            "normalization_domain": "aello.canary.norm/v1",
            "derivation": "uuid5(FORGE_TASK_NAMESPACE, canonical_content_hash)",
        }),
        "corpus": p["corpus"],
        "corpus_folds": p["corpus_folds"],
        "corpus_digests": p["corpus_digests"],
        "anchors": p["anchors"],
        "shards": p["shards"],
        "narrative": p["narrative"],
        "supersedes": p["supersedes"],
        "upstream_provenance": {k: v for k, v in p["upstream_record"].items()
                                if not k.endswith("_reason")},
        "reward_composition": p["reward_composition"],
        "screening_interval_days": p["screening_interval_days"],
        "screening_detector_version": p["screening_detector_version"],
        "derivation_instant": p["derivation_instant"],
        "screening_roots": p["screening_roots"],
        "empty_submission_result": p["empty_submission_result"],
        "resolved_closure": p["resolved_closure"],
        "applicability": p["applicability"],
        "measurement_tier": g["measurement_tier"],
        "gradeable": {"value": g["gradeable"]["value"], "basis": g["gradeable"]["basis"],
                      "unmeasured_constants": sorted(g["gradeable"]["unmeasured_constants"])},
        "knee_anchor_status": g["knee_anchor_status"],
        "ablation_verdicts": {
            "max_single_ablation": g["ablation_bounds"]["max_single_ablation_verdict"],
            "max_marginal_ablation": g["ablation_bounds"]["max_marginal_ablation_verdict"],
            "reference_share_of_best": g["ablation_bounds"]["reference_share_verdict"],
            "effective_levers": g["ablation_bounds"]["effective_levers_verdict"],
            "levers_to_90pct_of_best": g["ablation_bounds"]["levers_to_90pct_verdict"],
            "bounds_failed": g["ablation_bounds"]["bounds_failed"],
        },
    }
    # measured over the frozen tree by seed/build/screen_bind.py and carried across untouched
    for key in ("screening_measured_at", "screening_expires_at", "screening_result"):
        carrier[key] = existing.get(key)
    return carrier


# ---------------------------------------------------------------------------------------------
# CANARY-PRESERVING WRITERS
# ---------------------------------------------------------------------------------------------

TRUTH_MARK = "<!-- AELLO-CANARY-BLOCK"


def existing_truth_canary(path):
    """The planted HTML-comment block from the existing TRUTH.md, or the empty string.

    seed/identity.py writes  body.rstrip() + "\\n\\n" + block  into this carrier, so re-attaching
    the block after exactly one blank line reproduces the planter's own bytes and the tree hash
    does not move when the freeze re-runs this generator.
    """
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    return "\n" + text[text.index(TRUTH_MARK):] if TRUTH_MARK in text else ""


def existing_json_canary(path):
    """The planted wrapper-level canary key from the existing solution/rubrics.json."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get("canary")
    except Exception:
        return None


def write_json(path, doc):
    """indent=1, sort_keys=True and one trailing newline: byte-for-byte what seed/identity.py's
    planter writes, so planting and regenerating cannot disagree."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
        fh.write("\n")


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def write_yaml(path, doc):
    """The convention seed/build/screen_bind.py uses when it rewrites this carrier after the
    freeze. A different convention reorders keys and reports as drift on the next screen."""
    import yaml
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=True, default_flow_style=False, width=100,
                       allow_unicode=True)


# ---------------------------------------------------------------------------------------------
# DRIFT MODE
# ---------------------------------------------------------------------------------------------

HAND_AUTHORED_MODULES = ("verifier.py", "budget.py", "isolation.py", "ladder.py", "compose.py")


def drift_report(g, planned):
    """Compare every generated artifact against what is on disk, canary-aware.

    The canary is stripped from the on-disk copy before comparison and nothing else is. Stripping
    everything after the marker instead would let appended text hide behind the tripwire, which is
    the opposite of what a tripwire is for.
    """
    import yaml
    report = {}

    on_disk = existing_json_canary(os.path.join(HERE, "rubrics.json"))
    current = None
    path = os.path.join(HERE, "rubrics.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            current = json.load(fh)
        current.pop("canary", None)
    report["compiled_rubrics_drift"] = current != planned["compiled_rubrics"]
    report["canary_present"] = bool(on_disk)

    for name, key in (("rubrics.json", "council"), ("rubrics.jsonl", "council_lines"),
                      ("test_output.py", "test_module"), ("constants.json", "constants")):
        p = os.path.join(TESTS, name)
        if not os.path.exists(p):
            report["%s_drift" % key] = True
            continue
        with open(p, encoding="utf-8") as fh:
            raw = fh.read()
        want = planned[key]
        got = json.loads(raw) if name.endswith(".json") else raw
        report["%s_drift" % key] = got != want

    truth_path = os.path.join(HERE, "TRUTH.md")
    if os.path.exists(truth_path):
        with open(truth_path, encoding="utf-8") as fh:
            body = fh.read()
        if TRUTH_MARK in body:
            body = body[:body.index(TRUTH_MARK)]
        report["truth_drift"] = body.rstrip("\n") + "\n" != planned["truth"]
    else:
        report["truth_drift"] = True

    prov_path = os.path.join(HERE, "provenance.yaml")
    if os.path.exists(prov_path):
        with open(prov_path, encoding="utf-8") as fh:
            report["provenance_drift"] = yaml.safe_load(fh) != planned["provenance"]
    else:
        report["provenance_drift"] = True

    missing = [m for m in HAND_AUTHORED_MODULES if not os.path.exists(os.path.join(TESTS, m))]
    report["hand_authored_modules_missing"] = missing
    report["constants_disagree"] = disagreements(g)
    return report


# ---------------------------------------------------------------------------------------------

def main():
    g = load_grounding()

    # ---- compute everything first; nothing is opened for writing above this line ----
    planned = {}
    planned["compiled_rubrics"] = build_compiled_rubrics(g)
    planned["council"] = build_council(g)
    planned["council_lines"] = build_council_lines(planned["council"])
    planned["constants"] = build_constants(g)
    planned["test_module"] = build_test_module(g)
    planned["truth"] = build_truth(g)
    carrier_in = read_carrier()
    planned["provenance"] = build_provenance(g, carrier_in)
    truth_tail = existing_truth_canary(os.path.join(HERE, "TRUTH.md"))
    json_canary = existing_json_canary(os.path.join(HERE, "rubrics.json"))
    bad = disagreements(g)

    if "--check" in sys.argv:
        report = drift_report(g, planned)
        print(json.dumps(report, indent=1, sort_keys=True))
        failed = sorted(k for k, v in report.items() if k.endswith("_drift") and v)
        if report["constants_disagree"]:
            failed.append("constants_disagree")
        if report["hand_authored_modules_missing"]:
            failed.append("hand_authored_modules_missing")
        if not report["canary_present"]:
            failed.append("canary_absent")
        if failed:
            print("FAILED: %s" % ", ".join(failed), file=sys.stderr)
        return 1 if failed else 0

    if bad:
        print(json.dumps({"constants_disagree": bad}, indent=1), file=sys.stderr)
        print("REFUSING to regenerate: grounding.yaml publishes figures its own evidence does "
              "not reproduce", file=sys.stderr)
        return 1

    # ---- writes only from here ----
    compiled = dict(planned["compiled_rubrics"])
    if json_canary:
        compiled["canary"] = json_canary
    write_json(os.path.join(HERE, "rubrics.json"), compiled)
    write_text(os.path.join(HERE, "TRUTH.md"), planned["truth"] + truth_tail)
    write_yaml(os.path.join(HERE, "provenance.yaml"), planned["provenance"])
    write_json(os.path.join(TESTS, "rubrics.json"), planned["council"])
    write_text(os.path.join(TESTS, "rubrics.jsonl"), planned["council_lines"])
    write_text(os.path.join(TESTS, "test_output.py"), planned["test_module"])
    write_json(os.path.join(TESTS, "constants.json"), planned["constants"])

    digest = hashlib.sha256(planned["test_module"].encode()).hexdigest()[:16]
    print(json.dumps({
        "slot_id": g["slot_id"],
        "compiled_items": len(planned["compiled_rubrics"]["items"]),
        "judged_rubrics": len(planned["council"]["rubrics"]),
        "compiled_weight_share": planned["compiled_rubrics"]["compiled_weight_share"],
        "constants_null": sorted(k for k in ("floor", "knee", "dichotomizing_threshold",
                                             "reward_gate_pass_threshold")
                                 if planned["constants"][k] is None),
        "ablation_bounds_failed": g["ablation_bounds"]["bounds_failed"],
        "canary_carried": {"rubrics_json": bool(json_canary), "truth_md": bool(truth_tail)},
        "test_module_sha256_16": digest,
        "figures_recomputed_and_agreeing": True,
    }, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
