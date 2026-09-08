"""Recompute this slot's ablation ladder and check it against seed/contract.yaml's five bounds.

WHY A BUNDLE CARRIES ITS OWN ABLATION ARITHMETIC.

seed/contract.yaml records five bounds on a slot's lever set and, for four of them, the measured
distribution the threshold was calibrated against. AELLO-C8-S4 fails three of the five. That is not
a defect in the measurement and it is not a thing to be quietly absorbed into a disposition field;
it is a property of the slot, and a property that lives only in a sentence somewhere is a property
nobody can re-derive.

So the arithmetic ships here, over the 32 measured configurations, and it can be run:

    python solution/ladder.py

WHAT THE LADDER MEASURED. Five levers, full power set, one seed (20260831), the leaf budget held
fixed at 12000 rounds-times-leaves per class-tree series across every configuration, scored as
balanced accuracy on the 87153 private graded rows. seed/ladders/run_ladder.py is the runner and
seed/ladders/ladder_c8s4.json is the record; the 32 points are frozen into solution/grounding.yaml
so this module needs neither.

THE FIVE BOUNDS, and what each one is FOR. They bound different failure modes and no one of them is
sufficient, which is why the contract carries all five:

  max_single_ablation_fraction (0.60, standalone measure)
      catches the DISJUNCTIVE one-trick, where several levers substitute for one another so that
      any one alone recovers most of the ramp. THIS SLOT: 1.02236. FAIL.

  max_marginal_ablation_fraction (0.80, leave-one-out measure)
      catches the CONJUNCTIVE one-trick, where one indivisible mechanism is declared as several
      mandatory knobs. THIS SLOT: 0.43491. PASS.

  min_reference_share_of_best (0.50)
      a PRECONDITION, not a reward parameter: a lever set whose all-on configuration cannot capture
      half its own achievable gain is interfering destructively and its fractions are
      uninterpretable. THIS SLOT: 0.94241. PASS.

  min_effective_levers (2.0)
      inverse Herfindahl over positive marginal worth -- the effective number of levers carrying
      the ramp. THIS SLOT: 1.10978. FAIL.

  min_levers_to_90pct_of_best (3)
      the fewest levers whose best combination reaches 90 percent of the floor-to-best span. THIS
      SLOT: 1. FAIL.

THE DENOMINATOR IS FLOOR-TO-REFERENCE, never floor-to-knee. A knee-anchored denominator is chosen
by the slot author, which makes every fraction it produces either vacuous or self-referential. This
slot's knee is null in any case.

THE FINDING, in one line: imbalance reweighting alone scores 0.93639 and all five levers together
score 0.93394, so the strongest single lever lands ABOVE the whole set and the standalone fraction
exceeds 1. The reason is not subtle. Balanced accuracy divides the score seven ways regardless of
prevalence and L3 divides the training loss the same seven ways -- the lever IS the metric, restated
as a weight vector. The other four are capacity and regularisation knobs applied to a problem that
was never short of capacity.

WHY IT LIVES IN solution/ AND NOT IN tests/. The 32-point ladder names which lever does the work,
and that is the answer to the task. tests/ is built into the verifier image and is agent-readable;
solution/ is the oracle side and reaches the agent never. The three OUTCOME LEVELS -- floor 0.82445,
reference 0.93394, best 0.94063 -- do ship in tests/constants.json, because knowing what score is
reachable under the budget is a fair thing for a solver to know and says nothing about how. The
per-lever decomposition stays here.

WHAT THIS MODULE IS NOT. It is not a check in tests/test_output.py and it does not enter r_det. The
compiled surface grades a submission; this grades the SLOT, and a slot-level finding belongs in the
record rather than in an agent's reward. It is run by hand, by a reviewer, or from the oracle path.
"""
import json
import os
import sys

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
LEVERS = ("L1", "L2", "L3", "L4", "L5")

CONTRACT_BOUNDS = {
    "max_single_ablation_fraction": 0.60,
    "max_marginal_ablation_fraction": 0.80,
    "min_reference_share_of_best": 0.50,
    "min_effective_levers": 2.0,
    "min_levers_to_90pct_of_best": 3,
}

LEVER_NAMES = {
    "L1": "depth/leaf schedule under the leaf budget",
    "L2": "categorical encoding of the wilderness and soil indicators",
    "L3": "class-imbalance reweighting toward the rare cover types",
    "L4": "early-stopping criterion against the in-budget validation fold",
    "L5": "feature-subsample schedule across boosting rounds",
}


def load_ladder():
    """The 32 measured points, from solution/grounding.yaml.

    grounding.yaml is the derivation source for everything in this bundle, so reading the ladder
    from it rather than from a second copy here means the module and the published figures cannot
    disagree about what was measured.
    """
    import yaml
    path = os.path.join(HERE, "grounding.yaml")
    with open(path, encoding="utf-8") as fh:
        g = yaml.safe_load(fh)
    return {row["cfg"]: float(row["bacc"]) for row in g["ablation_ladder"]}


def measures(ladder):
    """Every contract measure, from the raw 32 points. No published figure is read back."""
    floor = ladder["00000"]
    reference = ladder["11111"]
    best = max(ladder.values())
    best_cfg = min(cfg for cfg, v in ladder.items() if v == best)
    den = reference - floor
    span = best - floor

    standalone, marginal, absolute = {}, {}, {}
    for i, lever in enumerate(LEVERS):
        alone = "".join("1" if j == i else "0" for j in range(5))
        without = "".join("0" if j == i else "1" for j in range(5))
        absolute[lever] = {"alone": round(ladder[alone], 5),
                           "all_but": round(ladder[without], 5)}
        standalone[lever] = round((ladder[alone] - floor) / den, 5)
        marginal[lever] = round((reference - ladder[without]) / den, 5)

    positive = [v for v in marginal.values() if v > 0]
    total = sum(positive)
    effective = round(1.0 / sum((v / total) ** 2 for v in positive), 5) if total else 0.0

    best_at_k, breadth = {}, None
    for k in range(1, 6):
        top = max(v for cfg, v in ladder.items() if cfg.count("1") == k)
        best_at_k[k] = round(top, 5)
        if breadth is None and (top - floor) / span >= 0.90:
            breadth = k

    return {
        "floor": round(floor, 5), "reference": round(reference, 5), "best": round(best, 5),
        "best_cfg": best_cfg, "denominator": round(den, 5), "span_to_best": round(span, 5),
        "absolute": absolute,
        "standalone_fraction": standalone,
        "marginal_fraction": marginal,
        "max_standalone": round(max(standalone.values()), 5),
        "max_standalone_lever": max(standalone, key=lambda k: standalone[k]),
        "max_marginal": round(max(marginal.values()), 5),
        "max_marginal_lever": max(marginal, key=lambda k: marginal[k]),
        "reference_share_of_best": round(den / span, 5),
        "effective_levers": effective,
        "levers_to_90pct_of_best": breadth,
        "best_at_k": best_at_k,
    }


def verdicts(m):
    """Each bound, its measured value, and pass or fail. Ordered so the report reads the same way
    every time."""
    b = CONTRACT_BOUNDS
    rows = [
        ("max_single_ablation_fraction", m["max_standalone"], "<=",
         b["max_single_ablation_fraction"],
         "disjunctive one-trick: one lever substitutes for the set"),
        ("max_marginal_ablation_fraction", m["max_marginal"], "<=",
         b["max_marginal_ablation_fraction"],
         "conjunctive one-trick: one mechanism declared as several mandatory knobs"),
        ("min_reference_share_of_best", m["reference_share_of_best"], ">=",
         b["min_reference_share_of_best"],
         "precondition: the all-on configuration must capture half its own achievable gain"),
        ("min_effective_levers", m["effective_levers"], ">=", b["min_effective_levers"],
         "inverse Herfindahl over positive marginal worth"),
        ("min_levers_to_90pct_of_best", m["levers_to_90pct_of_best"], ">=",
         b["min_levers_to_90pct_of_best"],
         "breadth: fewest levers reaching 90 percent of the floor-to-best span"),
    ]
    out = []
    for name, measured, sense, bound, catches in rows:
        ok = measured <= bound if sense == "<=" else measured >= bound
        out.append({"bound": name, "measured": measured, "sense": sense, "contract": bound,
                    "verdict": "pass" if ok else "fail", "catches": catches})
    return out


def implausibility_flag(measured_balanced_accuracy, m):
    """Report a submission that scores implausibly far above the best measured configuration.

    All 581012 Covertype feature vectors are distinct, so a join against the public corpus recovers
    the graded labels exactly and would score near 1.0. The best configuration measured under this
    slot's leaf budget scores 0.94063. A submission well above that is not proof of a lookup and
    this function does not treat it as one -- it REPORTS, it does not score, and the adjudication is
    owed to Phase 2. Anything that turned this into a reward decision would need a threshold this
    slot has not measured, and inventing one here is precisely what the null constants forbid.
    """
    if measured_balanced_accuracy is None:
        return {"flagged": False, "reason": "not_measured"}
    best = m["best"]
    return {
        "flagged": bool(measured_balanced_accuracy > best),
        "measured": round(float(measured_balanced_accuracy), 6),
        "best_measured_configuration": best,
        "margin_above_best": round(float(measured_balanced_accuracy) - best, 6),
        "disposition": "report_only",
        "adjudication": "owed_to_phase_2",
    }


def report():
    ladder = load_ladder()
    m = measures(ladder)
    v = verdicts(m)
    failed = [row["bound"] for row in v if row["verdict"] == "fail"]
    return {
        "slot_id": "AELLO-C8-S4",
        "points_measured": len(ladder),
        "seed": 20260831,
        "denominator_rule": "floor_to_reference",
        "levers": LEVER_NAMES,
        "measures": m,
        "bounds": v,
        "bounds_failed": sorted(failed),
        "finding": {
            "headline": "L3 alone beats the whole lever set",
            "l3_alone": m["absolute"]["L3"]["alone"],
            "all_levers_on": m["reference"],
            "standalone_fraction": m["standalone_fraction"]["L3"],
            "l4_alone_is_below_floor": m["absolute"]["L4"]["alone"] < m["floor"],
            "best_is_not_all_on": m["best_cfg"] != "11111",
        },
    }


def _main():
    r = report()
    print(json.dumps(r, indent=1, sort_keys=True))
    # A failing bound is the MEASURED STATE of this slot, not a broken run, so the exit code
    # reports the finding rather than an error: 0 when every bound holds, 2 when one or more does
    # not. A non-zero-but-not-1 exit keeps it distinguishable from a crash.
    return 0 if not r["bounds_failed"] else 2


if __name__ == "__main__":
    raise SystemExit(_main())
