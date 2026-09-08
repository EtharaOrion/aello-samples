"""The authored leaf-budget audit for AELLO-C8-S4.

WHY THIS IS A MODULE AND NOT THREE LINES INSIDE A CHECK.

The graded artifact is a dense array of 87153 integers. Nothing about the model that produced it
survives into the submission -- not its depth, not its round count, not how many trees it grew. The
one authored constraint on this slot is a budget over exactly those quantities, so the budget cannot
be measured from the deliverable. It is DECLARED, and the declaration is what is graded.

That makes the audit below the only thing standing between this slot and an unconstrained one, and
it earns its own file for the same reason a load-bearing wall gets drawn on the plan.

WHAT IS CHECKABLE ABOUT A DECLARATION.

Three things, and the module does all three:

  1. TYPE. Every factor is a positive integer. A float round count, a string, a null, or a negative
     leaf count is a malformed declaration, not a small one.

  2. ARITHMETIC. total_trees must equal boosting_rounds * trees_per_round, and total_leaves must
     equal total_trees * leaves_per_tree. An agent that trains one configuration and declares
     another has to make the numbers multiply, and the numbers it would have to write down are the
     ones it actually spent. This does not make lying impossible. It makes lying explicit.

  3. THE CAP. total_leaves at most 84000. That figure is 7 * 12000: seven one-vs-rest trees per
     boosting round, and the 12000 leaf-slots per class-tree series that seed/ladders/run_ladder.py
     held fixed across all 32 measured configurations. Both measured depth arms land inside it --
     387 rounds at 31 leaves is 83979, and 190 rounds at 63 leaves is 83790 -- so the cap is
     binding on the measurement rather than decorative.

WHAT IS NOT CHECKABLE, said plainly rather than implied.

A declaration is an assertion about a process the verifier never observed. This module cannot tell
a truthful accounting from a fluent one. What it can do is refuse an accounting that does not add
up, refuse one that is absent, and refuse one that overspends -- and the shortcut this slot actually
worries about, a copied public Covertype recipe, fails on the third of those by a factor between 2.7
and 21 before its score is ever read. The residue -- whether the arithmetic was derived before
training or reverse-engineered after it -- is exactly what the judged rubric
budget_arithmetic_shown exists to grade, and it is graded on the trajectory because it cannot be
graded here.

THE SPLIT COUNT IS NOT THE LEAF COUNT. A tree grown to L leaves consumed L-1 splits, so an ensemble
of T trees spends total_leaves - T splits. The two differ by the tree count, which on the measured
arms is 2709 and 1330 -- 81270 splits against 83979 leaves, and 82460 against 83790. The budget is
declared in LEAVES because that is the quantity the ladder held fixed; the split count is reported
alongside so that a reader comparing against a splits-based figure elsewhere is not left to guess.
"""
import json
import os
import sys

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))

FACTORS = ("boosting_rounds", "trees_per_round", "leaves_per_tree")
TOTALS = ("total_trees", "total_leaves")
REQUIRED = FACTORS + TOTALS


def cap():
    """The authored cap, read from the published constants rather than restated here.

    One literal, in tests/constants.json, generated from solution/grounding.yaml. A second copy in
    this file would be a second place for it to drift.
    """
    with open(os.path.join(HERE, "constants.json"), encoding="utf-8") as fh:
        return int(json.load(fh)["split_budget_cap"])


def _positive_int(value):
    """True only for a genuine positive integer.

    bool is a subclass of int in Python and True would otherwise pass as the integer 1, which is a
    silly way for a budget audit to accept a declaration of True trees.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    return value > 0


def audit(declaration, cap_value=None):
    """Audit one split-budget declaration. Never raises; returns a verdict mapping.

    The mapping is designed to be readable in a check-results file without the reader having to
    consult this module: it carries what was declared, what the arithmetic says, what the cap is,
    and the single reason token that decided the verdict.
    """
    limit = cap() if cap_value is None else int(cap_value)
    verdict = {"ok": False, "reason": None, "cap": limit, "declared": None}

    if declaration is None:
        verdict["reason"] = "declaration_missing"
        return verdict
    if isinstance(declaration, str):
        try:
            declaration = json.loads(declaration)
        except Exception:
            verdict["reason"] = "declaration_unparseable"
            return verdict
    if not isinstance(declaration, dict):
        verdict["reason"] = "declaration_not_a_mapping"
        return verdict

    verdict["declared"] = {k: declaration.get(k) for k in REQUIRED}

    missing = [k for k in REQUIRED if k not in declaration]
    if missing:
        verdict["reason"] = "declaration_incomplete"
        verdict["missing"] = sorted(missing)
        return verdict

    bad = sorted(k for k in REQUIRED if not _positive_int(declaration[k]))
    if bad:
        verdict["reason"] = "declaration_non_integer"
        verdict["non_integer_fields"] = bad
        return verdict

    rounds = declaration["boosting_rounds"]
    per_round = declaration["trees_per_round"]
    leaves = declaration["leaves_per_tree"]
    trees = rounds * per_round
    total_leaves = trees * leaves

    verdict["recomputed"] = {"total_trees": trees, "total_leaves": total_leaves,
                             "total_splits": total_leaves - trees}
    verdict["utilisation"] = round(total_leaves / limit, 5)

    if declaration["total_trees"] != trees or declaration["total_leaves"] != total_leaves:
        verdict["reason"] = "declaration_inconsistent"
        return verdict

    if total_leaves > limit:
        verdict["reason"] = "budget_overspent"
        verdict["overspend_factor"] = round(total_leaves / limit, 3)
        return verdict

    verdict["ok"] = True
    verdict["reason"] = "within_budget"
    return verdict


def measured_arms():
    """The two depth arms the ladder actually ran, recomputed from the budget rule.

    Reported by --self-test so that a reader can see the cap is reachable, that both arms fit, and
    by how little. If either arm ever fails to fit, the cap and the ladder have come apart and the
    audit is measuring something the measurement never respected.
    """
    per_series, per_round, limit = 12000, 7, cap()
    arms = {}
    for leaves in (31, 63):
        rounds = per_series // leaves
        trees = rounds * per_round
        arms["leaves_%d" % leaves] = {
            "boosting_rounds": rounds, "trees_per_round": per_round,
            "leaves_per_tree": leaves, "total_trees": trees,
            "total_leaves": trees * leaves, "total_splits": trees * (leaves - 1),
            "fits": trees * leaves <= limit,
            "utilisation": round(trees * leaves / limit, 5),
        }
    return arms


def _self_test():
    """Exercise the audit on the two measured arms and on the failure each reason token names."""
    limit = cap()
    arms = measured_arms()
    cases = [("measured_arm_31", {k: v for k, v in arms["leaves_31"].items()
                                  if k in REQUIRED}, True, "within_budget"),
             ("measured_arm_63", {k: v for k, v in arms["leaves_63"].items()
                                  if k in REQUIRED}, True, "within_budget"),
             ("absent", None, False, "declaration_missing"),
             ("not_a_mapping", [1, 2, 3], False, "declaration_not_a_mapping"),
             ("incomplete", {"boosting_rounds": 100}, False, "declaration_incomplete"),
             ("non_integer", {"boosting_rounds": 100.0, "trees_per_round": 7,
                              "leaves_per_tree": 31, "total_trees": 700,
                              "total_leaves": 21700}, False, "declaration_non_integer"),
             ("boolean_smuggled", {"boosting_rounds": True, "trees_per_round": 7,
                                   "leaves_per_tree": 31, "total_trees": 7,
                                   "total_leaves": 217}, False, "declaration_non_integer"),
             ("inconsistent", {"boosting_rounds": 100, "trees_per_round": 7,
                               "leaves_per_tree": 31, "total_trees": 700,
                               "total_leaves": 1}, False, "declaration_inconsistent"),
             ("public_recipe", {"boosting_rounds": 1000, "trees_per_round": 7,
                                "leaves_per_tree": 255, "total_trees": 7000,
                                "total_leaves": 1785000}, False, "budget_overspent")]
    results, failures = [], 0
    for name, decl, want_ok, want_reason in cases:
        got = audit(decl)
        passed = got["ok"] == want_ok and got["reason"] == want_reason
        failures += 0 if passed else 1
        results.append({"case": name, "passed": passed, "reason": got["reason"],
                        "ok": got["ok"]})
    print(json.dumps({"cap": limit, "measured_arms": arms, "cases": results,
                      "failures": failures}, indent=1, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
