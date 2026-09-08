"""Read a submitted pipeline's DECLARED sampling budget without executing it.

WHY THE BUDGET IS DECLARED RATHER THAN TIMED. The graded quantity is accuracy at a fixed
neighbourhood-sampling budget. That is deliberate: accuracy at a fixed budget does not
depend on how fast the steps ran, while a wall-clock bar makes every score a function of
whatever else was on the device. The envelope in task.toml bounds the attempt; it is not the
graded quantity.

WHAT THE BUDGET IS, ON THIS SUBSTRATE. environment/data/graph.npz ships neighbour_mean and
no edge list, so the fan-out half of the budget has already been spent by the environment,
once, at hop depth 1 and full fan-out. What remains for the agent to allocate is
presentations: how many times each training node is shown. The three published numbers are

    HOP_DEPTH               1
    EXPANSIONS_PER_EPOCH    90941     one expansion per training node
    EPOCH_BUDGET            12

and their product, 1091292, is the total neighbourhood expansions a graded run may spend.

THIS TOOL IS CORROBORATION, NOT ENFORCEMENT. It AST-parses submitted sources for the three
module-level constants. The judged rubric honest_sampling_budget carries the actual
obligation, because nothing here observes the loop that ran; a declaration and an execution
are different objects and only one of them is visible to a static reader. The compiled check
`sampling_budget_declared` in tests/test_output.py is the graded authority and imports the
reader below, so the tool and the check cannot drift apart.

MODULE-LEVEL BINDINGS ONLY. A dead branch or a function-local name earns nothing. Values are
reduced through a safe integer evaluator, so a product or a unary sign resolves and an
unevaluable right-hand side returns None rather than passing. A name that is rebound, or
reached through globals(), a `global` statement or setattr, returns None, because a
declaration that changes is not a declaration.
"""
import ast
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DECLARED_NAMES = ("EPOCH_BUDGET", "EXPANSIONS_PER_EPOCH", "HOP_DEPTH")


def published():
    with open(os.path.join(HERE, "constants.json")) as handle:
        constants = json.load(handle)
    return {name: constants[name.lower()] for name in DECLARED_NAMES}


def safe_int(node):
    """Reduce an integer expression without executing it. None when unevaluable."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = safe_int(node.operand)
        if value is None:
            return None
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)):
        left, right = safe_int(node.left), safe_int(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left // right if right != 0 else None
    return None


def mutation_dialects(tree, name):
    """The doors that reach a module constant without a module-level Assign."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Global) and name in node.names:
            found.append("global-declaration")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Call)
                        and getattr(target.value.func, "id", None) in ("globals", "vars")
                        and isinstance(target.slice, ast.Constant)
                        and target.slice.value == name):
                    found.append("globals-subscript")
        elif (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setattr"
              and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)
              and node.args[1].value == name):
            found.append("setattr")
    return found


def declared_constants(path, names=DECLARED_NAMES):
    """The EFFECTIVE module-level declarations in one submitted source file.

    Returns {name: int or None}. A name that never appears is absent from the mapping; a
    name that appears but cannot be resolved to a single stable integer maps to None, which
    the caller must treat as a failure rather than as silence.
    """
    with open(path, encoding="utf-8", errors="replace") as handle:
        source = handle.read()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    out = {}
    for name in names:
        if mutation_dialects(tree, name):
            out[name] = None
            continue
        values, seen = [], 0
        for node in tree.body:
            targets = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            for target in targets:
                if getattr(target, "id", None) != name:
                    continue
                seen += 1
                if isinstance(node, ast.AugAssign) or seen > 1:
                    values = [None]           # rebound after its first binding
                    break
                values.append(safe_int(node.value))
            if values == [None]:
                break
        if seen:
            out[name] = values[0] if values else None
    return out


def python_sources(root):
    for base, _, files in sorted(os.walk(root)):
        for name in sorted(files):
            if name.endswith(".py"):
                yield os.path.join(base, name)


def audit(root, expected=None):
    """Every declaration in a submitted tree, against the published budget."""
    expected = published() if expected is None else expected
    rows, ok = [], True
    for path in python_sources(root):
        found = declared_constants(path)
        if not found:
            continue
        bad = {k: v for k, v in found.items() if v is None or v != expected.get(k)}
        rows.append({"path": path, "declared": found, "disagrees": sorted(bad)})
        if bad:
            ok = False
    return {"ok": ok, "expected": expected, "sources_with_declarations": rows,
            "total_neighbourhood_expansions":
                expected["EXPANSIONS_PER_EPOCH"] * expected["EPOCH_BUDGET"]
                * expected["HOP_DEPTH"]}


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/agent_src"
    if not os.path.isdir(root):
        print(json.dumps({"ok": True, "note": "no submitted source tree at %s" % root,
                          "expected": published()}, indent=1, sort_keys=True))
        return 0
    report = audit(root)
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
