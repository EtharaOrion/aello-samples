"""Corroborate the declared training budget by static inspection.

The graded quantity is accuracy at a FIXED BUDGET, deliberately, because accuracy at a
fixed budget does not depend on how fast the steps ran. Wall clock is an envelope the
attempt should fit inside, never the graded quantity: grading on seconds would make every
score a function of whatever else is on the device.

This tool is CORROBORATION, not enforcement: it AST-parses a submitted train.py for the
declared BUDGET_STEPS constant. The judged honest_step_budget rubric carries the actual
obligation (at most 8000 optimizer updates and at most 1,024,000 sample presentations);
the wall-clock envelope is not yet enforced by a timer, and that residual is recorded in
solution/TRUTH.md.
"""
import ast, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONST = json.load(open(os.path.join(HERE, "constants.json")))


def _safe_eval(n):
    """Reduce an integer expression without executing it. Returns None when unevaluable."""
    if isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool):
        return n.value
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
        v = _safe_eval(n.operand)
        return None if v is None else (v if isinstance(n.op, ast.UAdd) else -v)
    if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)):
        a, b = _safe_eval(n.left), _safe_eval(n.right)
        if a is None or b is None:
            return None
        if isinstance(n.op, ast.Add):
            return a + b
        if isinstance(n.op, ast.Sub):
            return a - b
        if isinstance(n.op, ast.Mult):
            return a * b
        return a // b if b != 0 else None
    return None


def _mutation_dialects(tree):
    """The module-attribute doors that reach BUDGET_STEPS without a module-level Assign."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Global) and "BUDGET_STEPS" in node.names:
            out.append("global-declaration")
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Call)
                        and getattr(t.value.func, "id", None) in ("globals", "vars")
                        and isinstance(t.slice, ast.Constant)
                        and t.slice.value == "BUDGET_STEPS"):
                    out.append("globals-subscript")
        elif (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setattr"
              and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)
              and node.args[1].value == "BUDGET_STEPS"):
            out.append("setattr")
    return out


def declared_budget(path):
    """Read the EFFECTIVE declared BUDGET_STEPS out of a submitted pipeline without executing it.

    Same shape as the graded compiled check `step_budget_declared`, which is the authority: this
    tool only corroborates. MODULE-LEVEL bindings only, so a dead branch or a function-local name
    earns nothing; the value is reduced through a safe integer evaluator, so a product or a unary
    sign resolves and an unevaluable right-hand side returns None rather than passing; and a
    declaration that is rebound, or reached through globals(), a `global` statement or setattr,
    returns None because a declaration that changes is not a declaration.

    An earlier revision of this file matched only ast.Assign with an ast.Constant value under
    ast.walk. Every dialect the graded checker rejects passed here, which is the contradiction
    this port removes.
    """
    tree = ast.parse(open(path).read())
    if _mutation_dialects(tree):
        return None
    values, seen = [], 0
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for t in targets:
            if getattr(t, "id", None) != "BUDGET_STEPS":
                continue
            seen += 1
            if isinstance(node, ast.AugAssign) or seen > 1:
                return None                      # rebound after its first binding
            values.append(_safe_eval(node.value))
    return values[0] if values else None


def find_submitted(sub):
    """First train.py in the SUBMITTED tree, sorted walk. Never the baked-in starter."""
    for r, _, fs in sorted(os.walk(sub)):
        if "train.py" in fs:
            return os.path.join(r, "train.py")
    return None


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else "/submission"
    path = find_submitted(sub)
    if path is None:
        print(json.dumps({"ok": True, "note": "no train.py submitted; grading uses predictions"}))
        return 0
    got = declared_budget(path)
    want = CONST["graded_step_budget"]
    ok = got is None or got == want
    print(json.dumps({"ok": ok, "source": path, "declared_budget_steps": got,
                      "required": want,
                      "graded_sample_budget": CONST["graded_sample_budget"],
                      "per_attempt_seconds": CONST["per_attempt_seconds"]}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
