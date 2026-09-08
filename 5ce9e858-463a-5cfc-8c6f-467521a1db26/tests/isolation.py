"""Prove that the private future is absent from everything the agent can read.

THE OBLIGATION. This slot's red line is 'any use of observations after the private window's
start as training signal'. Every other control in the bundle assumes that using those
observations requires FINDING them first, and that assumption is exactly the kind of thing that
is true until one day it quietly is not: a corpus landed with one extra column, a starter that
helpfully stacks the segments, a debug array left in environment/, and the red line becomes
unenforceable without anyone editing a checker. So the withholding is proved here, over the
delivered bytes, on every run.

WHAT IS PROVED, in the order the proofs get weaker.

  1. STRUCTURAL. environment/data/ettm1_graded_context.npy carries six columns, not seven. The
     target column is not withheld by convention, it does not exist in the file. This is the
     only proof in the module that is a proof rather than a search, and it is the one that
     matters.

  2. BYTE REACHABILITY. No file on the agent-visible surface has the digest of
     tests/heldout/ettm1_graded_targets.npy. A copy under another name is still the private
     window.

  3. VALUE REACHABILITY. No array on the agent-visible surface reproduces the private target
     vector, at full length or as a contiguous run longer than one graded block. A vector that
     agrees with the private future for 96 consecutive rows did not arrive there by modelling.
     The comparison is on float32 values rather than bytes, so a re-saved, re-typed or
     transposed copy is caught along with a verbatim one.

WHAT IS NOT PROVED, said plainly because a checker that overstates its reach is worse than one
that admits its limit. This module bounds what is REACHABLE. It cannot bound what was KNOWN.
ETTm1 is a public series and the graded window is a contiguous segment of it, so a solver that
has memorised the published file could reconstruct OT for these rows without any byte of the
private file ever entering the sandbox. That residual is real, it is recorded in
solution/grounding.yaml under contamination_sources, and it is carried by a judged rubric rather
than pretended away here.

    python tests/isolation.py            # scan the delivered tree
    python tests/isolation.py /some/dir  # additionally scan a submitted tree
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BUNDLE, "environment"))
import ettm1_window as W  # noqa: E402

HELDOUT_DIR = os.path.join(HERE, "heldout")
HELDOUT_FILE = os.path.join(HELDOUT_DIR, "ettm1_graded_targets.npy")
# Everything the agent may read. tests/heldout/ is deliberately absent, and so is solution/.
AGENT_VISIBLE = ("environment", "instruction.md", "task.toml")
ARRAY_SUFFIXES = (".npy", ".npz")
# A run this long agreeing with the private future is not a forecast. One graded block.
MATCH_RUN_LIMIT = W.HORIZON


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def walk(roots):
    """Every regular file under the named roots, in a sorted, reproducible order."""
    for root in roots:
        if os.path.isfile(root):
            yield root
            continue
        for base, dirs, files in os.walk(root):
            dirs.sort()
            for name in sorted(files):
                yield os.path.join(base, name)


def structural_proof():
    """The target column is absent from the graded context file itself."""
    import numpy as np
    ctx = np.load(os.path.join(W.DATA_DIR, W.CONTEXT_FILE), allow_pickle=False)
    return {"shape": list(ctx.shape),
            "expected": [W.CONTEXT_ROWS, W.N_COVARIATES],
            "target_column_absent": list(ctx.shape) == [W.CONTEXT_ROWS, W.N_COVARIATES]}


def arrays_in(path):
    """Every numeric array a file yields, or nothing when it yields none."""
    import numpy as np
    try:
        loaded = np.load(path, allow_pickle=False)
    except Exception:
        return []
    if hasattr(loaded, "files"):
        return [loaded[k] for k in loaded.files]
    return [loaded]


def longest_agreement(candidate, truth):
    """Longest run of consecutive positions where a flattened array equals the private vector.

    Compared at float32, which is the dtype the private file is stored in, so a copy that has
    been through a float64 round trip still matches. Returns 0 when the candidate is shorter
    than one block or cannot be read as a float vector.
    """
    import numpy as np
    flat = np.asarray(candidate).ravel()
    if flat.size < MATCH_RUN_LIMIT or not np.issubdtype(flat.dtype, np.number):
        return 0
    best = run = 0
    limit = min(flat.size, truth.size)
    equal = flat[:limit].astype("float32") == truth[:limit]
    for hit in equal:
        run = run + 1 if hit else 0
        best = max(best, run)
    return int(best)


def scan(roots):
    """Byte and value reachability of the private window across the named roots."""
    import numpy as np
    if not os.path.exists(HELDOUT_FILE):
        return {"scanned": 0, "byte_copies": [], "value_copies": [],
                "note": "private-file-unreadable-scan-vacuous"}
    target_digest = digest(HELDOUT_FILE)
    truth = np.load(HELDOUT_FILE, allow_pickle=False).astype("float32")
    byte_copies, value_copies, scanned = [], [], 0
    for path in walk(roots):
        if os.path.abspath(path).startswith(os.path.abspath(HELDOUT_DIR) + os.sep):
            continue
        scanned += 1
        try:
            if digest(path) == target_digest:
                byte_copies.append(path)
                continue
        except OSError:
            continue
        if not path.endswith(ARRAY_SUFFIXES):
            continue
        for array in arrays_in(path):
            run = longest_agreement(array, truth)
            if run >= MATCH_RUN_LIMIT:
                value_copies.append([path, run])
    return {"scanned": scanned, "byte_copies": sorted(byte_copies),
            "value_copies": sorted(value_copies)}


def report(extra_roots=()):
    roots = [os.path.join(BUNDLE, r) for r in AGENT_VISIBLE if os.path.exists(
        os.path.join(BUNDLE, r))] + [r for r in extra_roots if os.path.exists(r)]
    structural = structural_proof()
    found = scan(roots)
    isolated = (structural["target_column_absent"] and not found["byte_copies"]
                and not found["value_copies"] and found["scanned"] > 0)
    return {"roots": [os.path.relpath(r, BUNDLE) if r.startswith(BUNDLE) else r for r in roots],
            "structural": structural, "reachability": found, "isolated": bool(isolated),
            "bounds_reachability_not_knowledge": True}


def main():
    out = report(tuple(sys.argv[1:]))
    print(json.dumps(out, indent=1, sort_keys=True))
    if not out["isolated"]:
        print("FAILED: the private future is reachable from the agent-visible surface",
              file=sys.stderr)
    return 0 if out["isolated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
