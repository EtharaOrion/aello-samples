"""Verifier isolation for AELLO-C8-S4: the private-label boundary, asserted rather than assumed.

THE BOUNDARY THIS SLOT ACTUALLY HAS.

Three folds ship. Two of them ship with their labels -- 406708 training rows and 87151 public
validation rows -- and one does not: environment/data/graded_features.npz carries an X and no y at
all. The 87153 graded labels live in tests/heldout/graded_labels.npz, which is inside the verifier
image's build context and is never uploaded to the agent container.

That is a real boundary and it is also a narrow one, so this module states what it does and does not
establish.

WHAT IT ESTABLISHES.

  1. THE GRADED FEATURES CARRY NO LABELS. Measured by opening the archive and reading its key set,
     not by trusting the filename. If a y ever appears there, the slot has stopped being a held-out
     task and the check says so.

  2. ONE READER. The private labels are reachable through exactly one path literal in this whole
     tree -- graded_labels(), below. The module counts that literal over its own source and fails
     if a second one appears. A widening surface is the failure mode this catches: nobody adds a
     second reader on purpose, they add it while fixing something else.

  3. THE FOLDS ARE NOT INTERCHANGEABLE. The public validation fold is 87151 rows and the graded
     fold is 87153. Two rows apart. A submission built against the wrong fold therefore fails on
     its LENGTH rather than being scored against a fold it was never aligned to, and that margin is
     small enough to be worth writing down: the row-count check compares against 87153 exactly and
     does not accept 87151.

  4. NO AGENT-VISIBLE COPY. Every .npz under environment/ is opened and its arrays are examined for
     the shape-and-dtype signature of the graded label vector. This is a structural scan, not a
     content comparison, and it is deliberately structural -- comparing content would require
     holding the private labels beside every agent-visible array, which is more exposure than the
     check is worth.

WHAT IT DOES NOT ESTABLISH, stated because a bounded check that implies more than it measures is
worse than no check.

  - It does not certify that the graded labels are unreachable. Covertype is public. All 581012
    feature vectors in this corpus are distinct, so a join against the published file recovers the
    graded labels exactly, and the agent container's network allowlist -- which reaches pypi.org --
    does not close the route where a package vendors the corpus. That route is recorded in
    solution/grounding.yaml under shortcut_controls.label_lookup with disposition OWED, and the
    judged rubric public_label_lookup_declined is the only instrument that observes an attempt.

  - It does not observe the graded run. guarded paths are read from the record the run itself
    produced, because a scan performed after the fact cannot distinguish a write made during the
    graded run from one made before it.
"""
import json
import os
import re
import sys

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)

# THE ONE PATH LITERAL. Every read of the private labels in this tree goes through graded_labels()
# below, and _reader_count() asserts that this string appears exactly once in this file.
_LABELS_RELATIVE = "heldout/graded_labels.npz"

# Prefixes and fragments that constitute a red line when they appear in the run's own record of
# what it touched. tests/ is the checker tree; the graded fragment covers the features and the
# labels under any spelling; the verifier log directory is the third named surface.
GUARDED_PREFIXES = ("tests/", "/tests/", "tests\\")
GUARDED_FRAGMENTS = ("graded_labels", "graded_features", "/logs/verifier", "heldout")


def labels_path():
    return os.path.join(HERE, _LABELS_RELATIVE)


def graded_labels():
    """The private graded labels. The only reader in this tree.

    allow_pickle stays False here for the same reason it does on the submission: a .npy or .npz may
    carry a pickled object and loading one executes it inside the verifier container.
    """
    import numpy as np
    with np.load(labels_path(), allow_pickle=False) as archive:
        return np.asarray(archive["y"]).astype("int64")


def constants():
    with open(os.path.join(HERE, "constants.json"), encoding="utf-8") as fh:
        return json.load(fh)


def is_guarded(path):
    """Does this path, as the run recorded it, cross one of the two declared red lines?"""
    p = str(path).replace("\\", "/").strip()
    if not p:
        return False
    norm = p.lstrip("./")
    if any(norm.startswith(x.replace("\\", "/").lstrip("./")) for x in GUARDED_PREFIXES):
        return True
    return any(fragment in p for fragment in GUARDED_FRAGMENTS)


def _reader_count():
    """How many times the private-label path literal appears in this file's own source.

    Counted over the source rather than declared, so that a second reader added later fails the
    isolation audit instead of quietly widening the surface. The literal is spelled once in an
    assignment and referenced by name everywhere else, so the expected count is exactly one; the
    regex below matches the quoted string form only, which is what a new reader would introduce.
    """
    with open(os.path.abspath(__file__), encoding="utf-8") as fh:
        source = fh.read()
    return len(re.findall(r'["\']heldout/graded_labels\.npz["\']', source))


def graded_features_keys():
    """The key set of the agent-visible graded archive. Measured, not assumed."""
    import numpy as np
    path = os.path.join(BUNDLE, "environment", "data", "graded_features.npz")
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as archive:
        return sorted(archive.files)


def _array_signatures(path):
    """(name, shape, dtype) for every array in a .npy or .npz, read from HEADERS ONLY.

    The training archive holds 406708 by 54 float32, which is 88 MB decompressed. Materialising it
    to learn its shape would make an isolation audit the most expensive thing the verifier does, so
    the .npy headers are parsed out of the zip members directly and no array data is ever read.
    """
    import numpy.lib.format as fmt
    out = []

    def header(fh):
        version = fmt.read_magic(fh)
        if version == (1, 0):
            return fmt.read_array_header_1_0(fh)
        if version == (2, 0):
            return fmt.read_array_header_2_0(fh)
        raise ValueError("unsupported npy version %r" % (version,))

    if path.endswith(".npy"):
        with open(path, "rb") as fh:
            shape, _fortran, dtype = header(fh)
        return [("", tuple(shape), dtype)]
    import zipfile
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".npy"):
                continue
            with archive.open(name) as fh:
                shape, _fortran, dtype = header(fh)
            out.append((name[:-4], tuple(shape), dtype))
    return out


def agent_visible_label_shaped_arrays():
    """Every array under environment/ whose shape and dtype match the graded label vector.

    A structural scan. The graded labels are (87153,) of an integer kind; anything agent-visible
    with that signature is either a copy or something that deserves an explanation, and either way
    the isolation audit should not pass silently over it. Content is deliberately NOT compared:
    doing so would require holding the private labels beside every agent-visible array, which is
    more exposure than the check is worth.
    """
    rows = constants()["graded_rows"]
    root = os.path.join(BUNDLE, "environment")
    hits = []
    for base, _dirs, files in os.walk(root):
        for name in sorted(files):
            if not name.endswith((".npz", ".npy")):
                continue
            full = os.path.join(base, name)
            rel = os.path.relpath(full, BUNDLE)
            try:
                signatures = _array_signatures(full)
            except Exception as exc:
                hits.append({"path": rel, "key": None,
                             "note": "unreadable:%s" % type(exc).__name__})
                continue
            for key, shape, dtype in signatures:
                if shape == (rows,) and dtype.kind in ("i", "u"):
                    hits.append({"path": rel, "key": key, "note": "label_shaped",
                                 "dtype": str(dtype)})
    return hits


def audit_tree():
    """The full isolation verdict over the delivered tree. Never raises."""
    c = constants()
    keys = graded_features_keys()
    readers = _reader_count()
    copies = agent_visible_label_shaped_arrays()
    findings = []
    if keys is None:
        findings.append("graded_features_missing")
    elif keys != ["X"]:
        findings.append("graded_features_carries_more_than_X")
    if readers != 1:
        findings.append("label_reader_count_%d" % readers)
    if copies:
        findings.append("label_shaped_array_on_agent_surface")
    if not os.path.exists(labels_path()):
        findings.append("private_labels_missing")
    if c["graded_rows"] == c["val_public_rows"]:
        findings.append("folds_indistinguishable_by_length")
    return {
        "ok": not findings,
        "findings": sorted(findings),
        "graded_features_keys": keys,
        "label_reader_count": readers,
        "agent_visible_label_shaped_arrays": copies,
        "graded_rows": c["graded_rows"],
        "val_public_rows": c["val_public_rows"],
        "row_count_separation": c["graded_rows"] - c["val_public_rows"],
        "bounded_claim": ("this audit bounds the shipped-bytes channel and certifies nothing "
                          "about the public-corpus join, which is recorded OWED in "
                          "solution/grounding.yaml"),
    }


def _main():
    report = audit_tree()
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
