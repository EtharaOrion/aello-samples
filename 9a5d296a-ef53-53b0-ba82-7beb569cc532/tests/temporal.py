"""Prove, by executing it, that the graded fold is a future and that its labels are private.

Authoring and freeze gate, run on the host against the bundle tree. The grade-time twin is
the compiled check `graded_labels_isolated` in tests/test_output.py, which scans the
AGENT-AUTHORED tree instead of the environment tree.

An isolation claim that is asserted rather than executed is worth nothing after freeze, and
a temporal-split claim is worse than worthless when it is only asserted, because a reader
has no way to tell an honest time split from a random split described as one. So this walks
the shipped tree and checks, with no sampling stride:

  1. train_idx, val_idx and graded_idx are pairwise disjoint and together cover every node,
     so nothing is graded that was also fitted and nothing is silently dropped;
  2. tests/heldout/graded_labels.npz carries exactly one label per graded node and nothing
     else -- no features, no indices, no second array that would let it be joined back to
     anything;
  3. no file under environment/ matches the graded label file by content digest;
  4. no integer array anywhere under environment/ has the graded fold's length and equals
     the graded label vector;
  5. the label prior actually MOVES between the fitted years and the graded year, by at
     least the total variation this bundle records. This is the check that makes "the
     graded fold is a genuine future" a measurement rather than a claim. It runs against
     the private labels, verifier-side, and its recorded value is 0.346777.

Point 5 is the one worth arguing about, so the argument is here rather than elsewhere. A
time split whose label distribution did not move would still be a time split, and the check
would be wrong to fail it. It is bound to this corpus, where the movement is a measured
0.346777 and the majority class changes from 28 to 24 across the boundary; the check
asserts that the SHIPPED bytes still carry that movement, and it fires when someone
reshuffles the split into a random one while leaving the index array names alone. That
substitution is the failure it exists to catch.
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)
ENVIRONMENT = os.path.join(BUNDLE, "environment")
GRADED_LABELS = os.path.join(HERE, "heldout", "graded_labels.npz")
GRAPH = os.path.join(ENVIRONMENT, "data", "graph.npz")

RECORDED_PRIOR_TV_TRAIN_TO_GRADED = 0.346777
RECORDED_TRAIN_MAJORITY = 28
RECORDED_GRADED_MAJORITY = 24
N_CLASSES = 40
TV_TOLERANCE = 0.01


def digest(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def total_variation(a, b):
    import numpy as np
    pa = np.asarray(a, dtype="float64")
    pb = np.asarray(b, dtype="float64")
    return float(0.5 * abs(pa / pa.sum() - pb / pb.sum()).sum())


def scan(environment=ENVIRONMENT, graded_labels=GRADED_LABELS, graph=GRAPH):
    import numpy as np
    findings, scanned = [], 0

    if not os.path.exists(graded_labels):
        return {"isolated": False, "files_scanned": 0,
                "findings": [{"reason": "graded labels missing", "path": graded_labels}]}
    held = np.load(graded_labels)
    if sorted(held.files) != ["y"]:
        findings.append({"reason": "graded label file must carry exactly one array named y",
                         "path": graded_labels, "keys": sorted(held.files)})
    y_graded = held["y"].astype("int64")
    graded_digest = digest(graded_labels)

    if not os.path.exists(graph):
        findings.append({"reason": "shipped graph missing", "path": graph})
        return {"isolated": False, "files_scanned": 0, "findings": findings}
    with np.load(graph) as data:
        train_idx = data["train_idx"]
        val_idx = data["val_idx"]
        graded_idx = data["graded_idx"]
        y_train = data["y_train"].astype("int64")
        n_nodes = data["X"].shape[0]

    # 1. the three folds partition the node set
    sets = {"train": set(train_idx.tolist()), "val": set(val_idx.tolist()),
            "graded": set(graded_idx.tolist())}
    for a, b in (("train", "val"), ("train", "graded"), ("val", "graded")):
        shared = sets[a] & sets[b]
        if shared:
            findings.append({"reason": "folds overlap", "folds": [a, b],
                             "count": len(shared)})
    covered = len(sets["train"] | sets["val"] | sets["graded"])
    if covered != n_nodes:
        findings.append({"reason": "folds do not cover the node set",
                         "covered": covered, "nodes": int(n_nodes)})

    # 2. one label per graded node, and nothing else
    if len(y_graded) != len(graded_idx):
        findings.append({"reason": "graded label count does not match the graded fold",
                         "labels": int(len(y_graded)), "graded_nodes": int(len(graded_idx))})

    # 3 and 4. the labels are reachable from nowhere the agent can read
    for root, _, files in os.walk(environment):
        for name in files:
            path = os.path.join(root, name)
            scanned += 1
            if digest(path) == graded_digest:
                findings.append({"reason": "graded label file present in environment",
                                 "path": path})
            if not name.endswith((".npz", ".npy")):
                continue
            loaded = np.load(path, allow_pickle=False)
            arrays = ({k: loaded[k] for k in loaded.files}
                      if hasattr(loaded, "files") else {"": loaded})
            for key, array in arrays.items():
                if (np.issubdtype(array.dtype, np.integer)
                        and array.shape == y_graded.shape
                        and bool((array.astype("int64") == y_graded).all())):
                    findings.append({"reason": "graded label vector present in environment",
                                     "path": path, "key": key})
    if scanned == 0:
        findings.append({"reason": "environment tree empty; the scan is vacuous",
                         "path": environment})

    # 5. the split is a calendar, and the calendar moved
    train_hist = np.bincount(y_train, minlength=N_CLASSES)
    graded_hist = np.bincount(y_graded, minlength=N_CLASSES)
    tv = total_variation(train_hist, graded_hist)
    majority_moved = int(train_hist.argmax()) != int(graded_hist.argmax())
    if tv < RECORDED_PRIOR_TV_TRAIN_TO_GRADED - TV_TOLERANCE:
        findings.append({"reason": "label prior no longer moves across the split boundary",
                         "measured_total_variation": round(tv, 6),
                         "recorded": RECORDED_PRIOR_TV_TRAIN_TO_GRADED})
    if not majority_moved:
        findings.append({"reason": "majority class no longer changes across the boundary",
                         "train_majority": int(train_hist.argmax()),
                         "graded_majority": int(graded_hist.argmax())})

    return {
        "isolated": not findings,
        "files_scanned": scanned,
        "folds": {"train": len(train_idx), "val": len(val_idx), "graded": len(graded_idx),
                  "nodes": int(n_nodes)},
        "prior_total_variation_train_to_graded": round(tv, 6),
        "train_majority_class": int(train_hist.argmax()),
        "graded_majority_class": int(graded_hist.argmax()),
        "findings": findings,
    }


def main():
    report = scan()
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if report["isolated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
