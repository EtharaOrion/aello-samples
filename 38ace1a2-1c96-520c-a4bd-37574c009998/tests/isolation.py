"""Prove the private graded labels are not reachable from anything the solving agent can read.

Authoring/freeze gate, run on the host against the bundle tree. The grade-time twin lives
in tests/test_output.py (graded_split_isolated) and scans the agent-authored tree instead.

An isolation claim that is asserted rather than executed is worth nothing after freeze, so
this walks the environment tree and checks, with NO sampling stride:

  1. no environment file matches the sealed graded split by content digest;
  2. no environment array equals the graded label vector;
  3. environment/data/graded_images.npz carries exactly the graded images, in the graded
     order, and no label array -- the images are public by design, the labels are the secret;
  4. no OTHER environment image content overlaps the graded split, except the recorded
     upstream duplicates in tests/heldout/known_overlap.json (byte-identical images that
     exist in both the official CIFAR-100 train and test splits).
"""
import hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = os.path.join(os.path.dirname(HERE), "environment")
GRADED = os.path.join(HERE, "heldout", "graded_test.npz")
GRADED_IMAGES = os.path.join(ENV, "data", "graded_images.npz")
KNOWN = os.path.join(HERE, "heldout", "known_overlap.json")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def main():
    findings = []
    g = np.load(GRADED)
    gx, gy = g["x"], g["y"].astype(np.int64)
    gd = sha(GRADED)
    known = json.load(open(KNOWN)) if os.path.exists(KNOWN) else {"known_duplicates": []}
    allowed_train = {d["train_index"] for d in known.get("known_duplicates", [])}

    scanned = 0
    for root, _, files in os.walk(ENV):
        for f in files:
            p = os.path.join(root, f)
            scanned += 1
            if sha(p) == gd:
                findings.append({"reason": "graded split present in environment", "path": p})
            if f.endswith((".npz", ".npy")) and os.path.abspath(p) != os.path.abspath(GRADED_IMAGES):
                d = np.load(p)
                arrs = {k: d[k] for k in d.files} if hasattr(d, "files") else {"": d}
                for k, a in arrs.items():
                    if np.issubdtype(a.dtype, np.integer) and a.shape == gy.shape and (a == gy).all():
                        findings.append({"reason": "graded label vector present", "path": p, "key": k})
    if scanned == 0:
        findings.append({"reason": "environment tree empty; isolation scan is vacuous", "path": ENV})

    # 3. the shipped graded images: exactly the graded x, in order, label-free
    if not os.path.exists(GRADED_IMAGES):
        findings.append({"reason": "graded images missing from environment", "path": GRADED_IMAGES})
    else:
        gi = np.load(GRADED_IMAGES)
        if set(gi.files) != {"x"}:
            findings.append({"reason": "graded_images.npz must carry exactly {'x'}",
                             "path": GRADED_IMAGES, "keys": sorted(gi.files)})
        elif gi["x"].shape != gx.shape or not (gi["x"] == gx).all():
            findings.append({"reason": "graded_images.npz does not align with the graded split",
                             "path": GRADED_IMAGES})

    # 4. full image-content overlap, whitelisting only the recorded upstream duplicates
    gset = {gx[i].tobytes() for i in range(len(gy))}
    for name in ("train_i20k.npz", "val_public.npz"):
        p = os.path.join(ENV, "data", name)
        if not os.path.exists(p):
            continue
        x = np.load(p)["x"]
        hits = [i for i in range(len(x)) if x[i].tobytes() in gset]
        unexplained = [i for i in hits if not (name == "train_i20k.npz" and i in allowed_train)]
        if unexplained:
            findings.append({"reason": "image content shared with the graded split",
                             "path": p, "indices": unexplained[:20],
                             "count": len(unexplained)})

    print(json.dumps({"isolated": not findings, "files_scanned": scanned,
                      "known_duplicates_whitelisted": sorted(allowed_train),
                      "findings": findings}, indent=1))
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
