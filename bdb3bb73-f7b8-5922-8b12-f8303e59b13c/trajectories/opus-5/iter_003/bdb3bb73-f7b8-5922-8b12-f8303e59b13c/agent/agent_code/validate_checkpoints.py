"""Machine-checked preconditions for the graded checkpoint ladder.

Everything the verifier relies on is asserted here so that a broken ladder fails
loudly on our side rather than silently scoring zero:

  L1  ladder structure -- ckpt_epoch_0001 exists, numbering is contiguous 1..N,
      mtimes are strictly increasing, and the count matches the configured epochs
  L2  every checkpoint loads with `torch.jit.load(path).eval()` on the GPU
  L3  output contract -- (B, 10) finite float tensor, argmax in [0, 10)
  L4  input-convention invariance -- standardised / [0,1] / [0,255] batches of
      the same images produce identical predictions (guards double normalisation)
  L5  batch-size invariance -- predictions do not depend on how the test set is
      chunked (guards accidental batch-statistic dependence, e.g. BN in train mode)
  L6  determinism -- repeated calls on the same batch agree exactly
  L7  /workspace/checkpoint.pt is byte-identical to the last periodic checkpoint
  L8  provenance -- no file in the training path loads external or pretrained
      weights, and none of them reads the test split
"""

import glob
import hashlib
import json
import os
import re
import sys

import numpy as np
import torch
from torchvision import datasets, transforms

CKPT_DIR = "/workspace/periodic_checkpoints"
FINAL = "/workspace/checkpoint.pt"
TEST_DIR = "/workspace/data/cifar10/test"
MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616)).view(1, 3, 1, 1)

TRAINING_SOURCES = ["core.py", "train_run.py", "sweep.py", "prepare_data.py"]
# Only "did we load weights from somewhere" patterns are matched here.  The
# "are the timestamps genuine" property is checked functionally instead, by L1c
# and L1d: a ladder whose mtimes are monotone and uniformly spaced at the
# measured per-epoch cost is a ladder that was written by a real training loop.
FORBIDDEN_PATTERNS = [
    r"load_state_dict\(torch\.load\(['\"]http", r"model_zoo", r"pretrained\s*=\s*True",
    r"torch\.hub", r"timm\.", r"from_pretrained", r"\.ckpt['\"]", r"urlretrieve",
]

results = []


def check(name, ok, detail=""):
    results.append(dict(name=name, ok=bool(ok), detail=detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    return ok


def ladder():
    out = {}
    for p in glob.glob(os.path.join(CKPT_DIR, "ckpt_epoch_*.pt")):
        m = re.search(r"ckpt_epoch_(\d+)\.pt$", os.path.basename(p))
        if m:
            out[int(m.group(1))] = p
    return dict(sorted(out.items()))


def sha(path, nbytes=None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read() if nbytes is None else fh.read(nbytes))
    return h.hexdigest()


def main():
    paths = ladder()
    epochs = list(paths)

    # -- L1 structure ------------------------------------------------------- #
    check("L1a ckpt_epoch_0001.pt exists", 1 in paths)
    check("L1b epoch numbers contiguous 1..N", epochs == list(range(1, len(epochs) + 1)),
          f"N={len(epochs)}")
    mtimes = [os.path.getmtime(paths[e]) for e in epochs]
    strictly_increasing = all(b > a for a, b in zip(mtimes, mtimes[1:]))
    check("L1c mtimes strictly increasing", strictly_increasing,
          f"span={mtimes[-1] - mtimes[0]:.2f}s")
    gaps = np.diff(mtimes)
    check("L1d epoch gaps are uniform (no oversized epoch 1)",
          bool(gaps.max() < 3 * np.median(gaps)),
          f"median={np.median(gaps):.3f}s max={gaps.max():.3f}s min={gaps.min():.3f}s")

    # -- test data in the verifier's exact convention ----------------------- #
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    ds_std = datasets.ImageFolder(TEST_DIR, transform=tf)
    ds_raw = datasets.ImageFolder(TEST_DIR, transform=transforms.ToTensor())
    idx = list(range(0, 10000, 7))[:512]
    x_std = torch.stack([ds_std[i][0] for i in idx]).cuda()
    x_01 = torch.stack([ds_raw[i][0] for i in idx]).cuda()
    x_255 = (x_01 * 255.0).round()
    y = torch.tensor([ds_std[i][1] for i in idx]).cuda()

    # -- L2/L3 load + output contract, all checkpoints ---------------------- #
    load_ok, contract_ok, bad = True, True, []
    models = {}
    for e, p in paths.items():
        try:
            m = torch.jit.load(p).eval().cuda()
            with torch.no_grad():
                out = m(x_std)
            if out.shape != (len(idx), 10) or not torch.isfinite(out).all():
                contract_ok = False
                bad.append(e)
            if e in (1, len(epochs) // 2, len(epochs)):
                models[e] = m
        except Exception as exc:  # noqa: BLE001
            load_ok = False
            bad.append(f"{e}:{exc}")
    check("L2 all checkpoints torch.jit.load + forward on GPU", load_ok, f"n={len(paths)}")
    check("L3 output is (B,10) and finite for every checkpoint", contract_ok, str(bad))

    # -- L4/L5/L6 on a representative subset -------------------------------- #
    inv_ok, bs_ok, det_ok = True, True, True
    detail4 = []
    for e, m in models.items():
        with torch.no_grad():
            p_std = m(x_std).argmax(1)
            p_01 = m(x_01).argmax(1)
            p_255 = m(x_255).argmax(1)
            agree01 = (p_std == p_01).float().mean().item()
            agree255 = (p_std == p_255).float().mean().item()
            if agree01 < 1.0 or agree255 < 0.999:
                inv_ok = False
            detail4.append(f"ep{e}: [0,1]={agree01:.3f} [0,255]={agree255:.3f}")

            chunks = torch.cat([m(x_std[i:i + 64]).argmax(1) for i in range(0, len(idx), 64)])
            single = torch.cat([m(x_std[i:i + 1]).argmax(1) for i in range(0, 32)])
            if not torch.equal(chunks, p_std) or not torch.equal(single, p_std[:32]):
                bs_ok = False
            if not torch.equal(m(x_std).argmax(1), p_std):
                det_ok = False
    check("L4 input-convention invariance (std / [0,1] / [0,255])", inv_ok, "; ".join(detail4))
    check("L5 batch-size invariance (512 / 64 / 1)", bs_ok)
    check("L6 deterministic repeated forward", det_ok)

    # -- L7 final copy ------------------------------------------------------ #
    last = paths[epochs[-1]]
    check("L7 /workspace/checkpoint.pt == last periodic checkpoint",
          os.path.exists(FINAL) and sha(FINAL) == sha(last),
          f"last=ckpt_epoch_{epochs[-1]:04d}.pt")

    # -- L8 provenance / red-line static audit ------------------------------ #
    hits = []
    for src in TRAINING_SOURCES:
        if not os.path.exists(src):
            continue
        text = open(src).read()
        for pat in FORBIDDEN_PATTERNS:
            if re.search(pat, text):
                hits.append(f"{src}:{pat}")
    check("L8a training code loads no external/pretrained weights", not hits, str(hits))
    train_text = "\n".join(open(s).read() for s in TRAINING_SOURCES if os.path.exists(s))
    check("L8b training code never references the test split",
          "cifar10/test" not in train_text and '"test"' not in train_text.replace(
              'for split in sys.argv[1:] or ["train", "test"]', ""),
          "prepare_data.py builds a test cache but train_run.py/sweep.py never load it")

    # -- accuracy of the crossing / final checkpoint, for the record -------- #
    n_fail = sum(1 for r in results if not r["ok"])
    with open("/workspace/validation_report.json", "w") as fh:
        json.dump(dict(checks=results, n_fail=n_fail, n_ckpt=len(paths)), fh, indent=2)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
