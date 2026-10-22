"""Independent re-evaluation of a checkpoint ladder, using the verifier's
exact input pipeline.

    ImageFolder(test) -> Compose([ToTensor(), Normalize(MEAN, STD)])
    -> float32 NCHW batches of 512 -> torch.jit.load(ckpt).eval()(x).argmax(1)

Nothing in this file is used to make training decisions; it exists to confirm
that the artefact on disk behaves for the verifier the way it does for us.
Hyper-parameter selection happens in sweep.py, on a held-out slice of the
*train* split only.
"""
import argparse
import glob
import json
import os
import re
import time

import torch
from torchvision import datasets, transforms

TEST_DIR = "/workspace/data/cifar10/test"
CACHE = "/workspace/cache/test_grader.pt"


def grader_test_tensors(force=False):
    """float32 NCHW test images exactly as the verifier feeds them."""
    if os.path.exists(CACHE) and not force:
        blob = torch.load(CACHE, map_location="cpu")
        return blob["x"], blob["y"]
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    ds = datasets.ImageFolder(TEST_DIR, transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False,
                                         num_workers=0)
    xs, ys = [], []
    for xb, yb in loader:
        xs.append(xb)
        ys.append(yb)
    x, y = torch.cat(xs), torch.cat(ys)
    torch.save({"x": x, "y": y, "classes": ds.classes}, CACHE)
    return x, y


@torch.no_grad()
def accuracy(path, x, y):
    model = torch.jit.load(path).eval().cuda()
    correct = 0
    for i in range(0, len(x), 512):
        xb = x[i:i + 512].cuda(non_blocking=True)
        correct += (model(xb).argmax(1).cpu() == y[i:i + 512]).sum().item()
    return correct / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckptdir")
    ap.add_argument("--json", default=None)
    ap.add_argument("--force-rebuild", action="store_true")
    args = ap.parse_args()

    x, y = grader_test_tensors(force=args.force_rebuild)
    paths = sorted(glob.glob(os.path.join(args.ckptdir, "ckpt_epoch_*.pt")))
    rows = []
    for p in paths:
        n = int(re.search(r"(\d+)\.pt$", p).group(1))
        acc = accuracy(p, x, y)
        rows.append({"epoch": n, "acc": acc, "mtime": os.path.getmtime(p),
                     "path": os.path.basename(p)})
        print(f"  {os.path.basename(p)}  acc={acc:.4f}")

    if rows:
        epochs = [r["epoch"] for r in rows]
        contiguous = epochs == list(range(1, len(epochs) + 1))
        first = next((r for r in rows if r["acc"] >= 0.94), None)
        t0 = rows[0]["mtime"]
        summary = {
            "n_ckpt": len(rows), "contiguous_from_1": contiguous,
            "best_acc": max(r["acc"] for r in rows),
            "final_acc": rows[-1]["acc"],
            "ladder_span_sec": rows[-1]["mtime"] - t0,
        }
        if first is not None:
            n = first["epoch"]
            summary["first_cross_epoch"] = n
            summary["first_cross_acc"] = first["acc"]
            summary["t_star_sec"] = ((first["mtime"] - t0) * n / (n - 1)) if n > 1 else 0.0
        print(json.dumps(summary, indent=2))
        if args.json:
            with open(args.json, "w") as f:
                json.dump({"summary": summary, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
