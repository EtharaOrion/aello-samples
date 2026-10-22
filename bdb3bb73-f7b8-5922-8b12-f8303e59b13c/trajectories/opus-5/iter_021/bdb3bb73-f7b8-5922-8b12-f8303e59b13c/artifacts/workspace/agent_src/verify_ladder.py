"""Replica of the verifier's evaluation path.

Loads every scripted checkpoint with `torch.jit.load(path).eval()`, feeds it
float32 NCHW batches of the test PNGs transformed by exactly
`Compose([ToTensor(), Normalize(CIFAR_MEAN, CIFAR_STD)])`, and takes argmax(1).

This is the only place the test split is ever read, and it runs only after a
configuration has been frozen -- it checks the artifact, it does not select it.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def test_tensor(device):
    x = np.load("/workspace/cache/test_x.npy")          # NHWC uint8, same PNGs
    y = np.load("/workspace/cache/test_y.npy")
    # ToTensor(): HWC uint8 -> CHW float32 in [0,1];  Normalize(): (x-m)/s
    t = torch.from_numpy(x).permute(0, 3, 1, 2).float().div_(255.0)
    m = torch.tensor(MEAN).view(1, 3, 1, 1)
    s = torch.tensor(STD).view(1, 3, 1, 1)
    return t.sub_(m).div_(s).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def acc_of(path, X, Y, bs=512):
    model = torch.jit.load(path).eval().to(X.device)
    correct = 0
    for i in range(0, len(X), bs):
        correct += (model(X[i:i + bs]).argmax(1) == Y[i:i + bs]).sum().item()
    return correct / len(X)


def main(ckpt_dir, out_json=""):
    dev = torch.device("cuda")
    X, Y = test_tensor(dev)
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_epoch_*.pt")))
    rows = []
    for p in paths:
        ep = int(os.path.basename(p)[11:15])
        a = acc_of(p, X, Y)
        rows.append(dict(epoch=ep, path=p, acc=a, mtime=os.path.getmtime(p)))
        print(f"epoch {ep:4d}  acc {a:.4f}", flush=True)
    cross = next((r for r in rows if r["acc"] >= 0.94), None)
    t0 = rows[0]["mtime"] if rows else None
    summary = dict(n_ckpt=len(rows), final_acc=rows[-1]["acc"] if rows else None,
                   best_acc=max((r["acc"] for r in rows), default=None),
                   first_cross_epoch=cross["epoch"] if cross else None)
    if cross and len(rows) > 1:
        n = cross["epoch"]
        # same reconstruction the verifier uses
        summary["t_star_sec"] = (cross["mtime"] - t0) * n / (n - 1)
        summary["ladder_span_sec"] = rows[-1]["mtime"] - t0
    print(json.dumps(summary, indent=2))
    if out_json:
        with open(out_json, "w") as f:
            json.dump(dict(summary=summary, rows=rows), f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/workspace/periodic_checkpoints",
         sys.argv[2] if len(sys.argv) > 2 else "")
