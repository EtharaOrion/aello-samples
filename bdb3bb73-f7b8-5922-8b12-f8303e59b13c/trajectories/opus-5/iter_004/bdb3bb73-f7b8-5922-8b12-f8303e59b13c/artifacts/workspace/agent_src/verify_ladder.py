"""Replay the verifier's exact evaluation over a checkpoint ladder.

Mirrors the stated verifier contract:
  * build the test batch from the 32x32 test PNGs with
      transforms.Compose([ToTensor(), Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
  * batch size 512, float32 NCHW, on GPU
  * model = torch.jit.load(path).eval();  pred = model(x).argmax(1)
and additionally reconstructs t* from the checkpoint mtimes the same way the
verifier does:  t* = (mtime[N] - mtime[1]) * N / (N-1).

Run with --torchvision to build the tensor through torchvision.ImageFolder
instead of the cached .npy, as a one-off check that the cache is faithful.
"""

import argparse
import json
import os
import re
import time

import numpy as np
import torch

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def test_tensor_from_torchvision():
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    ds = datasets.ImageFolder("/workspace/data/cifar10/test", transform=tf)
    xs = torch.stack([ds[i][0] for i in range(len(ds))])
    ys = torch.tensor([ds[i][1] for i in range(len(ds))])
    return xs, ys


def test_tensor_from_cache():
    x = np.load("/workspace/data_npy/test_x.npy")
    y = np.load("/workspace/data_npy/test_y.npy")
    x = torch.from_numpy(x).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
    return (x - mean) / std, torch.from_numpy(y)


@torch.no_grad()
def accuracy(path, x, y, batch=512):
    model = torch.jit.load(path).eval()
    correct = 0
    for i in range(0, x.size(0), batch):
        xb = x[i : i + batch].cuda()
        pred = model(xb).argmax(1)
        correct += (pred.cpu() == y[i : i + batch]).sum().item()
    return correct / x.size(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/periodic_checkpoints")
    ap.add_argument("--torchvision", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--every", type=int, default=1)
    args = ap.parse_args()

    t = time.time()
    x, y = test_tensor_from_torchvision() if args.torchvision else test_tensor_from_cache()
    print(f"[verify] test tensor {tuple(x.shape)} range=[{x.min():.3f},{x.max():.3f}] "
          f"built in {time.time()-t:.1f}s (torchvision={args.torchvision})")

    files = sorted(f for f in os.listdir(args.dir) if re.fullmatch(r"ckpt_epoch_\d{4}\.pt", f))
    eps = [int(f[11:15]) for f in files]
    contiguous = eps == list(range(1, len(eps) + 1))
    print(f"[verify] {len(files)} checkpoints, contiguous 1..N: {contiguous}")
    mt = {e: os.path.getmtime(os.path.join(args.dir, f)) for e, f in zip(eps, files)}

    rows, cross = [], None
    for e, f in zip(eps, files):
        if e % args.every and e != eps[-1]:
            continue
        acc = accuracy(os.path.join(args.dir, f), x, y)
        tstar = (mt[e] - mt[1]) * e / (e - 1) if e > 1 and 1 in mt else float("nan")
        flag = ""
        if acc >= 0.94 and cross is None:
            cross, flag = (e, acc, tstar), "  <== first >= 0.94"
        rows.append(dict(epoch=e, acc=acc, t_star=tstar))
        print(f"  epoch {e:4d}  acc={acc*100:6.2f}%  t*={tstar:6.2f}s{flag}")

    out = dict(contiguous=contiguous, n=len(files), rows=rows)
    if cross:
        e, acc, tstar = cross
        s_out = max(0.0, min(1.0, (70.0 - tstar) / (70.0 - 28.0)))
        out.update(cross_epoch=e, cross_acc=acc, t_star=tstar, s_outcome=s_out)
        print(f"[verify] FIRST >=94%: epoch {e} acc={acc*100:.2f}% t*={tstar:.2f}s -> S_outcome={s_out:.4f}")
    else:
        out.update(cross_epoch=None, s_outcome=0.0)
        print("[verify] NO checkpoint reached 94% -- S_outcome = 0")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
