"""Re-evaluate saved checkpoints through the EXACT verifier pipeline.

The verifier does:
    model = torch.jit.load(path).eval()            # on GPU
    ds    = ImageFolder(test, Compose([ToTensor(), Normalize(MEAN, STD)]))
    pred  = model(x).argmax(1)                     # batch size 512

``--strict`` runs literally that (torchvision ImageFolder + transforms).  The
default fast path builds the identical tensor from the uint8 cache; ``--strict``
is used at least once per session to prove the two agree to float tolerance.
"""

import argparse
import glob
import os
import re
import time

import numpy as np
import torch

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
TEST_DIR = "/workspace/data/cifar10/test"


def strict_test_tensor():
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ds = datasets.ImageFolder(TEST_DIR, transform=tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
    xs, ys = [], []
    for x, y in dl:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def fast_test_tensor():
    x = torch.from_numpy(np.load("/workspace/cache/test_x.npy")).permute(0, 3, 1, 2)
    y = torch.from_numpy(np.load("/workspace/cache/test_y.npy"))
    x = x.float().div_(255.0)
    m = torch.tensor(MEAN).view(1, 3, 1, 1)
    s = torch.tensor(STD).view(1, 3, 1, 1)
    return (x - m) / s, y


@torch.no_grad()
def acc_of(path, x, y, bs=512):
    model = torch.jit.load(path).eval().cuda()
    correct = 0
    for i in range(0, len(x), bs):
        pred = model(x[i:i + bs].cuda()).argmax(1)
        correct += (pred == y[i:i + bs].cuda()).sum().item()
    del model
    return correct / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/periodic_checkpoints")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--from-epoch", type=int, default=1)
    ap.add_argument("--every", type=int, default=1)
    args = ap.parse_args()

    if args.strict:
        xs, ys = strict_test_tensor()
        xf, yf = fast_test_tensor()
        print(f"[strict-vs-fast] max|dx|={float((xs - xf).abs().max()):.3e} "
              f"labels_equal={bool((ys == yf).all())}", flush=True)
        x, y = xs, ys
    else:
        x, y = fast_test_tensor()

    paths = sorted(glob.glob(os.path.join(args.dir, "ckpt_epoch_*.pt")))
    eps = [int(re.search(r"(\d{4})", os.path.basename(p)).group(1)) for p in paths]
    if eps:
        contiguous = eps == list(range(1, len(eps) + 1))
        print(f"[ladder] n={len(eps)} epochs {eps[0]}..{eps[-1]} contiguous={contiguous}",
              flush=True)
    mt = {p: os.path.getmtime(p) for p in paths}
    first_cross = None
    for p, e in zip(paths, eps):
        if e < args.from_epoch or (e - args.from_epoch) % args.every:
            continue
        a = acc_of(p, x, y)
        span = mt[p] - mt[paths[0]]
        tstar = span * e / (e - 1) if e > 1 else float("nan")
        flag = ""
        if a >= 0.94 and first_cross is None:
            first_cross = (e, a, tstar)
            flag = "  <-- FIRST >= 94%"
        print(f"epoch {e:4d}  acc {a*100:6.2f}%  span {span:7.2f}s  t*={tstar:7.2f}s{flag}",
              flush=True)
    if first_cross:
        e, a, t = first_cross
        s_out = max(0.0, min(1.0, (70.0 - t) / (70.0 - 28.0)))
        print(f"\nFIRST CROSSING epoch {e}  acc {a*100:.2f}%  t*={t:.2f}s  "
              f"S_outcome={s_out:.4f}", flush=True)
    else:
        print("\nNO CROSSING >= 94%", flush=True)


if __name__ == "__main__":
    main()
