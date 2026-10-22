"""Re-evaluate a checkpoint ladder through the EXACT pipeline the grader uses.

    torch.jit.load(path).eval()  ->  model(x).argmax(1)
    x = ToTensor() + Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)) applied to the
        32x32 PNGs, batch size 512, float32 NCHW  (i.e. ALREADY standardised)

--split holdout  : the 5,000-image train slice used for all model selection (safe to run
                   as often as we like; it is the same slice train.py holds out).
--split test     : the real test split. Used only for the final report, after the ladder
                   has already been chosen and promoted.

It also reconstructs the grader's clock from checkpoint save times:
    t_star = (save[N] - save[1]) * N / (N - 1)
"""
import argparse
import glob
import json
import os
import re

import torch
from PIL import Image
from torchvision import transforms

ROOT = "/workspace/data/cifar10"
TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


def load_images(split, holdout=0):
    classes = sorted(os.listdir(os.path.join(ROOT, "train")))
    paths, labels = [], []
    sub = "train" if split == "holdout" else split
    for ci, c in enumerate(classes):
        d = os.path.join(ROOT, sub, c)
        for fn in sorted(os.listdir(d)):
            paths.append(os.path.join(d, fn))
            labels.append(ci)
    keep = range(len(paths))
    if split == "holdout":
        g = torch.Generator().manual_seed(1234)  # identical to train.py's holdout split
        keep = torch.randperm(len(paths), generator=g)[:holdout].tolist()
    x = torch.stack([TF(Image.open(paths[i]).convert("RGB")) for i in keep])
    y = torch.tensor([labels[i] for i in keep])
    return x, y


@torch.no_grad()
def acc_of(path, x, y, bs=512):
    model = torch.jit.load(path).eval().cuda()
    correct = 0
    for i in range(0, x.shape[0], bs):
        pred = model(x[i:i + bs].cuda()).argmax(1)
        correct += (pred.cpu() == y[i:i + bs]).sum().item()
    return correct / x.shape[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="/workspace/periodic_checkpoints")
    p.add_argument("--split", default="holdout", choices=["holdout", "test"])
    p.add_argument("--holdout", type=int, default=5000)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--from-epoch", type=int, default=1, dest="from_epoch")
    p.add_argument("--json-out", default="", dest="json_out")
    a = p.parse_args()

    ckpts = sorted(glob.glob(os.path.join(a.dir, "ckpt_epoch_*.pt")))
    eps = [int(re.search(r"(\d{4})\.pt$", c).group(1)) for c in ckpts]
    assert eps == list(range(1, len(eps) + 1)), f"epochs not contiguous 1..N: {eps[:5]}...{eps[-3:]}"
    mt = [os.path.getmtime(c) for c in ckpts]

    x, y = load_images(a.split, a.holdout)
    print(f"[{a.split}] {x.shape[0]} images, {len(ckpts)} checkpoints, "
          f"input range [{x.min():.2f},{x.max():.2f}]", flush=True)

    rows, first = [], None
    for c, e, m in zip(ckpts, eps, mt):
        if e < a.from_epoch or (e % a.every and e != eps[-1]):
            continue
        acc = acc_of(c, x, y)
        tstar = (m - mt[0]) * e / (e - 1) if e > 1 else float("nan")
        rows.append({"epoch": e, "acc": acc, "t_star": round(tstar, 2)})
        flag = ""
        if acc >= 0.94 and first is None:
            first = rows[-1]
            flag = "  <-- first >= 94%"
        print(f"  ep{e:4d}  acc={acc:.4f}  t*={tstar:6.2f}s{flag}", flush=True)

    out = {"split": a.split, "n_ckpt": len(ckpts), "rows": rows, "first_cross": first,
           "ladder_span_sec": round(mt[-1] - mt[0], 2)}
    print(json.dumps(out["first_cross"] or {"first_cross": None}), flush=True)
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
