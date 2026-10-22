"""Evaluate a checkpoint ladder through the *exact* pipeline the grader uses.

    torchvision.datasets.ImageFolder(<test>, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616))]))
    -> DataLoader(batch_size=512)
    -> torch.jit.load(ckpt).eval() on cuda -> argmax(1)

Nothing here is shared with the training code path on purpose: the images are
re-decoded from the PNGs on disk and re-normalized by torchvision, so a
mismatch between our training-time normalization and the grader's would show up
as a low number rather than being silently papered over.
"""
import argparse, json, os, re, sys, time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TEST_DIR = "/workspace/data/cifar10/test"


def get_loader(batch_size=512, workers=0):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    ds = datasets.ImageFolder(TEST_DIR, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=True)


def eval_ckpt(path, batches):
    model = torch.jit.load(path).eval().cuda()
    correct = total = 0
    with torch.no_grad():
        for x, y in batches:
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    del model
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/periodic_checkpoints")
    ap.add_argument("--out", default=None)
    ap.add_argument("--target", type=float, default=0.94)
    a = ap.parse_args()

    loader = get_loader()
    # cache the grader-normalized tensors on the GPU once (identical values,
    # just avoids re-decoding 10k PNGs for every checkpoint in the ladder)
    batches = [(x.cuda(non_blocking=True), y.cuda(non_blocking=True))
               for x, y in loader]
    assert sum(y.numel() for _, y in batches) == 10000

    files = sorted(f for f in os.listdir(a.dir) if re.fullmatch(r"ckpt_epoch_\d{4}\.pt", f))
    rows, first_cross = [], None
    mtimes = {}
    for f in files:
        p = os.path.join(a.dir, f)
        ep = int(re.findall(r"\d{4}", f)[0])
        acc = eval_ckpt(p, batches)
        mtimes[ep] = os.path.getmtime(p)
        rows.append(dict(epoch=ep, file=f, acc=acc))
        if first_cross is None and acc >= a.target:
            first_cross = ep
        print(f"{f}  acc={acc:.4f}", flush=True)

    summary = dict(dir=a.dir, n_ckpt=len(files), rows=rows,
                   first_crossing_epoch=first_cross,
                   best_acc=max((r["acc"] for r in rows), default=None))
    eps = sorted(mtimes)
    if len(eps) > 1 and eps == list(range(1, len(eps) + 1)):
        summary["ckpt_span_sec"] = mtimes[eps[-1]] - mtimes[1]
        if first_cross and first_cross > 1:
            summary["t_star_sec"] = (mtimes[first_cross] - mtimes[1]) * first_cross / (first_cross - 1)
            t = summary["t_star_sec"]
            summary["S_outcome"] = max(0.0, min(1.0, (70.0 - t) / (70.0 - 28.0)))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2), flush=True)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
