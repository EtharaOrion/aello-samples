"""Re-evaluate a checkpoint ladder exactly the way the verifier does.

The verifier loads each ``ckpt_epoch_NNNN.pt`` with ``torch.jit.load(...).eval()``,
feeds it float32 NCHW batches of 512 test PNGs transformed by
``ToTensor() -> Normalize(CIFAR_MEAN, CIFAR_STD)`` (i.e. already standardized),
and takes ``argmax(1)``.  This script replicates that pipeline and additionally
reconstructs the verifier's timing formula from the checkpoint mtimes:

    t_star = (mtime[ckpt_N] - mtime[ckpt_0001]) * N / (N - 1)

for the first N whose accuracy >= the target.

This is a post-hoc verification tool: the test split is only ever read here,
never by train.py or sweep.py.
"""

import argparse
import json
import os
import re

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import CIFAR_MEAN, CIFAR_STD

TEST_DIR = "/workspace/data/cifar10/test"


CACHED_TEST = "/workspace/cache/test_verifier_batches.pt"


def verifier_loader(batch_size: int = 512) -> DataLoader:
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    )
    ds = ImageFolder(TEST_DIR, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def verifier_batches(batch_size: int = 512):
    """The exact tensors the verifier feeds, cached on the GPU.

    Built once by running the real ImageFolder + ToTensor + Normalize pipeline,
    then memoised on disk so repeated ladder evaluations are cheap.
    """
    if os.path.exists(CACHED_TEST):
        blob = torch.load(CACHED_TEST, map_location="cuda")
    else:
        blob = [(x, y) for x, y in verifier_loader(batch_size)]
        torch.save(blob, CACHED_TEST)
        blob = torch.load(CACHED_TEST, map_location="cuda")
    return [(x.cuda(non_blocking=True), y.cuda(non_blocking=True)) for x, y in blob]


@torch.no_grad()
def accuracy(path: str, batches) -> float:
    model = torch.jit.load(path, map_location="cuda").eval()
    correct = total = 0
    for x, y in batches:
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += len(y)
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/periodic_checkpoints")
    ap.add_argument("--target", type=float, default=0.94)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    names = sorted(n for n in os.listdir(args.dir) if re.fullmatch(r"ckpt_epoch_\d+\.pt", n))
    epochs = [int(re.findall(r"\d+", n)[0]) for n in names]
    assert epochs == list(range(1, len(epochs) + 1)), f"epochs not contiguous 1..N: {epochs}"

    # Cache the standardized test batches on the GPU once (identical tensors for
    # every checkpoint, so the comparison is apples to apples).
    batches = verifier_batches()

    mtimes = {n: os.path.getmtime(os.path.join(args.dir, n)) for n in names}
    t0 = mtimes[names[0]]
    rows, t_star, crossing = [], None, None
    for name, epoch in zip(names, epochs):
        acc = accuracy(os.path.join(args.dir, name), batches)
        elapsed = mtimes[name] - t0
        rows.append(dict(epoch=epoch, acc=acc, since_first_ckpt=round(elapsed, 3)))
        marker = ""
        if acc >= args.target and t_star is None and epoch > 1:
            t_star = elapsed * epoch / (epoch - 1)
            crossing = epoch
            marker = f"  <== first >= {args.target:.2f}, t_star={t_star:.2f}s"
        print(f"epoch {epoch:4d}  acc {acc*100:6.2f}%  +{elapsed:7.3f}s{marker}")

    summary = dict(
        dir=args.dir, n_checkpoints=len(names), final_acc=rows[-1]["acc"],
        best_acc=max(r["acc"] for r in rows), crossing_epoch=crossing, t_star_sec=t_star,
        ladder_span_sec=round(mtimes[names[-1]] - t0, 3), rows=rows,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
