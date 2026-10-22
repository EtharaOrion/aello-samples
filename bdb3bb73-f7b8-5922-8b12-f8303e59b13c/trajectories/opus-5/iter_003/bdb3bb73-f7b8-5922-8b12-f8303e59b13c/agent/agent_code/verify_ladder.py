"""Re-implement the verifier's evaluation and score a checkpoint ladder with it.

This deliberately uses `ImageFolder` over the raw test PNGs plus the exact
transform the task statement specifies, rather than our cached tensors, so that
any mismatch in class ordering, channel order, or normalisation shows up here
instead of in the graded run.

It also reconstructs the verifier's clock:

    t_star = (mtime[ckpt_N] - mtime[ckpt_1]) * N / (N - 1)

for the first N whose accuracy clears the threshold, and checks the structural
preconditions (ckpt_epoch_0001 present, contiguous numbering, monotone mtimes).
"""

import argparse
import glob
import json
import os
import re
import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TEST_DIR = "/workspace/data/cifar10/test"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
T_BASELINE, T_REFERENCE = 70.0, 28.0


def load_test_batches(batch_size=512, device="cuda"):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ds = datasets.ImageFolder(TEST_DIR, transform=tf)
    # num_workers=0: /dev/shm in this container is too small for worker IPC and
    # decoding 10k 32x32 PNGs single-threaded takes about a second anyway.
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    batches = [(x.to(device, non_blocking=True), y.to(device, non_blocking=True)) for x, y in dl]
    assert sum(len(y) for _, y in batches) == 10000
    return batches, ds.classes


@torch.no_grad()
def accuracy(path, batches):
    model = torch.jit.load(path).eval().cuda()
    correct = 0
    for x, y in batches:
        correct += (model(x).argmax(1) == y).sum().item()
    return correct / sum(len(y) for _, y in batches)


def ladder_paths(ckpt_dir):
    out = {}
    for p in glob.glob(os.path.join(ckpt_dir, "ckpt_epoch_*.pt")):
        m = re.search(r"ckpt_epoch_(\d+)\.pt$", os.path.basename(p))
        if m:
            out[int(m.group(1))] = p
    return dict(sorted(out.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="/workspace/periodic_checkpoints")
    ap.add_argument("--threshold", type=float, default=0.94)
    ap.add_argument("--out", default=None)
    ap.add_argument("--start", type=int, default=1, help="skip cheap early epochs")
    args = ap.parse_args()

    batches, classes = load_test_batches()
    print(f"test set: 10000 imgs, classes={classes}", flush=True)

    paths = ladder_paths(args.ckpt_dir)
    epochs = list(paths)
    problems = []
    if not epochs:
        problems.append("no checkpoints found")
    else:
        if epochs[0] != 1:
            problems.append(f"ladder starts at epoch {epochs[0]}, must start at 1")
        if epochs != list(range(1, len(epochs) + 1)):
            problems.append("epoch numbers are not contiguous 1..N")
        mtimes = [os.path.getmtime(paths[e]) for e in epochs]
        if any(b < a for a, b in zip(mtimes, mtimes[1:])):
            problems.append("checkpoint mtimes are not monotonically increasing")

    mt = {e: os.path.getmtime(p) for e, p in paths.items()}
    t0 = mt[epochs[0]] if epochs else 0.0
    results, crossing = [], None
    for e in epochs:
        if e < args.start:
            continue
        t_eval = time.time()
        acc = accuracy(paths[e], batches)
        rel = mt[e] - t0
        t_star = rel * e / (e - 1) if e > 1 else float("nan")
        results.append(dict(epoch=e, acc=acc, rel_sec=rel, t_star=t_star,
                            eval_sec=time.time() - t_eval))
        flag = ""
        if crossing is None and acc >= args.threshold and e > 1:
            crossing = results[-1]
            flag = "  <-- FIRST CROSSING"
        print(f"  epoch {e:3d}  acc={acc:.4f}  rel={rel:6.2f}s  t*={t_star:6.2f}s{flag}", flush=True)

    summary = dict(ckpt_dir=args.ckpt_dir, n_ckpt=len(epochs), problems=problems,
                   results=results, threshold=args.threshold)
    if crossing:
        s_out = min(max((T_BASELINE - crossing["t_star"]) / (T_BASELINE - T_REFERENCE), 0.0), 1.0)
        summary.update(crossing_epoch=crossing["epoch"], t_star_sec=crossing["t_star"],
                       s_outcome=s_out, final_acc=results[-1]["acc"])
        print(f"\nFIRST >= {args.threshold:.2f}: epoch {crossing['epoch']} "
              f"acc={crossing['acc']:.4f}  t*={crossing['t_star']:.2f}s  S_outcome={s_out:.4f}")
    else:
        summary.update(crossing_epoch=None, s_outcome=0.0,
                       final_acc=results[-1]["acc"] if results else None)
        print(f"\nNO checkpoint reached {args.threshold:.2f}  -> S_outcome = 0")
    print("structural problems:", problems or "none")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
