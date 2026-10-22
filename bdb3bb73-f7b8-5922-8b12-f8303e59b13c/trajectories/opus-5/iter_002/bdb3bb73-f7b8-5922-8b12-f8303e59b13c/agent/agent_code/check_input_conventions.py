"""Sanity-check that a saved checkpoint is correct under every input convention.

The verifier feeds standardized float32 NCHW tensors.  A model that silently
re-normalises them would still look fine on a [0,1] evaluation while scoring
~10-30% for the verifier, so we check all three conventions explicitly.
"""

import argparse

import torch

from evaluate_ladder import verifier_batches
from model import CIFAR_MEAN, CIFAR_STD


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/checkpoint.pt")
    args = ap.parse_args()

    model = torch.jit.load(args.ckpt, map_location="cuda").eval()
    batches = verifier_batches()
    mean = torch.tensor(CIFAR_MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device="cuda").view(1, 3, 1, 1)

    variants = {
        "standardized (verifier)": lambda x: x,
        "[0,1]": lambda x: x * std + mean,
        "[0,255]": lambda x: (x * std + mean) * 255.0,
    }
    for name, fn in variants.items():
        correct = total = 0
        for x, y in batches:
            correct += (model(fn(x)).argmax(1) == y).sum().item()
            total += len(y)
        print(f"{name:26s} acc = {100*correct/total:6.2f}%")

    # Batch-size independence: the verifier uses 512, but a final short batch is
    # a different shape, and the input-scale test must not depend on batch size.
    x, y = batches[0]
    for bs in (1, 7, 512):
        acc = (model(x[:bs]).argmax(1) == y[:bs]).float().mean().item()
        print(f"batch_size={bs:4d} on first {bs} test images: acc = {100*acc:6.2f}%")


if __name__ == "__main__":
    main()
