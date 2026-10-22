"""Re-evaluate a checkpoint ladder with the EXACT verifier convention.

    torch.jit.load(path).eval() on GPU; x = float32 NCHW batch of 512 of the 32x32 test
    PNGs under transforms.Compose([ToTensor(), Normalize((0.4914,0.4822,0.4465),
    (0.2470,0.2435,0.2616))]); prediction = model(x).argmax(1).

This is the ONLY code path in the project that reads the test split, and it is run only
after the promotion decision for a ladder has already been fixed (see artifacts/PROTOCOL.md).
"""
import argparse, json, os, sys, time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


def get_test_batches(dev="cuda", bs=512):
    ds = datasets.ImageFolder("/workspace/data/cifar10/test", transform=TF)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    return [(x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)) for x, y in dl]


@torch.no_grad()
def acc_of(path, batches):
    m = torch.jit.load(path).eval().cuda()
    c = t = 0
    for x, y in batches:
        c += (m(x).argmax(1) == y).sum().item()
        t += y.numel()
    return c / t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/periodic_checkpoints")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()

    batches = get_test_batches()
    n_test = sum(y.numel() for _, y in batches)
    print(f"test images: {n_test}, batches: {len(batches)}, "
          f"input range [{batches[0][0].min():.2f},{batches[0][0].max():.2f}]", flush=True)

    files = sorted(f for f in os.listdir(a.dir) if f.startswith("ckpt_epoch_"))
    nums = [int(f.split("_")[-1].split(".")[0]) for f in files]
    contiguous = nums == list(range(1, len(nums) + 1))
    mt = {f: os.path.getmtime(os.path.join(a.dir, f)) for f in files}
    t0 = mt[files[0]]

    res, first_cross = [], None
    for i, f in enumerate(files):
        if a.stride > 1 and (i % a.stride) and i != len(files) - 1 and first_cross is None:
            continue
        acc = acc_of(os.path.join(a.dir, f), batches)
        N = nums[i]
        tstar = (mt[f] - t0) * N / (N - 1) if N > 1 else float("nan")
        res.append(dict(file=f, epoch=N, acc=acc, t_star=tstar))
        flag = ""
        if acc >= 0.94 and first_cross is None:
            first_cross = res[-1]; flag = "  <== FIRST CROSSING"
        print(f"  {f}  acc={acc:.4f}  t_star={tstar:8.3f}s{flag}", flush=True)

    ladder_span = mt[files[-1]] - t0
    summary = dict(dir=a.dir, n_ckpt=len(files), contiguous=contiguous,
                   ladder_span_sec=ladder_span,
                   mean_epoch_sec=ladder_span / max(1, len(files) - 1),
                   first_cross=first_cross,
                   best_acc=max(r["acc"] for r in res),
                   results=res)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=1))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
