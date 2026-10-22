"""Machine-checked contract tests for the graded checkpoint ladder.

Every check is a hard pass/fail; the script exits non-zero if any fails. These
cover the failure modes the task description calls out (unloadable checkpoints,
non-contiguous ladders, double normalisation, stale checkpoints) plus a few of
our own (batch-size dependence, non-finite logits, mismatched final copy).
"""

import hashlib
import os
import re
import sys

import numpy as np
import torch

DIR = sys.argv[1] if len(sys.argv) > 1 else "/workspace/periodic_checkpoints"
FINAL = "/workspace/checkpoint.pt"
CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616)).view(1, 3, 1, 1)

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


# ---------------------------------------------------------------- ladder shape
files = sorted(f for f in os.listdir(DIR) if re.fullmatch(r"ckpt_epoch_\d{4}\.pt", f))
eps = [int(f[11:15]) for f in files]
check("ladder non-empty", len(files) > 0, f"{len(files)} checkpoints")
check("epochs contiguous 1..N", eps == list(range(1, len(eps) + 1)), f"1..{len(eps)}")
check("ckpt_epoch_0001.pt present", 1 in eps)

mt = [os.path.getmtime(os.path.join(DIR, f)) for f in files]
check("mtimes strictly increasing", all(b > a for a, b in zip(mt, mt[1:])),
      f"span {mt[-1]-mt[0]:.2f}s")
gaps = np.diff(mt)
check("per-epoch gaps plausible & uniform",
      gaps.min() > 0.05 and gaps.max() < 5 * gaps.min(),
      f"min={gaps.min():.3f}s max={gaps.max():.3f}s")

digests = [hashlib.md5(open(os.path.join(DIR, f), "rb").read()).hexdigest() for f in files]
check("no duplicate/stale checkpoints", len(set(digests)) == len(digests),
      f"{len(set(digests))} distinct of {len(digests)}")

check("final checkpoint.pt exists", os.path.exists(FINAL))
if os.path.exists(FINAL):
    d = hashlib.md5(open(FINAL, "rb").read()).hexdigest()
    check("checkpoint.pt == last periodic checkpoint", d == digests[-1])

# ---------------------------------------------------------------- test tensor
xr = np.load("/workspace/data_npy/test_x.npy")
y = torch.from_numpy(np.load("/workspace/data_npy/test_y.npy"))
x01 = torch.from_numpy(xr).permute(0, 3, 1, 2).float().div_(255.0)
xstd = (x01 - CIFAR_MEAN) / CIFAR_STD
x255 = x01 * 255.0
check("test tensor matches verifier convention",
      abs(float(xstd.min()) + 1.989) < 0.01 and abs(float(xstd.max()) - 2.126) < 0.01,
      f"range [{xstd.min():.3f}, {xstd.max():.3f}]")

# one-off cross-check that the .npy cache equals torchvision's ImageFolder pipeline
try:
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    ds = datasets.ImageFolder("/workspace/data/cifar10/test", transform=tf)
    idx = list(range(0, 10000, 971))
    tv = torch.stack([ds[i][0] for i in idx])
    lb = torch.tensor([ds[i][1] for i in idx])
    check("npy cache == torchvision ImageFolder pipeline",
          torch.allclose(tv, xstd[idx], atol=1e-6) and torch.equal(lb, y[idx]),
          f"{len(idx)} probe images, max|d|={float((tv-xstd[idx]).abs().max()):.2e}")
except Exception as e:  # pragma: no cover
    check("npy cache == torchvision ImageFolder pipeline", False, repr(e))


# ---------------------------------------------------------------- model contract
@torch.no_grad()
def predict(m, x, bs):
    return torch.cat([m(x[i : i + bs].cuda()).argmax(1).cpu() for i in range(0, x.size(0), bs)])


probe = slice(0, 2048)
for tag, path in [("epoch1", os.path.join(DIR, files[0])), ("last", os.path.join(DIR, files[-1]))]:
    m = torch.jit.load(path).eval()
    check(f"{tag}: torch.jit.load on GPU", True)
    with torch.no_grad():
        out = m(xstd[:512].cuda())
    check(f"{tag}: output is (512,10) float32 finite",
          tuple(out.shape) == (512, 10) and out.dtype == torch.float32 and bool(torch.isfinite(out).all()),
          f"{tuple(out.shape)} {out.dtype}")

    p512 = predict(m, xstd[probe], 512)
    check(f"{tag}: batch-size independent (512 vs 256 vs 1000)",
          torch.equal(p512, predict(m, xstd[probe], 256)) and torch.equal(p512, predict(m, xstd[probe], 1000)))
    a_std = (p512 == y[probe]).float().mean().item()
    a_01 = (predict(m, x01[probe], 512) == y[probe]).float().mean().item()
    a_255 = (predict(m, x255[probe], 512) == y[probe]).float().mean().item()
    check(f"{tag}: input-convention invariant (std / [0,1] / [0,255])",
          abs(a_std - a_01) < 1e-9 and abs(a_std - a_255) < 5e-3,
          f"std={a_std*100:.2f}% 01={a_01*100:.2f}% 255={a_255*100:.2f}%")

m1 = torch.jit.load(os.path.join(DIR, files[0])).eval()
mN = torch.jit.load(os.path.join(DIR, files[-1])).eval()
with torch.no_grad():
    d = float((m1(xstd[:512].cuda()) - mN(xstd[:512].cuda())).abs().max())
check("first and last checkpoint are different models", d > 1e-3, f"max|dlogit|={d:.3f}")

m_cpu = torch.jit.load(os.path.join(DIR, files[-1]), map_location="cpu").eval()
check("last checkpoint also loads on CPU", True)

nfail = sum(1 for _, ok, _ in results if not ok)
print(f"\n{len(results)-nfail}/{len(results)} checks passed")
sys.exit(1 if nfail else 0)
