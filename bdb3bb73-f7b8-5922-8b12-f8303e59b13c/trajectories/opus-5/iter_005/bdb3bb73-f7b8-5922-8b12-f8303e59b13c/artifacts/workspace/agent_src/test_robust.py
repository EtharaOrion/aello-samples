"""Robustness checks on the FINAL checkpoint, beyond plain accuracy.

Each check targets a way a checkpoint can look fine locally but fail (or be
judged invalid) under the verifier.
"""
import os
import sys

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.datasets import ImageFolder

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/checkpoint.pt"
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")


# ---- 1. loads on GPU and on CPU -------------------------------------------
m = torch.jit.load(CKPT, map_location="cuda").eval()
mc = torch.jit.load(CKPT, map_location="cpu").eval()
check("torch.jit.load on cuda and cpu", True)

# ---- 2. real torchvision pipeline, exactly as the verifier builds it -------
tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
ds = ImageFolder("/workspace/data/cifar10/test", transform=tf)
X = torch.stack([ds[i][0] for i in range(2048)]).cuda()
Y = torch.tensor([ds[i][1] for i in range(2048)]).cuda()
check("verifier input is float32 NCHW", X.dtype == torch.float32 and X.ndim == 4,
      f"shape={tuple(X.shape)} range=[{X.min():.2f},{X.max():.2f}]")

# ---- 3. cached-uint8 path is bit-identical to the torchvision path ---------
d = torch.load("/workspace/cache/test.pt", weights_only=False)
xu = d["x"][:2048].cuda().float().div_(255.0)
mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)
xu = (xu - mean) / std
# NOTE: not bit-identical. ToTensor()/Normalize() and the cached path apply the
# same maths in a different op order, so results differ by ~1 float32 ULP
# (~5e-7). What matters is that the model's PREDICTIONS are unchanged, which is
# asserted separately below.
dmax = (xu - X).abs().max().item()
check("cached pipeline == torchvision pipeline (float32 ULP)", dmax < 1e-5,
      f"max|d|={dmax:.3e}")
with torch.no_grad():
    check("predictions identical on both input paths",
          torch.equal(m(xu[:1024]).argmax(1), m(X[:1024]).argmax(1)))
check("cached labels == ImageFolder labels", torch.equal(d["y"][:2048].cuda(), Y))

# ---- 4. batch-size independence (verifier uses 512) -----------------------
with torch.no_grad():
    ref = m(X[:512]).argmax(1)
    outs = {}
    for bs in (1, 7, 512, 1024, 2048):
        pr = torch.cat([m(X[i:i + bs]).argmax(1) for i in range(0, 512, bs)])[:512]
        outs[bs] = (pr == ref).float().mean().item()
check("prediction independent of batch size", all(v == 1.0 for v in outs.values()), outs)

# ---- 5. determinism --------------------------------------------------------
with torch.no_grad():
    check("deterministic across repeat calls", torch.equal(m(X[:512]), m(X[:512])))

# ---- 6. GPU vs CPU agreement ----------------------------------------------
with torch.no_grad():
    agree = (mc(X[:256].cpu()).argmax(1) == m(X[:256]).argmax(1).cpu()).float().mean().item()
check("CPU and GPU predictions agree", agree > 0.99, f"agreement={agree:.4f}")

# ---- 7. eval() really uses BN running stats (no batch-stat leakage) --------
with torch.no_grad():
    single = torch.cat([m(X[i:i + 1]).argmax(1) for i in range(64)])
    batched = m(X[:64]).argmax(1)
check("batch-of-1 == batched (BN in eval mode)", torch.equal(single, batched))

# ---- 8. accuracy on the full test split, verifier-style --------------------
Xa = torch.stack([ds[i][0] for i in range(len(ds))]).cuda()
Ya = torch.tensor([ds[i][1] for i in range(len(ds))]).cuda()
with torch.no_grad():
    corr = sum((m(Xa[i:i + 512]).argmax(1) == Ya[i:i + 512]).sum().item()
               for i in range(0, len(ds), 512))
acc = corr / len(ds)
check("test accuracy >= 0.94", acc >= 0.94, f"acc={acc*100:.2f}%")

# ---- 9. output shape / no NaNs --------------------------------------------
with torch.no_grad():
    o = m(X[:512])
check("output shape (N,10), finite", o.shape == (512, 10) and torch.isfinite(o).all(),
      f"shape={tuple(o.shape)} dtype={o.dtype}")

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
