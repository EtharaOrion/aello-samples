"""Emit graded predictions from ONE run's checkpoint with test-time augmentation.

Uses only views of the images produced by a single training run (raw weights + that run's
EMA copies). No second training run is involved.
"""
import argparse, time, numpy as np, torch, torch.nn.functional as F
from train import Net, MEAN, STD, bal_acc, DATA

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--tta", default="flip_shift")
ap.add_argument("--out", default="")
args = ap.parse_args()

ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
a = ck["args"]
dev = "cuda"
mean, std = MEAN.to(dev), STD.to(dev)
depths = tuple(int(v) for v in a["depths"].split(","))
models = []
for sd in [ck["raw"]] + ck["emas"]:
    m = Net(a["width"], depths, drop=0.0).to(dev).to(memory_format=torch.channels_last)
    m.load_state_dict(sd)
    m.eval()
    models.append(m)
R = a["res"]

dv = np.load(f"{DATA}/val_public.npz")
xva = torch.from_numpy(dv["x"]).to(dev).permute(0, 3, 1, 2).contiguous()
yva = dv["y"]
xg = torch.from_numpy(np.load(f"{DATA}/graded_images.npz")["x"]).to(dev).permute(0, 3, 1, 2).contiguous()

SHIFTS = ((0, 0), (0, 2), (0, -2), (2, 0), (-2, 0))


def view_specs(kind):
    if kind == "flip":
        return [(f, (0, 0), R) for f in (0, 1)]
    if kind == "flip_shift":
        return [(f, s, R) for f in (0, 1) for s in SHIFTS]
    if kind == "flip_scale2":
        return [(f, (0, 0), r) for f in (0, 1) for r in (R, R + 8)]
    if kind == "flip_shift_scale":
        return [(f, s, r) for f in (0, 1) for s in SHIFTS for r in (R, R + 8)]
    raise ValueError(kind)


@torch.no_grad()
def probs(xu8, kind, bs=500):
    specs = view_specs(kind)
    acc = []
    for s in range(0, xu8.shape[0], bs):
        xb = xu8[s:s + bs].float().div(255.)
        xf = torch.flip(xb, dims=[3])
        p = 0
        for f, sh, r in specs:
            v = xf if f else xb
            if sh != (0, 0):
                v = torch.roll(v, shifts=sh, dims=(2, 3))
            z = (v - mean) / std
            z = F.interpolate(z, size=(r, r), mode="bilinear", align_corners=False)
            z = z.contiguous(memory_format=torch.channels_last)
            for m in models:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = p + F.softmax(m(z).float(), 1)
        acc.append(p / (len(specs) * len(models)))
    return torch.cat(acc)


t0 = time.time()
for kind in ["flip", "flip_shift", "flip_scale2", "flip_shift_scale"]:
    pv = probs(xva, kind)
    print(f"  {kind:18s} val_bal={bal_acc(pv.argmax(1).cpu().numpy(), yva):.4f}")
pv = probs(xva, args.tta)
print(f"CHOSEN {args.tta}: val_bal={bal_acc(pv.argmax(1).cpu().numpy(), yva):.4f}")
if args.out:
    pg = probs(xg, args.tta).argmax(1).cpu().numpy().astype(np.int64)
    np.save(args.out, pg)
    print("wrote", args.out, pg.shape, pg.dtype, pg.min(), pg.max(), len(np.unique(pg)),
          f"{time.time()-t0:.1f}s")
