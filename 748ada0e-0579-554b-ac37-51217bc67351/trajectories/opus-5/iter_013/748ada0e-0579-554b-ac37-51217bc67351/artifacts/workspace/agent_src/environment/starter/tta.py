"""Post-hoc TTA study on a single run's checkpoint (no extra training)."""
import sys, time, numpy as np, torch, torch.nn.functional as F
from train import Net, MEAN, STD, bal_acc, N_CLASSES, DATA

ck = torch.load(sys.argv[1], map_location="cuda")
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

dv = np.load(f"{DATA}/val_public.npz")
xva = torch.from_numpy(dv["x"]).to(dev).permute(0, 3, 1, 2).contiguous()
yva = dv["y"]
xg = torch.from_numpy(np.load(f"{DATA}/graded_images.npz")["x"]).to(dev).permute(0, 3, 1, 2).contiguous()
R = a["res"]


def views(xb, kind):
    """xb float (B,3,32,32) in [0,1] -> list of normalised, resized tensors."""
    out = []
    base = [xb, torch.flip(xb, dims=[3])]
    if kind == "flip":
        cand = [(v, R) for v in base]
    elif kind == "flip_scale":
        cand = [(v, r) for v in base for r in (R - 8, R, R + 8)]
    elif kind == "flip_shift":
        cand = []
        for v in base:
            cand.append((v, R))
            for dy, dx in ((0, 2), (0, -2), (2, 0), (-2, 0)):
                cand.append((torch.roll(v, shifts=(dy, dx), dims=(2, 3)), R))
    elif kind == "flip_scale2":
        cand = [(v, r) for v in base for r in (R, R + 8)]
    for v, r in cand:
        z = (v - mean) / std
        z = F.interpolate(z, size=(r, r), mode="bilinear", align_corners=False)
        out.append(z.contiguous(memory_format=torch.channels_last))
    return out


@torch.no_grad()
def probs(xu8, kind, bs=500):
    acc = []
    for s in range(0, xu8.shape[0], bs):
        xb = xu8[s:s + bs].float().div(255.)
        p = 0
        for z in views(xb, kind):
            for m in models:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = p + F.softmax(m(z).float(), 1)
        acc.append(p)
    return torch.cat(acc)


for kind in ["flip", "flip_scale2", "flip_scale", "flip_shift"]:
    t = time.time()
    pv = probs(xva, kind)
    print(f"{kind:12s} val_bal={bal_acc(pv.argmax(1).cpu().numpy(), yva):.4f}  ({time.time()-t:.1f}s)")
