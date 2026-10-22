"""Load a trained checkpoint and emit predictions with richer test-time augmentation.

TTA and ensembling at prediction time only -- no extra training.
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from train import Net, bal_acc, DATA, N_CLASSES

DEV = "cuda"


def build(ckpt, key):
    cfg = ckpt["cfg"]
    m = Net(cfg["width"], tuple(int(d) for d in cfg["depths"].split(","))).to(DEV)
    sd = ckpt[key]
    m.load_state_dict({k: v.to(dtype=p.dtype) for (k, v), p
                       in zip(sd.items(), m.state_dict().values())})
    return m.to(memory_format=torch.channels_last).eval()


@torch.no_grad()
def probs_views(model, x_u8, mean, std, views, bs=1000):
    """views: list of (res, shift_y, shift_x). Each view also gets its hflip."""
    acc = torch.zeros(x_u8.shape[0], N_CLASSES, device=DEV)
    for s in range(0, x_u8.shape[0], bs):
        xb0 = x_u8[s:s + bs].permute(0, 3, 1, 2).float().div_(255.)
        for (res, dy, dx) in views:
            xb = xb0
            if dy or dx:
                xb = F.pad(xb, (2, 2, 2, 2), mode="reflect")
                xb = xb[:, :, 2 + dy:2 + dy + 32, 2 + dx:2 + dx + 32]
            if res != 32:
                xb = F.interpolate(xb, size=(res, res), mode="bilinear",
                                   align_corners=False)
            xb = ((xb - mean) / std).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                acc[s:s + bs] += model(xb).float().softmax(1)
                acc[s:s + bs] += model(xb.flip(-1)).float().softmax(1)
    return acc


VIEWSETS = {
    "flip": [(48, 0, 0)],
    "scale3": [(48, 0, 0), (44, 0, 0), (52, 0, 0)],
    "shift5": [(48, 0, 0), (48, -2, 0), (48, 2, 0), (48, 0, -2), (48, 0, 2)],
    "scale3shift5": [(48, 0, 0), (44, 0, 0), (52, 0, 0),
                     (48, -2, 0), (48, 2, 0), (48, 0, -2), (48, 0, 2)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--views", default="scale3")
    ap.add_argument("--members", default="raw,ema1,ema2")
    ap.add_argument("--prior_pow", type=float, default=0.0)
    ap.add_argument("--dump", default="")
    ap.add_argument("--sweep", type=int, default=1)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location=DEV)
    mean, std = ck["mean"].to(DEV), ck["std"].to(DEV)
    va = np.load(os.path.join(DATA, "val_public.npz"))
    xva = torch.from_numpy(va["x"]).to(DEV); yva = va["y"]
    xg = torch.from_numpy(np.load(os.path.join(DATA, "graded_images.npz"))["x"]).to(DEV)
    tr = np.load(os.path.join(DATA, "train_i20k.npz"))
    prior = torch.bincount(torch.from_numpy(tr["y"]), minlength=N_CLASSES).float()
    prior = (prior / prior.sum()).to(DEV)

    members = a.members.split(",")
    vs = VIEWSETS[a.views]
    pv = torch.zeros(len(yva), N_CLASSES, device=DEV)
    pg = torch.zeros(xg.shape[0], N_CLASSES, device=DEV)
    per_member_v, per_member_g = {}, {}
    for k in members:
        m = build(ck, k)
        v = probs_views(m, xva, mean, std, vs)
        g = probs_views(m, xg, mean, std, vs)
        per_member_v[k], per_member_g[k] = v, g
        pv += v / v.sum(1, keepdim=True)
        pg += g / g.sum(1, keepdim=True)
        del m
    for k in members:
        print(f"  member {k:5s} {bal_acc(per_member_v[k].argmax(1).cpu().numpy(), yva):.4f}")
    for pp in ([0.0, 0.25, 0.5, 1.0] if a.sweep else [a.prior_pow]):
        adj = pv / prior.pow(pp).unsqueeze(0)
        print(f"  ens views={a.views} prior_pow={pp}: "
              f"{bal_acc(adj.argmax(1).cpu().numpy(), yva):.4f}")
    if a.dump:
        torch.save({"val": pv.cpu(), "graded": pg.cpu(), "yva": yva}, a.dump)
    if a.out:
        p = (pg / prior.pow(a.prior_pow).unsqueeze(0)).argmax(1).cpu().numpy().astype(np.int64)
        np.save(a.out, p)
        print(f"wrote {a.out} shape {p.shape} uniq {len(np.unique(p))}")


if __name__ == "__main__":
    raise SystemExit(main())
