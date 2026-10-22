"""Which shift offsets should the TTA average over? Paired comparison on val."""
import argparse, itertools, os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aello_data as data
from train import Net, bal_acc

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--models", default="raw,ema0.999")
a = ap.parse_args()
dev = "cuda"; torch.backends.cudnn.benchmark = True
ck = torch.load(a.ckpt, map_location="cpu"); cfg = ck["cfg"]
res = cfg["res"]; mean, std = ck["mean"].to(dev), ck["std"].to(dev)
depths = tuple(int(v) for v in cfg["depths"].split(","))
xv, yv = data.load_val()
XV = torch.from_numpy(np.ascontiguousarray(xv)).permute(0, 3, 1, 2).contiguous().to(dev)
G = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
names = a.models.split(",")
per = {}


@torch.no_grad()
def run(m, X, batch=1000):
    out = {}
    for s in range(0, X.shape[0], batch):
        x0 = X[s:s + batch].float().div_(255.)
        pad = F.pad(x0, (4, 4, 4, 4), mode="reflect")
        for dy in G:
            for dx in G:
                xs = x0 if (dy == 0 and dx == 0) else \
                    torch.roll(pad, shifts=(dy, dx), dims=(2, 3))[:, :, 4:-4, 4:-4]
                for f in (False, True):
                    xf = xs.flip(3) if f else xs
                    xf = F.interpolate(xf, size=(res, res), mode="bilinear", align_corners=False)
                    xf = ((xf - mean) / std).contiguous(memory_format=torch.channels_last)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        o = m(xf)
                    out.setdefault((dy, dx, f), []).append(F.log_softmax(o.float(), 1).half().cpu())
    return {k: torch.cat(v).float().numpy() for k, v in out.items()}


for nm in names:
    m = Net(data.N_CLASSES, cfg["width"], depths, cfg["stem"]).to(dev).to(memory_format=torch.channels_last)
    m.load_state_dict(ck["models"][nm]); m.eval(); per[nm] = run(m, XV); del m
    print("views done", nm, flush=True)
np.savez("/workspace/sv_val.npz", c=np.array(per, dtype=object))

def ev(offs):
    tot = sum(per[nm][(dy, dx, f)] for nm in names for (dy, dx) in offs for f in (False, True))
    return bal_acc((tot / (len(names) * len(offs) * 2)).argmax(1), yv)

sq = lambda k: [(dy, dx) for dy in (-k, 0, k) for dx in (-k, 0, k)]
dia = lambda k: [(0, 0), (k, 0), (-k, 0), (0, k), (0, -k)]
CAND = {
    "center": [(0, 0)],
    "diamond1": dia(1), "diamond2": dia(2), "diamond3": dia(3),
    "sq1": sq(1), "sq2": sq(2), "sq3": sq(3), "sq4": sq(4),
    "dia1+2": dia(1) + dia(2)[1:], "dia2+4": dia(2) + dia(4)[1:],
    "sq2+dia1": sq(2) + dia(1)[1:], "sq1+dia2": sq(1) + dia(2)[1:],
    "sq2+dia4": sq(2) + dia(4)[1:],
}
rows = sorted(((ev(v), k, len(v)) for k, v in CAND.items()), reverse=True)
for s, k, n in rows:
    print("  %.4f  %-10s (%d offsets, %d views/model)" % (s, k, n, 2 * n))
