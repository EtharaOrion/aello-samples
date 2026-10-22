"""Fixed-budget CIFAR-100 trainer: WRN-40-10, EMA, mixup/cutout and flip TTA."""
import argparse, math, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.augmentation.auto import RandAugment


STEPS = 8000
BATCH = 128
MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


class Block(nn.Module):
    def __init__(self, ni, no, stride, drop=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(ni)
        self.c1 = nn.Conv2d(ni, no, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(no)
        self.c2 = nn.Conv2d(no, no, 3, 1, 1, bias=False)
        self.short = nn.Identity() if ni == no and stride == 1 else nn.Conv2d(ni, no, 1, stride, bias=False)
        self.drop = drop

    def forward(self, x):
        z = F.relu(self.bn1(x), inplace=True)
        shortcut = self.short(z) if not isinstance(self.short, nn.Identity) else x
        z = self.c1(z)
        z = self.c2(F.dropout(F.relu(self.bn2(z), inplace=True), self.drop, self.training))
        return shortcut + z


class WRN(nn.Module):
    def __init__(self, depth=40, widen=10, classes=100):
        super().__init__()
        n = (depth - 4) // 6
        widths = [16, 16*widen, 32*widen, 64*widen]
        self.c0 = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        groups = []
        for gi, stride in enumerate((1, 2, 2)):
            blocks = [Block(widths[gi], widths[gi+1], stride)]
            blocks += [Block(widths[gi+1], widths[gi+1], 1) for _ in range(n-1)]
            groups.append(nn.Sequential(*blocks))
        self.g1, self.g2, self.g3 = groups
        self.bn = nn.BatchNorm2d(widths[-1])
        self.fc = nn.Linear(widths[-1], classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear): nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.g3(self.g2(self.g1(self.c0(x))))
        return self.fc(F.adaptive_avg_pool2d(F.relu(self.bn(x), inplace=True), 1).flatten(1))


def augment(x, gen, randaug):
    # reflect pad and independently crop/flip each image
    b, _, h, w = x.shape
    p = F.pad(x, (4, 4, 4, 4), mode="reflect")
    oy = torch.randint(0, 9, (b,), generator=gen, device=x.device)
    ox = torch.randint(0, 9, (b,), generator=gen, device=x.device)
    yy = oy[:, None] + torch.arange(h, device=x.device)[None]
    xx = ox[:, None] + torch.arange(w, device=x.device)[None]
    x = p[torch.arange(b, device=x.device)[:, None, None, None],
          torch.arange(3, device=x.device)[None, :, None, None], yy[:, None, :, None], xx[:, None, None, :]]
    flip = torch.rand(b, generator=gen, device=x.device) < .5
    x[flip] = x[flip].flip(3)
    cy = torch.randint(0, 32, (b,), generator=gen, device=x.device)
    cx = torch.randint(0, 32, (b,), generator=gen, device=x.device)
    gy = torch.arange(32, device=x.device)[None, :, None]
    gx = torch.arange(32, device=x.device)[None, None, :]
    mask = ((gy-cy[:, None, None]).abs() < 6) & ((gx-cx[:, None, None]).abs() < 6)
    return x.masked_fill(mask[:, None], 0.)


def norm(x):
    mean = x.new_tensor(MEAN)[None, :, None, None]
    std = x.new_tensor(STD)[None, :, None, None]
    return (x - mean) / std


@torch.no_grad()
def update_ema(ema, model, decay):
    for pe, p in zip(ema.parameters(), model.parameters()): pe.lerp_(p, 1-decay)
    for be, b in zip(ema.buffers(), model.buffers()): be.copy_(b)


@torch.no_grad()
def logits(model, arr, device, batch=512):
    model.eval(); outs=[]
    for s in range(0, len(arr), batch):
        x = torch.from_numpy(np.ascontiguousarray(arr[s:s+batch])).permute(0,3,1,2).to(device).float().div_(255)
        x = norm(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # Exact horizontal flip is a cheap, robust two-view TTA.
            z = (model(x) + model(x.flip(3))) * .5
        outs.append(z.float().cpu())
    return torch.cat(outs).numpy()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graded", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default="strong_ckpt.pt")
    ap.add_argument("--train", default="../data/train_i20k.npz")
    ap.add_argument("--val", default="../data/val_public.npz")
    a=ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark=True
    device="cuda"
    d=np.load(a.train); xt=d["x"]; yt=d["y"].astype(np.int64)
    vd=np.load(a.val); xv=vd["x"]; yv=vd["y"].astype(np.int64)
    xg=np.load(a.graded)["x"]
    model=WRN().to(device).to(memory_format=torch.channels_last)
    ema=WRN().to(device).to(memory_format=torch.channels_last); ema.load_state_dict(model.state_dict()); ema.requires_grad_(False)
    opt=torch.optim.SGD(model.parameters(), lr=.2, momentum=.9, nesterov=True, weight_decay=5e-4)
    rng=np.random.default_rng(a.seed); gen=torch.Generator(device=device); gen.manual_seed(a.seed+911)
    randaug=RandAugment(n=2, m=7).to(device)
    counts=np.bincount(yt, minlength=100)
    sample_prob=(1.0/counts[yt]); sample_prob/=sample_prob.sum()
    t0=time.time()
    for step in range(STEPS):
        # Sampling with replacement honors exactly 128 presentations/update.
        ii=rng.choice(len(yt), size=BATCH, replace=True, p=sample_prob)
        x=torch.from_numpy(np.ascontiguousarray(xt[ii])).permute(0,3,1,2).to(device).float().div_(255)
        x=x.contiguous(memory_format=torch.channels_last); x=augment(x, gen, randaug)
        y=torch.from_numpy(yt[ii]).to(device)
        perm=torch.randperm(BATCH, generator=gen, device=device)
        lam=float(rng.beta(.2,.2)); x=norm(x)
        x=x.mul(lam).add_(x[perm],alpha=1-lam)
        if step < 400: lr=.2*(step+1)/400
        else: lr=.2*.5*(1+math.cos(math.pi*(step-400)/(STEPS-400)))
        for pg in opt.param_groups: pg["lr"]=lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z=model(x); loss=lam*F.cross_entropy(z,y,label_smoothing=.02)+(1-lam)*F.cross_entropy(z,y[perm],label_smoothing=.02)
        loss.backward(); opt.step()
        if (step+1)%1000==0: print(f"step {step+1}/{STEPS} loss {loss.item():.4f} lr {lr:.5f} sec {time.time()-t0:.1f}", flush=True)
    scored=[]
    for name, candidate in (("raw",model),):
        lv=logits(candidate,xv,device); pv=lv.argmax(1)
        bal=float(np.mean([(pv[yv==c]==c).mean() for c in range(100)])); acc=float((pv==yv).mean())
        print(f"{name} val_acc {acc:.5f} val_bal {bal:.5f}",flush=True); scored.append((bal,name,candidate,acc))
    bal,name,best,acc=max(scored,key=lambda q:q[0])
    print(f"selected {name}",flush=True)
    torch.save({"model":best.state_dict(),"val_acc":acc,"val_bal":bal,"selected":name,"steps":STEPS,"batch":BATCH},a.checkpoint)
    pred=logits(best,xg,device).argmax(1).astype(np.int64)
    tmp=a.out+".tmp.npy"; np.save(tmp,pred); os.replace(tmp,a.out)
    print(f"wrote {a.out} shape={pred.shape} range={pred.min()}..{pred.max()} total_sec={time.time()-t0:.1f}",flush=True)

if __name__=="__main__": main()
