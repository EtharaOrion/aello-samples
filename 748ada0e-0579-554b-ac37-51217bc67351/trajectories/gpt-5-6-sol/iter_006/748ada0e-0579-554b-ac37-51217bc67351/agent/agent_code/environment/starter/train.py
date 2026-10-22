"""Fixed-budget CIFAR-100 trainer using exactly 8,000 updates of 128 examples."""
import argparse, copy, math, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aello_data as data

STEPS, BATCH = 8000, 128
MEAN, STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

class WideBasic(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.c1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.short = (nn.Conv2d(cin, cout, 1, stride=stride, bias=False)
                      if stride != 1 or cin != cout else None)
    def forward(self, x):
        z = F.relu(self.bn1(x), inplace=True)
        shortcut = x if self.short is None else self.short(z)
        z = self.c1(z)
        z = F.relu(self.bn2(z), inplace=True)
        return shortcut + self.c2(z)

class WideResNet(nn.Module):
    def __init__(self, depth=40, widen=10, classes=100):
        super().__init__()
        n = (depth - 4) // 6
        widths = [16, 16*widen, 32*widen, 64*widen]
        self.c0 = nn.Conv2d(3, widths[0], 3, padding=1, bias=False)
        self.g1 = self._group(n, widths[0], widths[1], 1)
        self.g2 = self._group(n, widths[1], widths[2], 2)
        self.g3 = self._group(n, widths[2], widths[3], 2)
        self.bn = nn.BatchNorm2d(widths[3])
        self.fc = nn.Linear(widths[3], classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear): nn.init.zeros_(m.bias)
    @staticmethod
    def _group(n, cin, cout, stride):
        return nn.Sequential(WideBasic(cin, cout, stride),
                             *[WideBasic(cout, cout, 1) for _ in range(n-1)])
    def forward(self, x):
        x = self.g3(self.g2(self.g1(self.c0(x))))
        return self.fc(F.adaptive_avg_pool2d(F.relu(self.bn(x), inplace=True), 1).flatten(1))

def augment(x):
    b, c, h, w = x.shape
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    oy = torch.randint(0, 9, (b,), device=x.device)
    ox = torch.randint(0, 9, (b,), device=x.device)
    yy = oy[:, None] + torch.arange(h, device=x.device)[None]
    xx = ox[:, None] + torch.arange(w, device=x.device)[None]
    x = xp[torch.arange(b, device=x.device)[:, None, None], :, yy[:, :, None], xx[:, None, :]].permute(0, 3, 1, 2)
    return torch.where(torch.rand(b, 1, 1, 1, device=x.device) < .5, x.flip(3), x)

def cutout(x, size=12):
    """Per-image square masking, used only on the gentler Mixup batches."""
    b, _, h, w = x.shape
    cy = torch.randint(0, h, (b,), device=x.device)
    cx = torch.randint(0, w, (b,), device=x.device)
    yy = torch.arange(h, device=x.device)[None, :, None]
    xx = torch.arange(w, device=x.device)[None, None, :]
    keep = ((yy-cy[:,None,None]).abs() >= size//2) | ((xx-cx[:,None,None]).abs() >= size//2)
    return x * keep[:,None]

def cutmix(x, y, alpha=1.0):
    b, _, h, w = x.shape
    p = torch.randperm(b, device=x.device)
    lam = np.random.beta(alpha, alpha); ratio = math.sqrt(1-lam)
    rw, rh = int(w*ratio), int(h*ratio)
    cx, cy = np.random.randint(w), np.random.randint(h)
    x1, x2 = max(cx-rw//2, 0), min(cx+rw//2, w)
    y1, y2 = max(cy-rh//2, 0), min(cy+rh//2, h)
    x[:, :, y1:y2, x1:x2] = x[p, :, y1:y2, x1:x2]
    lam = 1-((x2-x1)*(y2-y1))/(w*h)
    return x, y, y[p], lam

def mixup(x, y, alpha=.2):
    p = torch.randperm(len(y), device=x.device); lam = float(np.random.beta(alpha, alpha))
    return x.mul(lam).add_(x[p], alpha=1-lam), y, y[p], lam

@torch.no_grad()
def update_ema(ema, model, decay):
    ms = model.state_dict()
    for k, v in ema.state_dict().items():
        if v.is_floating_point(): v.lerp_(ms[k], 1-decay)
        else: v.copy_(ms[k])

def tensors(x, device):
    return torch.from_numpy(np.ascontiguousarray(x)).permute(0,3,1,2).to(device).float().div_(255.).contiguous(memory_format=torch.channels_last)

@torch.inference_mode()
def logits(model, x, device, batch=512, tta=True, multicrop=False):
    model.eval(); mean=torch.tensor(MEAN,device=device)[None,:,None,None]; std=torch.tensor(STD,device=device)[None,:,None,None]
    ans=[]
    for s in range(0,len(x),batch):
        z=(tensors(x[s:s+batch],device)-mean)/std
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            if multicrop:
                zp=F.pad(z,(2,2,2,2),mode="reflect"); a=0
                for yy,xx in ((0,0),(0,4),(4,0),(4,4),(2,2)):
                    v=zp[:,:,yy:yy+32,xx:xx+32]
                    a=a+model(v).float()+model(v.flip(3)).float()
                a=a/10
            else:
                a=model(z).float()
                if tta: a=(a+model(z.flip(3)).float())*.5
        ans.append(a.cpu())
    return torch.cat(ans).numpy()

def balanced_acc(pred,y):
    return float(np.mean([np.mean(pred[y==c]==c) for c in range(100)]))

def train(seed, device="cuda"):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    x,y=data.load_train("i20k")
    model=WideResNet().to(device).to(memory_format=torch.channels_last)
    ema=copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    opt=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,weight_decay=5e-4,nesterov=True)
    mean=torch.tensor(MEAN,device=device)[None,:,None,None]; std=torch.tensor(STD,device=device)[None,:,None,None]
    rng=np.random.default_rng(seed); started=time.time(); model.train()
    for step in range(STEPS):
        idx=rng.integers(0,len(y),size=BATCH)
        xb=augment(tensors(x[idx],device)); yb=torch.from_numpy(y[idx]).to(device)
        if step%3==0: xb,ya,yb2,lam=cutmix(xb,yb)
        else: xb,ya,yb2,lam=mixup(cutout(xb),yb)
        xb=(xb-mean)/std
        lr=.1*(step+1)/400 if step<400 else .05*(1+math.cos(math.pi*(step-400)/(STEPS-400)))
        for g in opt.param_groups: g["lr"]=lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            out=model(xb)
            loss=lam*F.cross_entropy(out,ya,label_smoothing=.05)+(1-lam)*F.cross_entropy(out,yb2,label_smoothing=.05)
        loss.backward(); opt.step(); update_ema(ema,model,.997)
        if (step+1)%1000==0: print(f"step {step+1}/{STEPS} loss {loss.item():.4f} lr {lr:.5f} elapsed {time.time()-started:.1f}s",flush=True)
    print(f"training_seconds {time.time()-started:.2f}")
    return model,ema

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--instance",default="i20k"); ap.add_argument("--seed",type=int,default=0); ap.add_argument("--graded",required=True); ap.add_argument("--out",required=True); a=ap.parse_args()
    model,ema=train(a.seed)
    vd=np.load(os.path.join(os.path.dirname(a.graded),"val_public.npz"))
    raw=logits(model,vd["x"],"cuda"); av=logits(ema,vd["x"],"cuda")
    pr,pe=raw.argmax(1),av.argmax(1)
    ar,br=(pr==vd["y"]).mean(),balanced_acc(pr,vd["y"]); ae,be=(pe==vd["y"]).mean(),balanced_acc(pe,vd["y"])
    print(f"val_raw accuracy={ar:.5f} balanced_accuracy={br:.5f}"); print(f"val_ema accuracy={ae:.5f} balanced_accuracy={be:.5f}")
    chosen=ema if be>=br else model
    mc=logits(chosen,vd["x"],"cuda",multicrop=True).argmax(1)
    am,bm=(mc==vd["y"]).mean(),balanced_acc(mc,vd["y"])
    print(f"val_multicrop accuracy={am:.5f} balanced_accuracy={bm:.5f}")
    use_mc=bm>=max(br,be)
    pred=logits(chosen,np.load(a.graded)["x"],"cuda",multicrop=use_mc).argmax(1).astype(np.int64)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)),exist_ok=True); np.save(a.out,pred)
    print(f"wrote {a.out} shape={pred.shape} range={pred.min()}..{pred.max()} classes={len(np.unique(pred))} source={'ema' if be>=br else 'raw'} tta={'multicrop' if use_mc else 'flip'}")
if __name__=="__main__": main()
