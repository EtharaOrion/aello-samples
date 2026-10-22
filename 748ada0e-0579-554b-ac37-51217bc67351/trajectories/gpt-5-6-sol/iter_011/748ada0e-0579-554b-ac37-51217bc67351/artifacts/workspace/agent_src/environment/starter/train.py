"""Budget-compliant CIFAR-100 trainer: exactly 8,000 x 128 presentations."""
import argparse, copy, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aello_data as data

STEPS, BATCH = 8000, 128
MEAN, STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

class WideBasic(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.short = nn.Conv2d(cin, cout, 1, stride, bias=False) if cin != cout or stride != 1 else None
    def forward(self, x):
        z = F.relu(self.bn1(x), inplace=True)
        shortcut = self.short(z) if self.short is not None else x
        return shortcut + self.c2(F.relu(self.bn2(self.c1(z)), inplace=True))

class WideResNet(nn.Module):
    def __init__(self, depth=40, widen=10):
        super().__init__(); n = (depth - 4) // 6; w = [16, 16*widen, 32*widen, 64*widen]
        self.conv = nn.Conv2d(3, w[0], 3, padding=1, bias=False)
        self.g1 = self.group(n, w[0], w[1], 1); self.g2 = self.group(n, w[1], w[2], 2)
        self.g3 = self.group(n, w[2], w[3], 2); self.bn = nn.BatchNorm2d(w[3]); self.fc = nn.Linear(w[3], 100)
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear): nn.init.zeros_(m.bias)
    @staticmethod
    def group(n, cin, cout, stride):
        return nn.Sequential(WideBasic(cin, cout, stride), *[WideBasic(cout, cout, 1) for _ in range(n-1)])
    def forward(self, x):
        x = self.g3(self.g2(self.g1(self.conv(x))))
        return self.fc(F.adaptive_avg_pool2d(F.relu(self.bn(x), inplace=True), 1).flatten(1))

def augment(x):
    b, c, h, w = x.shape; xp = F.pad(x, (4,)*4, mode='reflect')
    oy = torch.randint(0, 9, (b,), device=x.device); ox = torch.randint(0, 9, (b,), device=x.device)
    rows = oy[:,None] + torch.arange(h,device=x.device)[None]; cols = ox[:,None] + torch.arange(w,device=x.device)[None]
    x = xp[torch.arange(b,device=x.device)[:,None,None,None], torch.arange(c,device=x.device)[None,:,None,None], rows[:,None,:,None], cols[:,None,None,:]]
    x = torch.where(torch.rand(b,1,1,1,device=x.device)<.5, x.flip(3), x)
    # Canonical CIFAR Cutout: independent 16x16 masks. This is the sole strong
    # regularizer in the hard-label recipe.
    cy=torch.randint(0,h,(b,),device=x.device); cx=torch.randint(0,w,(b,),device=x.device)
    yy=torch.arange(h,device=x.device)[None,:,None]; xx=torch.arange(w,device=x.device)[None,None,:]
    keep=((yy-cy[:,None,None]).abs()>=8)|((xx-cx[:,None,None]).abs()>=8)
    return x*keep[:,None]

def mix(x, y):
    b=len(y); p=torch.randperm(b,device=x.device)
    if torch.rand((),device=x.device)<.60:
        lam=float(torch.rand((),device=x.device)); side=math.sqrt(1-lam)
        rw,rh=int(x.shape[3]*side),int(x.shape[2]*side)
        cx=int(torch.randint(0,x.shape[3],(),device=x.device)); cy=int(torch.randint(0,x.shape[2],(),device=x.device))
        x1,x2=max(cx-rw//2,0),min(cx+rw//2,x.shape[3]); y1,y2=max(cy-rh//2,0),min(cy+rh//2,x.shape[2])
        z=x.clone(); z[:,:,y1:y2,x1:x2]=x[p,:,y1:y2,x1:x2]
        lam=1-(x2-x1)*(y2-y1)/(x.shape[2]*x.shape[3])
    else:
        lam=float(torch.distributions.Beta(.2,.2).sample(())); z=x.mul(lam).add_(x[p],alpha=1-lam)
    return z,y,y[p],lam

@torch.no_grad()
def ema_update(ema, model, decay):
    for a,b in zip(ema.parameters(),model.parameters()): a.lerp_(b,1-decay)
    for a,b in zip(ema.buffers(),model.buffers()):
        if a.is_floating_point(): a.lerp_(b,1-decay)
        else: a.copy_(b)

def normalize_numpy(x, device):
    z=torch.from_numpy(np.ascontiguousarray(x)).permute(0,3,1,2).to(device).float().div_(255)
    return z.sub_(z.new_tensor(MEAN)[None,:,None,None]).div_(z.new_tensor(STD)[None,:,None,None]).contiguous(memory_format=torch.channels_last)

@torch.inference_mode()
def predict_logits(model,x,device='cuda',batch=512):
    model.eval(); out=[]
    for s in range(0,len(x),batch):
        z=normalize_numpy(x[s:s+batch],device)
        with torch.autocast('cuda',dtype=torch.bfloat16): q=model(z).float()+model(z.flip(3)).float()
        out.append(q.cpu())
    return torch.cat(out).numpy()/2

def train(instance,seed):
    x,y=data.load_train(instance); torch.manual_seed(seed); np.random.seed(seed); torch.backends.cudnn.benchmark=True
    model=WideResNet().cuda().to(memory_format=torch.channels_last); ema=copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    opt=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,nesterov=True,weight_decay=5e-4)
    rng=np.random.default_rng(seed); mean=torch.tensor(MEAN,device='cuda')[None,:,None,None]; std=torch.tensor(STD,device='cuda')[None,:,None,None]
    order=rng.permutation(len(y)); pos=0
    model.train(); start=time.time(); randaug=T.RandAugment(num_ops=2,magnitude=9)
    for step in range(STEPS):
        if pos+BATCH <= len(y):
            ix=order[pos:pos+BATCH];pos+=BATCH
        else:
            tail=order[pos:];order=rng.permutation(len(y));need=BATCH-len(tail)
            ix=np.concatenate((tail,order[:need]));pos=need
        xb=torch.from_numpy(np.ascontiguousarray(x[ix])).permute(0,3,1,2).cuda()
        xb=randaug(augment(xb)).float().div_(255).sub_(mean).div_(std).contiguous(memory_format=torch.channels_last)
        ya=torch.from_numpy(y[ix]).cuda()
        do_mix = False
        lr=.1*(step+1)/400 if step<400 else .05*(1+math.cos(math.pi*(step-400)/(STEPS-400)))
        for g in opt.param_groups: g['lr']=lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            q=model(xb)
            if do_mix:
                loss=lam*F.cross_entropy(q,ya,label_smoothing=.05)+(1-lam)*F.cross_entropy(q,yb,label_smoothing=.05)
            else: loss=F.cross_entropy(q,ya,label_smoothing=.05)
        loss.backward(); opt.step(); ema_update(ema,model,min(.999,(step+1)/(step+10)))
        if (step+1)%1000==0: print(f'step {step+1}/{STEPS} loss {loss.item():.4f} lr {lr:.5f} elapsed {time.time()-start:.1f}s',flush=True)
    print(f'training_seconds {time.time()-start:.2f}'); return model,ema

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--instance',required=True,choices=list(data.INSTANCES)); ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--graded',required=True); ap.add_argument('--out',required=True); ap.add_argument('--val',default='../data/val_public.npz'); a=ap.parse_args()
    base,ema=train(a.instance,a.seed); model=ema
    if a.val and os.path.exists(a.val):
        d=np.load(a.val); best=(-1,None,None)
        for name,m in [('base',base),('ema',ema)]:
            p=predict_logits(m,d['x']).argmax(1); rec=[np.mean(p[d['y']==c]==c) for c in range(100)]; ba=np.mean(rec)
            print(f"{name}_val_accuracy {np.mean(p==d['y']):.5f} val_balanced {ba:.5f}")
            if ba>best[0]: best=(ba,name,m)
        model=best[2]; print(f'selected {best[1]}')
    xg=np.load(a.graded)['x']; p=predict_logits(model,xg).argmax(1).astype(np.int64)
    tmp=a.out+'.tmp.npy'; np.save(tmp,p); os.replace(tmp,a.out); torch.save(model.state_dict(),os.path.join(os.path.dirname(a.out),'model_i20k.pt'))
    print(f'wrote {a.out} shape={p.shape} range={p.min()}..{p.max()} classes={len(np.unique(p))}')
if __name__=='__main__': main()
