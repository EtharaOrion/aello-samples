"""Budgeted CIFAR-100 trainer: one 8,000-update run, 128 presentations/update."""
import argparse, copy, math, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aello_data as data

STEPS, BATCH = 8000, 128
MEAN, STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

class PreActBlock(nn.Module):
    def __init__(self, ni, no, stride):
        super().__init__(); self.bn1=nn.BatchNorm2d(ni); self.c1=nn.Conv2d(ni,no,3,stride,1,bias=False)
        self.bn2=nn.BatchNorm2d(no); self.c2=nn.Conv2d(no,no,3,1,1,bias=False)
        self.short=nn.Conv2d(ni,no,1,stride,bias=False) if stride != 1 or ni != no else None
    def forward(self,x):
        z=F.relu(self.bn1(x),inplace=True); shortcut=self.short(z) if self.short is not None else x
        z=self.c1(z); z=F.relu(self.bn2(z),inplace=True)
        return shortcut+self.c2(z)

class WideResNet(nn.Module):
    def __init__(self, depth=40, widen=10, classes=100):
        super().__init__(); n=(depth-4)//6; w=[16,16*widen,32*widen,64*widen]
        self.stem=nn.Conv2d(3,w[0],3,1,1,bias=False); self.g1=self.group(n,w[0],w[1],1)
        self.g2=self.group(n,w[1],w[2],2); self.g3=self.group(n,w[2],w[3],2)
        self.bn=nn.BatchNorm2d(w[3]); self.fc=nn.Linear(w[3],classes)
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear): nn.init.zeros_(m.bias)
    @staticmethod
    def group(n,ni,no,s): return nn.Sequential(PreActBlock(ni,no,s),*[PreActBlock(no,no,1) for _ in range(n-1)])
    def forward(self,x):
        x=self.g3(self.g2(self.g1(self.stem(x))))
        return self.fc(F.adaptive_avg_pool2d(F.relu(self.bn(x),inplace=True),1).flatten(1))

def crop_flip(x):
    b,c,h,w=x.shape; x=F.pad(x,(4,4,4,4),mode='reflect')
    oy=torch.randint(0,9,(b,),device=x.device); ox=torch.randint(0,9,(b,),device=x.device)
    yy=oy[:,None]+torch.arange(h,device=x.device)[None]; xx=ox[:,None]+torch.arange(w,device=x.device)[None]
    x=x[torch.arange(b,device=x.device)[:,None,None,None],torch.arange(c,device=x.device)[None,:,None,None],yy[:,None,:,None],xx[:,None,None,:]]
    flip=torch.rand(b,device=x.device)<.5; x[flip]=x[flip].flip(3); return x

def soft_loss(logits,target,smooth=.04):
    lp=F.log_softmax(logits,1); return -((1-smooth)*(target*lp).sum(1)+smooth*lp.mean(1)).mean()

@torch.no_grad()
def update_ema(ema,model,d=.999):
    ep=list(ema.parameters()); mp=list(model.parameters()); torch._foreach_mul_(ep,d); torch._foreach_add_(ep,mp,alpha=1-d)
    for eb,mb in zip(ema.buffers(),model.buffers()):
        eb.lerp_(mb,1-d) if eb.is_floating_point() else eb.copy_(mb)

def balanced_indices(y,rng):
    pools=[np.flatnonzero(y==k) for k in range(100)]; cls=rng.integers(0,100,BATCH)
    return np.fromiter((pools[k][rng.integers(len(pools[k]))] for k in cls),np.int64,BATCH)

def normalize(x,mean,std): return (x.float().div_(255.)-mean)/std

@torch.inference_mode()
def logits_for(model,x,device,flip=False,batch=512):
    model.eval(); ans=[]; mean=torch.tensor(MEAN,device=device).view(1,3,1,1); std=torch.tensor(STD,device=device).view(1,3,1,1)
    for s in range(0,len(x),batch):
        z=torch.from_numpy(np.ascontiguousarray(x[s:s+batch])).to(device).permute(0,3,1,2)
        z=normalize(z,mean,std).contiguous(memory_format=torch.channels_last)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            a=model(z).float()
            if flip: a.add_(model(z.flip(3)).float()).mul_(.5)
        ans.append(a.cpu())
    return torch.cat(ans).numpy()

def bal_acc(pred,y): return float(np.mean([np.mean(pred[y==k]==k) for k in range(100)]))

def train(instance,seed):
    torch.manual_seed(seed); np.random.seed(seed); torch.backends.cudnn.benchmark=True; device='cuda'
    x,y=data.load_train(instance); x_gpu=torch.from_numpy(x).to(device); y_gpu=torch.from_numpy(y).to(device)
    mean=torch.tensor(MEAN,device=device).view(1,3,1,1); std=torch.tensor(STD,device=device).view(1,3,1,1)
    base=WideResNet().to(device).to(memory_format=torch.channels_last); ema=copy.deepcopy(base).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    opt=torch.optim.SGD(base.parameters(),lr=.18,momentum=.9,nesterov=True,weight_decay=5e-4); rng=np.random.default_rng(seed)
    pools=[np.flatnonzero(y==k) for k in range(100)]
    for step in range(STEPS):
        cls=rng.integers(0,100,BATCH); idx=np.fromiter((pools[k][rng.integers(len(pools[k]))] for k in cls),np.int64,BATCH)
        ii=torch.from_numpy(idx).to(device); xb=x_gpu[ii].permute(0,3,1,2).contiguous(memory_format=torch.channels_last)
        xb=normalize(crop_flip(xb),mean,std); target=F.one_hot(y_gpu[ii],100).float()
        if rng.random()<.5:
            perm=torch.randperm(BATCH,device=device); lam=rng.beta(1.,1.); cut=int(32*math.sqrt(1-lam)); cy,cx=rng.integers(0,32,2)
            y1,y2=max(0,cy-cut//2),min(32,cy+(cut+1)//2); x1,x2=max(0,cx-cut//2),min(32,cx+(cut+1)//2)
            xb[:,:,y1:y2,x1:x2]=xb[perm,:,y1:y2,x1:x2]; lam=1.-((y2-y1)*(x2-x1)/1024.)
            target=target.mul(lam).add_(target[perm],alpha=1-lam)
        lr=.18*(step+1)/400 if step<400 else .09*(1+math.cos(math.pi*(step-400)/(STEPS-400)))
        for g in opt.param_groups: g['lr']=lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): loss=soft_loss(base(xb),target)
        loss.backward(); opt.step(); update_ema(ema,base)
        if (step+1)%1000==0: print(f'step {step+1}/{STEPS} loss {loss.item():.4f} lr {lr:.5f}',flush=True)
    return base,ema

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--instance',required=True,choices=list(data.INSTANCES)); ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--graded',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); base,ema=train(a.instance,a.seed)
    xv,yv=data.load_val(); candidates=[]
    for name,model,flip in [('base',base,False),('ema',ema,False),('ema_flip',ema,True)]:
        lg=logits_for(model,xv,'cuda',flip); score=bal_acc(lg.argmax(1),yv)
        print(f'public {name}: bal_acc={score:.5f} acc={(lg.argmax(1)==yv).mean():.5f}',flush=True); candidates.append((score,name,model,flip))
    score,name,model,flip=max(candidates,key=lambda z:z[0]); xg=np.load(a.graded)['x']; pred=logits_for(model,xg,'cuda',flip).argmax(1).astype(np.int64)
    assert pred.shape==(10000,) and pred.min()>=0 and pred.max()<100
    tmp=a.out+'.tmp.npy'; np.save(tmp,pred); os.replace(tmp,a.out); print(f'selected {name} public_bal={score:.5f}; wrote {a.out}',flush=True)

if __name__=='__main__': main()
