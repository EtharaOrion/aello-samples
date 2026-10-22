"""Budget-honest CIFAR-100 trainer: WRN-40-10, 8000 x 128 presentations."""
import argparse, copy, math, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aello_data as data

STEPS, BATCH = 8000, 128
MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)

class Block(nn.Module):
    def __init__(self, ni, no, stride, drop=0.0):
        super().__init__(); self.bn1=nn.BatchNorm2d(ni); self.c1=nn.Conv2d(ni,no,3,stride,1,bias=False)
        self.bn2=nn.BatchNorm2d(no); self.c2=nn.Conv2d(no,no,3,1,1,bias=False)
        self.short=nn.Conv2d(ni,no,1,stride,bias=False) if ni!=no else None; self.drop=drop
    def forward(self,x):
        z=F.relu(self.bn1(x),inplace=True); shortcut=self.short(z) if self.short is not None else x
        z=self.c1(z); z=F.relu(self.bn2(z),inplace=True)
        if self.drop: z=F.dropout(z,self.drop,self.training)
        return shortcut+self.c2(z)

class WRN(nn.Module):
    def __init__(self,depth=40,widen=10,classes=100):
        super().__init__(); n=(depth-4)//6; w=[16,16*widen,32*widen,64*widen]
        self.c0=nn.Conv2d(3,w[0],3,1,1,bias=False); self.g1=self._group(n,w[0],w[1],1)
        self.g2=self._group(n,w[1],w[2],2); self.g3=self._group(n,w[2],w[3],2)
        self.bn=nn.BatchNorm2d(w[3]); self.fc=nn.Linear(w[3],classes)
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out')
            elif isinstance(m,nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear): nn.init.zeros_(m.bias)
    @staticmethod
    def _group(n,ni,no,stride): return nn.Sequential(Block(ni,no,stride),*[Block(no,no,1) for _ in range(n-1)])
    def forward(self,x):
        x=self.g3(self.g2(self.g1(self.c0(x))))
        return self.fc(F.adaptive_avg_pool2d(F.relu(self.bn(x),inplace=True),1).flatten(1))

def augment(x,gen):
    b,_,h,w=x.shape; x=F.pad(x,(4,4,4,4),mode='reflect')
    oy=torch.randint(9,(b,),generator=gen,device=x.device); ox=torch.randint(9,(b,),generator=gen,device=x.device)
    yy=oy[:,None,None]+torch.arange(h,device=x.device)[None,:,None]
    xx=ox[:,None,None]+torch.arange(w,device=x.device)[None,None,:]
    bi=torch.arange(b,device=x.device)[:,None,None]
    x=x[bi,:,yy,xx].permute(0,3,1,2).contiguous(memory_format=torch.channels_last)
    flip=torch.rand(b,generator=gen,device=x.device)<.5; x[flip]=x[flip].flip(3)
    cy=torch.randint(h,(b,),generator=gen,device=x.device); cx=torch.randint(w,(b,),generator=gen,device=x.device)
    gy=torch.arange(h,device=x.device)[None,:,None]; gx=torch.arange(w,device=x.device)[None,None,:]
    mask=(gy-cy[:,None,None]).abs().lt(8)&(gx-cx[:,None,None]).abs().lt(8)
    return x.masked_fill_(mask[:,None],0.)

def balanced_accuracy(pred,y): return np.mean([np.mean(pred[y==c]==c) for c in range(100)])

@torch.inference_mode()
def logits(model,x,device,flip_tta=True,batch=512):
    model.eval(); ans=[]; mean=torch.tensor(MEAN,device=device)[None,:,None,None]; std=torch.tensor(STD,device=device)[None,:,None,None]
    for s in range(0,len(x),batch):
        z=torch.from_numpy(np.ascontiguousarray(x[s:s+batch])).to(device).permute(0,3,1,2).float().div_(255); z=(z-mean)/std
        with torch.autocast('cuda',dtype=torch.bfloat16):
            q=model(z)
            if flip_tta: q=(q+model(z.flip(3)))*.5
        ans.append(q.float().cpu())
    return torch.cat(ans).numpy()

def train(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.benchmark=True
    x,y=data.load_train('i20k'); dev='cuda'; model=WRN().to(dev).to(memory_format=torch.channels_last); ema=copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    opt=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,nesterov=True,weight_decay=5e-4)
    rng=np.random.default_rng(seed); gen=torch.Generator(device=dev); gen.manual_seed(seed+991)
    mean=torch.tensor(MEAN,device=dev)[None,:,None,None]; std=torch.tensor(STD,device=dev)[None,:,None,None]; t0=time.time()
    for step in range(STEPS):
        ids=rng.integers(0,len(y),size=BATCH)
        z=torch.from_numpy(np.ascontiguousarray(x[ids])).to(dev).permute(0,3,1,2).float().div_(255); z=(z-mean)/std
        z=augment(z.contiguous(memory_format=torch.channels_last),gen); target=torch.from_numpy(y[ids]).to(dev)
        if rng.random()<.5:
            lam=float(rng.beta(.4,.4)); perm=torch.randperm(BATCH,generator=gen,device=dev); z=lam*z+(1-lam)*z[perm]; target2=target[perm]
        else: lam=1.; target2=target
        warm=400; lr=.2*(step+1)/warm if step<warm else .2*.5*(1+math.cos(math.pi*(step-warm)/(STEPS-warm)))
        for g in opt.param_groups: g['lr']=lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=model(z); loss=lam*F.cross_entropy(out,target,label_smoothing=.05)+(1-lam)*F.cross_entropy(out,target2,label_smoothing=.05)
        loss.backward(); opt.step()
        if step==6000: ema.load_state_dict(model.state_dict())
        elif step>6000:
            with torch.no_grad():
                for ep,p in zip(ema.parameters(),model.parameters()): ep.lerp_(p,.005)
                for eb,b in zip(ema.buffers(),model.buffers()): eb.copy_(b)
        if (step+1)%1000==0: print(f'step {step+1}/{STEPS} loss {loss.item():.4f} lr {lr:.5f} sec {time.time()-t0:.1f}',flush=True)
    return ema,x,y

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--instance',default='i20k'); ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--graded',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    model,x,y=train(a.seed); val=np.load(os.path.join(os.path.dirname(a.graded),'val_public.npz'))
    lv=logits(model,val['x'],'cuda'); counts=np.bincount(y,minlength=100).astype(np.float32); prior=np.log(counts/counts.mean()); best=None
    for tau in (0.,.25,.5,.75,1.):
        p=(lv-tau*prior).argmax(1); ba=balanced_accuracy(p,val['y']); acc=np.mean(p==val['y']); print(f'tau {tau:.2f}: val_acc {acc:.5f} val_bal_acc {ba:.5f}')
        if best is None or ba>best[0]: best=(ba,tau)
    xg=np.load(a.graded)['x']; lg=logits(model,xg,'cuda'); pred=(lg-best[1]*prior).argmax(1).astype(np.int64)
    tmp=a.out+'.tmp.npy'; np.save(tmp,pred); os.replace(tmp,a.out)
    print(f'selected tau {best[1]:.2f}; wrote {a.out} shape={pred.shape} range={pred.min()}..{pred.max()}')

if __name__=='__main__': main()
