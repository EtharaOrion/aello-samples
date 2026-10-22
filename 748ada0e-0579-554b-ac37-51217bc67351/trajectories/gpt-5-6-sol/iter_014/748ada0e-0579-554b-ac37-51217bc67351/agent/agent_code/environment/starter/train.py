"""Budget-faithful CIFAR-100 trainer: WRN-40-10, 8000 x 128 presentations."""
import argparse, os, sys, time
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
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.bn1=nn.BatchNorm2d(cin); self.c1=nn.Conv2d(cin,cout,3,stride,1,bias=False)
        self.bn2=nn.BatchNorm2d(cout); self.c2=nn.Conv2d(cout,cout,3,1,1,bias=False)
        self.short=nn.Conv2d(cin,cout,1,stride,bias=False) if cin!=cout or stride!=1 else None
    def forward(self,x):
        z=F.relu(self.bn1(x),inplace=True); shortcut=self.short(z) if self.short is not None else x
        z=self.c1(z); z=F.relu(self.bn2(z),inplace=True); return shortcut+self.c2(z)

class WRN(nn.Module):
    def __init__(self,depth=40,widen=10):
        super().__init__(); n=(depth-4)//6; w=[16,16*widen,32*widen,64*widen]
        self.c0=nn.Conv2d(3,w[0],3,1,1,bias=False); groups=[]
        for g,stride in enumerate((1,2,2)):
            groups.append(nn.Sequential(Block(w[g],w[g+1],stride),
                          *[Block(w[g+1],w[g+1],1) for _ in range(n-1)]))
        self.groups=nn.Sequential(*groups); self.bn=nn.BatchNorm2d(w[-1]); self.fc=nn.Linear(w[-1],100)
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out')
            elif isinstance(m,nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear): nn.init.zeros_(m.bias)
    def forward(self,x):
        x=self.groups(self.c0(x)); x=F.relu(self.bn(x),inplace=True)
        return self.fc(F.adaptive_avg_pool2d(x,1).flatten(1))

def crop_flip(x):
    b,_,h,w=x.shape; x=F.pad(x,(4,4,4,4),mode='reflect')
    oy=torch.randint(0,9,(b,),device=x.device); ox=torch.randint(0,9,(b,),device=x.device)
    yy=oy[:,None]+torch.arange(h,device=x.device)[None]; xx=ox[:,None]+torch.arange(w,device=x.device)[None]
    x=x[torch.arange(b,device=x.device)[:,None,None,None],torch.arange(3,device=x.device)[None,:,None,None],yy[:,None,:,None],xx[:,None,None,:]]
    mask=torch.rand(b,device=x.device)<.5; x[mask]=x[mask].flip(-1); return x

def mild_augment(x,rng):
    """One inexpensive RandAugment-style photometric operation, batch-randomized."""
    op=int(rng.integers(0,6)); mag=float(rng.uniform(.72,1.28))
    if op==0: x=x*mag
    elif op==1:
        mu=x.mean((2,3),keepdim=True); x=(x-mu)*mag+mu
    elif op==2:
        gray=(.299*x[:,0:1]+.587*x[:,1:2]+.114*x[:,2:3]); x=gray+(x-gray)*mag
    elif op==3:
        threshold=float(rng.uniform(.35,.75)); x=torch.where(x>threshold,1-x,x)
    elif op==4:
        bits=int(rng.integers(5,8)); levels=float(2**bits-1); x=torch.round(x*levels)/levels
    else:
        lo=x.amin((2,3),keepdim=True); hi=x.amax((2,3),keepdim=True); x=(x-lo)/(hi-lo+1e-5)
    return x.clamp_(0,1)

def soft_ce(logits,target): return -(target*F.log_softmax(logits.float(),1)).sum(1).mean()

def train(seed):
    torch.manual_seed(seed); np.random.seed(seed); torch.backends.cudnn.benchmark=True
    x,y=data.load_train('i20k'); dev=torch.device('cuda')
    mean=torch.tensor(MEAN,device=dev)[None,:,None,None]; std=torch.tensor(STD,device=dev)[None,:,None,None]
    model=WRN().to(dev).to(memory_format=torch.channels_last)
    opt=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,nesterov=True,weight_decay=5e-4)
    rng=np.random.default_rng(seed); swa=None; nswa=0; t0=time.time()
    for step in range(STEPS):
        ii=rng.integers(0,len(y),BATCH)
        xb=torch.from_numpy(np.ascontiguousarray(x[ii])).permute(0,3,1,2).to(dev).float().div_(255)
        xb=xb.contiguous(memory_format=torch.channels_last); xb=(mild_augment(crop_flip(xb),rng)-mean)/std
        yb=torch.from_numpy(y[ii]).to(dev); target=F.one_hot(yb,100).float().mul_(.95).add_(.0005)
        # CutMix on half the batches retains local fine-grained cues better than
        # always-on MixUp for CIFAR-100; plain batches receive a small random erasure.
        if rng.random() < .5:
            lam=float(rng.beta(1.,1.)); perm=torch.randperm(BATCH,device=dev)
            side=int(round(32*np.sqrt(1-lam))); cx=int(rng.integers(0,32)); cy=int(rng.integers(0,32))
            x1=max(0,cx-side//2); x2=min(32,cx+(side+1)//2); y1=max(0,cy-side//2); y2=min(32,cy+(side+1)//2)
            xb[:,:,y1:y2,x1:x2]=xb[perm,:,y1:y2,x1:x2]
            area=(x2-x1)*(y2-y1)/1024.; target=target.mul(1-area).add_(target[perm],alpha=area)
        lr=.1*(step+1)/400 if step<400 else .05*(1+np.cos(np.pi*(step-400)/(STEPS-400)))
        for pg in opt.param_groups: pg['lr']=lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): loss=soft_ce(model(xb),target)
        loss.backward(); opt.step()
        if step>=7600 and (step+1)%50==0:
            state=model.state_dict()
            if swa is None: swa={k:v.detach().float().cpu().clone() for k,v in state.items()}
            else:
                for k,v in state.items(): swa[k].lerp_(v.detach().float().cpu(),1/(nswa+1))
            nswa+=1
        if (step+1)%1000==0: print(f'step {step+1}/{STEPS} loss {loss.item():.4f} lr {lr:.5f} time {time.time()-t0:.1f}s',flush=True)
    final={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    return model,mean,std,final,swa

@torch.no_grad()
def logits(model,images,mean,std,batch=500):
    model.eval(); ans=[]
    for s in range(0,len(images),batch):
        z=torch.from_numpy(np.ascontiguousarray(images[s:s+batch])).permute(0,3,1,2).cuda().float().div_(255)
        z=z.contiguous(memory_format=torch.channels_last); z=(z-mean)/std
        with torch.autocast('cuda',dtype=torch.bfloat16): ans.append(((model(z).float()+model(z.flip(-1)).float())*.5).cpu())
    return torch.cat(ans).numpy()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--instance',default='i20k'); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--graded',required=True); p.add_argument('--out',required=True); p.add_argument('--val',default='../data/val_public.npz')
    a=p.parse_args(); model,mean,std,final,swa=train(a.seed); vd=np.load(a.val)
    best=(-1,None,None,None)
    for name,state in [('final',final),('swa',swa)]:
        model.load_state_dict(state); vl=logits(model,vd['x'],mean,std); yp=vl.argmax(1)
        counts=np.bincount(data.load_train('i20k')[1],minlength=100).astype(np.float32)
        prior=np.log(counts/counts.mean())
        for tau in (0.,.25,.5,.75,1.,1.25,1.5):
            yp=(vl-tau*prior[None]).argmax(1)
            rec=[(yp[vd['y']==c]==c).mean() for c in range(100)]; bal=float(np.mean(rec))
            if bal>best[0]: best=(bal,name,state,tau)
        print(f'{name} best balanced {best[0]:.5f}',flush=True)
    model.load_state_dict(best[2]); print('selected',best[1],'tau',best[3],flush=True)
    gl=logits(model,np.load(a.graded)['x'],mean,std)
    pred=(gl-best[3]*prior[None]).argmax(1).astype(np.int64)
    tmp=a.out+'.tmp.npy'; np.save(tmp,pred); os.replace(tmp,a.out)
    print(f'wrote {a.out}: {pred.shape} {pred.dtype} range {pred.min()}..{pred.max()}',flush=True)

if __name__=='__main__': main()
