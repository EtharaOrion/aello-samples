"""Budget-faithful CIFAR-100 trainer: exactly 8000 updates x 128 examples."""
import argparse, copy, math, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aello_data as data
STEPS, BATCH = 8000, 128

class WideBasic(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.bn1=nn.BatchNorm2d(cin); self.c1=nn.Conv2d(cin,cout,3,stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(cout); self.c2=nn.Conv2d(cout,cout,3,padding=1,bias=False)
        self.short=nn.Conv2d(cin,cout,1,stride,bias=False) if cin!=cout or stride!=1 else None
    def forward(self,x):
        z=F.relu(self.bn1(x),inplace=True); shortcut=x if self.short is None else self.short(z)
        return self.c2(F.relu(self.bn2(self.c1(z)),inplace=True))+shortcut

class WRN(nn.Module):
    def __init__(self, depth=40, widen=10):
        super().__init__(); n=(depth-4)//6; widths=[16,16*widen,32*widen,64*widen]
        self.c0=nn.Conv2d(3,16,3,padding=1,bias=False); blocks=[]; cin=16
        for cout,stride in zip(widths[1:],[1,2,2]):
            blocks.append(WideBasic(cin,cout,stride)); blocks += [WideBasic(cout,cout,1) for _ in range(n-1)]; cin=cout
        self.blocks=nn.Sequential(*blocks); self.bn=nn.BatchNorm2d(cin); self.fc=nn.Linear(cin,100)
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode="fan_out",nonlinearity="relu")
            elif isinstance(m,nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear): nn.init.zeros_(m.bias)
    def forward(self,x):
        x=self.blocks(self.c0(x)); x=F.relu(self.bn(x),inplace=True)
        return self.fc(F.adaptive_avg_pool2d(x,1).flatten(1))

def crop_flip(x):
    b,c,h,w=x.shape; xp=F.pad(x,(4,4,4,4),mode="reflect")
    oy=torch.randint(0,9,(b,),device=x.device); ox=torch.randint(0,9,(b,),device=x.device)
    yy=oy[:,None]+torch.arange(h,device=x.device)[None]; xx=ox[:,None]+torch.arange(w,device=x.device)[None]
    x=xp[torch.arange(b,device=x.device)[:,None,None,None],torch.arange(c,device=x.device)[None,:,None,None],yy[:,None,:,None],xx[:,None,None,:]]
    flip=torch.rand(b,device=x.device)<.5; x[flip]=x[flip].flip(3); return x

def mixed_loss(model,x,y):
    r=random.random()
    if r>=.50: return F.cross_entropy(model(x),y,label_smoothing=.04)
    perm=torch.randperm(len(y),device=y.device)
    if r<.25:
        lam=float(np.random.beta(.2,.2)); z=model(x.mul(lam).add(x[perm],alpha=1-lam))
    else:
        lam0=float(np.random.beta(1.,1.)); side=int(32*math.sqrt(1-lam0)); cx,cy=random.randrange(32),random.randrange(32)
        x1,x2=max(cx-side//2,0),min(cx+side//2,32); y1,y2=max(cy-side//2,0),min(cy+side//2,32)
        x[:,:,y1:y2,x1:x2]=x[perm,:,y1:y2,x1:x2]; lam=1-(x2-x1)*(y2-y1)/1024.; z=model(x)
    return lam*F.cross_entropy(z,y,label_smoothing=.04)+(1-lam)*F.cross_entropy(z,y[perm],label_smoothing=.04)

@torch.inference_mode()
def logits_tta(model,x,device,batch=512):
    model.eval(); mean=torch.tensor([.5071,.4867,.4408],device=device)[None,:,None,None]; std=torch.tensor([.2675,.2565,.2761],device=device)[None,:,None,None]; ans=[]
    for s in range(0,len(x),batch):
        z=torch.from_numpy(np.ascontiguousarray(x[s:s+batch])).to(device).permute(0,3,1,2).float().div_(255.); z=(z-mean)/std
        with torch.autocast("cuda",dtype=torch.bfloat16): ans.append(((model(z).float()+model(z.flip(3)).float())*.5).cpu())
    return torch.cat(ans).numpy()

def balanced_accuracy(pred,y): return np.mean([np.mean(pred[y==c]==c) for c in range(100)])

def train(seed,val_path):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.backends.cudnn.benchmark=True
    x,y=data.load_train("i20k"); by_class=[np.flatnonzero(y==c) for c in range(100)]; rng=np.random.default_rng(seed)
    model=WRN().cuda().to(memory_format=torch.channels_last); avg=copy.deepcopy(model).eval()
    for p in avg.parameters(): p.requires_grad_(False)
    opt=torch.optim.SGD(model.parameters(),lr=.20,momentum=.9,nesterov=True,weight_decay=5e-4,fused=True)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.20,total_steps=STEPS,pct_start=.10,anneal_strategy="cos",div_factor=20.,final_div_factor=100.)
    mean=torch.tensor([.5071,.4867,.4408],device="cuda")[None,:,None,None]; std=torch.tensor([.2675,.2565,.2761],device="cuda")[None,:,None,None]; t0=time.time()
    for step in range(STEPS):
        cls=rng.integers(0,100,BATCH); ids=np.fromiter((by_class[c][rng.integers(len(by_class[c]))] for c in cls),dtype=np.int64,count=BATCH)
        xb=torch.from_numpy(np.ascontiguousarray(x[ids])).cuda().permute(0,3,1,2).contiguous(memory_format=torch.channels_last).float().div_(255.)
        xb=(crop_flip(xb)-mean)/std; yb=torch.from_numpy(y[ids]).cuda(); opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16): loss=mixed_loss(model,xb,yb)
        loss.backward(); opt.step(); sched.step()
        if (step+1)%10==0:
            with torch.no_grad(): torch._foreach_lerp_(list(avg.parameters()),list(model.parameters()),.01)
        if (step+1)%1000==0: print(f"step {step+1}/{STEPS} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.5f} elapsed {time.time()-t0:.1f}s",flush=True)
    for ab,mb in zip(avg.buffers(),model.buffers()): ab.copy_(mb)
    v=np.load(val_path); vx,vy=v["x"],v["y"]; rawlog=logits_tta(model,vx,"cuda"); avglog=logits_tta(avg,vx,"cuda")
    rawba=balanced_accuracy(rawlog.argmax(1),vy); avgba=balanced_accuracy(avglog.argmax(1),vy)
    print(f"public raw BA {rawba:.5f} acc {(rawlog.argmax(1)==vy).mean():.5f}"); print(f"public ema BA {avgba:.5f} acc {(avglog.argmax(1)==vy).mean():.5f}")
    chosen=avg if avgba>=rawba else model; print("selected","ema" if chosen is avg else "raw","elapsed",time.time()-t0); return chosen

def main():
    p=argparse.ArgumentParser(); p.add_argument("--instance",default="i20k"); p.add_argument("--seed",type=int,default=0); p.add_argument("--graded",required=True); p.add_argument("--out",required=True); p.add_argument("--val",default="../data/val_public.npz"); a=p.parse_args()
    model=train(a.seed,a.val); gx=np.load(a.graded)["x"]; pred=logits_tta(model,gx,"cuda").argmax(1).astype(np.int64)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)),exist_ok=True); tmp=a.out+".tmp.npy"; np.save(tmp,pred); os.replace(tmp,a.out)
    print("wrote",a.out,pred.shape,pred.dtype,int(pred.min()),int(pred.max()))
if __name__=="__main__": main()
