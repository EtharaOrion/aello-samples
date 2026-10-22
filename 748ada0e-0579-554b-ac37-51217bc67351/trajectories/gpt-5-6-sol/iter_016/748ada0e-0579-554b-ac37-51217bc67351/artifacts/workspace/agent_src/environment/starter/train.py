"""Budget-exact CIFAR-100 trainer: WRN-40-10, balanced sampling, mixing and EMA."""
import argparse, copy, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aello_data as data
STEPS, BATCH = 8000, 128

class Block(nn.Module):
    def __init__(self, ni, no, stride):
        super().__init__(); self.equal=(ni==no and stride==1)
        self.b1=nn.BatchNorm2d(ni); self.c1=nn.Conv2d(ni,no,3,stride,1,bias=False)
        self.b2=nn.BatchNorm2d(no); self.c2=nn.Conv2d(no,no,3,1,1,bias=False)
        self.skip=nn.Identity() if self.equal else nn.Conv2d(ni,no,1,stride,bias=False)
    def forward(self,x):
        z=F.relu(self.b1(x),inplace=True); s=x if self.equal else self.skip(z)
        return s+self.c2(F.relu(self.b2(self.c1(z)),inplace=True))

class WRN(nn.Module):
    def __init__(self, depth=40, widen=10):
        super().__init__(); n=(depth-4)//6; ch=[16,16*widen,32*widen,64*widen]
        self.c=nn.Conv2d(3,ch[0],3,1,1,bias=False); groups=[]
        for g in range(3):
            groups.append(nn.Sequential(*[Block(ch[g] if j==0 else ch[g+1],ch[g+1],2 if g and j==0 else 1) for j in range(n)]))
        self.g=nn.ModuleList(groups); self.bn=nn.BatchNorm2d(ch[3]); self.fc=nn.Linear(ch[3],100)
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear): nn.init.zeros_(m.bias)
    def forward(self,x):
        x=self.c(x)
        for g in self.g: x=g(x)
        return self.fc(F.adaptive_avg_pool2d(F.relu(self.bn(x),inplace=True),1).flatten(1))

def crop_flip(x):
    b,c,h,w=x.shape; x=F.pad(x,(4,4,4,4),mode='reflect')
    oy=torch.randint(0,9,(b,),device=x.device); ox=torch.randint(0,9,(b,),device=x.device)
    yy=oy[:,None]+torch.arange(h,device=x.device)[None]; xx=ox[:,None]+torch.arange(w,device=x.device)[None]
    x=x[torch.arange(b,device=x.device)[:,None,None,None],torch.arange(c,device=x.device)[None,:,None,None],yy[:,None,:,None],xx[:,None,None,:]]
    mask=torch.rand(b,device=x.device)<.5; x[mask]=x[mask].flip(3)
    if torch.rand((),device=x.device)<.5: x=(x-.5)*torch.empty((),device=x.device).uniform_(.75,1.25)+.5
    if torch.rand((),device=x.device)<.35:
        gray=x.mean(1,keepdim=True); x=gray+(x-gray)*torch.empty((),device=x.device).uniform_(.7,1.3)
    return x.clamp_(0,1)

MEAN=(.5071,.4867,.4408); STD=(.2675,.2565,.2761)
def norm(x):
    return (x-x.new_tensor(MEAN)[None,:,None,None])/x.new_tensor(STD)[None,:,None,None]

@torch.no_grad()
def logits(model, arr, flip=False, bs=512):
    model.eval(); out=[]
    for s in range(0,len(arr),bs):
        x=torch.from_numpy(np.ascontiguousarray(arr[s:s+bs])).cuda().permute(0,3,1,2).float().div_(255)
        x=norm(x.contiguous(memory_format=torch.channels_last))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z=model(x).float()
            if flip: z=(z+model(x.flip(3)).float())/2
        out.append(z.cpu())
    model.train(); return torch.cat(out).numpy()

def bacc(pred,y): return np.mean([np.mean(pred[y==k]==k) for k in range(100)])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--instance',required=True); ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--graded',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.backends.cudnn.benchmark=True
    x,y=data.load_train(a.instance); vd=np.load('../data/val_public.npz'); xv,yv=vd['x'],vd['y']
    byclass=[np.flatnonzero(y==k) for k in range(100)]; rng=np.random.default_rng(a.seed)
    model=WRN().cuda().to(memory_format=torch.channels_last); ema=copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    opt=torch.optim.SGD(model.parameters(),lr=.12,momentum=.9,nesterov=True,weight_decay=5e-4)
    warm=400; t0=time.time()
    for step in range(STEPS):
        cls=rng.integers(0,100,BATCH); ids=np.array([rng.choice(byclass[k]) for k in cls])
        xb=torch.from_numpy(np.ascontiguousarray(x[ids])).cuda().permute(0,3,1,2).float().div_(255)
        xb=crop_flip(xb.contiguous(memory_format=torch.channels_last)); yb=torch.tensor(y[ids],device='cuda')
        perm=torch.randperm(BATCH,device='cuda'); lam=float(rng.beta(.5,.5))
        if rng.random()<.55:
            r=np.sqrt(1-lam); rw=int(32*r); rh=int(32*r); cx=int(rng.integers(32)); cy=int(rng.integers(32))
            x1=max(cx-rw//2,0); x2=min(cx+rw//2,32); y1=max(cy-rh//2,0); y2=min(cy+rh//2,32)
            xb[:,:,y1:y2,x1:x2]=xb[perm,:,y1:y2,x1:x2]; lam=1-(x2-x1)*(y2-y1)/1024
        else: xb=xb*lam+xb[perm]*(1-lam)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z=model(norm(xb)); loss=lam*F.cross_entropy(z,yb,label_smoothing=.04)+(1-lam)*F.cross_entropy(z,yb[perm],label_smoothing=.04)
        loss.backward(); opt.step()
        q=(step+1)/warm if step<warm else .5*(1+np.cos(np.pi*(step-warm)/(STEPS-warm)))
        for pg in opt.param_groups: pg['lr']=.12*q
        with torch.no_grad():
            for pe,p in zip(ema.parameters(),model.parameters()): pe.mul_(.999).add_(p,alpha=.001)
            for be,b in zip(ema.buffers(),model.buffers()): be.copy_(b)
        if (step+1)%1000==0: print(step+1,round(float(loss),4),round(opt.param_groups[0]['lr'],5),round(time.time()-t0,1),flush=True)
    lm=logits(model,xv,True); le=logits(ema,xv,True)
    candidates=[('last',lm),('ema',le),('blend',(lm+le)/2)]; scores=[bacc(z.argmax(1),yv) for _,z in candidates]
    print('validation',[(c,round(s,5)) for (c,_),s in zip(candidates,scores)],flush=True)
    chosen=candidates[int(np.argmax(scores))][0]; xg=np.load(a.graded)['x']
    if chosen=='last': gl=logits(model,xg,True)
    elif chosen=='ema': gl=logits(ema,xg,True)
    else: gl=(logits(model,xg,True)+logits(ema,xg,True))/2
    pred=gl.argmax(1).astype(np.int64); np.save(a.out,pred)
    print('wrote',a.out,'choice',chosen,'shape',pred.shape,'range',pred.min(),pred.max(),'seconds',round(time.time()-t0,1))
if __name__=='__main__': main()
