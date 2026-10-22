"""Budget-faithful CIFAR-100 trainer: one 8,000 update run, batch size 128."""
import argparse, copy, math, os, sys, time
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
    def __init__(self, ni, no, stride):
        super().__init__()
        self.bn1, self.bn2 = nn.BatchNorm2d(ni), nn.BatchNorm2d(no)
        self.c1 = nn.Conv2d(ni, no, 3, stride, 1, bias=False)
        self.c2 = nn.Conv2d(no, no, 3, 1, 1, bias=False)
        self.short = nn.Identity() if ni == no and stride == 1 else nn.Conv2d(ni, no, 1, stride, bias=False)
    def forward(self, x):
        z = F.relu(self.bn1(x), inplace=True)
        s = x if isinstance(self.short, nn.Identity) else self.short(z)
        z = self.c1(z)
        return s + self.c2(F.relu(self.bn2(z), inplace=True))

class WRN(nn.Module):
    def __init__(self, depth=40, widen=10, classes=100):
        super().__init__(); n=(depth-4)//6; ch=[16,16*widen,32*widen,64*widen]
        self.stem=nn.Conv2d(3,ch[0],3,1,1,bias=False)
        def group(a,b,stride): return nn.Sequential(Block(a,b,stride),*[Block(b,b,1) for _ in range(n-1)])
        self.g1,self.g2,self.g3=group(ch[0],ch[1],1),group(ch[1],ch[2],2),group(ch[2],ch[3],2)
        self.bn=nn.BatchNorm2d(ch[3]); self.fc=nn.Linear(ch[3],classes)
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear): nn.init.zeros_(m.bias)
    def forward(self,x):
        x=self.g3(self.g2(self.g1(self.stem(x))))
        return self.fc(F.adaptive_avg_pool2d(F.relu(self.bn(x),inplace=True),1).flatten(1))

def prep(x,device):
    x=torch.from_numpy(np.ascontiguousarray(x)).permute(0,3,1,2).to(device,non_blocking=True).float().div_(255)
    return x.contiguous(memory_format=torch.channels_last)

def normalize(x):
    return (x-x.new_tensor(MEAN)[None,:,None,None])/x.new_tensor(STD)[None,:,None,None]

def crop_flip(x):
    b,_,h,w=x.shape; x=F.pad(x,(4,4,4,4),mode='reflect')
    oy=torch.randint(0,9,(b,),device=x.device); ox=torch.randint(0,9,(b,),device=x.device)
    patches=x.unfold(2,h,1).unfold(3,w,1)
    x=patches[torch.arange(b,device=x.device),:,oy,ox]
    flip=torch.rand(b,device=x.device)<.5; x[flip]=x[flip].flip(3)
    return x.contiguous(memory_format=torch.channels_last)

def soft_ce(logits,y1,y2,lam,smoothing=.05):
    a=F.cross_entropy(logits,y1,label_smoothing=smoothing,reduction='none')
    b=F.cross_entropy(logits,y2,label_smoothing=smoothing,reduction='none')
    return (lam*a+(1-lam)*b).mean()

@torch.no_grad()
def update_ema(ema,model,decay):
    for e,m in zip(ema.parameters(),model.parameters()): e.lerp_(m,1-decay)
    for e,m in zip(ema.buffers(),model.buffers()): e.copy_(m)

@torch.no_grad()
def logits_tta(model,arr,device,batch=500,shifted=False):
    model.eval(); outs=[]; shifts=((0,0),(0,1),(1,0),(-1,0),(0,-1)) if shifted else ((0,0),)
    for s in range(0,len(arr),batch):
        x=prep(arr[s:s+batch],device); total=0
        for dy,dx in shifts:
            z=torch.roll(x,(dy,dx),(2,3)) if (dy or dx) else x
            with torch.autocast('cuda',dtype=torch.bfloat16): total+=model(normalize(z)).float()+model(normalize(z.flip(3))).float()
        outs.append((total/(2*len(shifts))).cpu())
    return torch.cat(outs).numpy()

def train(instance,seed,device='cuda'):
    torch.manual_seed(seed); np.random.seed(seed); torch.backends.cudnn.benchmark=True
    x,y=data.load_train(instance); model=WRN().to(device).to(memory_format=torch.channels_last)
    by_class=[np.flatnonzero(y==c) for c in range(100)]
    ema=copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    opt=torch.optim.SGD(model.parameters(),lr=.2,momentum=.9,nesterov=True,weight_decay=5e-4)
    rng=np.random.default_rng(seed); start=time.time()
    for step in range(STEPS):
        cls=rng.integers(0,100,BATCH); ind=np.fromiter((by_class[c][rng.integers(len(by_class[c]))] for c in cls),dtype=np.int64,count=BATCH)
        xb=crop_flip(prep(x[ind],device)); yb=torch.from_numpy(y[ind]).to(device)
        target2=yb; lam=1.0
        if step%4 != 0:
            perm=torch.randperm(BATCH,device=device); target2=yb[perm]
            raw=float(rng.beta(1.,1.)); cut=math.sqrt(1-raw); ww=int(32*cut); hh=int(32*cut); cx=int(rng.integers(32)); cy=int(rng.integers(32))
            x1=max(0,cx-ww//2); x2=min(32,cx+ww//2); y1=max(0,cy-hh//2); y2=min(32,cy+hh//2)
            xb[:,:,y1:y2,x1:x2]=xb[perm,:,y1:y2,x1:x2]; lam=1-((x2-x1)*(y2-y1)/1024.)
        lr=.1*(step+1)/400 if step<400 else 1e-4+(.1-1e-4)*.5*(1+math.cos(math.pi*(step-400)/(STEPS-400)))
        for g in opt.param_groups:g['lr']=lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): loss=soft_ce(model(normalize(xb)),yb,target2,lam)
        loss.backward(); opt.step(); update_ema(ema,model,.995)
        if (step+1)%1000==0: print(f'step {step+1}/{STEPS} loss {loss.item():.4f} lr {lr:.5f} sec {time.time()-start:.1f}',flush=True)
    return model,ema

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--instance',required=True,choices=list(data.INSTANCES)); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--graded',required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args(); raw,ema=train(a.instance,a.seed)
    xv=np.load('../data/val_public.npz'); best=(-1,None,None,None)
    for name,model in [('raw',raw),('ema',ema)]:
        for shifted in (False,True):
            lv=logits_tta(model,xv['x'],'cuda',shifted=shifted); pv=lv.argmax(1)
            bal=np.mean([(pv[xv['y']==c]==c).mean() for c in range(100)])
            print(f'{name} shifted={shifted} public accuracy {(pv==xv["y"]).mean():.5f} balanced {bal:.5f}',flush=True)
            if bal>best[0]: best=(bal,name,model,shifted)
    print('selected',best[1],'shifted',best[3],flush=True)
    pred=logits_tta(best[2],np.load(a.graded)['x'],'cuda',shifted=best[3]).argmax(1).astype(np.int64)
    np.save(a.out,pred); print('wrote',a.out,flush=True)
if __name__=='__main__': main()
