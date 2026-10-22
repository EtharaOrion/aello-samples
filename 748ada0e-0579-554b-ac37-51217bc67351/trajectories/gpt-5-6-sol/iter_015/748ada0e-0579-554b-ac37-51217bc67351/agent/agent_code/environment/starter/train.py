"""Budget-honest CIFAR-100 trainer: WRN-40-10, 8000 x 128 presentations."""
import argparse, copy, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCH, STEPS, NCLASS = 128, 8000, 100
MEAN=(.50614085,.48646389,.44178585); STD=(.26770826,.25607717,.27548141)

def normalize(x):
    mean=x.new_tensor(MEAN)[None,:,None,None]; std=x.new_tensor(STD)[None,:,None,None]
    return (x-mean)/std

class Block(nn.Module):
    def __init__(self, ni, no, stride, drop=0.0):
        super().__init__()
        self.bn1=nn.BatchNorm2d(ni); self.c1=nn.Conv2d(ni,no,3,stride,1,bias=False)
        self.bn2=nn.BatchNorm2d(no); self.c2=nn.Conv2d(no,no,3,1,1,bias=False)
        self.short = nn.Identity() if ni==no and stride==1 else nn.Conv2d(ni,no,1,stride,bias=False)
        self.drop=drop
    def forward(self,x):
        z=F.relu(self.bn1(x),inplace=True)
        shortcut=self.short(z) if not isinstance(self.short,nn.Identity) else x
        z=self.c1(z); z=self.c2(F.dropout(F.relu(self.bn2(z),inplace=True),self.drop,self.training))
        return shortcut+z

class WRN(nn.Module):
    def __init__(self, depth=40, widen=10, drop=.0):
        super().__init__(); n=(depth-4)//6; ch=[16,16*widen,32*widen,64*widen]
        self.stem=nn.Conv2d(3,ch[0],3,1,1,bias=False)
        def group(a,b,s): return nn.Sequential(Block(a,b,s,drop),*[Block(b,b,1,drop) for _ in range(n-1)])
        self.g1=group(ch[0],ch[1],1); self.g2=group(ch[1],ch[2],2); self.g3=group(ch[2],ch[3],2)
        self.bn=nn.BatchNorm2d(ch[3]); self.fc=nn.Linear(ch[3],NCLASS)
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear): nn.init.zeros_(m.bias)
    def forward(self,x):
        x=self.g3(self.g2(self.g1(self.stem(x))))
        return self.fc(F.adaptive_avg_pool2d(F.relu(self.bn(x),inplace=True),1).flatten(1))

def crop_flip(x, gen):
    b,c,h,w=x.shape; x=F.pad(x,(4,4,4,4),mode='reflect')
    oy=torch.randint(0,9,(b,),device=x.device,generator=gen); ox=torch.randint(0,9,(b,),device=x.device,generator=gen)
    yy=oy[:,None]+torch.arange(h,device=x.device)[None]; xx=ox[:,None]+torch.arange(w,device=x.device)[None]
    x=x[torch.arange(b,device=x.device)[:,None,None,None],torch.arange(c,device=x.device)[None,:,None,None],yy[:,None,:,None],xx[:,None,None,:]]
    flip=torch.rand(b,device=x.device,generator=gen)<.5; x[flip]=x[flip].flip(-1)
    return x

def soft_ce(logits, ya, yb, lam, smooth=.08):
    la=F.cross_entropy(logits,ya,label_smoothing=smooth,reduction='none')
    lb=F.cross_entropy(logits,yb,label_smoothing=smooth,reduction='none')
    return (lam*la+(1-lam)*lb).mean()

def train(seed, train_path):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    d=np.load(train_path); x=d['x']; y=d['y']; dev='cuda'; byclass=[np.flatnonzero(y==k) for k in range(NCLASS)]
    model=WRN(drop=.0).to(dev).to(memory_format=torch.channels_last)
    opt=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,nesterov=True,weight_decay=5e-4)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.1,total_steps=STEPS,pct_start=.08,div_factor=10,final_div_factor=500)
    rng=np.random.default_rng(seed); gen=torch.Generator(device=dev); gen.manual_seed(seed+1000)
    avg=copy.deepcopy(model).eval(); navg=0; t=time.time(); model.train()
    randaug=v2.AutoAugment(policy=v2.AutoAugmentPolicy.CIFAR10,fill=0.5)
    for step in range(STEPS):
        cls=np.concatenate((np.arange(NCLASS),rng.integers(0,NCLASS,BATCH-NCLASS))); rng.shuffle(cls)
        idx=np.fromiter((rng.choice(byclass[k]) for k in cls),np.int64,BATCH)
        xb=torch.from_numpy(np.ascontiguousarray(x[idx])).permute(0,3,1,2).to(dev).float().div_(255.).contiguous(memory_format=torch.channels_last)
        yb=torch.from_numpy(y[idx]).to(dev); xb=crop_flip(xb,gen)
        # Separate calls provide an independent policy draw for every image.
        xb=torch.cat([randaug(q) for q in xb.split(1)])
        xb=normalize(xb)
        cy=torch.randint(0,32,(BATCH,),device=dev,generator=gen); cx=torch.randint(0,32,(BATCH,),device=dev,generator=gen)
        gy=torch.arange(32,device=dev)[None,:,None]; gx=torch.arange(32,device=dev)[None,None,:]
        keep=((gy-cy[:,None,None]).abs()>=8)|((gx-cx[:,None,None]).abs()>=8); xb.mul_(keep[:,None])
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z=model(xb); loss=F.cross_entropy(z,yb,label_smoothing=.10)
        loss.backward(); opt.step(); sched.step()
        if step>=7000:
            navg+=1; a=1.0/navg
            with torch.no_grad():
                for q,p in zip(avg.parameters(),model.parameters()): q.lerp_(p,a)
                for q,p in zip(avg.buffers(),model.buffers()):
                    if q.is_floating_point(): q.lerp_(p,a)
                    else: q.copy_(p)
        if step%500==0 or step==STEPS-1:
            print(f'step {step+1}/{STEPS} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.5f} sec {time.time()-t:.1f}',flush=True)
    return model,avg

@torch.inference_mode()
def logits(model,x,batch=500):
    model.eval(); outs=[]; temperature=1.5
    for s in range(0,len(x),batch):
        z=torch.from_numpy(np.ascontiguousarray(x[s:s+batch])).permute(0,3,1,2).cuda().float().div_(255.).contiguous(memory_format=torch.channels_last)
        z=normalize(z)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=model(z).float().div(temperature).softmax(1)+model(z.flip(-1)).float().div(temperature).softmax(1)
        outs.append((out/2).cpu())
    return torch.cat(outs).numpy()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--instance',default='i20k'); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--graded',required=True); p.add_argument('--out',required=True); p.add_argument('--val',default='../data/val_public.npz')
    a=p.parse_args(); train_path=os.path.join(ROOT,'data',f'train_{a.instance}.npz'); raw,avg=train(a.seed,train_path)
    vd=np.load(a.val); vy=vd['y']; best=None
    for name,model in [('raw',raw),('average',avg)]:
        vp=logits(model,vd['x']).argmax(1); recalls=[(vp[vy==k]==k).mean() for k in range(NCLASS)]; ba=float(np.mean(recalls))
        print(f'public {name} accuracy {(vp==vy).mean():.5f} balanced {ba:.5f}',flush=True)
        if best is None or ba>best[0]: best=(ba,name,model)
    _,chosen,model=best; print('selected',chosen,flush=True)
    gp=logits(model,np.load(a.graded)['x']).argmax(1).astype(np.int64)
    tmp=a.out+'.tmp.npy'; np.save(tmp,gp); os.replace(tmp,a.out)
    torch.save({'model':model.state_dict(),'seed':a.seed,'steps':STEPS,'batch':BATCH},a.out+'.pt')
    print(f'wrote {a.out}: {gp.shape} {gp.dtype} range {gp.min()}..{gp.max()}',flush=True)

if __name__=='__main__': main()
