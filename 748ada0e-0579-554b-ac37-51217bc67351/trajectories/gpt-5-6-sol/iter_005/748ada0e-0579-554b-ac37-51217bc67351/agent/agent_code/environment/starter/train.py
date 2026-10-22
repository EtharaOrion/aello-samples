"""Budgeted CIFAR-100 trainer: 8000 updates, batch 128."""
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

class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride, drop=0.0):
        super().__init__(); self.bn1=nn.BatchNorm2d(cin)
        self.conv1=nn.Conv2d(cin,cout,3,stride,1,bias=False)
        self.bn2=nn.BatchNorm2d(cout); self.conv2=nn.Conv2d(cout,cout,3,1,1,bias=False)
        self.short=nn.Conv2d(cin,cout,1,stride,bias=False) if stride!=1 or cin!=cout else None
        self.drop=drop
    def forward(self,x):
        z=F.relu(self.bn1(x),inplace=True); shortcut=self.short(z) if self.short is not None else x
        z=self.conv1(z); z=self.conv2(F.dropout(F.relu(self.bn2(z),inplace=True),self.drop,self.training))
        return shortcut+z

class WideResNet(nn.Module):
    def __init__(self,depth=40,widen=10,classes=100):
        super().__init__(); n=(depth-4)//6; widths=[16,16*widen,32*widen,64*widen]
        self.conv=nn.Conv2d(3,widths[0],3,1,1,bias=False); groups=[]
        for gi,stride in enumerate((1,2,2)):
            blocks=[BasicBlock(widths[gi],widths[gi+1],stride)]
            blocks += [BasicBlock(widths[gi+1],widths[gi+1],1) for _ in range(n-1)]
            groups.append(nn.Sequential(*blocks))
        self.g1,self.g2,self.g3=groups; self.bn=nn.BatchNorm2d(widths[-1]); self.fc=nn.Linear(widths[-1],classes)
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear): nn.init.zeros_(m.bias)
    def forward(self,x):
        x=self.g3(self.g2(self.g1(self.conv(x))))
        return self.fc(F.adaptive_avg_pool2d(F.relu(self.bn(x),inplace=True),1).flatten(1))

def crop_flip(x):
    b,_,h,w=x.shape; x=F.pad(x,(4,4,4,4),mode='reflect')
    oy=torch.randint(0,9,(b,),device=x.device); ox=torch.randint(0,9,(b,),device=x.device)
    yy=oy[:,None]+torch.arange(h,device=x.device)[None]; xx=ox[:,None]+torch.arange(w,device=x.device)[None]
    x=x[torch.arange(b,device=x.device)[:,None,None,None],torch.arange(3,device=x.device)[None,:,None,None],yy[:,None,:,None],xx[:,None,None,:]]
    return torch.where((torch.rand(b,device=x.device)<.5)[:,None,None,None],x.flip(3),x)

def cutmix_or_mixup(x,y):
    b=x.size(0); perm=torch.randperm(b,device=x.device)
    if torch.rand((),device=x.device)<.65:
        lam=torch.rand((),device=x.device); cut=torch.sqrt(1-lam); cw=(32*cut).long(); ch=(32*cut).long()
        cx=torch.randint(0,32,(),device=x.device); cy=torch.randint(0,32,(),device=x.device)
        x1=torch.clamp(cx-cw//2,0,32); x2=torch.clamp(cx+cw//2,0,32); y1=torch.clamp(cy-ch//2,0,32); y2=torch.clamp(cy+ch//2,0,32)
        z=x.clone(); z[:,:,y1:y2,x1:x2]=x[perm,:,y1:y2,x1:x2]
        lam=1.0-((x2-x1)*(y2-y1)).float()/1024.0
    else:
        a=torch.tensor(.2,device=x.device); g1=torch._standard_gamma(a); g2=torch._standard_gamma(a); lam=g1/(g1+g2)
        z=x*lam+x[perm]*(1-lam)
    return z,y,y[perm],lam

def norm(x):
    return (x-x.new_tensor(MEAN)[None,:,None,None])/x.new_tensor(STD)[None,:,None,None]

@torch.no_grad()
def logits(model,x,device,batch=500,tta=True):
    model.eval(); ans=[]
    for s in range(0,len(x),batch):
        z=torch.from_numpy(np.ascontiguousarray(x[s:s+batch])).to(device).permute(0,3,1,2).float().div_(255.)
        z=norm(z.contiguous(memory_format=torch.channels_last))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            q=model(z).float(); q=(q+model(z.flip(3)).float())*.5 if tta else q
        ans.append(q.cpu())
    return torch.cat(ans).numpy()

def train(instance,seed):
    torch.manual_seed(seed); np.random.seed(seed); torch.backends.cudnn.benchmark=True
    x,y=data.load_train(instance); model=WideResNet().cuda().to(memory_format=torch.channels_last)
    ema=copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    opt=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,nesterov=True,weight_decay=5e-4)
    rng=np.random.default_rng(seed); start=time.time()
    for step in range(STEPS):
        lr=.1*(step+1)/400 if step<400 else .001+.099*.5*(1+math.cos(math.pi*(step-400)/(STEPS-400)))
        for pg in opt.param_groups: pg['lr']=lr
        ind=rng.integers(0,len(y),size=BATCH); xb=torch.from_numpy(np.ascontiguousarray(x[ind])).cuda().permute(0,3,1,2).float().div_(255.)
        xb=norm(crop_flip(xb.contiguous(memory_format=torch.channels_last))); yb=torch.from_numpy(y[ind]).cuda()
        xb,ya,yb2,lam=cutmix_or_mixup(xb,yb); opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred=model(xb); loss=lam*F.cross_entropy(pred,ya,label_smoothing=.05)+(1-lam)*F.cross_entropy(pred,yb2,label_smoothing=.05)
        loss.backward(); opt.step()
        if step==999: ema.load_state_dict(model.state_dict())
        elif step>999:
            for pe,p in zip(ema.parameters(),model.parameters()): pe.lerp_(p.detach(),.002)
            for be,b in zip(ema.buffers(),model.buffers()): be.copy_(b)
        if (step+1)%1000==0: print(f'step {step+1}/{STEPS} loss {loss.item():.4f} lr {lr:.5f} elapsed {time.time()-start:.1f}s',flush=True)
    print(f'training_seconds {time.time()-start:.2f}',flush=True); return ema

def main():
    p=argparse.ArgumentParser(); p.add_argument('--instance',required=True,choices=list(data.INSTANCES)); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--graded',required=True); p.add_argument('--out',required=True); p.add_argument('--val',default='../data/val_public.npz'); a=p.parse_args()
    model=train(a.instance,a.seed); vd=np.load(a.val); vp=logits(model,vd['x'],'cuda').argmax(1)
    rec=[(vp[vd['y']==c]==c).mean() for c in range(100)]
    print(f'val_accuracy {(vp==vd["y"]).mean():.5f} val_balanced_accuracy {np.mean(rec):.5f}',flush=True)
    pred=logits(model,np.load(a.graded)['x'],'cuda').argmax(1).astype(np.int64)
    tmp=a.out+'.tmp.npy'; np.save(tmp,pred); os.replace(tmp,a.out)
    print(f'wrote {a.out} shape={pred.shape} range={pred.min()}..{pred.max()} classes={len(np.unique(pred))}')

if __name__=='__main__': main()
