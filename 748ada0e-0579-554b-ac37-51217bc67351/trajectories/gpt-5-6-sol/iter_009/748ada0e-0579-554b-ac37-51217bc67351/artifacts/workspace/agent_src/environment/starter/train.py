"""Budget-exact CIFAR-100 trainer; validation is never used for optimization."""
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aello_data as data

STEPS, BATCH = 8000, 128
MEAN, STD = (0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761)

class WideBasic(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.bn1=nn.BatchNorm2d(cin); self.c1=nn.Conv2d(cin,cout,3,stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(cout); self.c2=nn.Conv2d(cout,cout,3,padding=1,bias=False)
        self.short=nn.Conv2d(cin,cout,1,stride,bias=False) if cin!=cout else nn.Identity()
    def forward(self,x):
        z=F.relu(self.bn1(x),inplace=True)
        s=self.short(z) if not isinstance(self.short,nn.Identity) else x
        return s+self.c2(F.relu(self.bn2(self.c1(z)),inplace=True))

class WideResNet(nn.Module):
    def __init__(self, depth=40, widen=10):
        super().__init__(); n=(depth-4)//6; w=[16,16*widen,32*widen,64*widen]
        self.c0=nn.Conv2d(3,w[0],3,padding=1,bias=False)
        self.g1=self.group(n,w[0],w[1],1); self.g2=self.group(n,w[1],w[2],2); self.g3=self.group(n,w[2],w[3],2)
        self.bn=nn.BatchNorm2d(w[3]); self.fc=nn.Linear(w[3],100)
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode="fan_out",nonlinearity="relu")
            elif isinstance(m,nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        nn.init.zeros_(self.fc.bias)
    @staticmethod
    def group(n,cin,cout,stride):
        return nn.Sequential(WideBasic(cin,cout,stride),*[WideBasic(cout,cout,1) for _ in range(n-1)])
    def forward(self,x):
        x=self.g3(self.g2(self.g1(self.c0(x))))
        return self.fc(F.adaptive_avg_pool2d(F.relu(self.bn(x),inplace=True),1).flatten(1))

def crop_flip(x):
    b,_,h,w=x.shape; x=F.pad(x,(4,4,4,4),mode="reflect")
    oy=torch.randint(0,9,(b,),device=x.device); ox=torch.randint(0,9,(b,),device=x.device)
    yy=oy[:,None]+torch.arange(h,device=x.device)[None]; xx=ox[:,None]+torch.arange(w,device=x.device)[None]
    x=x[torch.arange(b,device=x.device)[:,None,None,None],torch.arange(3,device=x.device)[None,:,None,None],yy[:,None,:,None],xx[:,None,None,:]]
    flip=torch.rand(b,device=x.device)<.5; x[flip]=x[flip].flip(-1)
    return x

def augment(x):
    x=crop_flip(x); b=len(x)
    if torch.rand((),device=x.device)<.8:
        br=1+(torch.rand(b,1,1,1,device=x.device)*.30-.15); ct=1+(torch.rand(b,1,1,1,device=x.device)*.30-.15)
        mu=x.mean((1,2,3),keepdim=True); x=(x*br-mu)*ct+mu
    if torch.rand((),device=x.device)<.5:
        sz=int(torch.randint(4,11,(),device=x.device)); yy=torch.randint(0,33-sz,(b,),device=x.device); xx=torch.randint(0,33-sz,(b,),device=x.device)
        gy=torch.arange(32,device=x.device)[None,:,None]; gx=torch.arange(32,device=x.device)[None,None,:]
        mask=(gy>=yy[:,None,None])&(gy<yy[:,None,None]+sz)&(gx>=xx[:,None,None])&(gx<xx[:,None,None]+sz)
        x=x.masked_fill(mask[:,None],.5)
    return x.clamp_(0,1)

def soft_ce(logits,target): return -(target*F.log_softmax(logits,1)).sum(1).mean()

def train(seed):
    x,y=data.load_train("i20k"); device=torch.device("cuda")
    torch.manual_seed(seed); np.random.seed(seed); torch.backends.cudnn.benchmark=True
    model=WideResNet().to(device).to(memory_format=torch.channels_last)
    opt=torch.optim.SGD(model.parameters(),lr=.12,momentum=.9,nesterov=True,weight_decay=5e-4)
    mean=torch.tensor(MEAN,device=device)[None,:,None,None]; std=torch.tensor(STD,device=device)[None,:,None,None]
    gen=np.random.default_rng(seed); swa=None; n_swa=0; started=time.time(); model.train()
    for step in range(STEPS):
        ids=gen.integers(0,len(y),size=BATCH)
        xb=torch.from_numpy(np.ascontiguousarray(x[ids])).permute(0,3,1,2).to(device).float().div_(255.)
        xb=augment(xb); xb=((xb-mean)/std).contiguous(memory_format=torch.channels_last)
        labels=torch.from_numpy(y[ids]).to(device); target=F.one_hot(labels,100).float().mul_(.95).add_(.0005)
        perm=torch.randperm(BATCH,device=device); lam=float(np.random.beta(.5,.5))
        if step%2==0: xb=xb.mul(lam).add_(xb[perm],alpha=1-lam)
        else:
            cut=int(32*np.sqrt(1-lam)); cy,cx=gen.integers(0,32,size=2)
            y0,y1=max(0,cy-cut//2),min(32,cy+cut//2); x0,x1=max(0,cx-cut//2),min(32,cx+cut//2)
            xb[:,:,y0:y1,x0:x1]=xb[perm,:,y0:y1,x0:x1]; lam=1-((y1-y0)*(x1-x0))/1024
        target=target.mul(lam).add_(target[perm],alpha=1-lam)
        lr=.12*(step+1)/400 if step<400 else .12*.5*(1+np.cos(np.pi*(step-400)/(STEPS-400)))
        for pg in opt.param_groups: pg["lr"]=lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16): loss=soft_ce(model(xb),target)
        loss.backward(); opt.step()
        if step>=7000 and step%100==0:
            state=model.state_dict()
            if swa is None: swa={k:v.detach().clone() for k,v in state.items()}
            else:
                for k,v in state.items():
                    if swa[k].is_floating_point(): swa[k].lerp_(v.detach(),1/(n_swa+1))
                    else: swa[k].copy_(v)
            n_swa+=1
        if (step+1)%1000==0: print(f"step {step+1}/{STEPS} loss {loss.item():.4f} lr {lr:.5f} elapsed {time.time()-started:.1f}s",flush=True)
    if swa is not None: model.load_state_dict(swa)
    print(f"training presentations={STEPS*BATCH} checkpoints_averaged={n_swa}",flush=True)
    return model,mean,std

@torch.inference_mode()
def get_logits(model,images,mean,std,tta=True,batch=512):
    model.eval(); ans=[]
    for s in range(0,len(images),batch):
        z=torch.from_numpy(np.ascontiguousarray(images[s:s+batch])).permute(0,3,1,2).cuda().float().div_(255.)
        z=((z-mean)/std).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            q=model(z).float(); q=(q+model(z.flip(-1)).float())/2 if tta else q
        ans.append(q.cpu().numpy())
    return np.concatenate(ans)

def balanced_bias(logit,target_count):
    z=logit.astype(np.float64); bias=np.zeros(z.shape[1])
    for _ in range(100):
        q=z+bias; q=np.exp(q-q.max(1,keepdims=True)); q/=q.sum(1,keepdims=True)
        bias+=.5*np.log(target_count/(q.sum(0)+1e-12))
    return (z+bias).argmax(1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--instance",required=True); ap.add_argument("--seed",type=int,default=0); ap.add_argument("--graded",required=True); ap.add_argument("--out",required=True); a=ap.parse_args()
    model,mean,std=train(a.seed); base=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vp=np.load(os.path.join(base,"data","val_public.npz")); vl=get_logits(model,vp["x"],mean,std)
    raw=vl.argmax(1); vb=balanced_bias(vl,len(vl)/100)
    def metrics(p):
        yy=vp["y"]; return float((p==yy).mean()),float(np.mean([(p[yy==c]==c).mean() for c in range(100)]))
    mr,mb=metrics(raw),metrics(vb); print("public raw acc/bacc",mr,"balanced-prior",mb,flush=True)
    gl=get_logits(model,np.load(a.graded)["x"],mean,std); use_bal=mb[1]>mr[1]
    pred=balanced_bias(gl,len(gl)/100) if use_bal else gl.argmax(1)
    os.makedirs(os.path.dirname(a.out),exist_ok=True); np.save(a.out,pred.astype(np.int64))
    print(f"wrote {a.out}; balanced_prior={use_bal}; range={pred.min()}..{pred.max()}",flush=True)
if __name__=="__main__": main()
