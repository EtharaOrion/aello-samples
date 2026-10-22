## Refinement Loop - Attempt 2
This is attempt number 2. 1 attempt(s) have been made before this one.

**BEST SO FAR: Iteration 1 (reward 0.4496)** — The agent audited the setup, ran a budget-constrained job, used public validation for refinements, and saved checked predictions.. Build on this result; do not discard it.

## Prior Attempts (bounded digest: top 3 scored + last 3)
### Iteration 1 | Reward: 0.4496
- Approach: The agent audited the setup, ran a budget-constrained job, used public validation for refinements, and saved checked predictions.
- Measured: reward 0.4496
- What happened: The agent’s stated workflow began with auditing the starter materials, hardware, and data layout. It then conducted a training run intended to comply with the available update budget, used the public validation split for measured refinements, and retained a prediction file for submission. The activity digest recorded 16 steps, 11 agent turns, and 9 tool calls, all using `exec`. The authoritative grader verdict was: “this submission earned reward 0.4496 (45% of the maximum achievable 1.00); it was not a valid scoreable success.” The following figures are unvalidated agent claims rather than grader-verified results. The agent claimed a public balanced accuracy of **0.61074** and public accuracy of **0.61000**. It also claimed that the run used exactly **8,000 updates × 128 samples** and took **86.1 seconds**. In its final report, it identified `/submission/preds_i20k.npy` as the submitted file and claimed that it had verified a shape of `(10000,)`, an `int64` data type, label values from `0` through `99`, and representation of all **100 classes**. No model architecture, learning rate, optimizer configuration, width, depth, or layer count was stated in the provided excerpt.

### Best solution so far — code from iteration 1 (reward 0.4496)
This is YOUR OWN prior best source. Build on it (write your chosen code to /workspace); do not restart from scratch unless you have a reason. It will be re-trained and re-graded from scratch.
```python
# ===== agent_src/environment/starter/train.py =====
"""Budget-compliant CIFAR-100 trainer: exactly 8000 updates x batch 128."""
import argparse, copy, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aello_data as data
BUDGET_STEPS=8000; BATCH=128
MEAN=(.5071,.4867,.4408); STD=(.2675,.2565,.2761)

class Block(nn.Module):
 def __init__(self,ni,no,stride):
  super().__init__(); self.b1=nn.BatchNorm2d(ni); self.c1=nn.Conv2d(ni,no,3,stride,1,bias=False); self.b2=nn.BatchNorm2d(no); self.c2=nn.Conv2d(no,no,3,1,1,bias=False); self.skip=nn.Identity() if ni==no and stride==1 else nn.Conv2d(ni,no,1,stride,bias=False)
 def forward(self,x):
  z=F.relu(self.b1(x),inplace=True); s=self.skip(z if not isinstance(self.skip,nn.Identity) else x); z=self.c1(z); z=self.c2(F.dropout(F.relu(self.b2(z),inplace=True),.1,self.training)); return z+s
class WRN(nn.Module):
 def __init__(self,w=6):
  super().__init__(); ch=[16,16*w,32*w,64*w]; self.stem=nn.Conv2d(3,16,3,1,1,bias=False); gs=[]
  for g in range(3): gs.append(nn.Sequential(Block(ch[g],ch[g+1],1 if g==0 else 2),*[Block(ch[g+1],ch[g+1],1) for _ in range(3)]))
  self.gs=nn.Sequential(*gs); self.bn=nn.BatchNorm2d(ch[-1]); self.fc=nn.Linear(ch[-1],100)
  for m in self.modules():
   if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out')
 def forward(self,x): return self.fc(F.adaptive_avg_pool2d(F.relu(self.bn(self.gs(self.stem(x))),inplace=True),1).flatten(1))

def norm(x): return (x-x.new_tensor(MEAN)[None,:,None,None])/x.new_tensor(STD)[None,:,None,None]
def aug(x):
 b,c,h,w=x.shape; xp=F.pad(x,(4,)*4,mode='reflect'); oy,ox=torch.randint(0,9,(2,b),device=x.device); yy=oy[:,None]+torch.arange(h,device=x.device)[None]; xx=ox[:,None]+torch.arange(w,device=x.device)[None]
 x=xp[torch.arange(b,device=x.device)[:,None,None,None],torch.arange(c,device=x.device)[None,:,None,None],yy[:,None,:,None],xx[:,None,None,:]]
 x=torch.where((torch.rand(b,1,1,1,device=x.device)<.5),x.flip(3),x)
 br=torch.empty(b,1,1,1,device=x.device).uniform_(.8,1.2); co=torch.empty(b,1,1,1,device=x.device).uniform_(.8,1.2); sa=torch.empty(b,1,1,1,device=x.device).uniform_(.8,1.2)
 x=x*br; mu=x.mean((2,3),keepdim=True); x=(x-mu)*co+mu; gray=x.mean(1,keepdim=True); x=(x-gray)*sa+gray
 use=torch.rand(b,device=x.device)<.5; cy=torch.randint(0,25,(b,),device=x.device); cx=torch.randint(0,25,(b,),device=x.device); gy=torch.arange(32,device=x.device)[None,:,None]; gx=torch.arange(32,device=x.device)[None,None,:]; hole=(gy>=cy[:,None,None])&(gy<cy[:,None,None]+8)&(gx>=cx[:,None,None])&(gx<cx[:,None,None]+8)
 return x.clamp(0,1)*((~(hole&use[:,None,None]))[:,None])
def mix(x,y,step):
 b=len(y); p=torch.randperm(b,device=x.device)
 if step%2==0:
  l=torch.distributions.Beta(.2,.2).sample((b,)).to(x.device).clamp(.05,.95); return x*l[:,None,None,None]+x[p]*(1-l[:,None,None,None]),y,y[p],l
 l0=torch.rand(b,device=x.device); side=(32*torch.sqrt(1-l0)).long().clamp(1,32); cx=torch.randint(0,32,(b,),device=x.device); cy=torch.randint(0,32,(b,),device=x.device); x1=(cx-side//2).clamp(0,32); x2=(cx+side//2).clamp(0,32); y1=(cy-side//2).clamp(0,32); y2=(cy+side//2).clamp(0,32); gy=torch.arange(32,device=x.device)[None,:,None]; gx=torch.arange(32,device=x.device)[None,None,:]; m=(gy>=y1[:,None,None])&(gy<y2[:,None,None])&(gx>=x1[:,None,None])&(gx<x2[:,None,None]); return torch.where(m[:,None],x[p],x),y,y[p],1-m.float().mean((1,2))
def lossfn(z,a,b,l): return (l*F.cross_entropy(z,a,label_smoothing=.05,reduction='none')+(1-l)*F.cross_entropy(z,b,label_smoothing=.05,reduction='none')).mean()
@torch.no_grad()
def logits(model,x,dev,batch=512):
 model.eval(); out=[]
 for s in range(0,len(x),batch):
  z=norm(torch.from_numpy(np.ascontiguousarray(x[s:s+batch])).permute(0,3,1,2).to(dev).float().div_(255))
  with torch.autocast('cuda',dtype=torch.bfloat16): p=(model(z).float()+model(z.flip(3)).float())/2
  out.append(p.cpu())
 return torch.cat(out).numpy()
def train(instance,seed):
 x,y=data.load_train(instance); torch.manual_seed(seed); rng=np.random.default_rng(seed); model=WRN().cuda().to(memory_format=torch.channels_last); ema=copy.deepcopy(model).eval()
 for p in ema.parameters(): p.requires_grad_(False)
 opt=torch.optim.SGD(model.parameters(),lr=.2,momentum=.9,nesterov=True,weight_decay=5e-4); st=time.time()
 for step in range(BUDGET_STEPS):
  ids=rng.integers(0,len(y),BATCH); xb=torch.from_numpy(np.ascontiguousarray(x[ids])).permute(0,3,1,2).cuda().float().div_(255); ya=torch.from_numpy(y[ids]).cuda(); xb,ya,yb,l=mix(aug(xb),ya,step); xb=norm(xb).contiguous(memory_format=torch.channels_last)
  lr=.2*(step+1)/400 if step<400 else .1*(1+math.cos(math.pi*(step-400)/(BUDGET_STEPS-400)))
  for pg in opt.param_groups: pg['lr']=lr
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16): loss=lossfn(model(xb),ya,yb,l)
  loss.backward(); opt.step()
  with torch.no_grad():
   for pe,p in zip(ema.parameters(),model.parameters()): pe.lerp_(p,.001)
   for be,b in zip(ema.buffers(),model.buffers()): be.copy_(b)
  if (step+1)%1000==0: print(f'step {step+1} loss {loss.item():.3f} lr {lr:.4f} elapsed {time.time()-st:.1f}s',flush=True)
 return ema
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--instance',required=True,choices=list(data.INSTANCES)); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--graded',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); model=train(a.instance,a.seed)
 v=np.load('../data/val_public.npz'); pv=logits(model,v['x'],'cuda').argmax(1); recalls=[(pv[v['y']==c]==c).mean() for c in range(100)]; print(f'public accuracy {(pv==v["y"]).mean():.5f} balanced {np.mean(recalls):.5f}')
 pred=logits(model,np.load(a.graded)['x'],'cuda').argmax(1).astype(np.int64); os.makedirs(os.path.dirname(os.path.abspath(a.out)),exist_ok=True); np.save(a.out,pred); print(f'wrote {a.out}: {pred.shape} {pred.dtype} [{pred.min()},{pred.max()}]')
if __name__=='__main__': main()

# ===== agent_src/environment/aello_data.py =====
"""Shipped data access. Read-only helper; the graded pipeline is starter/train.py."""
import os
import numpy as np

DATA = os.path.dirname(os.path.abspath(__file__))
INSTANCES = ("i20k",)
SIZES = {"i20k": 20000}
N_CLASSES = 100


def load_train(instance):
    """Training images and labels."""
    if instance not in INSTANCES:
        raise ValueError("unknown instance %r; expected one of %s" % (instance, list(INSTANCES)))
    d = np.load(os.path.join(DATA, "data", "train_%s.npz" % instance))
    return d["x"], d["y"].astype(np.int64)


def load_val():
    """Public validation split, disjoint from the training set.

    Use it to measure locally. It is NOT the graded split, and instruction.md rule 4
    forbids training on it.
    """
    d = np.load(os.path.join(DATA, "data", "val_public.npz"))
    return d["x"], d["y"].astype(np.int64)


def load_graded_images():
    """The graded images, labels withheld. Predict on these and write
    /submission/preds_i20k.npy (int array, shape (10000,), values 0..99)."""
    return np.load(os.path.join(DATA, "data", "graded_images.npz"))["x"]
```