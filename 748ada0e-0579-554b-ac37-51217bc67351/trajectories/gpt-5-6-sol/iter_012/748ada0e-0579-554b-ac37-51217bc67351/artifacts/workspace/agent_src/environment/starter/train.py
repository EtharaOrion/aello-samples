"""Budget-faithful CIFAR-100 trainer: 8000 updates, batch 128."""
import argparse, copy, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.augmentation.auto import RandAugment
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aello_data as data
STEPS, BATCH = 8000, 128
MEAN=(.5071,.4867,.4408); STD=(.2675,.2565,.2761)

class Block(nn.Module):
 def __init__(self,a,b,s):
  super().__init__(); self.b1=nn.BatchNorm2d(a); self.c1=nn.Conv2d(a,b,3,s,1,bias=False); self.b2=nn.BatchNorm2d(b); self.c2=nn.Conv2d(b,b,3,1,1,bias=False); self.short=nn.Conv2d(a,b,1,s,bias=False) if a!=b else None
 def forward(self,x):
  z=F.relu(self.b1(x),inplace=True); skip=self.short(z) if self.short is not None else x
  return self.c2(F.relu(self.b2(self.c1(z)),inplace=True))+skip
class WRN(nn.Module):
 def __init__(self,depth=40,widen=10):
  super().__init__(); n=(depth-4)//6; ws=[16,16*widen,32*widen,64*widen]; self.c=nn.Conv2d(3,16,3,1,1,bias=False); q=[]
  for g in range(3):
   for j in range(n): q.append(Block(ws[g] if j==0 else ws[g+1],ws[g+1],2 if g and j==0 else 1))
  self.body=nn.Sequential(*q); self.bn=nn.BatchNorm2d(ws[-1]); self.fc=nn.Linear(ws[-1],100)
  for m in self.modules():
   if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
   elif isinstance(m,nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
 def forward(self,x): return self.fc(F.adaptive_avg_pool2d(F.relu(self.bn(self.body(self.c(x))),inplace=True),1).flatten(1))

def norm(x):
 m=torch.tensor(MEAN,device=x.device)[None,:,None,None]; s=torch.tensor(STD,device=x.device)[None,:,None,None]; return (x-m)/s
def augment(x):
 b,c,h,w=x.shape; z=F.pad(x,(4,4,4,4),mode='reflect'); oy=torch.randint(0,9,(b,),device=x.device); ox=torch.randint(0,9,(b,),device=x.device)
 yy=oy[:,None,None]+torch.arange(h,device=x.device)[None,:,None]; xx=ox[:,None,None]+torch.arange(w,device=x.device)[None,None,:]
 idx=(yy*40+xx).reshape(b,1,-1).expand(-1,c,-1); z=z.reshape(b,c,-1).gather(2,idx).reshape(b,c,h,w)
 flip=torch.rand(b,device=x.device)<.5; z[flip]=z[flip].flip(-1)
 return z
@torch.no_grad()
def update(avg,model,d=.999):
 torch._foreach_lerp_(list(avg.parameters()),list(model.parameters()),1-d)
 for a,b in zip(avg.buffers(),model.buffers()): a.copy_(b)
def train(seed):
 x,y=data.load_train('i20k'); rng=np.random.default_rng(seed); torch.manual_seed(seed); model=WRN().cuda().to(memory_format=torch.channels_last); avg=copy.deepcopy(model).eval(); [p.requires_grad_(False) for p in avg.parameters()]
 opt=torch.optim.SGD(model.parameters(),lr=.12,momentum=.9,nesterov=True,weight_decay=5e-4); sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.12,total_steps=STEPS,pct_start=.1,div_factor=12,final_div_factor=1000); strong=RandAugment(2,14).cuda()
 order=rng.permutation(len(y)); pos=0; start=time.time()
 for step in range(STEPS):
  if pos+BATCH>len(y): order=rng.permutation(len(y)); pos=0
  ids=np.sort(order[pos:pos+BATCH]); pos+=BATCH; xb=torch.from_numpy(np.ascontiguousarray(x[ids])).cuda().permute(0,3,1,2).to(memory_format=torch.channels_last).float().div_(255); yb=torch.from_numpy(y[ids]).cuda(); xb=norm(strong(augment(xb)))
  lam=float(rng.beta(.2,.2)); perm=torch.randperm(BATCH,device='cuda'); xb=xb.mul(lam).add_(xb[perm],alpha=1-lam); opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):
   z=model(xb); loss=lam*F.cross_entropy(z,yb,label_smoothing=.05)+(1-lam)*F.cross_entropy(z,yb[perm],label_smoothing=.05)
  loss.backward(); opt.step(); sch.step()
  if step==5999: avg.load_state_dict(model.state_dict())
  elif step>5999: update(avg,model,.995)
  if (step+1)%500==0: print(f'{step+1}/{STEPS} loss {loss.item():.3f} lr {sch.get_last_lr()[0]:.5f} sec {time.time()-start:.1f}',flush=True)
 return model.eval(),avg
@torch.no_grad()
def predict(model,a,batch=500):
 model.eval(); out=[]
 for i in range(0,len(a),batch):
  x=torch.from_numpy(np.ascontiguousarray(a[i:i+batch])).cuda().permute(0,3,1,2).to(memory_format=torch.channels_last).float().div_(255); x=norm(x)
  with torch.autocast('cuda',dtype=torch.bfloat16): z=(model(x).float()+model(x.flip(-1)).float())/2
  out.append(z.cpu())
 return torch.cat(out).numpy().argmax(1)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--instance',default='i20k'); p.add_argument('--seed',type=int,default=0); p.add_argument('--graded',required=True); p.add_argument('--out',required=True); a=p.parse_args(); online,avg=train(a.seed)
 v=np.load('../data/val_public.npz'); scored=[]
 for name,m in [('online',online),('ema',avg)]:
  q=predict(m,v['x']); rec=[(q[v['y']==c]==c).mean() for c in range(100)]; score=float(np.mean(rec)); scored.append((score,m)); print(f'{name} public accuracy {(q==v["y"]).mean():.5f} balanced {score:.5f}',flush=True)
 model=max(scored,key=lambda z:z[0])[1]
 torch.save(model.state_dict(),'/workspace/last_model.pt')
 g=np.load(a.graded)['x']; q=predict(model,g).astype(np.int64); np.save(a.out,q); print('wrote',a.out,q.shape,q.min(),q.max(),flush=True)
if __name__=='__main__': main()
