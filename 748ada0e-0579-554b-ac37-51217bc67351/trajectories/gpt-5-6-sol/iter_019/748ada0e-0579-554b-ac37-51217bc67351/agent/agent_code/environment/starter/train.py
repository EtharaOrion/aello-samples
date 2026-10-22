"""Budget-honest CIFAR-100 trainer: exactly 8000 updates of 128 images."""
import argparse, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aello_data as data
STEPS,BATCH=8000,128
MEAN=(.5071,.4867,.4408); STD=(.2675,.2565,.2761)

class Block(nn.Module):
 def __init__(self,a,b,s):
  super().__init__(); self.b1=nn.BatchNorm2d(a); self.c1=nn.Conv2d(a,b,3,s,1,bias=False); self.b2=nn.BatchNorm2d(b); self.c2=nn.Conv2d(b,b,3,1,1,bias=False); self.sh=nn.Identity() if a==b and s==1 else nn.Conv2d(a,b,1,s,bias=False)
 def forward(self,x):
  z=F.relu(self.b1(x),inplace=True); sh=x if isinstance(self.sh,nn.Identity) else self.sh(z); z=self.c1(z); z=self.c2(F.dropout(F.relu(self.b2(z),inplace=True),.1,self.training)); return z+sh
class WRN(nn.Module):
 def __init__(self):
  super().__init__(); ch=[16,160,320,640]; self.c=nn.Conv2d(3,16,3,1,1,bias=False)
  def group(a,b,s): return nn.Sequential(Block(a,b,s),Block(b,b,1),Block(b,b,1),Block(b,b,1))
  self.g1=group(ch[0],ch[1],1); self.g2=group(ch[1],ch[2],2); self.g3=group(ch[2],ch[3],2); self.bn=nn.BatchNorm2d(640); self.fc=nn.Linear(640,100)
  for m in self.modules():
   if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
   elif isinstance(m,nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
   elif isinstance(m,nn.Linear): nn.init.zeros_(m.bias)
 def forward(self,x): return self.fc(F.adaptive_avg_pool2d(F.relu(self.bn(self.g3(self.g2(self.g1(self.c(x))))),inplace=True),1).flatten(1))

def norm(x):
 m=torch.tensor(MEAN,device=x.device)[None,:,None,None]; s=torch.tensor(STD,device=x.device)[None,:,None,None]; return (x-m)/s
def augment(x):
 b,c,h,w=x.shape; x=F.pad(x,(4,4,4,4),mode='reflect'); oy=torch.randint(0,9,(b,),device=x.device); ox=torch.randint(0,9,(b,),device=x.device); yy=oy[:,None]+torch.arange(h,device=x.device)[None]; xx=ox[:,None]+torch.arange(w,device=x.device)[None]; x=x[torch.arange(b,device=x.device)[:,None,None,None],torch.arange(c,device=x.device)[None,:,None,None],yy[:,None,:,None],xx[:,None,None,:]]
 flip=torch.rand(b,device=x.device)<.5; x[flip]=x[flip].flip(3); x=(x-.5)*torch.empty(b,1,1,1,device=x.device).uniform_(.8,1.2)+.5; x=x+torch.empty(b,1,1,1,device=x.device).uniform_(-.1,.1)
 cut=torch.rand(b,device=x.device)<.5; cy=torch.randint(0,32,(b,),device=x.device); cx=torch.randint(0,32,(b,),device=x.device); gy=torch.arange(32,device=x.device)[None,:,None]; gx=torch.arange(32,device=x.device)[None,None,:]; mask=((gy-cy[:,None,None]).abs()<8)&((gx-cx[:,None,None]).abs()<8)&cut[:,None,None]; return x.masked_fill(mask[:,None],.5).clamp_(0,1)
def lossfn(model,x,y):
 b=len(y); p=torch.randperm(b,device=x.device)
 if torch.rand((),device=x.device)<.5: lam=float(np.random.beta(.4,.4)); x=lam*x+(1-lam)*x[p]
 else:
  lam=float(np.random.beta(1.,1.)); r=math.sqrt(1-lam); rw=int(32*r); rh=int(32*r); cx=int(torch.randint(0,32,()).item()); cy=int(torch.randint(0,32,()).item()); x1=max(cx-rw//2,0); x2=min(cx+rw//2,32); y1=max(cy-rh//2,0); y2=min(cy+rh//2,32); x=x.clone(); x[:,:,y1:y2,x1:x2]=x[p,:,y1:y2,x1:x2]; lam=1-(x2-x1)*(y2-y1)/1024
 z=model(norm(x)); return lam*F.cross_entropy(z,y,label_smoothing=.05)+(1-lam)*F.cross_entropy(z,y[p],label_smoothing=.05)
def train(seed):
 torch.manual_seed(seed); np.random.seed(seed); x,y=data.load_train('i20k'); xt=torch.from_numpy(x).permute(0,3,1,2).contiguous().pin_memory(); yt=torch.from_numpy(y); bc=[np.flatnonzero(y==k) for k in range(100)]; rng=np.random.default_rng(seed)
 model=WRN().cuda().to(memory_format=torch.channels_last); opt=torch.optim.SGD(model.parameters(),lr=.2,momentum=.9,nesterov=True,weight_decay=5e-4); avg=None; navg=0; t=time.time()
 for step in range(STEPS):
  cls=rng.integers(0,100,BATCH); ids=np.array([rng.choice(bc[k]) for k in cls]); xb=xt[ids].cuda(non_blocking=True).float().div_(255).contiguous(memory_format=torch.channels_last); yb=yt[ids].cuda(non_blocking=True); xb=augment(xb); lr=.2*(step+1)/400 if step<400 else .1*(1+math.cos(math.pi*(step-400)/(STEPS-400)))
  for g in opt.param_groups:g['lr']=lr
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16): loss=lossfn(model,xb,yb)
  loss.backward(); opt.step()
  if step>=6500 and (step+1)%100==0:
   st=model.state_dict()
   if avg is None: avg={k:v.detach().clone() for k,v in st.items()}
   else:
    for k,v in st.items(): avg[k].lerp_(v,1/(navg+1)) if avg[k].is_floating_point() else avg[k].copy_(v)
   navg+=1
  if (step+1)%1000==0: print(f'step {step+1} loss {loss.item():.4f} lr {lr:.5f} sec {time.time()-t:.1f}',flush=True)
 raw=model
 if avg is not None: raw.load_state_dict(avg)
 return raw
@torch.inference_mode()
def logits(model,x,batch=500):
 model.eval(); out=[]
 for s in range(0,len(x),batch):
  z=torch.from_numpy(np.ascontiguousarray(x[s:s+batch])).permute(0,3,1,2).cuda().float().div_(255)
  with torch.autocast('cuda',dtype=torch.bfloat16): q=(model(norm(z))+model(norm(z.flip(3)))).float()/2
  out.append(q.cpu())
 return torch.cat(out).numpy()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--instance',required=True); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--graded',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); model=train(a.seed)
 xv,yv=data.load_val(); pv=logits(model,xv).argmax(1); rec=[(pv[yv==k]==k).mean() for k in range(100)]; print(f'public accuracy {(pv==yv).mean():.5f} balanced {np.mean(rec):.5f}',flush=True)
 pred=logits(model,np.load(a.graded)['x']).argmax(1).astype(np.int64); assert pred.shape==(10000,) and pred.min()>=0 and pred.max()<100; tmp=a.out+'.tmp.npy'; np.save(tmp,pred); os.replace(tmp,a.out); print('wrote',a.out,'range',pred.min(),pred.max(),'classes',len(np.unique(pred)),flush=True)
if __name__=='__main__': main()
