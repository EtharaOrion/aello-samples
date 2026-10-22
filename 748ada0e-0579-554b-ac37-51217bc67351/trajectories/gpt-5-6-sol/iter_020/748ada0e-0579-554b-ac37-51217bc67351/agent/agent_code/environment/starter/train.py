"""Budgeted CIFAR-100 trainer: WRN-28-10, MixUp, cosine, weight averaging."""
import argparse, os, sys, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aello_data as data
STEPS, BATCH = 8000, 128
MEAN=(.5071,.4867,.4408); STD=(.2675,.2565,.2761)

class Block(nn.Module):
 def __init__(self,a,b,stride):
  super().__init__(); self.bn1=nn.BatchNorm2d(a); self.c1=nn.Conv2d(a,b,3,stride,1,bias=False); self.bn2=nn.BatchNorm2d(b); self.c2=nn.Conv2d(b,b,3,1,1,bias=False); self.skip=nn.Identity() if a==b and stride==1 else nn.Conv2d(a,b,1,stride,bias=False)
 def forward(self,x):
  z=F.relu(self.bn1(x),inplace=True); s=x if isinstance(self.skip,nn.Identity) else self.skip(z); return self.c2(F.relu(self.bn2(self.c1(z)),inplace=True))+s
class WRN(nn.Module):
 def __init__(self,width=10,depth=28):
  super().__init__(); n=(depth-4)//6; ch=[16,16*width,32*width,64*width]; self.c=nn.Conv2d(3,16,3,1,1,bias=False); blocks=[]
  for g in range(3):
   for j in range(n): blocks.append(Block(ch[g] if j==0 else ch[g+1],ch[g+1],2 if g and j==0 else 1))
  self.blocks=nn.Sequential(*blocks); self.bn=nn.BatchNorm2d(ch[-1]); self.fc=nn.Linear(ch[-1],100)
  for m in self.modules():
   if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
   elif isinstance(m,nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
 def forward(self,x): return self.fc(F.adaptive_avg_pool2d(F.relu(self.bn(self.blocks(self.c(x))),inplace=True),1).flatten(1))

def augment(x,gen):
 b,c,h,w=x.shape; xp=F.pad(x,(4,4,4,4),mode='reflect'); oy=torch.randint(0,9,(b,),device=x.device,generator=gen); ox=torch.randint(0,9,(b,),device=x.device,generator=gen); yy=oy[:,None]+torch.arange(h,device=x.device)[None,:]; xx=ox[:,None]+torch.arange(w,device=x.device)[None,:]; bi=torch.arange(b,device=x.device)[:,None,None]
 x=xp[bi,:,yy[:,:,None],xx[:,None,:]].permute(0,3,1,2).contiguous(); flip=torch.rand(b,device=x.device,generator=gen)<.5; x[flip]=x[flip].flip(-1)
 br=1+(torch.rand(b,1,1,1,device=x.device,generator=gen)-.5)*.20; ct=1+(torch.rand(b,1,1,1,device=x.device,generator=gen)-.5)*.20; mu=x.mean((2,3),keepdim=True); return ((x-mu)*ct+mu)*br
def norm(x): return (x-x.new_tensor(MEAN)[None,:,None,None])/x.new_tensor(STD)[None,:,None,None]
@torch.no_grad()
def logits(model,arr,device,batch=500):
 model.eval(); ans0=[]; ans1=[]
 for s in range(0,len(arr),batch):
  z=torch.from_numpy(np.ascontiguousarray(arr[s:s+batch])).to(device).permute(0,3,1,2).float().div_(255); z=norm(z.contiguous(memory_format=torch.channels_last))
  with torch.autocast('cuda',dtype=torch.bfloat16): q0=model(z).float(); q1=model(z.flip(-1)).float()
  ans0.append(q0.cpu()); ans1.append(q1.cpu())
 return torch.cat(ans0).numpy(),torch.cat(ans1).numpy()
def train(seed):
 torch.manual_seed(seed); np.random.seed(seed); device='cuda'; x,y=data.load_train('i20k'); x=torch.from_numpy(x).to(device).permute(0,3,1,2).contiguous().float().div_(255); y=torch.from_numpy(y).to(device); model=WRN().to(device).to(memory_format=torch.channels_last); opt=torch.optim.SGD(model.parameters(),lr=.12,momentum=.9,nesterov=True,weight_decay=5e-4); gen=torch.Generator(device=device); gen.manual_seed(seed+123); avg={k:v.detach().clone() for k,v in model.state_dict().items()}; navg=0; start=time.time()
 for step in range(STEPS):
  frac=(step+1)/STEPS; lr=.12*min(1.,(step+1)/400)*(.005+.995*.5*(1+math.cos(math.pi*frac)))
  for pg in opt.param_groups: pg['lr']=lr
  idx=torch.randint(0,len(y),(BATCH,),device=device,generator=gen); xb=norm(augment(x[idx],gen).contiguous(memory_format=torch.channels_last)); yb=y[idx]; opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):
   coin=float(torch.rand((),device=device,generator=gen)); perm=torch.randperm(BATCH,device=device,generator=gen)
   if coin<.45:
    # CutMix retains CIFAR's sharp local structure better than pixelwise interpolation.
    lam=float(np.random.beta(1.,1.)); side=int(32*math.sqrt(1-lam)); cy=int(torch.randint(0,32,(),device=device,generator=gen)); cx=int(torch.randint(0,32,(),device=device,generator=gen)); y0=max(0,cy-side//2); y1=min(32,cy+side//2); x0=max(0,cx-side//2); x1=min(32,cx+side//2); mixed=xb.clone(); mixed[:,:,y0:y1,x0:x1]=xb[perm,:,y0:y1,x0:x1]; lam=1-(y1-y0)*(x1-x0)/1024; out=model(mixed); loss=lam*F.cross_entropy(out,yb,label_smoothing=.04)+(1-lam)*F.cross_entropy(out,yb[perm],label_smoothing=.04)
   elif coin<.70:
    lam=float(np.random.beta(.4,.4)); out=model(lam*xb+(1-lam)*xb[perm]); loss=lam*F.cross_entropy(out,yb,label_smoothing=.04)+(1-lam)*F.cross_entropy(out,yb[perm],label_smoothing=.04)
   else:
    # Cutout on unmixed examples.
    cy=int(torch.randint(0,32,(),device=device,generator=gen)); cx=int(torch.randint(0,32,(),device=device,generator=gen)); masked=xb.clone(); masked[:,:,max(0,cy-6):min(32,cy+6),max(0,cx-6):min(32,cx+6)]=0; loss=F.cross_entropy(model(masked),yb,label_smoothing=.05)
  loss.backward(); opt.step()
  if step>=6500 and step%25==0:
   navg+=1
   for k,v in model.state_dict().items(): avg[k].lerp_(v,1/navg) if avg[k].is_floating_point() else avg[k].copy_(v)
  if (step+1)%1000==0: print(f'step {step+1} loss {loss.item():.3f} lr {lr:.5f} sec {time.time()-start:.1f}',flush=True)
 final=WRN().to(device).to(memory_format=torch.channels_last); final.load_state_dict(model.state_dict()); model.load_state_dict(avg); return model,final

def balanced_pred(z,temp=None):
 if temp is None: return z.argmax(1)
 q=np.exp((z-z.max(1,keepdims=True))/temp)
 for _ in range(30): q/=q.sum(1,keepdims=True); q/=q.sum(0,keepdims=True)
 return q.argmax(1)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--instance',default='i20k'); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--graded',required=True); ap.add_argument('--out',required=True); ap.add_argument('--val',default='../data/val_public.npz'); a=ap.parse_args(); avg,final=train(a.seed); vd=np.load(a.val); vy=vd['y']; av0,av1=logits(avg,vd['x'],'cuda'); fi0,fi1=logits(final,vd['x'],'cuda'); best=None
 for wa in (.5,.75,1.):
  for wf in (.25,.5):
   z=wa*((1-wf)*av0+wf*av1)+(1-wa)*((1-wf)*fi0+wf*fi1)
   for temp in (None,.5,1.,2.):
    p=balanced_pred(z,temp); score=np.mean([(p[vy==c]==c).mean() for c in range(100)])
    if best is None or score>best[0]: best=(score,wa,wf,temp)
 print(f'public best balanced={best[0]:.5f} avg_weight={best[1]} flip_weight={best[2]} balance_temp={best[3]}',flush=True); _,wa,wf,temp=best; zv=wa*((1-wf)*av0+wf*av1)+(1-wa)*((1-wf)*fi0+wf*fi1)
 gd=np.load(a.graded)['x']; av0,av1=logits(avg,gd,'cuda'); fi0,fi1=logits(final,gd,'cuda'); z=wa*((1-wf)*av0+wf*av1)+(1-wa)*((1-wf)*fi0+wf*fi1); pred=balanced_pred(z,temp).astype(np.int64); os.makedirs(os.path.dirname(os.path.abspath(a.out)),exist_ok=True); np.save(a.out,pred); np.savez_compressed('/workspace/latest_logits.npz',val=zv.astype(np.float16),graded=z.astype(np.float16)); print(f'wrote {a.out} shape={pred.shape} range={pred.min()}..{pred.max()}',flush=True)
if __name__=='__main__': main()
