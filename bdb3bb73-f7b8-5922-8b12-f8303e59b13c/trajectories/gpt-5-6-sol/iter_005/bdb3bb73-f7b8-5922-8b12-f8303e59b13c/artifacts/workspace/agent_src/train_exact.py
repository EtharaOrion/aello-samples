import json, math, os, time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.transforms import PILToTensor

torch.backends.cudnn.benchmark = True
ROOT=Path('/workspace'); OUT=Path(os.environ.get('OUTPUT_DIR',ROOT/'periodic_checkpoints'))
MEAN=torch.tensor((.4914,.4822,.4465),device='cuda').half().view(1,3,1,1)
STD=torch.tensor((.2470,.2435,.2616),device='cuda').half().view(1,3,1,1)

def load_train():
 d=ImageFolder(ROOT/'data/cifar10/train',transform=PILToTensor()); assert len(d)==50000
 x=torch.empty(len(d),3,32,32,dtype=torch.uint8); y=torch.empty(len(d),dtype=torch.long)
 for i,(a,b) in enumerate(d): x[i]=a; y[i]=b
 x=x.cuda().half().div_(255).to(memory_format=torch.channels_last); y=y.cuda()
 return x,y

class BN(nn.BatchNorm2d):
 def __init__(self,n):
  super().__init__(n,eps=1e-12,momentum=.4); self.weight.requires_grad=False
class Conv(nn.Conv2d):
 def __init__(self,a,b): super().__init__(a,b,3,padding='same',bias=False)
 def reset_parameters(self):
  super().reset_parameters(); nn.init.dirac_(self.weight.data[:self.weight.size(1)])
class Group(nn.Module):
 def __init__(self,a,b):
  super().__init__(); self.c1=Conv(a,b); self.n1=BN(b); self.c2=Conv(b,b); self.n2=BN(b)
 def forward(self,x):
  x=F.gelu(self.n1(F.max_pool2d(self.c1(x),2))); return F.gelu(self.n2(self.c2(x)))
class Net(nn.Module):
 def __init__(self):
  super().__init__(); self.whiten=nn.Conv2d(3,24,2,bias=True); self.whiten.weight.requires_grad=False
  self.g1=Group(24,64); self.g2=Group(64,256); self.g3=Group(256,256); self.head=nn.Linear(256,10,bias=False)
  for m in self.modules():
   if isinstance(m,BN): m.float()
   else: m.half()
 def reset(self):
  for m in self.modules():
   if type(m) in (nn.Conv2d,Conv,BN,nn.Linear): m.reset_parameters()
  self.head.weight.data.mul_(1/self.head.weight.data.std())
 def init_whiten(self,x):
  p=x[:5000].unfold(2,2,1).unfold(3,2,1).transpose(1,3).reshape(-1,3,2,2).float(); q=p.reshape(len(p),-1)
  cov=q.T@q/len(q); e,v=torch.linalg.eigh(cov,UPLO='U'); w=v.T.reshape(-1,3,2,2)/torch.sqrt(e.view(-1,1,1,1)+5e-4)
  self.whiten.weight.data.copy_(torch.cat((w,-w)).half())
 def forward(self,x):
  x=F.gelu(self.whiten(x)); x=self.g1(x); x=self.g2(x); x=self.g3(x); x=F.max_pool2d(x,3).view(len(x),-1)
  return self.head(x)/x.size(-1)
class Export(nn.Module):
 def __init__(self,n): super().__init__(); self.net=n
 def forward(self,x):
  x=x.half(); a=(self.net(x)+self.net(x.flip(3)))*.5
  p=F.pad(x,(1,1,1,1),'reflect'); u=p[:,:,:32,:32]; d=p[:,:,2:34,2:34]
  b=(self.net(u)+self.net(u.flip(3))+self.net(d)+self.net(d.flip(3)))*.25
  return (a+b)*.5

@torch.compile
def zero(g):
 X=g.bfloat16(); X/=X.norm()+1e-7
 if g.size(0)>g.size(1): X=X.T
 for _ in range(3):
  A=X@X.T; X=3.4445*X+(-4.775*A+2.0315*A@A)@X
 if g.size(0)>g.size(1): X=X.T
 return X
class Muon(torch.optim.Optimizer):
 def __init__(self,p): super().__init__(p,dict(lr=.24,momentum=.6,nesterov=True))
 @torch.no_grad()
 def step(self):
  for group in self.param_groups:
   for p in group['params']:
    if p.grad is None: continue
    g=p.grad; s=self.state[p]; buf=s.setdefault('momentum_buffer',torch.zeros_like(g)); buf.mul_(.6).add_(g); g=g.add(buf,alpha=.6)
    p.mul_(len(p)**.5/p.norm()); p.add_(zero(g.reshape(len(g),-1)).view_as(g),alpha=-group['lr'])

def crop(x):
 shifts=torch.randint(-2,3,(len(x),2),device='cuda'); out=torch.empty(len(x),3,32,32,device='cuda',dtype=x.dtype,memory_format=torch.channels_last)
 for sy in range(-2,3):
  for sx in range(-2,3):
   m=(shifts[:,0]==sy)&(shifts[:,1]==sx); out[m]=x[m,:,2+sy:34+sy,2+sx:34+sx]
 return out
def save(n,e):
 n.eval(); torch.jit.script(Export(n)).save(str(OUT/f'ckpt_epoch_{e:04d}.pt')); n.train()

def main():
 torch.manual_seed(int(os.environ.get('SEED','42'))); x,y=load_train(); x=(x-MEAN)/STD
 flip=torch.rand(len(x),device='cuda')<.5; x[flip]=x[flip].flip(3); padded=F.pad(x,(2,2,2,2),'reflect')
 n=Net().cuda().to(memory_format=torch.channels_last); n.reset(); n.init_whiten(x); compiled=torch.compile(n,mode='max-autotune')
 filt=[p for p in n.parameters() if p.requires_grad and p.ndim==4]; nb=[p for k,p in n.named_parameters() if 'n1' in k or 'n2' in k if p.requires_grad]
 wd=.004
 sgd=torch.optim.SGD([dict(params=[n.whiten.bias],lr=.053,weight_decay=wd/.053),dict(params=nb,lr=.053,weight_decay=wd/.053),dict(params=[n.head.weight],lr=.67,weight_decay=wd/.67)],momentum=.85,nesterov=True,fused=True)
 mu=Muon(filt); OUT.mkdir(parents=True,exist_ok=True)
 for f in OUT.glob('ckpt_epoch_*.pt'): f.unlink()
 step=0; start=time.time(); bs=2000
 for ep in range(1,9):
  images=crop(padded); images=images.flip(3) if ep%2==0 else images; order=torch.randperm(len(y),device='cuda')
  n.train(); acc=0
  for j in range(0,len(y),bs):
   idx=order[j:j+bs]; logits=compiled(images[idx]); loss=F.cross_entropy(logits,y[idx],label_smoothing=.2,reduction='sum'); loss.backward()
   sgd.param_groups[0]['lr']=.053*max(0,1-step/75)
   for g in sgd.param_groups[1:]: g['lr']=g.get('initial_lr',g['lr'])*(1-step/200) if step==0 else g['lr']
   # Set exact linear schedules directly.
   sgd.param_groups[1]['lr']=.053*(1-step/200); sgd.param_groups[2]['lr']=.67*(1-step/200); mu.param_groups[0]['lr']=.24*(1-step/200)
   sgd.step(); mu.step(); n.zero_grad(set_to_none=True); step+=1; acc+=(logits.argmax(1)==y[idx]).sum().item()
  torch.cuda.synchronize(); save(n,ep); print(json.dumps({'epoch':ep,'train_acc':acc/50000,'elapsed':time.time()-start}),flush=True)
 if OUT.resolve()==(ROOT/'periodic_checkpoints').resolve():
  dst=ROOT/'checkpoint.pt'; dst.unlink(missing_ok=True); os.link(OUT/'ckpt_epoch_0008.pt',dst)

if __name__=='__main__': main()
