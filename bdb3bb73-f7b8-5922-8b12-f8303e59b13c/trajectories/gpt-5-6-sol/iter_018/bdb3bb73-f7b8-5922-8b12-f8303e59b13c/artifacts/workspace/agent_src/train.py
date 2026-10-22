import os, time, json, shutil
from math import ceil
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
ROOT = "/workspace"
MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device="cuda").view(1,3,1,1)
STD = torch.tensor((0.2470, 0.2435, 0.2616), device="cuda").view(1,3,1,1)

def load_train():
    base = os.path.join(ROOT, "data/cifar10/train")
    classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    ims, ys = [], []
    for y, cls in enumerate(classes):
        files = sorted(os.path.join(base, cls, f) for f in os.listdir(os.path.join(base, cls)) if f.endswith(".png"))
        ims.extend(np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8) for f in files)
        ys.extend([y]*len(files))
    x = torch.from_numpy(np.stack(ims)).cuda().permute(0,3,1,2).half().div_(255).contiguous(memory_format=torch.channels_last)
    x = (x - MEAN.half()) / STD.half()
    return x, torch.tensor(ys, device="cuda")

def augment(x, epoch):
    # One preselected mirror assignment, reversed on alternate epochs, plus random 2px translation.
    if epoch == 0:
        augment.flips = torch.rand(len(x), device=x.device) < .5
    z = torch.where(augment.flips[:,None,None,None], x.flip(-1), x)
    if epoch & 1: z = z.flip(-1)
    z = F.pad(z, (2,)*4, mode="reflect")
    shifts = torch.randint(0, 5, (len(x),2), device=x.device)
    out = torch.empty_like(x)
    for sy in range(5):
        for sx in range(5):
            m = (shifts[:,0] == sy) & (shifts[:,1] == sx)
            out[m] = z[m,:,sy:sy+32,sx:sx+32]
    return out

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, n):
        super().__init__(n, eps=1e-12, momentum=.4)
        self.weight.requires_grad = False

class Conv(nn.Conv2d):
    def __init__(self, ci, co): super().__init__(ci, co, 3, padding=1, bias=False)
    def reset_parameters(self):
        super().reset_parameters(); nn.init.dirac_(self.weight.data[:self.weight.size(1)])

class ConvGroup(nn.Module):
    def __init__(self, ci, co):
        super().__init__(); self.conv1=Conv(ci,co); self.pool=nn.MaxPool2d(2); self.norm1=BatchNorm(co)
        self.conv2=Conv(co,co); self.norm2=BatchNorm(co); self.activ=nn.GELU()
    def forward(self,x):
        x=self.activ(self.norm1(self.pool(self.conv1(x))))
        return self.activ(self.norm2(self.conv2(x)))

class CifarNet(nn.Module):
    def __init__(self):
        super().__init__(); self.whiten=nn.Conv2d(3,24,2,bias=True); self.whiten.weight.requires_grad=False
        self.layers=nn.Sequential(nn.GELU(),ConvGroup(24,64),ConvGroup(64,256),ConvGroup(256,256),nn.MaxPool2d(3))
        self.head=nn.Linear(256,10,bias=False)
        for m in self.modules():
            if isinstance(m,BatchNorm): m.float()
            else: m.half()
    def reset(self):
        for m in self.modules():
            if type(m) in (nn.Conv2d,Conv,BatchNorm,nn.Linear): m.reset_parameters()
        self.head.weight.data.mul_(1/self.head.weight.data.std())
    def init_whiten(self, images):
        p=images[:5000].unfold(2,2,1).unfold(3,2,1).transpose(1,3).reshape(-1,3,2,2).float()
        pf=p.view(len(p),-1); cov=pf.T@pf/len(pf); vals,vecs=torch.linalg.eigh(cov,UPLO="U")
        w=vecs.T.reshape(-1,3,2,2)/torch.sqrt(vals[:,None,None,None]+5e-4)
        self.whiten.weight.data.copy_(torch.cat((w,-w)).half())
    def forward(self, x: torch.Tensor, whiten_bias_grad: bool=True):
        x=x.half()
        b=self.whiten.bias
        assert b is not None
        b=b if whiten_bias_grad else b.detach()
        x=F.conv2d(x,self.whiten.weight,b)
        x=self.layers(x); x=x.view(x.size(0),-1)
        return self.head(x)/x.size(-1)

class InferenceNet(nn.Module):
    def __init__(self, core):
        super().__init__(); self.core=core
    def mirror(self, x: torch.Tensor):
        return .5*self.core(x)+.5*self.core(x.flip(-1))
    def forward(self, x: torch.Tensor):
        # Six deterministic views: mirror, then the two diagonal one-pixel translations.
        logits=self.mirror(x)
        p=F.pad(x,(1,1,1,1),mode="reflect")
        translated=.5*self.mirror(p[:,:,0:32,0:32])+.5*self.mirror(p[:,:,2:34,2:34])
        return .5*logits+.5*translated

@torch.compile
def zeropower(g, steps: int=3):
    a,b,c=3.4445,-4.7750,2.0315; x=g.bfloat16(); x=x/(x.norm()+1e-7)
    tr=x.size(0)>x.size(1)
    if tr: x=x.T
    for _ in range(steps):
        A=x@x.T; x=a*x+(b*A+c*A@A)@x
    return x.T if tr else x

class Muon(torch.optim.Optimizer):
    def __init__(self,p,lr=.24,momentum=.6): super().__init__(p,dict(lr=lr,momentum=momentum))
    @torch.no_grad()
    def step(self):
        for gr in self.param_groups:
            for p in gr['params']:
                if p.grad is None: continue
                s=self.state[p]; buf=s.setdefault('momentum_buffer',torch.zeros_like(p.grad))
                buf.mul_(gr['momentum']).add_(p.grad); g=p.grad.add(buf,alpha=gr['momentum'])
                p.mul_(len(p)**.5/p.norm()); u=zeropower(g.reshape(len(g),-1)).view_as(g)
                p.add_(u,alpha=-gr['lr'])

def make_opts(model, bs=2000):
    wd=2e-6*bs; bias_lr=.053; head_lr=.67
    conv=[p for p in model.parameters() if p.ndim==4 and p.requires_grad]
    norms=[p for n,p in model.named_parameters() if 'norm' in n and p.requires_grad]
    sgd=torch.optim.SGD([dict(params=[model.whiten.bias],lr=bias_lr,weight_decay=wd/bias_lr),dict(params=norms,lr=bias_lr,weight_decay=wd/bias_lr),dict(params=[model.head.weight],lr=head_lr,weight_decay=wd/head_lr)],momentum=.85,nesterov=True,fused=True)
    mu=Muon(conv)
    for o in (sgd,mu):
        for g in o.param_groups:g['initial_lr']=g['lr']
    return sgd,mu

def train_run(x,y,outdir,epochs=8,warm=False):
    bs=2000; steps_epoch=len(x)//bs; total=epochs*steps_epoch; bias_steps=3*steps_epoch
    base=CifarNet().cuda().to(memory_format=torch.channels_last); base.reset(); base.init_whiten(x)
    net=torch.compile(base,mode="max-autotune"); sgd,mu=make_opts(base)
    os.makedirs(outdir,exist_ok=True); start=time.time(); step=0
    for ep in range(epochs):
        base.train(); ax=augment(x,ep); order=torch.randperm(len(x),device="cuda")
        for j in range(steps_epoch):
            ix=order[j*bs:(j+1)*bs]; pred=net(ax[ix],step<bias_steps)
            F.cross_entropy(pred,y[ix],label_smoothing=.2,reduction='sum').backward()
            for g in sgd.param_groups[:1]:g['lr']=g['initial_lr']*(1-step/bias_steps)
            for g in sgd.param_groups[1:]+mu.param_groups:g['lr']=g['initial_lr']*(1-step/total)
            sgd.step(); mu.step(); base.zero_grad(set_to_none=True); step+=1
        torch.cuda.synchronize(); base.eval()
        scripted=torch.jit.script(InferenceNet(base))
        scripted.save(os.path.join(outdir,f"ckpt_epoch_{ep+1:04d}.pt"))
        print(f"epoch {ep+1} elapsed {time.time()-start:.3f}",flush=True)
    return time.time()-start

if __name__=='__main__':
    x,y=load_train(); out=os.path.join(ROOT,'periodic_checkpoints')
    for f in os.listdir(out): os.remove(os.path.join(out,f))
    elapsed=train_run(x,y,out,10)
    shutil.copy(os.path.join(out,'ckpt_epoch_0010.pt'),os.path.join(ROOT,'checkpoint.pt'))
    with open(os.path.join(ROOT,'agent_reported_metrics.json'),'w') as f: json.dump({'accuracy':0.0,'wall_clock_sec':elapsed},f)
