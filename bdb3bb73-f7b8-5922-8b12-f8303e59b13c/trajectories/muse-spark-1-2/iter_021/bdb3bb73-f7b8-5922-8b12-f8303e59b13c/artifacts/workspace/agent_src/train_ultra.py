import os, time, random, shutil
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

EPOCHS=18
BATCH=4096
LR=3.0
WD=5e-4
MEAN=(0.4914,0.4822,0.4465); STD=(0.2470,0.2435,0.2616)
CKPT="/workspace/scratch_ckpt"
os.makedirs(CKPT,exist_ok=True)
for f in os.listdir(CKPT): os.remove(os.path.join(CKPT,f))
torch.backends.cudnn.benchmark=True
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True

class Block(nn.Module):
    def __init__(self,i,o,s):
        super().__init__()
        self.bn1=nn.BatchNorm2d(i); self.r1=nn.ReLU(inplace=True)
        self.c1=nn.Conv2d(i,o,3,s,1,bias=False)
        self.bn2=nn.BatchNorm2d(o); self.r2=nn.ReLU(inplace=True)
        self.c2=nn.Conv2d(o,o,3,1,1,bias=False)
        self.sc=nn.Conv2d(i,o,1,s,bias=False) if i!=o or s!=1 else nn.Identity()
    def forward(self,x):
        out=self.r1(self.bn1(x)); sc=self.sc(x); out=self.c1(out); out=self.c2(self.r2(self.bn2(out))); return out+sc
class WRN(nn.Module):
    def __init__(self,d=28,w=10,nc=10):
        super().__init__()
        n=(d-4)//6
        self.ic=nn.Conv2d(3,16,3,1,1,bias=False)
        c1,c2,c3=16*w,32*w,64*w
        def mk(i,o,n,s): return nn.Sequential(*([Block(i,o,s)]+[Block(o,o,1) for _ in range(n-1)]))
        self.l1=mk(16,c1,n,1); self.l2=mk(c1,c2,n,2); self.l3=mk(c2,c3,n,2)
        self.bn=nn.BatchNorm2d(c3); self.r=nn.ReLU(inplace=True); self.fc=nn.Linear(c3,nc)
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear): nn.init.zeros_(m.bias)
    def forward(self,x):
        o=self.ic(x); o=self.l1(o); o=self.l2(o); o=self.l3(o); o=self.r(self.bn(o)); o=F.adaptive_avg_pool2d(o,1).flatten(1); return self.fc(o)

print("Caching...")
train_ds_pil = ImageFolder("/workspace/data/cifar10/train")
N=len(train_ds_pil)
train_imgs = torch.empty(N,3,32,32,dtype=torch.uint8)
train_lbls = torch.empty(N,dtype=torch.long)
for i,(img,lbl) in enumerate(train_ds_pil):
    train_imgs[i]=torch.from_numpy(np.array(img)).permute(2,0,1)
    train_lbls[i]=lbl
train_imgs_gpu=train_imgs.cuda()
train_lbls_gpu=train_lbls.cuda()
print(f"cached {N}")

test_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
test_ds=ImageFolder("/workspace/data/cifar10/test",transform=test_tf)
test_loader=DataLoader(test_ds,batch_size=1024,shuffle=False,num_workers=2,pin_memory=True,persistent_workers=True)

device=torch.device("cuda")
model=WRN().to(device).to(memory_format=torch.channels_last)
# no compile for speed
crit=nn.CrossEntropyLoss(label_smoothing=0.1)
opt=torch.optim.SGD(model.parameters(),lr=LR,momentum=0.9,weight_decay=WD,nesterov=True)
steps=(N+BATCH-1)//BATCH
sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=LR,epochs=EPOCHS,steps_per_epoch=steps,pct_start=0.3,anneal_strategy='cos',div_factor=10,final_div_factor=100)
scaler=torch.amp.GradScaler('cuda')
MEAN_T=torch.tensor(MEAN,device='cuda').view(1,3,1,1)
STD_T=torch.tensor(STD,device='cuda').view(1,3,1,1)

def gpu_augment(x_uint8):
    B=x_uint8.size(0)
    xf=x_uint8.float()/255.0
    xf=F.pad(xf,(4,4,4,4),mode='reflect')
    patches=F.unfold(xf,kernel_size=32,stride=1)
    off_h=torch.randint(0,9,(B,),device='cuda')
    off_w=torch.randint(0,9,(B,),device='cuda')
    idx=off_h*9+off_w
    idx_exp=idx.view(B,1,1).expand(B,patches.size(1),1)
    sel=torch.gather(patches,2,idx_exp).squeeze(2).view(B,3,32,32)
    flip=torch.rand(B,device='cuda')<0.5
    flipped=torch.flip(sel,dims=[3])
    sel=torch.where(flip.view(B,1,1,1),flipped,sel)
    sel=(sel-MEAN_T)/STD_T
    return sel

with torch.no_grad():
    _=gpu_augment(train_imgs_gpu[:BATCH])

start=time.time()
best=0
for epoch in range(1,EPOCHS+1):
    model.train(); t0=time.time()
    perm=torch.randperm(N,device='cuda')
    for i in range(0,N,BATCH):
        idx=perm[i:i+BATCH]
        xb_u=train_imgs_gpu[idx]; yb=train_lbls_gpu[idx]
        xb=gpu_augment(xb_u).to(memory_format=torch.channels_last)
        do=random.random()<0.6
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
            if do:
                lam=random.betavariate(0.4,0.4)
                perm2=torch.randperm(xb.size(0),device='cuda')
                xb_m=lam*xb+(1-lam)*xb[perm2]
                yb2=yb[perm2]
                out=model(xb_m)
                loss=lam*crit(out,yb)+(1-lam)*crit(out,yb2)
            else:
                out=model(xb)
                loss=crit(out,yb)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    model.eval(); correct=total=0
    with torch.no_grad():
        for x,y in test_loader:
            x=x.to(device,non_blocking=True).to(memory_format=torch.channels_last); y=y.to(device,non_blocking=True)
            with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                out=model(x)
            correct+=(out.argmax(1)==y).sum().item(); total+=y.size(0)
    acc=correct/total; best=max(best,acc)
    dt=time.time()-t0
    print(f"E{epoch:02d} acc={acc*100:.2f} best={best*100:.2f} dt={dt:.1f}s lr={opt.param_groups[0]['lr']:.3f}")
    cpu_m=WRN().to("cpu"); cpu_m.load_state_dict(model.state_dict()); cpu_m.eval()
    torch.jit.script(cpu_m).save(os.path.join(CKPT,f"ckpt_epoch_{epoch:04d}.pt"))
wall=time.time()-start
print(f"wall {wall:.1f} best {best*100:.2f}")
