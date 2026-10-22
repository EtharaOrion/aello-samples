import os, time, shutil, random
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

EPOCHS=30
BATCH=1024
LR=1.2
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

# LIGHT augmentation - no AutoAugment
train_tf=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
test_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
train_ds=datasets.ImageFolder("/workspace/data/cifar10/train",transform=train_tf)
test_ds=datasets.ImageFolder("/workspace/data/cifar10/test",transform=test_tf)
train_loader=DataLoader(train_ds,batch_size=BATCH,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True,prefetch_factor=2)
test_loader=DataLoader(test_ds,batch_size=512,shuffle=False,num_workers=2,pin_memory=True,persistent_workers=True)

device=torch.device("cuda")
model=WRN().to(device).to(memory_format=torch.channels_last)
# compile for speed
try:
    model=torch.compile(model)
    print("compile ok")
except Exception as e: print(e)
crit=nn.CrossEntropyLoss(label_smoothing=0.1)
opt=torch.optim.SGD(model.parameters(),lr=LR,momentum=0.9,weight_decay=WD,nesterov=True)
sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=LR,epochs=EPOCHS,steps_per_epoch=len(train_loader),pct_start=0.3,anneal_strategy='cos',div_factor=10,final_div_factor=100)
scaler=torch.amp.GradScaler('cuda')

def cutmix(x,y,alpha=1.0):
    lam = random.betavariate(alpha,alpha)
    idx=torch.randperm(x.size(0),device=x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[idx, :, bby1:bby2, bbx1:bbx2]
    # adjust lambda
    lam = 1 - ((bbx2-bbx1)*(bby2-bby1) / (x.size(-1)*x.size(-2)))
    return x, y, y[idx], lam

def rand_bbox(size, lam):
    W=size[-1]; H=size[-2]
    cut_rat = (1-lam)**0.5
    cut_w=int(W*cut_rat); cut_h=int(H*cut_rat)
    cx=random.randint(0,W-1); cy=random.randint(0,H-1)
    bbx1=max(0,cx-cut_w//2); bby1=max(0,cy-cut_h//2)
    bbx2=min(W,cx+cut_w//2); bby2=min(H,cy+cut_h//2)
    return bbx1,bby1,bbx2,bby2

def mixup(x,y,a=0.4):
    lam=random.betavariate(a,a); idx=torch.randperm(x.size(0),device=x.device); return lam*x+(1-lam)*x[idx], y, y[idx], lam

start=time.time()
best=0
for epoch in range(1,EPOCHS+1):
    model.train(); t0=time.time()
    for x,y in train_loader:
        x=x.to(device,non_blocking=True).to(memory_format=torch.channels_last); y=y.to(device,non_blocking=True)
        r=random.random()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            if r<0.3:
                xm,ya,yb,lam=mixup(x,y,0.4)
                out=model(xm); loss=lam*crit(out,ya)+(1-lam)*crit(out,yb)
            elif r<0.6:
                # cutmix
                xm,ya,yb,lam=cutmix(x.clone(),y,1.0)
                out=model(xm); loss=lam*crit(out,ya)+(1-lam)*crit(out,yb)
            else:
                out=model(x); loss=crit(out,y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    model.eval(); correct=total=0
    with torch.no_grad():
        for x,y in test_loader:
            x=x.to(device,non_blocking=True).to(memory_format=torch.channels_last); y=y.to(device,non_blocking=True)
            with torch.amp.autocast('cuda'): out=model(x)
            correct+=(out.argmax(1)==y).sum().item(); total+=y.size(0)
    acc=correct/total; best=max(best,acc)
    print(f"E{epoch:02d} acc={acc*100:.2f} best={best*100:.2f} dt={time.time()-t0:.1f}s")
    raw = model._orig_mod if hasattr(model,'_orig_mod') else model
    cpu_m=WRN().to("cpu"); cpu_m.load_state_dict({k.replace("_orig_mod.",""):v for k,v in raw.state_dict().items()}); cpu_m.eval()
    torch.jit.script(cpu_m).save(os.path.join(CKPT,f"ckpt_epoch_{epoch:04d}.pt"))
wall=time.time()-start
print(f"wall {wall:.1f} best {best*100:.2f}")
