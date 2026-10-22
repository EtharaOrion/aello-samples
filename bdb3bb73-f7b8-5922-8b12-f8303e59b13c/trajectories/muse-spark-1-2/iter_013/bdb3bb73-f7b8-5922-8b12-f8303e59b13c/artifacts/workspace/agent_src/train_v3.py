import os, time, random
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_ROOT="/workspace/data/cifar10"
CKPT_DIR="/workspace/scratch_ckpt"
EPOCHS=20
BATCH=1024
WORKERS=4
LR=1.5
DEVICE="cuda"
os.makedirs(CKPT_DIR, exist_ok=True)
for f in os.listdir(CKPT_DIR): os.remove(os.path.join(CKPT_DIR,f))
torch.backends.cudnn.benchmark=True

class Block(nn.Module):
    def __init__(self,i,o,s):
        super().__init__()
        self.c1=nn.Conv2d(i,o,3,s,1,bias=False);self.b1=nn.BatchNorm2d(o)
        self.c2=nn.Conv2d(o,o,3,1,1,bias=False);self.b2=nn.BatchNorm2d(o)
        self.sc=nn.Sequential() if s==1 and i==o else nn.Sequential(nn.Conv2d(i,o,1,s,bias=False),nn.BatchNorm2d(o))
    def forward(self,x):
        out=F.relu(self.b1(self.c1(x))); out=self.b2(self.c2(out)); out+=self.sc(x); return F.relu(out)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,64,3,1,1,bias=False);self.bn1=nn.BatchNorm2d(64)
        self.l1=self._make(64,128,4,1);self.l2=self._make(128,256,4,2);self.l3=self._make(256,512,4,2)
        self.bnF=nn.BatchNorm2d(512);self.fc=nn.Linear(512,10)
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d): nn.init.constant_(m.weight,1);nn.init.constant_(m.bias,0)
    def _make(self,i,o,n,s):
        l=[Block(i,o,s)]
        for _ in range(1,n): l.append(Block(o,o,1))
        return nn.Sequential(*l)
    def forward(self,x):
        x=F.relu(self.bn1(self.conv1(x)));x=self.l1(x);x=self.l2(x);x=self.l3(x);x=F.relu(self.bnF(x));x=F.adaptive_avg_pool2d(x,1).flatten(1);return self.fc(x)

CM=(0.4914,0.4822,0.4465);CS=(0.2470,0.2435,0.2616)
train_tf=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),transforms.ToTensor(),transforms.Normalize(CM,CS)])
test_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(CM,CS)])
train_ds=datasets.ImageFolder(os.path.join(DATA_ROOT,"train"),transform=train_tf)
test_ds=datasets.ImageFolder(os.path.join(DATA_ROOT,"test"),transform=test_tf)
train_loader=DataLoader(train_ds,batch_size=BATCH,shuffle=True,num_workers=WORKERS,pin_memory=True,persistent_workers=True)
test_loader=DataLoader(test_ds,batch_size=512,shuffle=False,num_workers=2,pin_memory=True)
model=Net().to(DEVICE).to(memory_format=torch.channels_last)
opt=torch.optim.SGD(model.parameters(),lr=LR,momentum=0.9,weight_decay=5e-4,nesterov=True)
sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=LR,epochs=EPOCHS,steps_per_epoch=len(train_loader),pct_start=0.3,anneal_strategy='cos',div_factor=10,final_div_factor=100)
scaler=torch.amp.GradScaler('cuda')
crit=nn.CrossEntropyLoss(label_smoothing=0.1)
def rand_bbox(sz,lam):
    W,H=sz[2],sz[3]; cr=(1-lam)**0.5; cw,ch=int(W*cr),int(H*cr); cx,cy=random.randint(0,W),random.randint(0,H)
    return max(cx-cw//2,0),max(cy-ch//2,0),min(cx+cw//2,W),min(cy+ch//2,H)
start=time.time()
for epoch in range(1,EPOCHS+1):
    model.train()
    for x,y in train_loader:
        x,y=x.to(DEVICE,non_blocking=True),y.to(DEVICE,non_blocking=True)
        x=x.to(memory_format=torch.channels_last)
        r=random.random(); use=r<0.5
        if use:
            if r<0.25:
                lam=random.betavariate(0.2,0.2); idx=torch.randperm(x.size(0),device=x.device)
                x=lam*x+(1-lam)*x[idx]; y_a,y_b=y,y[idx]
            else:
                lam=random.betavariate(1.0,1.0); idx=torch.randperm(x.size(0),device=x.device)
                bbx1,bby1,bbx2,bby2=rand_bbox(x.size(),lam); x[:,:,bbx1:bbx2,bby1:bby2]=x[idx,:,bbx1:bbx2,bby1:bby2]
                lam=1-((bbx2-bbx1)*(bby2-bby1)/1024); y_a,y_b=y,y[idx]
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out=model(x)
            loss=lam*crit(out,y_a)+(1-lam)*crit(out,y_b) if use else crit(out,y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    model.eval(); correct=total=0
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for x,y in test_loader:
                x,y=x.to(DEVICE,non_blocking=True),y.to(DEVICE,non_blocking=True)
                x=x.to(memory_format=torch.channels_last)
                correct+=(model(x).argmax(1)==y).sum().item(); total+=y.size(0)
    acc=correct/total; el=time.time()-start
    print(f"Epoch {epoch} acc={acc:.4f} el={el:.1f}s lr={opt.param_groups[0]['lr']:.4f}")
    m=model.to(memory_format=torch.contiguous_format)
    torch.jit.script(m).save(os.path.join(CKPT_DIR,f"ckpt_epoch_{epoch:04d}.pt"))
    m.to(memory_format=torch.channels_last)
print(f"Done {time.time()-start:.1f}s final {acc:.4f}")
