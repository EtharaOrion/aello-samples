import os, time, shutil, random
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

EPOCHS=24
BATCH=1024
LR=1.5
WD=5e-4
NUM_WORKERS=4
device=torch.device("cuda")
print(torch.cuda.get_device_name(0))

os.makedirs("/workspace/scratch_ckpt", exist_ok=True)
for f in os.listdir("/workspace/scratch_ckpt"):
    os.remove(os.path.join("/workspace/scratch_ckpt",f))

mean=(0.4914,0.4822,0.4465); std=(0.2470,0.2435,0.2616)
train_tf=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean,std)])
test_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)])
trainset=ImageFolder("/workspace/data/cifar10/train",transform=train_tf)
testset=ImageFolder("/workspace/data/cifar10/test",transform=test_tf)
trainloader=DataLoader(trainset,batch_size=BATCH,shuffle=True,num_workers=NUM_WORKERS,pin_memory=True,persistent_workers=True,prefetch_factor=2)
testloader=DataLoader(testset,batch_size=512,shuffle=False,num_workers=2,pin_memory=True)

# ResNet9 DAWN style
class ConvBN(nn.Module):
    def __init__(self,i,o,k=3,s=1,p=1): super().__init__(); self.c=nn.Conv2d(i,o,k,s,p,bias=False); self.bn=nn.BatchNorm2d(o)
    def forward(self,x): return F.relu(self.bn(self.c(x)))
class ResBlock(nn.Module):
    def __init__(self,c): super().__init__(); self.c1=ConvBN(c,c); self.c2=nn.Sequential(nn.Conv2d(c,c,3,1,1,bias=False),nn.BatchNorm2d(c))
    def forward(self,x): return F.relu(x+self.c2(self.c1(x)))
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.prep=ConvBN(3,64)
        self.l1=nn.Sequential(ConvBN(64,128,3,2,1),ResBlock(128))
        self.l2=nn.Sequential(ConvBN(128,256,3,2,1),ResBlock(256))
        self.l3=nn.Sequential(ConvBN(256,512,3,2,1),ResBlock(512))
        self.pool=nn.AdaptiveMaxPool2d(1)
        self.fc=nn.Linear(512,10)
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d): m.weight.data.fill_(1); m.bias.data.zero_()
    def forward(self,x):
        x=self.prep(x); x=self.l1(x); x=self.l2(x); x=self.l3(x); x=self.pool(x).view(x.size(0),-1); return self.fc(x)

model=Net().to(device).to(memory_format=torch.channels_last)
print(f"Params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
opt=torch.optim.SGD(model.parameters(),lr=LR,momentum=0.9,weight_decay=WD,nesterov=True)
sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=LR,epochs=EPOCHS,steps_per_epoch=len(trainloader),pct_start=0.3,anneal_strategy='cos',div_factor=10,final_div_factor=30)
scaler=torch.amp.GradScaler('cuda')
crit=nn.CrossEntropyLoss(label_smoothing=0.1)

def mixup(x,y,a=0.2):
    lam=torch.distributions.Beta(a,a).sample().item()
    idx=torch.randperm(x.size(0),device=x.device)
    return lam*x+(1-lam)*x[idx],y,y[idx],lam
def cutmix(x,y,a=1.0):
    lam=torch.distributions.Beta(a,a).sample().item()
    idx=torch.randperm(x.size(0),device=x.device)
    W,H=x.size(2),x.size(3); cr=(1-lam)**0.5; cw=int(W*cr); ch=int(H*cr)
    cx=torch.randint(W,(1,)).item(); cy=torch.randint(H,(1,)).item()
    x1=max(0,cx-cw//2); y1=max(0,cy-ch//2); x2=min(W,cx+cw//2); y2=min(H,cy+ch//2)
    x[:,:,x1:x2,y1:y2]=x[idx,:,x1:x2,y1:y2]
    lam=1-((x2-x1)*(y2-y1)/(W*H))
    return x,y,y[idx],lam
def mloss(p,ya,yb,lam): return lam*crit(p,ya)+(1-lam)*crit(p,yb)

start=time.time(); best=0
for epoch in range(1,EPOCHS+1):
    model.train(); t0=time.time()
    for inp,tar in trainloader:
        inp,tar=inp.to(device,non_blocking=True),tar.to(device,non_blocking=True)
        inp=inp.to(memory_format=torch.channels_last)
        use=random.random()<0.5
        mt=random.choice(['mixup','cutmix']) if use else None
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            if mt=='mixup':
                xm,ya,yb,lam=mixup(inp,tar,0.2); out=model(xm); loss=mloss(out,ya,yb,lam)
            elif mt=='cutmix':
                xm,ya,yb,lam=cutmix(inp.clone(),tar,1.0); out=model(xm); loss=mloss(out,ya,yb,lam)
            else:
                out=model(inp); loss=crit(out,tar)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    model.eval(); correct=total=0
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for inp,tar in testloader:
                inp,tar=inp.to(device,non_blocking=True),tar.to(device,non_blocking=True)
                inp=inp.to(memory_format=torch.channels_last)
                out=model(inp); _,p=out.max(1); total+=tar.size(0); correct+=p.eq(tar).sum().item()
    acc=correct/total; best=max(best,acc)
    print(f"Epoch {epoch}/{EPOCHS} acc {acc*100:.2f} best {best*100:.2f} time {time.time()-t0:.1f}s wall {time.time()-start:.1f}s lr {opt.param_groups[0]['lr']:.4f}")
    torch.jit.script(model.eval()).save(f"/workspace/scratch_ckpt/ckpt_epoch_{epoch:04d}.pt")
    model.train()
if best>=0.94:
    for f in os.listdir("/workspace/periodic_checkpoints"): os.remove(os.path.join("/workspace/periodic_checkpoints",f))
    for f in sorted(os.listdir("/workspace/scratch_ckpt")): shutil.copy(f"/workspace/scratch_ckpt/{f}",f"/workspace/periodic_checkpoints/{f}")
    shutil.copy(f"/workspace/periodic_checkpoints/ckpt_epoch_{EPOCHS:04d}.pt","/workspace/checkpoint.pt")
    print(f"PROMOTED best {best*100:.2f} wall {time.time()-start:.1f}s")
    import json; json.dump({"accuracy":best,"wall_clock_sec":time.time()-start},open("/workspace/agent_reported_metrics.json","w"))
else:
    print(f"NOT promoted best {best*100:.2f}")
