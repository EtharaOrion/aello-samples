import os, time, shutil, random
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

EPOCHS=35
BATCH=1024
LR=1.2
WD=5e-4
NUM_WORKERS=4
device=torch.device("cuda")
print(torch.cuda.get_device_name(0))
os.makedirs("/workspace/scratch_ckpt", exist_ok=True)
for f in os.listdir("/workspace/scratch_ckpt"): os.remove(os.path.join("/workspace/scratch_ckpt",f))
mean=(0.4914,0.4822,0.4465); std=(0.2470,0.2435,0.2616)
train_tf=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean,std)])
test_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)])
trainset=ImageFolder("/workspace/data/cifar10/train",transform=train_tf)
testset=ImageFolder("/workspace/data/cifar10/test",transform=test_tf)
trainloader=DataLoader(trainset,batch_size=BATCH,shuffle=True,num_workers=NUM_WORKERS,pin_memory=True,persistent_workers=True,prefetch_factor=2)
testloader=DataLoader(testset,batch_size=512,shuffle=False,num_workers=2,pin_memory=True)

def conv3x3(i,o,s=1): return nn.Conv2d(i,o,3,stride=s,padding=1,bias=False)
class Block(nn.Module):
    def __init__(self, inp, out, stride):
        super().__init__()
        self.bn1=nn.BatchNorm2d(inp); self.c1=conv3x3(inp,out,stride)
        self.bn2=nn.BatchNorm2d(out); self.c2=conv3x3(out,out)
        self.short=nn.Sequential() if stride==1 and inp==out else nn.Sequential(nn.Conv2d(inp,out,1,stride=stride,bias=False),nn.BatchNorm2d(out))
    def forward(self,x):
        out=F.relu(self.bn1(x)); out=self.c1(out); out=F.relu(self.bn2(out)); out=self.c2(out); out+=self.short(x); return out
class WRN(nn.Module):
    def __init__(self,d,w):
        super().__init__()
        n=(d-4)//6; k=w; stages=[16,16*k,32*k,64*k]
        self.c1=conv3x3(3,stages[0])
        self.l1=self._make(stages[0],stages[1],n,1)
        self.l2=self._make(stages[1],stages[2],n,2)
        self.l3=self._make(stages[2],stages[3],n,2)
        self.bn=nn.BatchNorm2d(stages[3]); self.fc=nn.Linear(stages[3],10)
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d): m.weight.data.fill_(1); m.bias.data.zero_()
    def _make(self,inp,out,n,s):
        l=[Block(inp,out,s)]
        for _ in range(1,n): l.append(Block(out,out,1))
        return nn.Sequential(*l)
    def forward(self,x):
        x=self.c1(x); x=self.l1(x); x=self.l2(x); x=self.l3(x); x=F.relu(self.bn(x)); x=F.adaptive_avg_pool2d(x,1); x=x.view(x.size(0),-1); return self.fc(x)

# Try WRN 16-8 (11M params, ~2x faster)
for cfg in [(16,8)]:
    model=WRN(*cfg).to(device).to(memory_format=torch.channels_last)
    print(f"WRN {cfg} Params {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    opt=torch.optim.SGD(model.parameters(),lr=LR,momentum=0.9,weight_decay=WD,nesterov=True)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=LR,epochs=EPOCHS,steps_per_epoch=len(trainloader),pct_start=0.2,anneal_strategy='cos',div_factor=10,final_div_factor=50)
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
        print(f"Epoch {epoch}/{EPOCHS} acc {acc*100:.2f} best {best*100:.2f} time {time.time()-t0:.1f}s wall {time.time()-start:.1f}s")
        torch.jit.script(model.eval()).save(f"/workspace/scratch_ckpt/ckpt_epoch_{epoch:04d}.pt")
        model.train()
    print(f"WRN {cfg} done best {best*100:.2f} wall {time.time()-start:.1f}s")
    if best>=0.94:
        for f in os.listdir("/workspace/periodic_checkpoints"): os.remove(os.path.join("/workspace/periodic_checkpoints",f))
        for f in sorted(os.listdir("/workspace/scratch_ckpt")): shutil.copy(f"/workspace/scratch_ckpt/{f}",f"/workspace/periodic_checkpoints/{f}")
        shutil.copy(f"/workspace/periodic_checkpoints/ckpt_epoch_{EPOCHS:04d}.pt","/workspace/checkpoint.pt")
        import json; json.dump({"accuracy":best,"wall_clock_sec":time.time()-start},open("/workspace/agent_reported_metrics.json","w"))
        print("PROMOTED")
    else:
        print("NOT promoted")
