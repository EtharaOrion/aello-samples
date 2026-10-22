import os, time, random, math
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
DATA_TRAIN="/workspace/data/cifar10/train"
DATA_TEST="/workspace/data/cifar10/test"
CKPT_DIR="/workspace/scratch_ckpt"
EPOCHS=30
BATCH=1024
LR=1.0
WD=5e-4
MOM=0.9
MEAN=(0.4914,0.4822,0.4465)
STD=(0.2470,0.2435,0.2616)
torch.backends.cudnn.benchmark=True
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
class BasicBlock(nn.Module):
    def __init__(self,i,o,s=1):
        super().__init__()
        self.bn1=nn.BatchNorm2d(i);self.conv1=nn.Conv2d(i,o,3,s,1,bias=False)
        self.bn2=nn.BatchNorm2d(o);self.conv2=nn.Conv2d(o,o,3,1,1,bias=False)
        self.shortcut=nn.Sequential()
        if s!=1 or i!=o:
            self.shortcut=nn.Sequential(nn.Conv2d(i,o,1,s,bias=False),nn.BatchNorm2d(o))
    def forward(self,x):
        out=F.relu(self.bn1(x))
        sc=self.shortcut(out) if len(self.shortcut)>0 else x
        out=self.conv1(out)
        out=self.conv2(F.relu(self.bn2(out)))
        out+=sc
        return out
class CifarResNet(nn.Module):
    def __init__(self,d=28,w=4):
        super().__init__()
        n=(d-4)//6;k=w;self.in_ch=16
        self.conv1=nn.Conv2d(3,16,3,1,1,bias=False)
        self.layer1=self._make(16*k,n,1)
        self.layer2=self._make(32*k,n,2)
        self.layer3=self._make(64*k,n,2)
        self.bn=nn.BatchNorm2d(64*k);self.fc=nn.Linear(64*k,10)
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d): nn.init.constant_(m.weight,1);nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu');nn.init.constant_(m.bias,0)
    def _make(self,o,b,s):
        ls=[]
        for i in range(b):
            ls.append(BasicBlock(self.in_ch,o,s if i==0 else 1));self.in_ch=o
        return nn.Sequential(*ls)
    def forward(self,x):
        x=self.conv1(x);x=self.layer1(x);x=self.layer2(x);x=self.layer3(x)
        x=F.relu(self.bn(x));x=F.adaptive_avg_pool2d(x,1);x=x.view(x.size(0),-1);x=self.fc(x);return x
def rand_bbox(size, lam):
    W=size[-1];H=size[-2]
    cut_rat=(1.-lam)**0.5
    cut_w=int(W*cut_rat);cut_h=int(H*cut_rat)
    cx=random.randint(0,W-1);cy=random.randint(0,H-1)
    bbx1=max(0,cx-cut_w//2);bby1=max(0,cy-cut_h//2)
    bbx2=min(W,cx+cut_w//2);bby2=min(H,cy+cut_h//2)
    return bbx1,bby1,bbx2,bby2
def mixup_cutmix(bx,by,alpha=1.0):
    if random.random()>0.5:
        lam=random.betavariate(alpha,alpha) if alpha>0 else 1.0
        idx=torch.randperm(bx.size(0),device=bx.device)
        mx=lam*bx+(1-lam)*bx[idx]
        ya,yb=by,by[idx]
        return mx,ya,yb,lam
    else:
        lam=random.betavariate(alpha,alpha) if alpha>0 else 1.0
        idx=torch.randperm(bx.size(0),device=bx.device)
        bbx1,bby1,bbx2,bby2=rand_bbox(bx.size(),lam)
        mx=bx.clone()
        mx[:,:,bby1:bby2,bbx1:bbx2]=bx[idx,:,bby1:bby2,bbx1:bbx2]
        lam=1-((bbx2-bbx1)*(bby2-bby1)/(bx.size(-1)*bx.size(-2)))
        ya,yb=by,by[idx]
        return mx,ya,yb,lam
def main():
    os.makedirs(CKPT_DIR,exist_ok=True)
    for f in os.listdir(CKPT_DIR):
        if f.startswith("ckpt_epoch_"): os.remove(os.path.join(CKPT_DIR,f))
    device=torch.device("cuda")
    train_tf=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    test_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    train_ds=datasets.ImageFolder(DATA_TRAIN,transform=train_tf)
    test_ds=datasets.ImageFolder(DATA_TEST,transform=test_tf)
    train_loader=DataLoader(train_ds,batch_size=BATCH,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True,prefetch_factor=2)
    test_loader=DataLoader(test_ds,batch_size=512,shuffle=False,num_workers=2,pin_memory=True)
    model=CifarResNet(28,4).to(device).to(memory_format=torch.channels_last)
    crit=nn.CrossEntropyLoss(label_smoothing=0.1)
    opt=torch.optim.SGD(model.parameters(),lr=LR,momentum=MOM,weight_decay=WD,nesterov=True)
    warm=3
    def lam(e):
        ee=e+1
        if ee<=warm: return ee/warm
        p=(ee-warm)/(EPOCHS-warm)
        return 0.5*(1+math.cos(p*math.pi))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lam)
    scaler=torch.amp.GradScaler('cuda')
    start=time.time();first=None;last=None;best=0
    for epoch in range(1,EPOCHS+1):
        model.train();t0=time.time()
        for x,y in train_loader:
            x=x.to(device,non_blocking=True).to(memory_format=torch.channels_last);y=y.to(device,non_blocking=True)
            if random.random()<0.5:
                x,ya,yb,lamv=mixup_cutmix(x,y,alpha=1.0)
                with torch.amp.autocast('cuda'):
                    out=model(x);loss=lamv*crit(out,ya)+(1-lamv)*crit(out,yb)
            else:
                with torch.amp.autocast('cuda'):
                    out=model(x);loss=crit(out,y)
            opt.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.step(opt);scaler.update()
        sched.step()
        model.eval();corr=0;tot=0
        with torch.no_grad():
            for x,y in test_loader:
                x=x.to(device,non_blocking=True).to(memory_format=torch.channels_last);y=y.to(device,non_blocking=True)
                with torch.amp.autocast('cuda'): out=model(x)
                corr+=(out.argmax(1)==y).sum().item();tot+=y.size(0)
        acc=corr/tot;best=max(best,acc)
        print(f"Epoch {epoch:03d} acc={acc*100:.2f} best={best*100:.2f} lr={opt.param_groups[0]['lr']:.4f} t={time.time()-t0:.1f}s")
        scripted=torch.jit.script(model.eval());scripted.save(os.path.join(CKPT_DIR,f"ckpt_epoch_{epoch:04d}.pt"))
        if first is None: first=time.time()
        last=time.time()
    print(f"Done best {best*100:.2f} wall {last-start:.1f}s")
if __name__=="__main__": main()
