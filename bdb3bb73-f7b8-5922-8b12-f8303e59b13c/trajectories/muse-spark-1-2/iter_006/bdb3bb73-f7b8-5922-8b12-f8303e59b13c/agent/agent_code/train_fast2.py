import os, time, shutil, random
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
DATA_TRAIN="/workspace/data/cifar10/train"
DATA_TEST="/workspace/data/cifar10/test"
CKPT_DIR="/workspace/scratch_ckpt"
EPOCHS=35
BATCH=1024
LR=0.15
WD=5e-4
MEAN=(0.4914,0.4822,0.4465); STD=(0.2470,0.2435,0.2616)
device=torch.device("cuda")
torch.backends.cudnn.benchmark=True
class BasicBlock(nn.Module):
    def __init__(self,i,o,s):
        super().__init__()
        self.bn1=nn.BatchNorm2d(i); self.r1=nn.ReLU(inplace=True); self.c1=nn.Conv2d(i,o,3,s,1,bias=False)
        self.bn2=nn.BatchNorm2d(o); self.r2=nn.ReLU(inplace=True); self.c2=nn.Conv2d(o,o,3,1,1,bias=False)
        self.sc=nn.Identity() if i==o and s==1 else nn.Conv2d(i,o,1,s,bias=False)
    def forward(self,x):
        out=self.r1(self.bn1(x)); sc=self.sc(x); out=self.c1(out); out=self.c2(self.r2(self.bn2(out))); return out+sc
class WRN(nn.Module):
    def __init__(self,d=28,w=10):
        super().__init__()
        n=(d-4)//6
        self.ic=nn.Conv2d(3,16,3,1,1,bias=False)
        def mk(i,o,n,s):
            l=[BasicBlock(i,o,s)]
            for _ in range(1,n): l.append(BasicBlock(o,o,1))
            return nn.Sequential(*l)
        c1,c2,c3=16*w,32*w,64*w
        self.s1=mk(16,c1,n,1); self.s2=mk(c1,c2,n,2); self.s3=mk(c2,c3,n,2)
        self.bn=nn.BatchNorm2d(c3); self.relu=nn.ReLU(inplace=True); self.fc=nn.Linear(c3,10)
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d): nn.init.constant_(m.weight,1); nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu'); nn.init.constant_(m.bias,0)
    def forward(self,x):
        x=self.ic(x); x=self.s1(x); x=self.s2(x); x=self.s3(x); x=self.relu(self.bn(x)); x=F.adaptive_avg_pool2d(x,1).flatten(1); return self.fc(x)
train_tf=transforms.Compose([transforms.RandomCrop(32,padding=4,padding_mode='reflect'),transforms.RandomHorizontalFlip(),transforms.TrivialAugmentWide(),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
test_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
train_ds=datasets.ImageFolder(DATA_TRAIN,transform=train_tf)
test_ds=datasets.ImageFolder(DATA_TEST,transform=test_tf)
train_loader=DataLoader(train_ds,batch_size=BATCH,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True,prefetch_factor=2,drop_last=True)
test_loader=DataLoader(test_ds,batch_size=512,shuffle=False,num_workers=4,pin_memory=True)
print(f"Train {len(train_ds)}")
model=WRN(28,10).to(device).to(memory_format=torch.channels_last)
print(f"Params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
opt=torch.optim.SGD(model.parameters(),lr=LR,momentum=0.9,weight_decay=WD,nesterov=True)
crit=nn.CrossEntropyLoss(label_smoothing=0.1)
scaler=torch.amp.GradScaler('cuda')
if os.path.exists(CKPT_DIR): shutil.rmtree(CKPT_DIR)
os.makedirs(CKPT_DIR,exist_ok=True)
start=time.time()
first=None; last=None
for epoch in range(1,EPOCHS+1):
    model.train()
    if epoch<=5: lr=LR*epoch/5
    else:
        prog=(epoch-5)/(EPOCHS-5)
        lr=1e-4+0.5*(LR-1e-4)*(1+torch.cos(torch.tensor(prog*3.14159265)).item())
    for pg in opt.param_groups: pg['lr']=lr
    for x,y in train_loader:
        x=x.to(device,non_blocking=True).to(memory_format=torch.channels_last); y=y.to(device,non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out=model(x); loss=crit(out,y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    model.eval(); correct=0; total=0
    with torch.no_grad():
        for x,y in test_loader:
            x=x.to(device,non_blocking=True).to(memory_format=torch.channels_last); y=y.to(device,non_blocking=True)
            with torch.amp.autocast('cuda'): out=model(x)
            correct+=(out.argmax(1)==y).sum().item(); total+=y.size(0)
    acc=correct/total; elapsed=time.time()-start
    print(f"Epoch {epoch:02d} acc={acc*100:.2f}% lr={lr:.4f} time={elapsed:.1f}s")
    ckpt=os.path.join(CKPT_DIR,f"ckpt_epoch_{epoch:04d}.pt")
    m_cpu=model.to("cpu"); m_cpu.eval(); torch.jit.script(m_cpu).save(ckpt); model.to(device).to(memory_format=torch.channels_last)
    if first is None: first=time.time()
    last=time.time()
    if acc>=0.94: print(f"Hit 94 at {epoch}"); break
ckpts=sorted([f for f in os.listdir(CKPT_DIR) if f.startswith("ckpt_epoch_")])
if ckpts: shutil.copy(os.path.join(CKPT_DIR,ckpts[-1]),"/workspace/scratch_final.pt")
total=last-start if last else time.time()-start
print(f"Done wall {total:.1f}s")
