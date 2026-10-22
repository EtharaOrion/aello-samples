import os, time, shutil, random, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

EPOCHS=30
BATCH=1024
LR=0.8
WD=5e-4
WARMUP=5
CKPT_DIR="/workspace/scratch_ckpt"
for f in os.listdir(CKPT_DIR):
    os.remove(os.path.join(CKPT_DIR,f))

device=torch.device("cuda")
print(torch.cuda.get_device_name(0))

mean=(0.4914,0.4822,0.4465)
std=(0.2470,0.2435,0.2616)

train_transform=transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean,std),
])
test_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean,std),
])

train_set=ImageFolder("/workspace/data/cifar10/train",transform=train_transform)
test_set=ImageFolder("/workspace/data/cifar10/test",transform=test_transform)
train_loader=DataLoader(train_set,batch_size=BATCH,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True)
test_loader=DataLoader(test_set,batch_size=512,shuffle=False,num_workers=2,pin_memory=True)

class BasicBlock(nn.Module):
    def __init__(self,i,o,s):
        super().__init__()
        self.bn1=nn.BatchNorm2d(i); self.relu1=nn.ReLU(inplace=True)
        self.conv1=nn.Conv2d(i,o,3,s,1,bias=False)
        self.bn2=nn.BatchNorm2d(o); self.relu2=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(o,o,3,1,1,bias=False)
        self.eq=(i==o)
        self.shortcut=nn.Conv2d(i,o,1,s,0,bias=False) if not self.eq else nn.Identity()
    def forward(self,x):
        out=self.relu1(self.bn1(x)); out=self.conv1(out); out=self.relu2(self.bn2(out)); out=self.conv2(out)
        return out+self.shortcut(x)
class NetworkBlock(nn.Module):
    def __init__(self,n,i,o,s):
        super().__init__()
        self.layer=nn.Sequential(*[BasicBlock(i if k==0 else o,o,s if k==0 else 1) for k in range(n)])
    def forward(self,x): return self.layer(x)
class WRN(nn.Module):
    def __init__(self,d=28,w=10):
        super().__init__()
        nC=[16,16*w,32*w,64*w]
        n=(d-4)//6
        self.conv1=nn.Conv2d(3,nC[0],3,1,1,bias=False)
        self.b1=NetworkBlock(n,nC[0],nC[1],1); self.b2=NetworkBlock(n,nC[1],nC[2],2); self.b3=NetworkBlock(n,nC[2],nC[3],2)
        self.bn1=nn.BatchNorm2d(nC[3]); self.relu=nn.ReLU(inplace=True); self.fc=nn.Linear(nC[3],10); self.nC=nC[3]
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d): m.weight.data.fill_(1); m.bias.data.zero_()
            elif isinstance(m,nn.Linear): m.bias.data.zero_()
    def forward(self,x):
        out=self.conv1(x); out=self.b1(out); out=self.b2(out); out=self.b3(out)
        out=self.relu(self.bn1(out)); out=F.adaptive_avg_pool2d(out,1); out=out.view(-1,self.nC)
        return self.fc(out)

model=WRN(28,10).to(device).to(memory_format=torch.channels_last)
print(f"Params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
torch.jit.script(model); print("script ok")
criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer=torch.optim.SGD(model.parameters(),lr=LR,momentum=0.9,weight_decay=WD,nesterov=True)
def lr_lambda(epoch):
    e=epoch+1
    if e<=WARMUP: return e/WARMUP
    progress=(e-WARMUP)/(EPOCHS-WARMUP)
    return 0.5*(1+torch.cos(torch.tensor(progress*3.14159265))).item()
scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler=torch.amp.GradScaler('cuda')
start_wall=time.time()
for epoch in range(1,EPOCHS+1):
    model.train(); t0=time.time()
    for x,y in train_loader:
        x=x.to(device,non_blocking=True).to(memory_format=torch.channels_last); y=y.to(device,non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out=model(x); loss=criterion(out,y)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
    scheduler.step()
    epoch_train=time.time()-t0
    model.eval(); correct=total=0
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for x,y in test_loader:
                x=x.to(device,non_blocking=True).to(memory_format=torch.channels_last); y=y.to(device,non_blocking=True)
                out=model(x); correct+=(out.argmax(1)==y).sum().item(); total+=y.size(0)
    acc=correct/total
    total_time=time.time()-start_wall
    print(f"Epoch {epoch}/{EPOCHS} acc {acc*100:.2f}% lr {optimizer.param_groups[0]['lr']:.4f} train {epoch_train:.1f}s total {total_time:.1f}s")
    model.eval(); scripted=torch.jit.script(model); ckpt=os.path.join(CKPT_DIR,f"ckpt_epoch_{epoch:04d}.pt"); scripted.save(ckpt)
    loaded=torch.jit.load(ckpt,map_location="cpu"); loaded.eval()
    with torch.no_grad(): _=loaded(torch.randn(2,3,32,32))
    print(f" -> saved {ckpt}")
    model.train()
    if acc>=0.94: print(f"*** Hit 94% at epoch {epoch} ***")
wall=time.time()-start_wall
print(f"Done wall={wall:.1f}s acc={acc:.4f}")
with open("/tmp/scratch_metrics.json","w") as f: json.dump({"accuracy":acc,"wall_clock_sec":wall},f)
