import os, time, shutil, random, json
import torch, numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_TRAIN="/workspace/data/cifar10/train"
DATA_TEST="/workspace/data/cifar10/test"
CKPT_DIR="/workspace/periodic_checkpoints"
FINAL_CKPT="/workspace/checkpoint.pt"
EPOCHS=80
BATCH=512
LR=1.0
WD=5e-4
MOM=0.9
NW=8
MEAN=(0.4914,0.4822,0.4465)
STD=(0.2470,0.2435,0.2616)
device=torch.device("cuda")
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.allow_tf32=True
torch.backends.cuda.matmul.allow_tf32=True

class Mul(nn.Module):
    def __init__(self,w): super().__init__(); self.w=w
    def forward(self,x): return x*self.w
class Residual(nn.Module):
    def __init__(self,m): super().__init__(); self.m=m
    def forward(self,x): return x+self.m(x)
def conv_bn(c_in,c_out,k=3,s=1,p=1):
    return nn.Sequential(nn.Conv2d(c_in,c_out,k,s,p,bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0=conv_bn(3,64,3,1,1)
        self.conv1=conv_bn(64,128,3,1,1)
        self.pool1=nn.MaxPool2d(2)
        self.res1a=Residual(nn.Sequential(conv_bn(128,128,3,1,1),conv_bn(128,128,3,1,1)))
        self.res1b=Residual(nn.Sequential(conv_bn(128,128,3,1,1),conv_bn(128,128,3,1,1)))
        self.res1c=Residual(nn.Sequential(conv_bn(128,128,3,1,1),conv_bn(128,128,3,1,1)))
        self.conv2=conv_bn(128,256,3,1,1)
        self.pool2=nn.MaxPool2d(2)
        self.conv3=conv_bn(256,512,3,1,1)
        self.pool3=nn.MaxPool2d(2)
        self.res2a=Residual(nn.Sequential(conv_bn(512,512,3,1,1),conv_bn(512,512,3,1,1)))
        self.res2b=Residual(nn.Sequential(conv_bn(512,512,3,1,1),conv_bn(512,512,3,1,1)))
        self.res2c=Residual(nn.Sequential(conv_bn(512,512,3,1,1),conv_bn(512,512,3,1,1)))
        self.pool4=nn.MaxPool2d(2)
        self.fc=nn.Linear(512*2*2,10,bias=False)
        self.mul=Mul(0.2)
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d): nn.init.constant_(m.weight,1); nn.init.constant_(m.bias,0)
    def forward(self,x):
        x=self.conv0(x);x=self.conv1(x);x=self.pool1(x)
        x=self.res1a(x);x=self.res1b(x);x=self.res1c(x)
        x=self.conv2(x);x=self.pool2(x)
        x=self.conv3(x);x=self.pool3(x)
        x=self.res2a(x);x=self.res2b(x);x=self.res2c(x)
        x=self.pool4(x)
        x=x.reshape(x.size(0),-1)
        x=self.fc(x);x=self.mul(x)
        return x

train_tf=transforms.Compose([
    transforms.RandomCrop(32,padding=4,padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(MEAN,STD),
])
test_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
train_ds=datasets.ImageFolder(DATA_TRAIN,transform=train_tf)
test_ds=datasets.ImageFolder(DATA_TEST,transform=test_tf)
train_loader=DataLoader(train_ds,batch_size=BATCH,shuffle=True,num_workers=NW,pin_memory=True,persistent_workers=True,prefetch_factor=4,drop_last=True)
test_loader=DataLoader(test_ds,batch_size=512,shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True)

model=Net().to(device).to(memory_format=torch.channels_last)
criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer=torch.optim.SGD(model.parameters(),lr=LR,momentum=MOM,weight_decay=WD,nesterov=True)
scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=LR,epochs=EPOCHS,steps_per_epoch=len(train_loader),pct_start=0.3,anneal_strategy='cos',div_factor=10,final_div_factor=100)
scaler=torch.amp.GradScaler('cuda')

def cutmix(x,y,alpha=1.0):
    lam=np.random.beta(alpha,alpha)
    b=x.size(0)
    idx=torch.randperm(b,device=x.device)
    y_a,y_b=y,y[idx]
    W=x.size(3);H=x.size(2)
    cut_rat=np.sqrt(1.-lam)
    cut_w=int(W*cut_rat);cut_h=int(H*cut_rat)
    cx=np.random.randint(W);cy=np.random.randint(H)
    bbx1=np.clip(cx-cut_w//2,0,W);bby1=np.clip(cy-cut_h//2,0,H)
    bbx2=np.clip(cx+cut_w//2,0,W);bby2=np.clip(cy+cut_h//2,0,H)
    x[:,:,bby1:bby2,bbx1:bbx2]=x[idx][:,:,bby1:bby2,bbx1:bbx2]
    lam=1-((bbx2-bbx1)*(bby2-bby1)/(W*H))
    return x,y_a,y_b,lam

if os.path.exists(CKPT_DIR): shutil.rmtree(CKPT_DIR)
os.makedirs(CKPT_DIR,exist_ok=True)
start=time.time()
print(f"Net deeper {EPOCHS}ep B{BATCH} LR{LR} steps={len(train_loader)}")
best=0
for epoch in range(1,EPOCHS+1):
    model.train()
    t0=time.time()
    for x,y in train_loader:
        x=x.to(device,non_blocking=True).to(memory_format=torch.channels_last)
        y=y.to(device,non_blocking=True)
        use=random.random()<0.5
        if use: x,y_a,y_b,lam=cutmix(x,y,1.0)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out=model(x)
            loss=(lam*criterion(out,y_a)+(1-lam)*criterion(out,y_b)) if use else criterion(out,y)
        scaler.scale(loss).backward()
        scaler.step(optimizer);scaler.update();scheduler.step()
    model.eval();correct=total=0
    with torch.no_grad():
        for x,y in test_loader:
            x=x.to(device,non_blocking=True).to(memory_format=torch.channels_last)
            y=y.to(device,non_blocking=True)
            with torch.amp.autocast('cuda'):
                out=model(x)
            correct+=(out.argmax(1)==y).sum().item();total+=y.size(0)
    acc=correct/total;best=max(best,acc)
    print(f"Ep {epoch:02d} acc={acc*100:.2f} best={best*100:.2f} lr={optimizer.param_groups[0]['lr']:.4f} {time.time()-t0:.1f}s")
    torch.jit.script(model.eval()).save(os.path.join(CKPT_DIR,f"ckpt_epoch_{epoch:04d}.pt"))

shutil.copy(os.path.join(CKPT_DIR,f"ckpt_epoch_{EPOCHS:04d}.pt"),FINAL_CKPT)
wall=time.time()-start
print(f"Done wall={wall:.1f}s best={best*100:.2f}")
with open("/workspace/agent_reported_metrics.json","w") as f: json.dump({"accuracy":best,"wall_clock_sec":wall},f)
m=torch.jit.load(FINAL_CKPT);print("load ok",m(torch.randn(2,3,32,32,device=device)).shape)
